import argparse
import os
import random
import statistics

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from collections import OrderedDict
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.visualization import simple_vis
from opencood.utils.box_utils import boxes_to_corners_3d
import torch.nn.functional as F
import itertools
from opencood.visualization import simple_vis
from torch.utils.data import SequentialSampler, DataLoader, Sampler
from opencood.utils import common_utils
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')

    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_n', type=int, required=False, default=10,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--save_start', type=int, required=False, default=0,
                        help='save start number of visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--eval_epoch', type=str, default=None,
                        help='Set the checkpoint')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Communication confidence threshold')
    opt = parser.parse_args()
    return opt


def todevice(batch_data, device):
    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(device)
    elif isinstance(batch_data, list):
        return [todevice(data, device) for data in batch_data]
    elif isinstance(batch_data, dict):
        return {k: todevice(k, device) for k, v in batch_data.items()}
    else:
        raise NotImplementedError("Not support data type %s" % type(batch_data))


def predict(input, model):
    model.eval()
    ouput_dict = model(input)
    return todevice(ouput_dict, device=DEVICE)


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


class RangeSampler(Sampler):
    r"""Samples elements sequentially from a given range, always in the same order.
    Arguments:
        start (int): The starting index of the range.
        end (int): The ending index of the range (exclusive).
    """

    def __init__(self, start, end, step=1):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step

    def __iter__(self):
        return iter(range(self.start, self.end, self.step))

    def __len__(self):
        return len(range(self.start, self.end, self.step))


def apply_gaussian_kernel(bbxs, comm_map, offset=0, pred_scores=None, normalize=True, fake=False):
    H, W = comm_map.shape
    if pred_scores is None:
        pred_scores = [1] * len(bbxs)
    for bbx, pred_score in zip(bbxs, pred_scores):
        center = np.mean(bbx, axis=0)
        x_center, y_center = center
        # 计算坐标集的边界框
        bbox_min_x = np.min(bbx[:, 0])
        bbox_max_x = np.max(bbx[:, 0])
        bbox_min_y = np.min(bbx[:, 1])
        bbox_max_y = np.max(bbx[:, 1])
        # 计算宽度和高度
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        # 使用宽度或高度来确定高斯分布的标准差
        sigma = max(bbox_width, bbox_height)  # 半径考虑是给出的坐标的宽或高的一半

        # 可以选择在中心点或稍微偏离的地方应用高斯分布
        x_center += offset
        y_center += offset

        # 高斯分布
        range_x = np.arange(bbox_min_x, bbox_max_x)
        range_y = np.arange(bbox_min_y, bbox_max_y)
        r_y, r_x = np.meshgrid(range_y, range_x)
        distances = np.sqrt((r_x - x_center) ** 2 + (r_y - y_center) ** 2)
        # 归一化距离，确保最大值为1
        normalized_distances = 1 - distances / np.max(distances)
        gaussian_kernel = np.zeros((H, W))
        pred_score = pred_score.detach().item()
        # gaussian_kernel[bbox_min_x:bbox_max_x,bbox_min_y:bbox_max_y] = math.random(0.9,1)
        if fake:
            gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = np.exp(
                -((r_x - x_center) ** 2 + (r_y - y_center) ** 2) / (2 * sigma ** 2)) + 0.47 + abs(
                pred_score - 0.47) * random.random()
        else:
            if pred_score < 0.5:
                gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = 0
            elif pred_score < 0.6:
                gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = comm_map[bbox_min_x:bbox_max_x,
                                                                                bbox_min_y:bbox_max_y] * pred_score * 0.01
            else:
                gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = normalized_distances * pred_score * 0.3
            radius = min(bbox_width, bbox_height) // 2
            mask = ((r_x - x_center) >= radius) & ((r_y - y_center) >= radius)
            gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y][mask] = 0
        # 应用到注意力矩阵
        comm_map += gaussian_kernel

    # 归一化
    if normalize:
        comm_map = (comm_map - np.min(comm_map)) / (
                np.max(comm_map) - np.min(comm_map))
    return comm_map


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'intermediate_with_comm', 'no']

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre

    # assert "test" in hypes['validate_dir']
    left_hand = True if "OPV2V" in hypes['validate_dir'] else False
    print(f"Left hand visualizing: {left_hand}")

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print('Dataset Built:length:', len(opencood_dataset), "dataset type:", hypes["validate_dir"])
    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    if opt.eval_epoch is not None:
        epoch_id = opt.eval_epoch
        epoch_id, model = train_utils.load_saved_model(saved_path, model, epoch_id)
    else:
        epoch_id, model = train_utils.load_saved_model(saved_path, model)

    model.eval()
    # count = opt.save_vis_n
    step = 1
    count = 10
    start_index = 1510
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             sampler=RangeSampler(start_index, start_index + count, step),
                             drop_last=False)
    print('explain start')
    # used to help schedule learning rate

    vis_save_path = "./explain_comm_{}.png"
    for i, batch_data in enumerate(data_loader, start=start_index):
        count += 1
        batch_data = train_utils.to_device(batch_data, device)
        cav_content = batch_data['ego']
        ego_lidar_poses = cav_content["ego_lidar_poses"]
        self_id_list = cav_content["self_id_list"]
        print(len(self_id_list[0]))
        if len(self_id_list[0]) != 2:
            continue
        output_dict = OrderedDict()
        output_dict['ego'] = model(cav_content)
        pred_box_tensor, pred_score, gt_box_tensor = \
            opencood_dataset.post_process(batch_data,
                                          output_dict)
        comm_map = output_dict['ego']["comm_map"]
        com_rates = output_dict['ego']["com_rates"] if 'com_rates' in output_dict['ego'] else None
        print(com_rates)
        # print(comm_map.shape)
        bev_map = simple_vis.visualize(pred_box_tensor,
                                       gt_box_tensor,
                                       batch_data['ego'][
                                           'origin_lidar'][0],
                                       hypes['postprocess']['anchor_args']['cav_lidar_range'],
                                       vis_save_path.format(i),
                                       method='bev',
                                       left_hand=True if "OPV2V" in hypes['validate_dir'] else False,
                                       vis_pred_box=False if pred_box_tensor is None else True,
                                       ego_lidar_pose=ego_lidar_poses,
                                       vis_ego_box=True,
                                       return_map=True,
                                       ego_label=self_id_list[0])
        plt.close()
        bev_map = bev_map / 255.0
        plt.imshow(bev_map, vmax=1, vmin=0)
        plt.axis("off")
        plt.show()
        # fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        # print(len(comm_map))
        # axes[0].imshow(comm_map[-1][-1].detach().cpu().numpy(), vmin=0, vmax=1)
        # axes[0].axis('off')
        # axes[1].imshow(comm_map[-1][0].detach().cpu().numpy(), vmin=0, vmax=1)
        # axes[1].axis('off')
        # plt.show()

        gt_box = pred_box_tensor[:, :4, :2]
        if gt_box is not None and not isinstance(gt_box, np.ndarray):
            gt_box = common_utils.torch_tensor_to_numpy(gt_box)
        L1, W1, H1, L2, W2, H2 = opencood_dataset.params["preprocess"]["cav_lidar_range"]
        bev_origin = np.array([L1, W1]).reshape(1, -1)
        ratio = opencood_dataset.params["preprocess"]["args"]["res"] if "res" in \
                                                                        opencood_dataset.params["preprocess"][
                                                                            "args"] else 0.1

        comm_map_matrix = comm_map[-1][-1].detach().cpu().numpy()[::-1]
        comm_map_matrix = cv2.resize(comm_map_matrix, bev_map.shape[:2][::-1])
        bbxs = []
        for j in range(gt_box.shape[0]):
            bbx = gt_box[j][:4, :2]
            bbx = ((bbx - bev_origin) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            bbxs.append(bbx)
        # comm_map_matrix = apply_gaussian_kernel(bbxs, comm_map=comm_map_matrix, pred_scores=pred_score)
        plt.imshow(comm_map_matrix)
        plt.axis("off")
        plt.colorbar()
        plt.show()

        if len(self_id_list[0]) >= 3:
            comm_map_matrix = comm_map[-1][-2].detach().cpu().numpy()[::-1]
            comm_map_matrix = cv2.resize(comm_map_matrix, bev_map.shape[:2][::-1])
            plt.imshow(comm_map_matrix)
            plt.axis("off")
            plt.colorbar()
            plt.show()

        comm_map_matrix = comm_map[-1][0].detach().cpu().numpy()[::-1]
        comm_map_matrix = cv2.resize(comm_map_matrix, bev_map.shape[:2][::-1])
        plt.imshow(comm_map_matrix)
        plt.axis("off")
        plt.colorbar()
        plt.show()
        # comm_map_mask = comm_map_matrix > 0.1
        # plt.imshow(comm_map_mask)
        # plt.axis("off")
        # plt.colorbar()
        # plt.show()


if __name__ == '__main__':
    main()
