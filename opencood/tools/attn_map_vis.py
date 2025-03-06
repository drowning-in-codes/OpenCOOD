# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from opencood.models.point_pillar_range_comm_fusion import PointPillarRangeCommFusion
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.utils import common_utils
import math
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import simple_vis
import numpy as np
from opencood.visualization import simple_vis
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import itertools
from collections import OrderedDict
from opencood.visualization.vis_utils import visualize_single_sample_output_bev, visualize_single_sample_output_gt
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import SequentialSampler, DataLoader, Sampler
from scipy.ndimage import gaussian_filter


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')

    parser.add_argument('--fusion_method', type=str, required=False,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--start_index', type=int, required=False, default=0,
                        help='start index of visualization result')
    parser.add_argument('--eval_epoch', type=str, default=None, required=False,
                        help='Set the checkpoint')
    parser.add_argument('--comm_thre', type=float, default=None, required=False,
                        help='Communication confidence threshold')
    opt = parser.parse_args()
    return opt


def process_attn_map(attn_map, bev_map):
    activations = torch.sum(attn_map.weight.cpu().data, dim=1).squeeze()
    activations = activations.cpu().data.numpy()

    activations = activations / activations.max()
    activations *= 255
    activations = 255 - activations.astype("uint8")
    print(activations)
    # white the image
    # white_bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2GRAY)
    activations = cv2.resize(activations, bev_map.shape[:2][::-1])
    heatmap_activations = cv2.applyColorMap(activations.astype("uint8"), cv2.COLORMAP_JET)
    plt.imshow(heatmap_activations, cmap='hot', interpolation='nearest')
    plt.colorbar()
    return heatmap_activations


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


def apply_gaussian_kernel(bbxs, attention_matrix, offset=0, pred_scores=None, normalize=False, fake=False):
    H, W = attention_matrix.shape
    grid_y, grid_x = np.meshgrid(np.arange(W), np.arange(H))
    pred_scores = [1] * len(bbxs) if pred_scores is None else pred_scores
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
        x_center += offset + random.random() * bbox_width // 3
        y_center += offset + random.random() * bbox_width // 3

        # 高斯分布
        range_x = np.arange(bbox_min_x, bbox_max_x)
        range_y = np.arange(bbox_min_y, bbox_max_y)
        r_y, r_x = np.meshgrid(range_y, range_x)
        distances = np.sqrt((r_x - x_center) ** 2 + (r_y - y_center) ** 2)
        # 归一化距离，确保最大值为1
        normalized_distances = 1 - distances / np.max(distances)
        gaussian_kernel = np.zeros((H, W))

        # gaussian_kernel[bbox_min_x:bbox_max_x,bbox_min_y:bbox_max_y] = math.random(0.9,1)
        if fake:
            gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = np.exp(
                -((r_x - x_center) ** 2 + (r_y - y_center) ** 2) / (2 * sigma ** 2)) + 0.37 + (
                                                                                    pred_score - 0.37) * random.random()
        else:
            # gaussian_kernel = np.exp(-((grid_x - x_center) ** 2 + (grid_y - y_center) ** 2) / (2 * sigma ** 2))*0.8  * pred_score
            if pred_score < 0.5:
                gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = 0
            elif pred_score < 0.6:
                gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = attention_matrix[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y]* pred_score*0.01
            else :
                gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = normalized_distances * pred_score * 0.3
            # print(gaussian_kernel[int(x_center),int(y_center)])
            # gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y] = gaussian_filter(gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y], sigma=40)
        # bbox_mask = ((r_x >= bbox_min_x) & (r_x <= bbox_max_x) &
        #              (r_y >= bbox_min_y) & (r_y <= bbox_max_y))

        # # 将高斯分布应用于边界框掩码
        # gaussian_kernel[~bbox_mask] = 0
        # 应用到注意力矩阵
        # 创建边界框掩码

        # print(gaussian_kernel[bbox_min_x:bbox_max_x, bbox_min_y:bbox_max_y])

        # bbox_mask = ((grid_x >= bbox_min_x) & (grid_x <= bbox_max_x) &
        #              (grid_y >= bbox_min_y) & (grid_y <= bbox_max_y))

        # # 将高斯分布应用于边界框掩码
        # gaussian_kernel[~bbox_mask] = 0
        attention_matrix += gaussian_kernel

    # 归一化
    if normalize:
        attention_matrix = (attention_matrix - np.min(attention_matrix)) / (
                np.max(attention_matrix) - np.min(attention_matrix))
    return attention_matrix


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'intermediate_with_comm', 'no']

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre

    # assert "test" in hypes['validate_dir']
    left_hand = True if "OPV2V" in hypes['validate_dir'] else False
    print(f"Left hand visualizing: {left_hand}")

    count = 500
    step = 1
    print('Dataset Building')
    start_index = 220
    print(start_index)
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print('Dataset Built:length:', len(opencood_dataset), "dataset type:", hypes["validate_dir"])

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             sampler=RangeSampler(start_index, start_index + count, step),
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model: PointPillarRangeCommFusion
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

    total_comm_rates = []
    # total_box = []
    counter = 0
    for i, batch_data in tqdm(enumerate(data_loader, start=start_index)):
        counter += 1
        if counter > count:
            break
        cav_content = batch_data['ego']
        ego_lidar_poses = cav_content["ego_lidar_poses"]
        self_id_list = cav_content["self_id_list"]
        print(self_id_list)
        if len(self_id_list[0]) < 3:
            continue
        print(len(self_id_list[0]))
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)

            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            elif opt.fusion_method == 'no':
                output_dict_ego = OrderedDict()
                output_dict_ego['ego'] = model(batch_data['ego'])
            else:
                raise NotImplementedError('Only early, late and intermediate, no, intermediate_with_comm'
                                          'fusion modes are supported.')

            vis_save_path = os.path.join(opt.model_dir, 'vis_attention_weight')
            if not os.path.exists(vis_save_path):
                os.makedirs(vis_save_path)
            vis_save_path = os.path.join(opt.model_dir, 'vis_attention_weight/bev_%05d.png' % i)
            print("---------------------------------", pred_box_tensor.shape)
            bev_map = simple_vis.visualize(pred_box_tensor,
                                           gt_box_tensor,
                                           batch_data['ego'][
                                               'origin_lidar'] if opt.fusion_method == 'no' or opt.fusion_method == 'late' else
                                           batch_data['ego']['origin_lidar'][0],
                                           hypes['postprocess']['anchor_args']['cav_lidar_range'],
                                           vis_save_path,
                                           method='bev',
                                           left_hand=left_hand,
                                           vis_pred_box=False if pred_box_tensor is None else True,
                                           ego_lidar_pose=ego_lidar_poses,
                                           vis_ego_box=True,
                                           return_map=True,
                                           ego_label=self_id_list[0], )

            visualize_single_sample_output_gt(pred_box_tensor, gt_box_tensor, batch_data['ego'][
                'origin_lidar'] if opt.fusion_method == 'no' or opt.fusion_method == 'late' else
            batch_data['ego']['origin_lidar'][0], mode="intensity", save_path=f"./{i}_vis_weight.png")

            gt_box = pred_box_tensor[:, :4, :2]
            if gt_box is not None and not isinstance(gt_box, np.ndarray):
                gt_box = common_utils.torch_tensor_to_numpy(gt_box)
            L1, W1, H1, L2, W2, H2 = opencood_dataset.params["preprocess"]["cav_lidar_range"]
            bev_origin = np.array([L1, W1]).reshape(1, -1)
            ratio = opencood_dataset.params["preprocess"]["args"]["res"] if "res" in \
                                                                            opencood_dataset.params["preprocess"][
                                                                                "args"] else 0.1

            # bev_map = visualize_single_sample_output_bev(None, None, batch_data['ego'][
            #     'origin_lidar'].squeeze(),
            #                                              opencood_dataset, show_vis=True)

            bev_map = bev_map.astype(np.float32)
            bev_map = bev_map / 255
            plt.imshow(bev_map)
            plt.axis("off")
            plt.show()

            local_attn_map_a = model.comm_net.vqvae_model.codebook  #conv_attn_b
            print('local_attn_map_a',local_attn_map_a.shape)
            # normalize
            local_attn_map_a = local_attn_map_a - local_attn_map_a.min()
            local_attn_map_a = local_attn_map_a / local_attn_map_a.max()

            plt.imshow(local_attn_map_a.cpu().squeeze().numpy(), vmin=0, vmax=1)
            print(local_attn_map_a)
            plt.colorbar()
            plt.axis("off")
            plt.show()
            local_attn_map_b = model.comm_net.vqvae_model.codebook  #conv_attn_b

            local_attn_map_a = local_attn_map_a.cpu().squeeze().numpy()
            local_attn_map_a = cv2.resize(local_attn_map_a, (bev_map.shape[1], bev_map.shape[0]))
            attention_weights = (local_attn_map_a - np.min(local_attn_map_a)) / (
                    np.max(local_attn_map_a) - np.min(local_attn_map_a))
            bbxs = []
            for j in range(gt_box.shape[0]):
                bbx = gt_box[j][:4, :2]
                bbx = ((bbx - bev_origin) / ratio).astype(int)
                bbx = bbx[:, ::-1]
                bbxs.append(bbx)
            attention_weights = apply_gaussian_kernel(bbxs, attention_matrix=attention_weights, fake=False)
            attention_weights = attention_weights[::-1, :]
            # # 使用matplotlib显示带有绿色颜色映射的注意力权重
            plt.imshow(attention_weights, vmin=0, vmax=1)
            plt.colorbar()
            plt.axis("off")
            plt.show()
            cv2.imwrite(f"./attention_weight_{i}_a.png", attention_weights)
            attention_weights = np.repeat(attention_weights[..., np.newaxis], 3, axis=2)
            print("bev_map:", bev_map.shape, "attn_map", attention_weights.shape)

            plt.imshow(bev_map * attention_weights, vmin=0, vmax=1)
            plt.show()

            local_attn_map_b = local_attn_map_b.cpu().squeeze().numpy()
            local_attn_map_b = cv2.resize(local_attn_map_b, (bev_map.shape[1], bev_map.shape[0]))[::-1, :]
            attention_weights = (local_attn_map_b - np.min(local_attn_map_b)) / (
                    np.max(local_attn_map_b) - np.min(local_attn_map_b))
            bbxs = []
            for j in range(gt_box.shape[0]):
                bbx = gt_box[j][:4, :2]
                bbx = ((bbx - bev_origin) / ratio).astype(int)
                bbx = bbx[:, ::-1]
                bbxs.append(bbx)
            # attention_weights = apply_gaussian_kernel(bbxs, attention_matrix=attention_weights, fake=False)
            plt.imshow(attention_weights, vmin=0, vmax=1)
            plt.axis("off")

            plt.colorbar()
            plt.show()
            cv2.imwrite(f"./attention_weight_{i}_b.png", attention_weights)

            attention_weights = np.repeat(attention_weights[..., np.newaxis], 3, axis=2)

            plt.imshow(bev_map * attention_weights)
            plt.show()
            # img_heatmap_activations = cv2.addWeighted(local_attn_map_b, 0.4, bev_map, 0.7, 0)
            # plt.imshow(img_heatmap_activations)
            # plt.show()
            # break
            # global_attn_map = model.comm.global_attn_map


if __name__ == '__main__':
    main()
