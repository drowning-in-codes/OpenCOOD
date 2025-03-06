import argparse


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from collections import OrderedDict
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.visualization import simple_vis
from torch.utils.data import  DataLoader, Sampler

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
    return todevice(ouput_dict, DEVICE)


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())
    # return torch.log(image)/torch.log(image.max())


def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()
    # we want the gradient of the input x
    x.requires_grad_()
    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    # saliencies = x.grad.abs().detach().cpu()
    saliencies, _ = torch.max(x.grad.data.abs().detach().cpu(), dim=1)

    # We need to normalize each image, because their gradients might vary in scale
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies


def smooth_grad(x, y, model,loss_func, epoch, param_sigma_multiplier):
  model.eval()
  #x = x.cuda().unsqueeze(0)

  mean = 0
  sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
  smooth = np.zeros(x.cuda().unsqueeze(0).size())
  for i in range(epoch):
    # call Variable to generate random noise
    noise = torch.normal(mean,sigma*2,x.size(),device=x.device,requires_grad=True)
    x_mod = (x+noise).unsqueeze(0).cuda()
    x_mod.requires_grad_()

    y_pred = model(x_mod)
    loss = loss_func(y_pred, y.cuda().unsqueeze(0))
    loss.backward()

    # like the method in saliency map
    smooth += x_mod.grad.abs().detach().cpu().data.numpy()
  smooth = normalize(smooth / epoch) # don't forget to normalize
  # smooth = smooth / epoch # try this line to answer the question
  return smooth


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(opt.model_dir, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    print('---------------Creating Model------------------')
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
    # model.eval()

    print('-----------------Dataset Building------------------')
    count = 1455
    step = 1
    print('Dataset Building')
    # 925
    start_index = 1
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print('Dataset Built:length:', len(opencood_dataset), "dataset type:", hypes["validate_dir"])
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             sampler=RangeSampler(start_index, start_index + count, step),
                             pin_memory=False,
                             drop_last=False)
    vis_save_path = "./explain_saliency_{}.png"
    criterion = train_utils.create_loss(hypes)
    for i, batch_data in enumerate(data_loader, start=start_index):
        batch_data = train_utils.to_device(batch_data, device)
        cav_content = batch_data['ego']
        ego_lidar_poses = cav_content["ego_lidar_poses"]
        self_id_list = cav_content["self_id_list"]
        print(self_id_list)
        if len(self_id_list[0]) < 2:
            continue
        #  只取两个
        output_dict = OrderedDict()
        final_loss = torch.zeros(1).to(device)
        for _ in range(1):
            output_dict['ego'] = model(cav_content, keep_grad=True)
        # define the loss
            loss = criterion(output_dict['ego'],
                                   cav_content['label_dict'])
            print(loss.item())
            final_loss += loss
        final_loss.backward()
        decoded_feat = output_dict["ego"]['comm_map']
        # feat = output_dict["ego"]['before_comm_feat']
        feat = output_dict["ego"]['x_hats']
        print("decoded_feat", decoded_feat.shape,'feat',feat.shape)
        # saliencies = x.grad.abs().detach().cpu()
        # print(decoded_feat.grad.data)
        saliencies = torch.mean(decoded_feat.grad.data.abs().cpu(), dim=1)
        ego_feat = torch.sum(feat.data.abs().detach().cpu(), dim=1)
        pred_box_tensor, pred_score, gt_box_tensor = \
            opencood_dataset.post_process(batch_data,
                                                   output_dict)
        # We need to normalize each image, because their gradients might vary in scale
        saliencies = torch.stack([normalize(item) for item in saliencies])
        print(saliencies.shape)
        # plt.imshow(saliencies[0])
        # plt.show()
        # plt.imshow(saliencies[1])
        # plt.show()

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
                                       ego_label=self_id_list[0],
                                       )

        bev_map = bev_map / 255.0
        cv2.imwrite(vis_save_path.format(i), bev_map)
        # images
        # saliencies = compute_saliency_maps(images, labels, model)
        plt.close()
        # # visualize
        ego_saliencies = saliencies[0].detach().cpu().numpy()[::-1, :]
        agent_saliencies = saliencies[-1].detach().cpu().numpy()[::-1, :]

        plt.imshow(bev_map, vmin=0, vmax=1)
        plt.axis("off")
        plt.show()
        ego_saliencies = cv2.resize(ego_saliencies, (bev_map.shape[1], bev_map.shape[0]))
        plt.imshow(ego_saliencies, cmap="hot", vmin=0, vmax=1)
        plt.axis('off')

        plt.colorbar()

        plt.show()

        plt.imshow(ego_feat[0].detach().cpu().numpy()[::-1, :])
        plt.axis('off')
        plt.show()

        plt.imshow(ego_feat[-1].detach().cpu().numpy()[::-1, :])
        plt.axis('off')
        plt.show()

        # axes[1].imshow(saliencies[1], cmap="hot")
        # axes[1].axis('off')
        agent_saliencies = cv2.resize(agent_saliencies, (bev_map.shape[1], bev_map.shape[0]))
        plt.imshow(agent_saliencies, cmap="hot", vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        plt.show()

        # fig.colorbar(pcm, ax=axes[1:3])
        # axes[1].imshow(bev_map, vmin=0, vmax=1)
        # axes[1].axis('off')
        # ego_saliencies = np.repeat(ego_saliencies[:, :, np.newaxis], 3, axis=2)
        # ego_saliencies[:, :, 1] = 0

        # heat_map = bev_map + ego_saliencies
        # heat_map = np.clip(heat_map, 0, 1)
        # normalize
        # axes[2].imshow(heat_map, vmin=0, vmax=1)
        # axes[2].axis('off')
        plt.show()


if __name__ == '__main__':
    main()
