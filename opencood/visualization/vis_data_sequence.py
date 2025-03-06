# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import argparse
import copy

from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
import time
import numpy as np
from opencood.utils import box_utils
from opencood.utils import common_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils
from opencood.data_utils.datasets.early_fusion_vis_dataset import \
    EarlyFusionVisDataset
from opencood.data_utils.datasets.early_fusion_dataset import \
EarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import \
IntermediateFusionDataset
from opencood.visualization.vis_utils import visualize_single_sample_dataloader
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
save_path = "./"
def bbx2linset(bbx_corner, order='hwl', color=(1, 0, 0)):
    """
    Convert the torch tensor bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    line_set : list
        The list containing linsets.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)

    return bbx_linset


def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    oabbs : list
        The list containing all oriented bounding boxes.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)
    oabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)


        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs


def bbx2aabb(bbx_center, order):
    """
    Convert the torch tensor bounding box to o3d aabb for visualization.

    Parameters
    ----------
    bbx_center : torch.Tensor
        shape: (n, 7).

    order: str
        hwl or lwh.

    Returns
    -------
    aabbs : list
        The list containing all o3d.aabb
    """
    if not isinstance(bbx_center, np.ndarray):
        bbx_center = common_utils.torch_tensor_to_numpy(bbx_center)
    bbx_corner = box_utils.boxes_to_corners_3d(bbx_center, order)

    aabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        aabb = tmp_pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 0, 1)
        aabbs.append(aabb)

    return aabbs


def vis_parser():
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument('--color_mode', type=str, default="intensity",
                        help='lidar color rendering mode, e.g. intensity,'
                             'z-value or constant.')
    opt = parser.parse_args()
    return opt

def color_encoding(intensity, mode='intensity',special=False):
    """
    Encode the single-channel intensity to 3 channels rgb color.

    Parameters
    ----------
    intensity : np.ndarray
        Lidar intensity, shape (n,)

    mode : str
        The color rendering mode. intensity, z-value and constant are
        supported.

    Returns
    -------
    color : np.ndarray
        Encoded Lidar color, shape (n, 3)
    """
    assert mode in ['intensity', 'z-value', 'constant']

    if mode == 'intensity':
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        if special:
            if special == "yellow":
                cmap = plt.get_cmap('autumn')
                colors = cmap(np.linspace(0, 1, 256))
                # 创建离散 colormap
                discrete_cmap = ListedColormap(colors)
                VIRIDIS = discrete_cmap.colors
                VID_RANGE =  np.linspace(0.0, 1.0, 256)
                # VIRIDIS = np.array(cm.get_cmap('tab10').colors)
                # VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

            elif special == "red":
                VIRIDIS = np.array(cm.get_cmap('Set1').colors)
                VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
                VIRIDIS[:, 2] = 0
                VIRIDIS[:, 1] = 0
            else:
                VIRIDIS = np.array(cm.get_cmap('Paired').colors)
                VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
                VIRIDIS[:, 0] = 0
                VIRIDIS[:, 2] = 0
        else:
            VIRIDIS = np.array(cm.get_cmap('plasma').colors)
            VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
            # cmap = plt.get_cmap('autumn')
            # colors = cmap(np.linspace(0, 1, 256))
            # # 创建离散 colormap
            # discrete_cmap = ListedColormap(colors)
            # VIRIDIS = discrete_cmap.colors
            # VID_RANGE = np.linspace(0.0, 1.0, 256)
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    elif mode == 'z-value':
        min_value = -1.5
        max_value = 0.5
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(intensity)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5
        int_color = colors[:, :3]

    elif mode == 'constant':
        # regard all point cloud the same color
        int_color = np.ones((intensity.shape[0], 3))
        int_color[:, 0] *= 247 / 255
        int_color[:, 1] *= 244 / 255
        int_color[:, 2] *= 237 / 255

    return int_color

def save_o3d_visualization(element, save_path):
    """
    Save the open3d drawing to folder.

    Parameters
    ----------
    element : list
        List of o3d.geometry objects.

    save_path : str
        The save path.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = 1.0

    for i in range(len(element)):
        vis.add_geometry(element[i])
        vis.update_geometry(element[i])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_path)
    vis.destroy_window()


def lineset_assign(lineset1, lineset2):
    """
    Assign the attributes of lineset2 to lineset1.

    Parameters
    ----------
    lineset1 : open3d.LineSet
    lineset2 : open3d.LineSet

    Returns
    -------
    The lineset1 object with 2's attributes.
    """

    lineset1.points = lineset2.points
    lineset1.lines = lineset2.lines
    lineset1.colors = lineset2.colors

    return lineset1

from torch.utils.data.sampler import Sampler
# 自定义采样器，只返回偶数索引
class EvenIndexSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # 只迭代偶数索引
        return iter([i for i in range(len(self.data_source)) if i % 2 == 0])

    def __len__(self):
        # 返回偶数索引的数量
        return (len(self.data_source) + 1) // 2

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/visualization.yaml'))

    opencda_dataset = EarlyFusionDataset(params, visualize=True,
                                            train=False)
    data_loader = DataLoader(opencda_dataset, batch_size=1,
                             collate_fn=opencda_dataset.collate_batch_train,
                             shuffle=True,pin_memory=False
                             )
    opt = vis_parser()
    mode = opt.color_mode
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis_opt = vis.get_render_option()
    vis_opt.background_color = np.array([1.0, 1.0, 1.0])
    vis_opt.point_size = 10.0
    # used to visualize lidar points
    vis_pcd = o3d.geometry.PointCloud()
    # used to visualize object bounding box, maximum 50
    vis_aabbs = []
    order = 'hwl'
    for _ in range(50):
        vis_aabbs.append(o3d.geometry.LineSet())
    while True:
        for i_batch, batched in enumerate(data_loader):

            batch_data = batched['ego']
            print(batch_data["cav_id_list"][0])
            if '-1' not in batch_data["cav_id_list"][0]:
                continue
            split_lidar = batch_data['split_lidar'] # np.narray [[lidar_num,lidar_dim],[],...]
            if  isinstance(split_lidar, list) and not isinstance(split_lidar[-1],np.ndarray):
                split_lidar = common_utils.torch_tensor_to_numpy(split_lidar)
            if isinstance(split_lidar,np.ndarray):
                split_lidar = common_utils.torch_tensor_to_numpy(split_lidar)
            if len(split_lidar) != 3:
                continue
            print(len(split_lidar))
            # we only visualize the first cav for single sample
            for i_cav,cav_lidar in enumerate(split_lidar):
                if i_cav == 0:
                    special = "yellow"
                elif i_cav == 1:
                    special = "red"
                else:
                    special = "green"
                origin_lidar_intcolor = \
                    color_encoding(cav_lidar[:, -1] if mode == 'intensity'
                                   else cav_lidar[:, 2], mode=mode,special=special)
                # left -> right hand
                cav_lidar[:, :1] = -cav_lidar[:, :1]

                vis_pcd.points = o3d.utility.Vector3dVector(cav_lidar[:, :3])
                vis_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

                object_bbx_center = batch_data['object_bbx_center']
                object_bbx_mask = batch_data['object_bbx_mask']
                object_bbx_center = object_bbx_center[object_bbx_mask == 1]
                oabb =  False
                aabbs = bbx2linset(object_bbx_center, order) if not oabb else \
                    bbx2oabb(object_bbx_center, order)
                visualize_elements = [vis_pcd] # + aabbs
                #  visualize
                o3d.visualization.draw_geometries(visualize_elements)
                # save
                save_o3d_visualization(visualize_elements, f"./{i_cav}.jpg") # 保存一个场景下某辆车的数据和gt

                if i_batch == 0:
                    vis.add_geometry(vis_pcd)
                    for i in range(len(vis_aabbs)):
                        index = i if i < len(aabbs) else -1
                        vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                        vis.add_geometry(vis_aabbs[i])

                for i in range(len(vis_aabbs)):
                    index = i if i < len(aabbs) else -1
                    vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                    vis.update_geometry(vis_aabbs[i])

                vis.update_geometry(vis_pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

            points = np.array(batched["ego"]["origin_lidar"][0]) # ego cav
            object_bbx_center = np.array(batched["ego"]['object_bbx_center'])
            object_bbx_mask = np.array(batched["ego"]['object_bbx_mask'])
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]
            oabb = False
            aabbs = bbx2linset(object_bbx_center, order="hwl") if not oabb else \
                bbx2oabb(object_bbx_center, order="hwl")
            vis_pcd.points = o3d.utility.Vector3dVector(points[:,:3])
            origin_lidar_intcolor = \
                color_encoding(points[:, -1] if mode == 'intensity'
                               else points[:, 2], mode=mode)
            vis_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)
            visualize_elements = [vis_pcd] + aabbs

            save_o3d_visualization(visualize_elements, f"./gt.jpg")
            break

        break
    vis.destroy_window()
