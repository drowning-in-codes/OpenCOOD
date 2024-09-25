# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.fuse_modules.fuse_utils import regroup

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone

from opencood.models.point_pillar import PointPillar as PointPillarBase


class PointPillarEarlyFusionDA(PointPillarBase):


    def forward(self, data_dict):
        # during testing, the whole code is the same as the original code
        if not self.training:
            return super().forward(data_dict)

        feature = []
        # in domain adaption training, data_dict is a list
        for data_per in data_dict:
            voxel_features = data_per["processed_lidar"]["voxel_features"]
            voxel_coords = data_per["processed_lidar"]["voxel_coords"]
            voxel_num_points = data_per["processed_lidar"]["voxel_num_points"]
            # spatial_correction_matrix = data_per["spatial_correction_matrix"]

            batch_dict = {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
            }
            # n, 4 -> n, c
            batch_dict = self.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)

            spatial_features_2d = batch_dict["spatial_features_2d"]

            feature.append(spatial_features_2d)

        output_dict = {
            "psm": self.cls_head(feature[0]),  # source_feature
            "rm": self.reg_head(feature[0]),  # source_feature
            "source_feature": feature[0],  # source_feature
            "target_feature": feature[1],  # target_feature
            "target_psm": self.cls_head(feature[1]),
            # target_feature
            "target_rm": self.reg_head(feature[1]),  # target_feature
        }

        return output_dict
