# -*- coding: utf-8 -*-
# Author: Qiuhao shu
# License: TDG-Attribution-NonCommercial-NoDistrib
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from opencood.models.point_pillar_range_comm_fusion import PointPillarRangeCommFusion
from .utils import regroup


class PointPillarRangeCommFusionDA(PointPillarRangeCommFusion):
    """
    Range-aware implementation with point pillar backbone.
    """
    def forward(self, data_dict):
        # during testing, the whole code is the same as the original code
        if not self.training:
            return super().forward(data_dict)

        feature = []
        # in domain adaption training, data_dict is a list
        for data_per in data_dict:
            voxel_features = data_per['processed_lidar']['voxel_features']
            voxel_coords = data_per['processed_lidar']['voxel_coords']
            voxel_num_points = data_per['processed_lidar']['voxel_num_points']
            record_len = data_per['record_len']
            # spatial_correction_matrix = data_per['spatial_correction_matrix']
            distance_to_ego = data_per["distance_to_ego"]
                   # B, max_cav, 3(dt dv infra), 1, 1
            prior_encoding =\
                        data_per['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
            prior_encoding_stack = []
            for batch,rlen in enumerate(record_len):
                prior_encoding_stack.append(prior_encoding[batch][:rlen])
            prior_encoding_stack = torch.cat(prior_encoding_stack,dim=0) # (sum(cavvs),3,1,1)
            batch_dict = {'voxel_features': voxel_features,
                          'voxel_coords': voxel_coords,
                          'voxel_num_points': voxel_num_points,
                          'record_len': record_len}
            # n, 4 -> n, c
            batch_dict = self.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)

            spatial_features_2d = batch_dict['spatial_features_2d']
             # downsample feature to reduce memory
                        # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.compressor(spatial_features_2d)
            # communication
            prior_encoding_stack = prior_encoding_stack.repeat(1,1,spatial_features_2d.shape[2],spatial_features_2d.shape[3])
            

            if self.multi_scale:
                fused_feature, communication_rates, communication_feat = self.comm(
                    batch_dict["spatial_features"],
                    record_len=record_len,
                    backbone=self.backbone,
                    distance=distance_to_ego,prior_encoding=prior_encoding
                )
                # downsample feature to reduce memory
                # if self.shrink_flag:
                #     fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates, communication_feat = self.comm(
                    spatial_features_2d, record_len=record_len, distance=distance_to_ego,prior_encoding_stack=prior_encoding_stack
                )
                # downsample feature to reduce memory
                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            ####[source_feature, target_feature]####
            feature.append(fused_feature)

        output_dict = {'psm': self.cls_head(feature[0]),  # source_feature
                       'rm': self.reg_head(feature[0]),  # source_feature
                       'source_feature': feature[0],  # source_feature
                       'target_feature': feature[1],  # target_feature
                       'target_psm': self.cls_head(feature[1]),
                       # target_feature
                       'target_rm': self.reg_head(feature[1])  # target_feature
                       }

        return output_dict

 
