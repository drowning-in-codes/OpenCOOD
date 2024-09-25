# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
from einops import rearrange, repeat
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from .utils import regroup
from opencood.models.point_pillar_fcooper import PointPillarFCooper as PointPillarFCooperBase


class SpatialFusion(nn.Module):
    def __init__(self):
        super(SpatialFusion, self).__init__()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        # x: N, C, H, W record_len []

        split_x = self.regroup(x, record_len)
        out = []

        for xx in split_x:
            xx = torch.max(xx, dim=0, keepdim=True)[0]
            out.append(xx)
        return torch.cat(out, dim=0)
    
class PointPillarFCooper(PointPillarFCooperBase):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super().__init__(args)
        self.fusion_net = SpatialFusion()
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
            spatial_correction_matrix = data_per['spatial_correction_matrix']

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
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor(
                    spatial_features_2d)

            # N, C, H, W -> B,  L, C, H, W
            # regroup_feature, mask = regroup(spatial_features_2d,
            #                                 record_len,
            #                                 self.max_cav)
            # com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3) # [B, L]-> [B,H,W,1,L]
            # com_mask = repeat(com_mask,
            #                   'b h w c l -> b (h new_h) (w new_w) c l',
            #                   new_h=regroup_feature.shape[3],
            #                   new_w=regroup_feature.shape[4])   #[B, L, C, H, W],[B,H,W,1,L]

            fused_feature = self.fusion_net(spatial_features_2d, record_len) #[N, C, H, W],[B,L]
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
    
    