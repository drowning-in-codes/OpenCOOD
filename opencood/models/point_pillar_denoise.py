# -*- coding: utf-8 -*-
# Author: Qiuhao shu 
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.denoiseModel import MultiDenoiseModel,DenoiseModel
from opencood.models.fuse_modules.range_attn_fusion import RangeAttentionFusion
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.sub_modules.covisformer import CoVisFormer
from torch.nn import functional as F
from einops import repeat
from opencood.models.fuse_modules.fuse_utils import regroup,splitgroup


class PointPillarDenoise(nn.Module):
    """
    Range-aware implementation with point pillar backbone.
    """
    def __init__(self, args):
        super().__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            # H W 200 704 and feature stride is 2. hardcode here
            # FIXME: hardcode
            self.compressor = CoVisFormer(num_vehicles=5,feature_height=200//2,feature_width=704//2,channels=args['shrink_header']['dim'][-1],attention_mode="bias")
                
        
        output_fusion_feature_dim = args['head_dim']
        self.fusion_net = SpatialFusion()

        self.cls_head = nn.Conv2d(output_fusion_feature_dim, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(output_fusion_feature_dim, 7 * args['anchor_number'],
                                  kernel_size=1)


        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        distance_to_ego = data_dict['distance_to_ego']


        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      }
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        _,_,H,W = spatial_features_2d.shape
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        origin_feat = spatial_features_2d
        # compressor
        if self.compression:
             # N, C, H, W -> B, L, C, H, W
            regroup_feature, mask = regroup(spatial_features_2d,
                                            record_len,
                                            self.max_cav)

            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            com_mask = repeat(com_mask,
                            'b h w c l -> b (h new_h) (w new_w) c l', # [B,1,1,1,5] -> [B,H,W,1,5]
                            new_h=regroup_feature.shape[3],
                            new_w=regroup_feature.shape[4])
            # regroup_feature: [B, L, C, H, W]. B is batch_size,L is the max_cav
            spatial_features_2d = self.compressor(regroup_feature,distances=distance_to_ego,mask=com_mask)
            spatial_features_2d = splitgroup(spatial_features_2d,record_len)
        # spatial_features_2d = F.interpolate(spatial_features_2d,size=(H,W),mode='bilinear')
        fused_feature = self.fusion_net(spatial_features_2d,record_len)
        batch_dict['spatial_features_2d'] = fused_feature
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {'psm': psm,
                       'rm': rm,
                    #    'spatial_features_2d_downsample':spatial_features_2d,
                    #    'spatial_features_2d_origin':origin_feat
                    }

        return output_dict
