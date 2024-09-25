import torch
import torch.nn as nn
from einops import rearrange, repeat
from opencood.models.fuse_modules.fuse_utils import regroup

from opencood.models.fuse_modules.coalign_fuse import Att_w_Warp, normalize_pairwise_tfm
from opencood.models.point_pillar_coalign import (
    PointPillarCoAlign as PointPillarCoAlignBase,
)


class PointPillarCoAlignDA(PointPillarCoAlignBase):

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
            # get affine matrix for feature warping
            _, _, H0, W0 = batch_dict[
                "spatial_features"
            ].shape  # original feature map shape H0, W0.
            normalized_affine_matrix = normalize_pairwise_tfm(
                data_per["pairwise_t_matrix"], H0, W0, self.voxel_size[0]
            )

            spatial_features = batch_dict["spatial_features"]
            # multiscale fusion
            # The first scale feature 'feature_list[0]' for transmission. Default 100*352*64*32/1000000 = 72.0896
            feature_list = []
            feature_list.append(
                self.backbone.get_layer_i_feature(spatial_features, layer_i=0)
            )
            if self.compression:
                feature_list[0] = self.naive_compressor(feature_list[0])
            for i in range(1, self.backbone.num_levels):
                feature_list.append(
                    self.backbone.get_layer_i_feature(feature_list[i - 1], layer_i=i)
                )

            fused_feature_list = []
            for i, fuse_module in enumerate(self.fusion_net):
                fused_feature_list.append(
                    fuse_module(feature_list[i], record_len, normalized_affine_matrix)
                )
            fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

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
    
    