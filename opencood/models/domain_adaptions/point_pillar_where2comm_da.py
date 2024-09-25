import torch
import torch.nn as nn
from .utils import regroup
from einops import rearrange, repeat

from opencood.models.point_pillar_where2comm import (
    PointPillarWhere2comm as PointPillarWhere2commBase,
)


class PointPillarWhere2commDA(PointPillarWhere2commBase):

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
            record_len = data_per["record_len"]
            pairwise_t_matrix = data_per["pairwise_t_matrix"]

            batch_dict = {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
                "record_len": record_len,
            }
                # n, 4 -> n, c
            batch_dict = self.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)

            # N, C, H', W': [N, 256, 48, 176]
            spatial_features_2d = batch_dict["spatial_features_2d"]
            # Down-sample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)

            psm_single = self.cls_head(spatial_features_2d)

            # Compressor
            if self.compression:
                # The ego feature is also compressed
                spatial_features_2d = self.naive_compressor(spatial_features_2d)

            if self.multi_scale:
                # Bypass communication cost, communicate at high resolution, neither shrink nor compress
                fused_feature, communication_rates = self.fusion_net(
                    batch_dict["spatial_features"],
                    psm_single,
                    record_len,
                    pairwise_t_matrix,
                    self.backbone,
                )
                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates = self.fusion_net(
                    spatial_features_2d, psm_single, record_len, pairwise_t_matrix
                )
            ####[source_feature, target_feature]####
            feature.append(fused_feature)

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
