
import torch
import torch.nn as nn
from einops import rearrange, repeat
from .utils import regroup


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.point_pillar_intermediate import PointPillarIntermediate as PointePillarIntermediateBase
class PointPillarIntermediateDA(PointePillarIntermediateBase):
    
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
            ####[source_feature, target_feature]####
            feature.append(spatial_features_2d)

        output_dict = {'psm': self.cls_head(feature[0]),  # source_feature
                       'rm': self.reg_head(feature[0]),  # source_feature
                       'source_feature': feature[0],  # source_feature
                       'target_feature': feature[1],  # target_feature
                       'target_psm': self.cls_head(feature[1]),
                       # target_feature
                       'target_rm': self.reg_head(feature[1])  # target_feature
                       }

        return output_dict