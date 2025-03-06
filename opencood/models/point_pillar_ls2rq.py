import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.stack_s2rq_fusion import StackTrans
from opencood.models.communication_modules.point_pillar_vq import NaiveCommVQ


class PointPillarLS2RQ(nn.Module):
    def __init__(self, args):
        super().__init__()
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])

        self.compression = False
        if args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])
        self.freeze_comm = args['comm_vq']['freeze_comm']

        self.comm_net = NaiveCommVQ(args["comm_vq"]) if not self.freeze_comm else None
        self.cls_head = nn.Conv2d(128 * 3, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args["anchor_number"], kernel_size=1)
        self.max_cav = args["max_cav"] if 'max_cav' in args else 5
        self.fusion_net = StackTrans(384, args['t_v2x_fusion'], self.max_cav)
        self.freeze_fusion = args['t_v2x_fusion']['freeze_fusion'] if 'freeze_fusion' in args['t_v2x_fusion'] else True
        if self.freeze_fusion:
            for param in self.fusion_net.parameters():
                param.requires_grad = False

    def forward(self, data_dict,keep_grad=False):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding = \
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        extra_loss = None
        if not self.freeze_comm:
            spatial_features_2d.requires_grad_()
            spatial_features_2d.retain_grad()
            extra_loss, comm_features_2d,x_hats = self.comm_net(spatial_features_2d, record_len)


        # transformer fusion
        fused_feature = self.fusion_net(spatial_features_2d, record_len=record_len, prior_encoding=prior_encoding,
                                        domain_adaptation=False)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {'psm': psm,
                       'rm': rm}

        output_dict['before_comm_feat'] = spatial_features_2d
        output_dict['comm_map'] = spatial_features_2d
        output_dict['x_hats'] = x_hats

        if extra_loss is not None:
            if not isinstance(extra_loss, torch.Tensor):
                if isinstance(extra_loss, list):
                    output_dict['extra_loss'] = self.lamb * sum(extra_loss)
                elif isinstance(extra_loss, dict):
                    for k, v in extra_loss.items():
                        if isinstance(v, list):
                            extra_loss[k] = self.lamb * sum(v)
                        elif not isinstance(v, torch.Tensor):
                            raise ValueError('Unrecognized type of extra loss {}. v is {}', type(v), v)
                    output_dict['extra_loss'] = extra_loss
                else:
                    raise ValueError('Unrecognized type of extra loss', type(extra_loss))
        return output_dict