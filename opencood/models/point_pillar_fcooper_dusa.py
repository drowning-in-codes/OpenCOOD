import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.fuse_utils import extract_ego
from opencood.models.da_modules.dusa_classifier import DAImageHead, DAClasHead



class PointPillarFCooper(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.max_cav = args['max_cav']
        self.pillar_vfe = PillarVFE(args["pillar_vfe"],num_point_features=4,
                                    voxel_size=args['voxel_size'],point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'],64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        spatial_map_xs = torch.linspace(args['lidar_range'][0],args['lidar_range'][3],256)
        spatial_map_ys = torch.linspace(args['lidar_range'][1],args['lidar_range'][4],96)

        x, y = torch.meshgrid(spatial_map_xs, spatial_map_ys,indexing='ij')
        spatial_map_tmp = [y, x]
        spatial_map_tmp = torch.stack(spatial_map_tmp,dim=0)
        spatial_map = spatial_map_tmp.unsqueeze(0)
        spatial_map_tmp = spatial_map_tmp.abs()






















