# Author: proanimer
# Email: <bukalala174@gmail.com>
# License: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.models.sub_modules.denoiseModel import AutoEncoder, DenoiseModel
from opencood.models.fuse_modules.range_fusion_block import RangeAttentionBlock
from opencood.models.sub_modules.base_raa_bev_backbone import BaseRAABEVBackbone
# goals : combine local and global attention,multiscale attention,distance(range) encoding
class RangeAttentionFusion(nn.Module):
    def __init__(self, model_cfg: dict,input_dim: int,feature_len: int = 3):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False
        if 'compression' in model_cfg and model_cfg['compression'] > 0:
            self.compress = True
            self.compress_layer = model_cfg['compression']

        if self.compress:
            self.compression_modules = nn.ModuleList()

        layer_nums = model_cfg['layer_nums']
        num_filters = model_cfg['num_filters']
        self.num_levels = len(layer_nums)
        # feature_pyramid for multi-scale
        if "feature_pyramid" in self.model_cfg:
            self.feature_pyramid = self.model_cfg["feature_pyramid"]
        else:
            self.feature_pyramid = True
        # feature_pyramid for multi-scale
        if self.feature_pyramid :
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.compress and self.compress_layer - idx > 0:
                    self.compression_modules.append(AutoEncoder(num_filters[idx],
                                                            self.compress_layer - idx))
                fuse_network = RangeAttentionBlock(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = RangeAttentionBlock(model_cfg['in_channels'])
     

    def forward(self, spatial_features,record_len,backbone:BaseRAABEVBackbone=None):
        """
        fusion feature
        if use  feature_pyramid, use same backbone as the backbone of the feature generator network
        """
        ups = []
        ret_dict = {}
        x = spatial_features
        if self.feature_pyramid:
            for i in range(self.num_levels):
                # downsample
                x = backbone.blocks[i](x)
                # TODO: add denoise model to denoise the feature
                # compression
                if self.compress and i < len(self.compression_modules):
                    x = self.compression_modules[i](x)
                # fuse feature
                x_fuse = self.fuse_modules[i](x, record_len)
                stride = int(spatial_features.shape[2] / x.shape[2])
                ret_dict['spatial_features_%dx' % stride] = x
                
                # deconv 
                if len(backbone.deblocks) > 0:
                    x = backbone.deblocks[-1](x)
                    ups.append(x)
                else:
                    ups.append(x_fuse)

                if len(ups) > 1:
                    out = torch.cat(ups, dim=1)
                else:
                    out = ups[0]

                if len(backbone.deblocks) > self.num_levels:
                    out = backbone.deblocks[-1](out)
        else:
            out = self.fuse_modules(x, record_len)
        return out

