# Author: proanimer
# Email: <bukalala174@gmail.com>
# License: MIT

import torch
import torch.nn as nn
import numpy as np
from opencood.models.fuse_modules.range_fusion_block import RangeAttentionBlock



class RangeAttentionFusion(nn.Module):
    def __init__(self,input_channels):
        super().__init__()  
        self.rangeAttnBlock = RangeAttentionBlock(input_channels)
    

    def forward(self,spatial_features,record_len):
        out = self.rangeAttnBlock(spatial_features,record_len)
        return out