# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import numpy as np

from einops import rearrange
from opencood.utils.common_utils import torch_tensor_to_numpy

def splitgroup(regroup_feature, record_len):
    """
    Split the regroup_feature into a list of features based on the record_len.
    :param regroup_feature: 
    :param record_len: 
    :return: 
    """
    # [B,L,C,H,W] -> [N,C,H,W]
    assert regroup_feature.ndim == 5, "The dimension of regroup_feature should be 5."
    B,L,C,H,W = regroup_feature.shape
    dense_feature = []
    for batch_feature in regroup_feature:
        # [L,C,H,W] -> [N,C,H,W]
        batch_feature = batch_feature[:record_len[0]]
        dense_feature.append(batch_feature) 
    dense_feature = torch.cat(dense_feature,dim=0)
    return dense_feature


def regroup(dense_feature, record_len, max_len,fill_val=None):
    """
    Regroup the data based on the record_len.

    Parameters
    ----------
    dense_feature : torch.Tensor
        N, C, H, W
    record_len : list
        [sample1_len, sample2_len, ...]
    max_len : int
        Maximum cav number

    Returns
    -------
    regroup_feature : torch.Tensor
        B, L, C, H, W
    """
    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    split_features = torch.tensor_split(dense_feature,
                                        cum_sum_len[:-1])
    regroup_features = []
    mask = []

    for split_feature in split_features:
        # M, C, H, W
        feature_shape = split_feature.shape

        # the maximum M is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                     feature_shape[2], feature_shape[3]).fill_(fill_val if fill_val is not None else 0)
        padding_tensor = padding_tensor.to(split_feature.device)

        split_feature = torch.cat([split_feature, padding_tensor],
                                  dim=0)

        # 1, 5C, H, W
        split_feature = split_feature.view(-1,
                                           feature_shape[2],
                                           feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    # B, 5C, H, W
    regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W
    regroup_features = rearrange(regroup_features,
                                 'b (l c) h w -> b l c h w',
                                 l=max_len)
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return regroup_features, mask
