import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.fuse_modules.range_attn_fusion import RangeAttentionFusion
from opencood.models.fuse_modules.range_attn_light import (
    RangeAttentionFusion as lightRangeAttentionFusion,
)
from opencood.models.sub_modules.multi_comm import MultiRangeComm


class RangeComm(nn.Module):
    def __init__(
        self,
        H=None,
        W=None,
        ratio: float = 0.9,
        multi_scale=True,
        input_dim: int = 384,
        args=None,
        fusion_args=None,
        cls_head=None,
        with_comm=False,
        with_prior=True
    ) -> None:
        super().__init__()
        assert cls_head is not None, "cls_head must be provided"
        self.multi_scale = multi_scale
        self.with_prior = with_prior
        self.lam = nn.Parameter(torch.tensor(ratio))
        self.with_comm = with_comm
        if H is not None and W is not None:
            self.fc = nn.Linear(W, H)  # [W,H]
        self.fusion_model = nn.ModuleList()

        self.fusion_model = RangeAttentionFusion(
        model_cfg=fusion_args, input_channels=input_dim  #TODO: add prior encoding
    )

    @DeprecationWarning
    def generate_communication_map(self, conf_map):
        # params conf_map: torch.Tensor (V,H,W).
        req_map = 1 - conf_map  # [V,H,W]
        request = req_map[0].unsqueeze(0)  # M~j->i~ R~i~ # [1,H,W]

        diff_map = conf_map * request  # [V,H,W]
        sim_map = self.get_sim(conf_map, conf_map)  # [V,H,W]
        communication_map = self.lam * diff_map + (1 - self.lam) * sim_map  # [V,H,W]
        return communication_map

    @DeprecationWarning
    def get_sim(self, input_feat, target_feat, type=0):
        # find similarity between two agents
        # [V,H,W]
        if type == 0:
            tf = self.fc(target_feat)
            sim = torch.matmul(tf, input_feat)  # [V,H,W]
        elif type == 1:
            sim = input_feat * target_feat
        else:
            sim = torch.matmul(input_feat, target_feat.transpose(1, 2))

        return sim

    @DeprecationWarning
    def naive_communicaton(self, x, record_len, cls_head):
        B = len(record_len)
        _, _, H, W = x.shape
        communication_masks = []
        communication_rates = []
        conf_map = cls_head(x)
        # conf_map = cls_head(x) # [B,anchor_num,H, W]
        # if idx == self.layer_num - 1:
        #     last_conf_map = conf_map
        batch_confidence_maps = self.regroup(
            conf_map, record_len
        )  # [(V,anchor_num,H, W),...]
        for batch in range(B):
            V = record_len[batch]
            confidence_map, _ = (
                batch_confidence_maps[batch].sigmoid().max(dim=1, keepdim=True)
            )  # confidence map
            confidence_map = confidence_map.squeeze()  # [V,H, W]
            if confidence_map.ndim == 2:
                confidence_map = confidence_map.unsqueeze(0)
            communication_maps = self.generate_communication_map(confidence_map).to(
                confidence_map.device
            )  # [V, H, W]
            # generate mask for each batch
            if self.training:
                K = int(H * W * random.uniform(0.1, 1))
                communication_maps = communication_maps.reshape(-1, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(
                    communication_maps.device
                )
                ones_fill = torch.ones(
                    V,
                    K,
                    dtype=communication_maps.dtype,
                    device=communication_maps.device,
                )
                communication_mask = torch.scatter(
                    communication_mask, -1, indices, ones_fill
                ).reshape(V, 1, H, W)
            elif self.threshold:
                print(self.threshold)
                ones_mask = torch.ones_like(communication_maps).to(
                    communication_maps.device
                )
                zeros_mask = torch.zeros_like(communication_maps).to(
                    communication_maps.device
                )
                communication_mask = torch.where(
                    communication_maps > self.threshold, ones_mask, zeros_mask
                ).reshape(V, 1, H, W)
            else:
                communication_mask = torch.ones_like(communication_maps).to(
                    communication_maps.device
                )

            communication_rate = communication_mask.sum() / (V * H * W)
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        if x.shape[-1] != communication_masks.shape[-1]:
            communication_masks = F.interpolate(
                communication_masks,
                size=(x.shape[-2], x.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        return communication_masks, communication_rates

    def forward(
        self,
        x: torch.Tensor,
        record_len,
        backbone=None,
        distance=None,
        prior_encoding_stack=None
    ):
        communication_rates = None
        # params x: torch.Tensor (sum(n_cav), C, H, W). -> [B,L,C,H,W] with_prior
        # params conf_map: torch.Tensor (H, W).
        B = len(record_len)
        communication_feat = x
        if prior_encoding_stack is not None and self.with_prior:
            # prior encoding added
            x = torch.cat([x, prior_encoding_stack], dim=1)
        x_fuse = self.fusion_model(x, record_len, distance)
        return (
            x_fuse,
            communication_rates if self.with_comm else 1,
            communication_feat,
        )

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
