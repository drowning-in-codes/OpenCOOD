from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import numpy as np


def linear_map(x, range=[-1.0, 1.0]):
    span = range[1] - range[0]
    k = span / (torch.max(x) - torch.min(x))
    b = range[1] - torch.max(x) * k
    return k * x + b


@DeprecationWarning
class CrossAttentionWithEmbedding(nn.Module):
    def __init__(self, input_dim, heads: int = 3, use_distance=False) -> None:
        super().__init__()
        inner_dim = input_dim * heads
        self.inner_dim = inner_dim
        self.heads = heads
        self.use_distance = use_distance
        if use_distance:
            self.raan = LocalRangeAttentionBlock(input_dim)
        self.to_q = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=inner_dim, kernel_size=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=inner_dim, kernel_size=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=inner_dim, kernel_size=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
        )
        self.W_o = nn.Sequential(
            nn.Conv2d(in_channels=inner_dim, out_channels=input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=inner_dim, out_channels=input_dim, kernel_size=1),
        )

    def forward(self, query, key, value, pos_embedding=None):
        # [1,C,H,W]
        assert (
            query.shape[0] == 1 and key.shape[0] == 1 and value.shape[0] == 1
        ), "batch size must be 1"
        assert (
            query.ndim == 4 and key.ndim == 4 and value.ndim == 4
        ), "tensor dim not correct"
        dim = query.shape[1]
        query = self.to_q(query)  # [cav_num,C,H,W]
        key = self.to_k(key)  # [cav_num,C,H,W]
        value = self.to_v(value)  # [cav_num,C,H,W]
        query = rearrange(
            query, "b (c h) x y -> b h (x y) c", h=self.heads
        )  # [1,heads,H*Wvisio,C]
        if pos_embedding is not None:
            # [1,H,W]
            context = pos_embedding.unsqueeze(0).repeat(1, self.inner_dim, 1, 1)
            context = rearrange(context, "b (c h) x y -> b h (x y) c", h=self.heads)
            context = torch.matmul(
                query, context.transpose(-1, -2)
            )  # [cav_num,heads,H*W,H*W]
        key = rearrange(
            key, "b (c h) x y -> b h (x y) c", h=self.heads
        )  # [1,heads,H*W,C]
        if self.use_distance:
            value = self.raan(value)
        value = rearrange(
            value, "b (c h) x y -> b h (x y) c", h=self.heads
        )  # [1,heads,H*W,C]
        score = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(dim)
        if pos_embedding is not None:
            score = score + context  # [cav_num,heads,H*W,H*W]
        attn = F.softmax(score, -1)  # [cav_num,heads,H*W,H*W]
        out = torch.matmul(attn, value)  #  # [cav_num,heads,H*W,C]
        x = query.shape[2]
        out = rearrange(
            out, "b h (x y) c -> b (c h) x y", h=self.heads, x=x
        )  # [1,C,H,W]
        out = self.W_o(out)
        return out


class CrossAgentChannelAttention(nn.Module):
    def __init__(self, input_dim, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor
        self.to_qkv = nn.Sequential(
            nn.ConvTranspose2d(
                input_dim,
                input_dim * 3,
                kernel_size=factor,
                stride=factor,
            ),
        )
        self.W_o = nn.Conv2d(input_dim, input_dim, kernel_size=factor, stride=factor)

    def forward(self, batch_spatial_feature, distance=None, mode="context"):
        # params [V,C,H,W]
        # params [max_cav_num]
        assert batch_spatial_feature.ndim == 4, "tensor dim not correct"  # [cavs,C,H,W]
        cav_num, C, H, W = batch_spatial_feature.shape
        qkv = self.to_qkv(batch_spatial_feature).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t,
                "b c (x_h x) (y_h y) -> b (x_h y_h) c (x y)",
                x_h=self.factor,
                y_h=self.factor,
            ),
            qkv,
        )
        # [cav_num,num_head,C,H*W]   [cav_num,num_head,H*W,C]-> [cav_num,num_head,C,C]
        score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
            C
        )  # [cav_num,num_head,C,C]
        if distance is not None:
            distance = distance[:cav_num]
            distance = (
                distance.unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, self.factor**2, C, H * W)
            )  # [cav_num,num_head,C,H*W]
            if mode == "context":
                context = torch.matmul(
                    q, distance.transpose(-1, -2)
                )  # [cav_num,num_head,C,H*W]
                score += context
            else:
                score += distance
        attn = F.softmax(score, -1)  #  [cav_num,num_head,C,C]
        out = torch.matmul(attn, v)  #  [cav_num,num_head,C,C]  [cav_num,num_head,C,H*W]
        # out [cav_num,factor**2,C,H*W]
        out = out.transpose(1, 2).reshape(cav_num, C, -1)  # [cav_num,C,H*W]
        out = out.reshape(cav_num, C, self.factor * H, self.factor * W)  # [V,C,H,W]
        out = self.W_o(out)
        return out


class CrossAgentAttention(nn.Module):
    def __init__(self, input_dim, heads: int = 4) -> None:
        super().__init__()
        inner_dim = input_dim * heads
        self.inner_dim = inner_dim
        self.heads = heads
        self.to_qkv = nn.Sequential(
            nn.Conv2d(input_dim, inner_dim * 3, kernel_size=1),
        )
        self.W_o = nn.Conv2d(inner_dim, input_dim, kernel_size=1)

    def forward(self, batch_spatial_feature, distance=None, mode="context"):
        # params [V,C,H,W]
        # params [max_cav_num]
        assert batch_spatial_feature.ndim == 4, "tensor dim not correct"  # [cavs,C,H,W]
        cav_num, C, H, W = batch_spatial_feature.shape
        qkv = self.to_qkv(batch_spatial_feature).chunk(3, dim=1)  # [cav_num,C,H,W]
        if distance is not None:
            distance = distance[:cav_num]
            distance = distance.unsqueeze(-1).repeat(
                1, self.inner_dim
            )  # [max_cav_num,inner_dim]
            distance = rearrange(distance, "b (c h) -> 1 h b c", h=self.heads)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (x y) h b c", h=self.heads), qkv
        )
        score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(cav_num)  # [V,C,C]
        if distance is not None:
            if mode == "context":
                context = torch.matmul(
                    q, distance.transpose(-1, -2)
                )  # [H*W,head,cav_num,cav_num]
                score += context
            else:
                score += distance
        attn = F.softmax(score, -1)  #  [H*W,head,cav_num,dim]
        out = torch.matmul(
            attn, v
        )  #  [H*W,head,cav_num,cav_num] [H*W,head,cav_num,dim]-> [H*W,head,cav_num,dim]
        out = out.permute(2, 1, 3, 0).reshape(
            cav_num, self.inner_dim, H, W
        )  # [H*W,cav_num,C]
        out = self.W_o(out)
        return out


class CrossAgentWiseAttentionBlock(nn.Module):
    def __init__(self, intput_dim):
        super().__init__()
        self.bn_1 = nn.BatchNorm2d(intput_dim)
        self.bn_2 = nn.BatchNorm2d(intput_dim)
        self.ca = CrossAgentAttention(intput_dim)
        self.cca = CrossAgentChannelAttention(intput_dim)
        self.mlp = nn.Conv2d(intput_dim, intput_dim, kernel_size=1)

    def forward(self, feature, distance=None):
        # identity = feature
        # out = self.ca(feature, distance)
        # out = self.bn_1(out + identity)
        # identity = out
        # out = self.mlp_1(out)
        # out = self.bn_1(out + identity)

        # identity = out
        # out = self.cca(out, distance)
        # out = self.bn_2(out + identity)
        # identity = out
        # out = self.mlp_2(out)
        # out = self.bn_2(out + identity)
        # return out
        out = self.bn_1(feature)
        identity = out
        out = self.ca(out, distance)
        identity = identity + out
        out = self.bn_1(identity + out)
        out = self.mlp(out)
        out = identity + out

        out = self.bn_2(feature)
        identity = out
        out = self.cca(out, distance)
        identity = identity + out
        out = self.bn_2(identity + out)
        out = self.mlp(out)
        out = identity + out
        return out


class CrossAgentWiseAttention(nn.Module):
    def __init__(self, input_dim, depth: int = 2) -> None:
        super().__init__()
        self.blks = nn.ModuleList()
        for _ in range(depth):
            caa_blk = CrossAgentWiseAttentionBlock(input_dim)
            self.blks.append(caa_blk)

    def forward(self, spatial_features, distance=None):
        # params spatial_features: [V,C,H,W]
        # params distance: [1,max_cav]
        assert spatial_features.ndim == 4, "tensor dim not correct"
        out = spatial_features
        for index, blk in enumerate(self.blks):
            if index != 0:
                identity = out
                out = blk(out, distance)
                out = out + identity
            else:
                out = blk(out, distance)
                out = spatial_features + out
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    modified from https://arxiv.org/abs/2103.02907
    """

    def __init__(self, inp, oup=None, reduction=16):
        super().__init__()
        if oup is None:
            oup = inp
        # self.channel_weight = nn.Sequential(
        #     nn.AdaptiveMaxPool2d(1), nn.Sigmoid()
        # )  # [B,C,1,1]
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))

        mip = max(16, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # c_weight = self.channel_weight(x)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out_attn = identity * a_w * a_h
        return out_attn


class LocalInvertedAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = LocalRangeAttentionBlock(input_dim, kernel_size=3, padding=1)
        self.conv2 = LocalRangeAttentionBlock(input_dim, kernel_size=3, padding=1)
        self.dwise = LocalRangeAttentionBlock(
            input_dim, kernel_size=1, groups=input_dim // 2
        )
        self.att = CoordinateAttention(input_dim)

    def forward(self, batch_feature):
        identity = batch_feature
        out = self.conv1(batch_feature)
        out = self.dwise(out)
        attn_out = self.att(out)
        out = out * attn_out
        out = self.conv2(out)
        return out + identity


class LocalSandGlassAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = LocalRangeAttentionBlock(input_dim, kernel_size=3, padding=1)
        self.conv2 = LocalRangeAttentionBlock(input_dim, kernel_size=3, padding=1)
        self.dwise = LocalRangeAttentionBlock(
            input_dim, kernel_size=1, groups=input_dim // 2
        )
        self.att = CoordinateAttention(input_dim)

    def forward(self, batch_feature):
        identity = batch_feature
        out = self.dwise(batch_feature)
        attn_out = self.att(out)
        out = out * attn_out
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dwise(out)
        return out + identity


class LocalRangeAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        scale: float = 1.0,
        kernel_size=1,
        padding=0,
        groups=1,
        last_attention: bool = True,
    ):
        super().__init__()
        assert input_dim % 2 == 0, "channel must be even number"
        if output_dim is None:
            output_dim = input_dim
        else:
            assert output_dim % 2 == 0, "channel must be even number"
        branch_out_channel = output_dim // 2
        self.last_attention = last_attention

        self.conv_a = nn.Sequential(
            nn.Conv2d(
                input_dim,
                branch_out_channel,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(branch_out_channel),
        )

        self.conv_b = nn.Sequential(
            nn.Conv2d(
                input_dim,
                branch_out_channel,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(branch_out_channel),
        )

        self.conv_attn_a = nn.Sequential(
            nn.Conv2d(
                in_channels=branch_out_channel + 2,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_attn_b = nn.Sequential(
            nn.Conv2d(
                in_channels=branch_out_channel + 2,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_attention_a = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        self.conv_attention_b = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        # self.lam = nn.Parameter(torch.zeros(scale))
        self.scale_a = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.scale_b = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, spatial_features):  # spatial_features [1,C,H,W]

        x_a = self.conv_a(spatial_features)  # [1,C//2,H,W]
        x_b = self.conv_b(spatial_features)  # [1,C//2,H,W]

        grid_h_batch, grid_w_batch, distance_tensor = self.range_map(x_a)

        att_a = self.conv_attn_a(
            torch.cat([x_a, grid_h_batch, grid_w_batch], dim=1)
        )  # [1,C//2+2,H,W]

        att_b = self.conv_attn_b(
            torch.cat([x_b, 1 - grid_h_batch, 1 - grid_w_batch], dim=1)
        )  # [1,C//2+2,H,W]

        max_att_a = torch.max(x_a, 1)[0].unsqueeze(1)  # [1,1,H,W]
        avg_att_a = torch.mean(x_a, 1).unsqueeze(1)  # [1,1,H,W]

        att_maps_a = self.conv_attention_a(
            torch.cat([att_a, max_att_a, avg_att_a, distance_tensor], dim=1)
        )

        max_att_b = torch.max(x_b, 1)[0].unsqueeze(1)
        avg_att_b = torch.mean(x_b, 1).unsqueeze(1)
        att_maps_b = self.conv_attention_b(
            torch.cat([att_b, max_att_b, avg_att_b, -1.0 * distance_tensor], dim=1)
        )

        if self.last_attention:
            x_a = (1 + self.scale_a * att_maps_a) * x_a
            x_b = (1 + self.scale_b * att_maps_b) * x_b
        else:
            x_a = self.scale_a * att_maps_a * x_a
            x_b = self.scale_a * att_maps_b * x_b

        out = torch.cat((x_a, x_b), dim=1)
        return out  # spatial_features [1,C,H,W]

    def range_map(self, spatial_features, norm: bool = True):
        # (H,W) distance map
        H, W = spatial_features.shape[-2:]

        lin_h = torch.linspace(0, H - 1, H).cuda()
        lin_w = torch.linspace(0, W - 1, W).cuda()
        y, x = torch.meshgrid(lin_h, lin_w)
        y = torch.abs(y - H / 2 + 0.5) if H % 2 == 1 else torch.abs(y - H / 2)
        x = torch.abs(x - W / 2 + 0.5) if W % 2 == 1 else torch.abs(x - W / 2)
        y = y / float(H // 2)
        x = x / float(W // 2)
        distance_tensor = (
            torch.sqrt(x**2 + y**2)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(spatial_features.shape[0], 1, 1, 1)
        )
        if norm:
            distance_tensor = linear_map(distance_tensor)
        grid_h_batch = (
            y.unsqueeze(0).unsqueeze(0).repeat(spatial_features.shape[0], 1, 1, 1)
        )
        grid_w_batch = (
            x.unsqueeze(0).unsqueeze(0).repeat(spatial_features.shape[0], 1, 1, 1)
        )
        return grid_h_batch, grid_w_batch, distance_tensor


class LocalAttention(nn.Module):
    """
    implement local attention with range encoding
    """

    def __init__(self, input_dim, depth: int = 2):
        super().__init__()
        self.lsgs = nn.ModuleList()
        for i in range(depth):
            lra = LocalInvertedAttention(input_dim)
            self.lsgs.append(lra)
        # add lateral connection or residue connection to avoid gradient vanishing

    def forward(self, spatial_features):  # [V,C,H,W]
        for lsg in self.lsgs:
            spatial_features = lsg(spatial_features)
        return spatial_features


class RangeAttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.agent_att = CrossAgentWiseAttention(input_dim)
        self.local_att = LocalAttention(input_dim)
        self.gen_local = nn.Conv2d(
            input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim
        )
        self.gen_agent = nn.AdaptiveAvgPool2d(1)

        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gen_agent = nn.Sequential(
        #     nn.Conv2d(input_dim, input_dim // 2, kernel_size=1),
        #     nn.ReLU(),
        # )
        # self.pw_agent = nn.Sequential(
        #     nn.Conv2d(input_dim // 2, input_dim, kernel_size=1),
        # )

        # self.gen_local = nn.Sequential(
        #     nn.Conv2d(input_dim, input_dim // 2, kernel_size=1),
        #     nn.BatchNorm2d(input_dim // 2),
        #     nn.ReLU(),
        # )
        # self.pw_local = nn.Sequential(
        #     nn.Conv2d(input_dim // 2, input_dim, kernel_size=1),
        #     nn.BatchNorm2d(input_dim),
        # )

 

    def forward_(self, spatial_features, record_len, distance=None):
        assert spatial_features.ndim == 4, "tensor dim not correct"
        split_x = self.regroup(
            spatial_features, record_len
        )  # spatial_features [5,C,H,W]
        out = []
        for idx, batch_spatial_feature in enumerate(split_x):  # [2,C,H,W]
            """
            combine agent and local attention
            """
            local_feat = []
            for vehicle_feature in batch_spatial_feature:  # [C,H,W]
                local_feature = self.local_att(
                    vehicle_feature.unsqueeze(0)
                )  # [1,C,H,W]
                local_feat.append(local_feature)
            local_feat = torch.cat(local_feat, dim=0)  # [V,C,H,W]
            agent_feat = self.agent_att(
                local_feat, distance[idx]
            )  # [V,C,H,W] V is sum of cavs in the scene which can be detected

            ego_feat = batch_spatial_feature[0:1]  # [1,C,H,W]

            local_sum_feat = torch.sum(local_feat, dim=0, keepdim=True)# [1,C,H,W]

            agent_sum_feat = torch.sum(agent_feat, dim=0, keepdim=True)# [1,C,H,W]

            local_output = self.pw_local(self.gen_local(local_sum_feat))
            agent_output = self.pw_agent(self.gen_agent(self.gap(agent_sum_feat)))
            output = local_output + agent_output 
            local_weight = output.sigmoid()  # [1,C,H,W]
            
            out_feature = (
                local_weight * local_sum_feat
                + local_sum_feat
                + agent_sum_feat
                + ego_feat
            )
            out.append(out_feature)

        out = torch.cat(out, dim=0)
        return out

    def forward(self, spatial_features, record_len, distance=None):
        assert spatial_features.ndim == 4, "tensor dim not correct"
        split_x = self.regroup(
            spatial_features, record_len
        )  # spatial_features [5,C,H,W]
        out = []
        for idx, batch_spatial_feature in enumerate(split_x):  # [2,C,H,W]
            """
            combine agent and local attention
            """
            local_feat = []
            for vehicle_feature in batch_spatial_feature:  # [C,H,W]
                local_feature = self.local_att(
                    vehicle_feature.unsqueeze(0)
                )  # [1,C,H,W]
                local_feat.append(local_feature)
            local_feat = torch.cat(local_feat, dim=0)  # [V,C,H,W]
            agent_feat = self.agent_att(
                local_feat, distance[idx]
            )  # [V,C,H,W] V is sum of cavs in the scene which can be detected
            ego_feat = batch_spatial_feature[0:1]  # [1,C,H,W]
            local_sum_feat = torch.sum(local_feat, dim=0, keepdim=True)
            agent_sum_feat = torch.sum(agent_feat, dim=0, keepdim=True)
            # local_max_feat = torch.max(local_feat,dim=0)[0].unsqueeze(0) # [1,C,H,W]
            # local_mean_feat = torch.mean(local_feat,dim=0).unsqueeze(0)
            # local_feats = torch.cat([local_max_feat,local_mean_feat],dim=0) # [2,C,H,W]
            local_weight = torch.max(
                self.gen_local(local_sum_feat), dim=1, keepdim=True
            )[
                0
            ].sigmoid()  # [1,1,H,W]

            # agent_max_feat = torch.max(agent_feature,dim=0)[0].unsqueeze(0)
            # agent_feats = torch.cat([agent_max_feat,agent_mean_feat],dim=0)# [2,C,H,W]
            agent_weight = self.gen_agent(agent_sum_feat).sigmoid()  # [1,C,1,1]

            out_feature = (
                local_weight * local_sum_feat
                + local_sum_feat
                + agent_weight * agent_sum_feat
                + agent_sum_feat
                + ego_feat
            )
            out.append(out_feature)

        out = torch.cat(out, dim=0)
        return out

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
