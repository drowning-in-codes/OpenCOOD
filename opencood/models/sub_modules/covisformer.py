#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/2/18 上午10:16
#   lastModifiedTime:2024/2/18 上午10:16
#   file:test.py
#   software: classicNets
#
# compression and denoise model
#
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


def spatial_sampling(agent_feature, compress_ratio: float = 0.8):
    """
    spatial sampling except the ego feature
    """
    agent_num, _, H, W = agent_feature.size()
    reduced_pixels = int(H * W * compress_ratio)

    for i in range(
            1, agent_num
    ):  # Start from index 1 to skip the ego feature # agent_feature[i] [C,H,W]
        aggregate_features = torch.sum(agent_feature[i], dim=0)  # [H,W]
        aggregate_features = aggregate_features.reshape(-1)
        _, indices = torch.topk(
            aggregate_features,
            reduced_pixels,
        )

        mask = torch.zeros_like(aggregate_features)
        mask[indices] = 1
        mask = mask.reshape(H, W)
        agent_feature[i] *= mask

    return agent_feature


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout: float = 0.8):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class UpMerge(nn.Module):
    r"""Patch Merging Layer.
    Modified from Swin Transformer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.BatchNorm2d
    """

    def __init__(self, dim, output_dim=None,scale_factor=2,drop_rate=0.8):
        super().__init__()
        assert scale_factor in [1,2, 4], "scale factor must be [1,2,4]"
        if output_dim is None:
            output_dim = dim
        self.expansion = nn.Sequential(
            nn.ConvTranspose2d(dim,  dim, kernel_size=scale_factor, stride=scale_factor),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        """
        x: B,L,C,H,W
        """
        B, L, C, H, W = x.shape
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.expansion(x)
        x = rearrange(x, "(B L) C H W -> B L C H W", B=B, L=L)
        return x

class DownMerge(nn.Module): # [B,L,C,H,W] -> [B,L,C,H//2,W//2]
    r"""Patch Merging Layer.
    Modified from Swin Transformer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim,output_dim=None,norm_layer=nn.LayerNorm,downsample_rate=2,drop_rate=0.8):
        super().__init__()
        assert downsample_rate in [1,2,4], "downsample rate must be [1,2,4]"
        if output_dim is None:
            output_dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim,  output_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.downsample_rate = downsample_rate

    def forward(self, x):
        """
        x: B,L,C,H,W
        """
        B, L,C,H,W = x.shape
        assert H % self.downsample_rate == 0 and W % self.downsample_rate == 0, f"x size ({H}*{W}) are not even."
        x = x.reshape(B*L, H, W, C)
        if self.downsample_rate == 4:
            x0 = x[..., 0::4, 0::4]  # B H/4 W/4 C
            x1 = x[..., 1::4, 0::4]  # B H/4 W/4 C
            x2 = x[..., 0::4, 1::4]  # B H/4 W/4 C
            x3 = x[..., 1::4, 1::4]  # B H/4 W/4 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            x = x.reshape(B,L, -1, 4 * C)  # B L H/4*W/4 4*C
        elif self.downsample_rate == 2:
            x0 = x[..., 0::2, 0::2]  # B H/2 W/2 C
            x1 = x[..., 1::2, 1::2]  # B H/2 W/2 C
            x2 = x[..., 0::2, 1::2]  # B H/2 W/2 C
            x3 = x[..., 1::2, 1::2]  # B H/2 W/2 C
            x = torch.cat([x0, x1,x2,x3], -1)  # B H/2 W/2 2*C
            x = x.reshape(B,L, -1, 4 * C)  # B L H/2*W/2 4*C
        else:
            x = x.reshape(B, L, -1, C)
        x = self.norm(x)
        x = self.reduction(x)
        x = self.dropout(x)
        x = rearrange(x, "B L (H W) C-> B L C H W", H=H//self.downsample_rate, W=W//self.downsample_rate)
        return x


class Attention(nn.Module):
    """
    #  add sr_ratio to reduce the computation cost
    """

    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            attn_dropout: float = 0.8,
            proj_dropout: float = 0.8,
            mode="context",
            sr_ratio=None,
            win_h=None,
            win_w=None,
            agent_size=None,
    ):
        super().__init__() # [B,L,H//w,W//w,w,w,C]
        # assert dim_head % heads == 0, f"dim {dim_head} must be divisible by heads {heads}"
        if agent_size is None:
            agent_size = 5
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.mode = mode
        self.sr_ratio = sr_ratio
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        if sr_ratio is None or sr_ratio[0] == 1 or sr_ratio[1] == 1:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, dim_head, bias=False)
        else:
            assert sr_ratio is not None, "sr_ratio must be set"
            N = agent_size*win_h*win_w
            self.sr = nn.Conv2d(
                dim,
                dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.to_out = nn.Linear(inner_dim, dim_head, bias=False)

            self.sr_norm = nn.LayerNorm(dim)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

    def forward(self, x, context=None,mask=None): # -> b l x y w1 w2 d
        assert mask is None or isinstance(mask, torch.Tensor), "mask must be a tensor or None"
        if mask is not None:
            assert mask.ndim == 5, "mask must be 5D" # [B,H,W,1,L:5]
        batch, agent_size, height, width, window_height, window_width,dim = x.shape
        # flatten
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')  # [B,L,H//w,W//w,w,w,C] -> [B*H//w*W//w,w*w*L,C]
        B,N,D = x.shape
        x = self.norm(x)
        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        if self.sr_ratio is None or self.sr_ratio[0] == 1 or self.sr_ratio[1] == 1:
            kv = self.to_kv(x).chunk(2, dim=-1)
            k, v = map(
                lambda t: rearrange(t, " b n (h d)-> b h n d ", h=self.heads), kv
            )
            if context is None:
                dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            else:
                context = rearrange(context, "b n (h d) -> b h n d", h=self.heads)
                # add context to attention. bias or context mode
                if self.mode == "bias":
                    dots = torch.matmul(q, k.transpose(-1, -2))
                    dots += context
                else:
                    dots = torch.matmul(q, k.transpose(-1, -2))
                    pos_attn = torch.matmul(q, context)
                    dots += pos_attn
        else:
            x_ = x.transpose(1,2).reshape(B, D, window_height,-1)
            x_ = self.sr(x_).reshape(B, D, -1).transpose(1,2)
            x_ = self.sr_norm(x_)
            kv = self.to_kv(x_).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, " b n (h d)->b h n d", h=self.heads), kv)

            if context is None:
                dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            else:
                # add context to attention. bias or context mode
                if self.mode == "bias":
                    # context
                    context = context.unsqueeze(1).repeat(1, self.heads, 1, 1)
                    dots = torch.matmul(q, k.transpose(-1, -2))
                    dots += context
                else:
                    context = rearrange(context, "b n (h d) -> b h n d", h=self.heads)
                    dots = torch.matmul(q, k.transpose(-1, -2))
                    pos_attn = torch.einsum('b c h w, b c j k -> b c h j', q, context)
                    dots += pos_attn
            # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # b x y w1 w2 e l -> (b x y) 1 (l w1  w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            dots = dots.masked_fill(mask == 0, -float('inf'))
        attn = self.attend(dots)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        # out = rearrange(out, "b h n d -> b n (h d)")
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)
        out = self.proj_drop(out)
        out = self.to_out(out)
        out = rearrange(out,'(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height , y=width)
        return out


class simpleTransformer(nn.Module):
    def __init__(self,
                 dim,
                 depth: int,
                 dim_head,
                 heads=3,
                 mlp_dim=None,
                 sr_ratio=None,
                 embded_H=None,
                 embded_W=None,
                 mode="context") -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim,
                              heads=heads,
                              dim_head=dim_head,
                              sr_ratio=sr_ratio,
                              win_h=embded_H,
                              win_w=embded_W,
                              mode=mode),
                    FeedForward(dim, mlp_dim),
                ]
                )
            )

    def forward(self, x, mask=None,context=None):
        for attn, ff in self.layers:
            x = attn(x, context) + x
            x = ff(x) + x
        return self.norm(x)


class RangeEncoding(nn.Module):
    def __init__(self, max_range=70, dim=None, mode="context") -> None:
        """
        encode relative position information. e.g. distance,transform_matrix
        """
        super().__init__()
        self.max_range = max_range
        if mode == "context":
            self.embedding = nn.Embedding(max_range, dim)
        else:
            self.embedding = nn.Embedding(max_range, 1)

    def forward(self, prior_info):
        if prior_info.ndim == 1:
            prior_info = prior_info.unsqueeze(0)
        prior_info = prior_info.clamp(0, self.max_range - 1)
        device = prior_info.device
        prior_info = prior_info.type(torch.int).to(device)
        return self.embedding(prior_info)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PEG(nn.Module):
    """
    similar to https://arxiv.org/pdf/2102.10882.pdf
    """

    def __init__(self, dim, k: int = 3):
        super().__init__()
        self.proj = nn.Conv3d(
            dim, dim, kernel_size=k, stride=1, padding=k // 2, groups=dim
        )

    def forward(self, features):
        assert features.ndim == 7, "input tensor must be 7D"  # # -> [B,L,x,y,w,h,C]
        B,L,x,y,w,h,C = features.shape
        features = rearrange(features, 'B L x y w h C -> B L C (x w) (y h)')
        features = self.proj(features) + features
        features = rearrange(features, 'B L C (x w) (y h) -> B L x y w h C', x=x, w=w, h=h)
        return features


class PatchEmbeddding(nn.Module):
    def __init__(self, patch_size: Tuple[int, int],dim,hidden_dim=None,output_dim=None,drop_rate=0.8):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        if output_dim is None:
            output_dim = dim
        self.patch_size = patch_size
        self.fc1 = nn.Linear(dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self,x):
        B,L,C,H,W = x.shape
        w1,w2 = self.patch_size
        x = rearrange(x, "B L C (H w1) (W w2) -> B L H W w1 w2 C",w1=w1,w2=w2)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x

class simpleConvEmbed(nn.Module):
    def __init__(
            self,
            input_dim,
            feature_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            dropout: float = 0.5,
    ):
        super().__init__()  # [B,L,C,H,W] -> [B,L,H//w,W//w,w,w,C]
        self.patch_size = patch_size
        feature_h, feature_w = feature_size
        kernel_size = (feature_h // patch_size[0], feature_w // patch_size[1])
        if kernel_size[0] == 0 or kernel_size[1] == 0:
            kernel_size = (patch_size[0] // feature_h, patch_size[1] // feature_w)
            self.depthwise_conv = nn.ConvTranspose2d(
                input_dim,
                input_dim,
                kernel_size=kernel_size,
                stride=kernel_size,
                groups=input_dim)
        else:
            self.depthwise_conv = nn.Conv2d(
                input_dim,
                input_dim,
                kernel_size=kernel_size,
                stride=kernel_size,
                groups=input_dim,
            )  # -> [1,V,H//kernel_size,W//kernel_size]
        self.pointwise_conv = nn.Conv2d(
            input_dim, input_dim, kernel_size=1, stride=1
        )  # -> [1,V,H//kernel_size,W//kernel_size]
        dim = patch_size[0] * patch_size[1]
        self.ln_1 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def get_patch_size(self):
        return self.patch_size

    def forward(self, x):
        assert x.ndim ==5, "input tensor must be 5D" # [B,L,C,H,W]
        x = x.rearrange("B L C (H w1) (W w2) -> B L H W w1 w2 C",w1=self.patch_size[0],w2=self.patch_size[1]) # [B,L,C,H,W] -> [B,L,H//w,W//w,w,w,C]
        # flatten
        x = rearrange(x, 'B L x y w1 w2 C -> (B x y) (L w1 w2) C')  # [B,L,H//w,W//w,w,w,C] -> [B*H//w*W//w,w*w*L,C]
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        out = rearrange(out, "1 V H W -> 1 V (H W)")
        if self.dim != out.shape[2]:
            """
            sometimes the patch size is not equal to the feature size. align the dim.
            """
            out = F.interpolate(out, size=[self.dim], mode="nearest")
        out = self.ln_1(out)
        out = self.dropout(out)
        return out



class SeparableConv(nn.Module):
    def __init__(self, input_dim, output_dim=None, kernel_size=3, stride=1) -> None:
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if kernel_size > 1:
            self.depthwiseconv = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, stride=stride,
                                           groups=input_dim)
        else:
            self.depthwiseconv = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=kernel_size, stride=stride,
                                                    groups=input_dim)
        self.pointwiseconv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        out = self.pointwiseconv(self.norm(self.depthwiseconv(x)))
        return out



class AbsoluteSinoPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq=5, mode="sino") -> None:
        super().__init__()
        self.dim = dim
        self.max_seq = max_seq
        self.mode = mode
        self.embedding = nn.Embedding(max_seq, dim)
        self.sinusoid_table = torch.zeros((max_seq, dim))
        position = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        self.sinusoid_table[:, 0::2] = torch.sin(position * div_term)
        self.sinusoid_table[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        if self.mode == "sino":
            return self.sinusoid_table
        else:
            return self.embedding(x)

class PosCNN(nn.Module):
    """
    from Twins:https://arxiv.org/abs/2104.13840
    """
    def __int__(self, in_chans, embed_dim=768, s=1):
        super().__int__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),
        )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).reshape(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

class CoVisFormer(nn.Module):
    """
    similar arch with vit and pvt.
    encode distance information to features.  multi-scale
    """

    # patch embedding
    # [N, C, H, W] -> [N, C, H*W] -> [N, H*W, C]
    def __init__(
            self,
            num_vehicles,
            feature_width,
            feature_height,
            channels: int,
            patch_size=None,
            num_heads: int = 1,
            embed_dim: list = [128, 256, 256],
            layer_num=None,
            num_stages: int = 3,
            sr_ratios=None,
            attention_mode="context",
            scale_factor=[2,4,1]
    ) -> None:
        super().__init__()
        if sr_ratios is None:
            sr_ratios = [(2, 2), (1, 1), (2,2)]
        if patch_size is None:
            patch_size = [
                ( 2,   8),
                ( 5,   8),
                ( 4,  8),
            ]
        if layer_num is None:
            layer_num = [3, 3, 3]
        self.num_stages = num_stages
        self.patch_embed = nn.ModuleList()
        self.pos_embed = nn.ModuleList()
        self.additional_embed = nn.ModuleList()
        self.blks = nn.ModuleList()
        self.pegs = nn.ModuleList()
        self.patch_merge = nn.ModuleList()
        self.up_merge = nn.ModuleList()

        for i in range(num_stages):
            
            """
            downsample resolution to avoid OOM
            """
            input_resolution = (feature_height // scale_factor[i], feature_width // scale_factor[i])
            patchMerge = DownMerge(dim=embed_dim[i-1] if i!=0 else channels,downsample_rate=scale_factor[i]) if scale_factor[i] != 1 else nn.Identity()
            
            """
            patch embedding first.
            """ # [B,L,C,H,W] -> [B*H//w*W//w,w*w*L,C]
            patch_embed = PatchEmbeddding(
                dim=channels if i == 0 else embed_dim[i-1],
                output_dim=embed_dim[i],
                patch_size=patch_size[i],
            )
            """
            positional encoding
            """
            pos_embed = RangeEncoding(max_range=70, dim=embed_dim[i])
            additional_embed = RangeEncoding(max_range=70, dim=embed_dim[i], mode=attention_mode)
            """
            transformer
            """
            transformerBlock = simpleTransformer(
                dim=embed_dim[i],
                heads=num_heads,
                dim_head=embed_dim[i],
                depth=layer_num[i],
                sr_ratio=sr_ratios[i],
                embded_H=patch_size[i][0],
                embded_W=patch_size[i][1],
                mode=attention_mode
            ) # -> [B,L,x,y,w,h,C]
            """
            PEG model
            """
            peg = PEG(num_vehicles, k=3) if i == 0 else nn.Identity()
          
            """
            upscale to keep the same resolution
            """
            up_patch_merge = UpMerge(dim=embed_dim[i],scale_factor=scale_factor[i]) if scale_factor[i] != 1 else nn.Identity()

            self.patch_embed.append(patch_embed)
            self.pos_embed.append(pos_embed)
            self.additional_embed.append(additional_embed)
            self.blks.append(transformerBlock)
            self.pegs.append(peg)
            self.patch_merge.append(patchMerge)
            self.up_merge.append(up_patch_merge)
        # self.apply(_init_weights)

    def forward(self, spatial_features, distances, mask=None): # distances: [B,L]
        # spatial_features [B,L,C,H,W]. L is the max_cav,default:5
        for i in range(self.num_stages):
            spatial_features = self.patch_merge[i](spatial_features)
            spatial_features = self.patch_embed[i](spatial_features) # -> [B,L,x,y,w,h,C]
            distance_embed = self.pos_embed[i](distances)
            distance_embed = distance_embed.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            spatial_features = spatial_features + distance_embed # distances [B,L,C]
            context = self.additional_embed[i](distances)
            spatial_features = self.blks[i](spatial_features, context) # -> [B,L,x,y,w,h,C]
            spatial_features = self.pegs[i](spatial_features)
            spatial_features = rearrange(spatial_features, "B L x y w h C -> B L C (x w) (y h)")
            spatial_features = self.up_merge[i](spatial_features)

        return spatial_features


# if __name__ == "__main__":
#     data = torch.randn(2, 5, 256, 100, 352).to("cuda")
#     model = CoVisFormer(5, 50, 176, 256, num_heads=3, embed_dim=[256, 128, 256], layer_num=[3, 3, 3], num_stages=2).to('cuda')
#     distances = torch.randn(2, 5).to("cuda")*3
#     output = model(data,distances)  # [B,L,C,H,W] -> [V,C,H,W]
    
    
#     print(output)
