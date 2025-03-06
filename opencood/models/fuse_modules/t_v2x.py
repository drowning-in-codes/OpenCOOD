import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import einsum

from einops import rearrange
from opencood.models.fuse_modules.fuse_utils import splitgroup


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


def window_partition(x, window_size, h_w, w_w):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, h_w, w_w, B):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    # B = int(windows.shape[0] // (H * W // window_size // window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def to_channels_first(x):
    assert x.ndim == 4
    return x.permute(0, 3, 1, 2)  # shape [B,H,W,C] -> [B,C,H,W]


def to_channels_last(x):
    assert x.ndim == 4
    return x.permute(0, 2, 3, 1)  # shape [B,C,H,W] -> [B,H,W,C]


def make_tuple(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormFeedResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.mlp = FeedForward(dim, dim // 4)

    def forward(self, x, **kwargs):
        return self.mlp(self.fn(self.norm(x), **kwargs)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, 'dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.att_drop(attn)

        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.projection(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# def get_relative_distances(window_size):
#     indices = torch.tensor(np.array(
#         [[x, y] for x in range(window_size) for y in range(window_size)]))
#     distances = indices[None, :, :] - indices[:, None, :]
#     return distances

def get_relative_distances(window_height, window_width):
    indices = torch.tensor(
        [[x, y] for x in range(window_height) for y in range(window_width)]
    )
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


# class BaseWindowAttention(nn.Module):
#     def __init__(self, dim, heads, window_size, dim_head=None, drop_out=.1,
#                  relative_pos_embedding=True):
#         super().__init__()
#         dim_head = dim_head or dim // heads
#         inner_dim = dim_head * heads
#         self.dim_head = dim_head
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.window_size = window_size
#         self.relative_pos_embedding = relative_pos_embedding

#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

#         if self.relative_pos_embedding:
#             self.relative_indices = get_relative_distances(window_size) + \
#                                     window_size - 1
#             self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1,
#                                                           2 * window_size - 1))
#         else:
#             self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2,
#                                                           window_size ** 2))
#         self.ego_query_net = EgoQueryGen(dim, output_dim=inner_dim)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(drop_out)
#         )

#     def forward(self, x):
#         #  shape [B,L,H,W,C]
#         b, l, h, w, c, m = *x.shape, self.heads
#         assert h % self.window_size == 0, f'H {h} not divisible by window size {self.window_size}'
#         assert w % self.window_size == 0, f'W {w} not divisible by window size {self.window_size}'
#         kv = self.to_kv(x).chunk(2, dim=-1)
#         new_h = h // self.window_size
#         new_w = w // self.window_size
#         # q : (b, l, m, new_h*new_w, window_size^2, c_head)
#         k, v = map(
#             lambda t: rearrange(t,
#                                 'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
#                                 m=m, w_h=self.window_size,
#                                 w_w=self.window_size), kv)

#         ego_query = self.ego_query_net(x[:, 0])  # ego feat shape [1,H,W,C]
#         ego_query = ego_query.unsqueeze(1).expand(-1, l, -1, -1, -1)
#         ego_query = rearrange(ego_query, 'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c', m=m,
#                               w_h=self.window_size, w_w=self.window_size)
#         assert ego_query.shape == k.shape, "ego_query shape {},k shape {}".format(ego_query.shape, k.shape)
#         # b l m h window_size window_size
#         dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j',
#                             ego_query, k) * self.scale
#         # consider prior knowledge of the local window
#         if self.relative_pos_embedding:
#             dots += self.pos_embedding[self.relative_indices[:, :, 0],
#             self.relative_indices[:, :, 1]]
#         else:
#             dots += self.pos_embedding

#         attn = dots.softmax(dim=-1)
#         out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)

#         # b l h w c
#         out = rearrange(out,
#                         'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
#                         m=self.heads, w_h=self.window_size,
#                         w_w=self.window_size,
#                         new_w=new_w, new_h=new_h)
#         out = self.to_out(out)
#         return out

class BaseWindowAttention(nn.Module):
    def __init__(self, dim, heads, window_size, dim_head=None, drop_out=.1,
                 relative_pos_embedding=True):
        super().__init__()
        dim_head = dim_head or dim // heads
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        if isinstance(window_size, int):
            window_height = window_size
            window_width = window_size
        elif isinstance(window_size, tuple) or isinstance(window_size, list):
            window_height, window_width = window_size
        else:
            raise ValueError("window_size must be int or tuple or list")
        self.window_height = window_height
        self.window_width = window_width
        self.relative_pos_embedding = relative_pos_embedding

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_height, window_width)
            self.relative_indices += torch.tensor([window_height - 1, window_width - 1])
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_height - 1,
                                                          2 * window_width - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_height * window_width,
                                                          window_height * window_width))

        self.ego_query_net = EgoQueryGen(dim, output_dim=inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        #  shape [B,L,H,W,C]
        b, l, h, w, c, m = *x.shape, self.heads
        assert h % self.window_height == 0, f'H {h} not divisible by window height {self.window_height}'
        assert w % self.window_width == 0, f'W {w} not divisible by window width {self.window_width}'
        kv = self.to_kv(x).chunk(2, dim=-1)
        new_h = h // self.window_height
        new_w = w // self.window_width
        # q : (b, l, m, new_h*new_w, window_height*window_width, c_head)
        k, v = map(
            lambda t: rearrange(t,
                                'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=self.window_height,
                                w_w=self.window_width), kv)

        ego_query = self.ego_query_net(x[:, 0])  # ego feat shape [1,H,W,C]
        ego_query = ego_query.unsqueeze(1).expand(-1, l, -1, -1, -1)
        ego_query = rearrange(ego_query, 'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c', m=m,
                              w_h=self.window_height, w_w=self.window_width)
        assert ego_query.shape == k.shape, "ego_query shape {},k shape {}".format(ego_query.shape, k.shape)
        # b l m h window_height window_width
        dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j',
                            ego_query, k) * self.scale
        # consider prior knowledge of the local window
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0],
            self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)

        # b l h w c
        out = rearrange(out,
                        'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                        m=self.heads, w_h=self.window_height,
                        w_w=self.window_width,
                        new_w=new_w, new_h=new_h)
        out = self.to_out(out)
        return out


class CAVAttention(nn.Module):
    def __init__(self, dim, args, num_types=2, num_relations=4, dropout=.1):
        super().__init__()
        dim_head = args['dim_heads']
        heads = args['heads']
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.drop_out = nn.Dropout(dropout)
        self.num_types = num_types
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv_list = nn.ModuleList([])
        self.to_out_list = nn.ModuleList([])
        for t in range(num_types):
            to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
            self.to_qkv_list.append(to_qkv)
            self.to_out_list.append(to_out)
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, heads, dim_head, dim_head))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, heads, dim_head, dim_head))

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, x, prior_encoding, mask):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, H,W,L,1)
        # prior_encoding: (B,L,H,W,3)
        *_, types = [item.squeeze(-1) for item in
                     prior_encoding[:, :, 0, 0, :].split(
                         [1, 1, 1], dim=-1)]
        # mask: (B, 1, H, W, L, 1)
        mask = mask.unsqueeze(1)
        types = types.long()  # shape [B,L]
        qkv = self.to_qkv(x, types)  # shape [B,L,H,W,C]
        # (B,head,L,L,C_head,C_head)
        w_att, w_msg = self.get_edge_weights(x, types)

        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, 'b l h w (m c) -> b m h w l c',
                                          m=self.heads), (qkv))
        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum(
            'b m h w i p, b m i j p q, b m h w j q -> b m h w i j',
            [q, w_att, k]) * self.scale
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float('inf'))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        v_msg = torch.einsum('b m i j p c, b m h w j p -> b m h w i j c',
                             w_msg, v)
        out = torch.einsum('b m h w i j, b m h w i j c -> b m h w i c',
                           att_map, v_msg)

        out = rearrange(out, 'b m h w l c -> b h w l (m c)',
                        m=self.heads)

        out = self.to_out(out, types)
        out = self.drop_out(out)

        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)

        return out

    def to_qkv(self, cav_feats, types):
        # cav_feat shape: [B,L,H,W,C]
        B, L, H, W, C = cav_feats.shape
        q_batch = []
        k_batch = []
        v_batch = []
        for batch_index in range(B):
            q_list = []
            k_list = []
            v_list = []
            for cav_index in range(L):  # cav_feat shape: [H,W,C]
                cav_feat = cav_feats[batch_index, cav_index, :, :]
                q, k, v = self.to_qkv_list[types[batch_index, cav_index]](cav_feat.unsqueeze(0)).chunk(3,
                                                                                                       dim=-1)  # shape [1,H,W,3*D]
                q_list.append(q)
                k_list.append(k)
                v_list.append(v)
            q_batch.append(torch.cat(q_list, dim=0).unsqueeze(0))  # shape [1,L,H,W,C]
            k_batch.append(torch.cat(k_list, dim=0).unsqueeze(0))  # shape [1,L,H,W,C]
            v_batch.append(torch.cat(v_list, dim=0).unsqueeze(0))  # shape [1,L,H,W,C]

        # (B,L,H,W,C)
        q = torch.cat(q_batch, dim=0)
        k = torch.cat(k_batch, dim=0)
        v = torch.cat(v_batch, dim=0)

        return q, k, v

    def to_out(self, x, types):
        out_batch = []
        B, H, W, L, C = x.shape
        for batch_index in range(B):
            out_list = []
            for cav_index in range(L):
                cav_feat = x[batch_index, :, :, cav_index, :].unsqueeze(2)  # shape [H,W,1,C]
                out_list.append(
                    self.to_out_list[types[batch_index, cav_index]](cav_feat))  # shape [H,W,1,C]
            out_batch.append(torch.cat(out_list, dim=2).unsqueeze(0))  # shape [1,H,W,L,C]
        out = torch.cat(out_batch, dim=0)  # shape [B,H,W,L,C]
        return out

    def get_edge_weights(self, x, types):
        w_att_batch = []
        w_msg_batch = []

        B, L, C, H, W = x.shape
        for batch_index in range(B):
            w_att_list = []
            w_msg_list = []
            for cav_index_i in range(L):
                w_att_i_list = []
                w_msg_i_list = []

                for cav_index_j in range(L):
                    e_type = self.get_relation_type_index(types[batch_index, cav_index_i],
                                                          types[batch_index, cav_index_j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))  # shape [1,H,D,D] heads dim_head

                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))  # shape [1,L,H,D,D]

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))  # shape [1,L,L,H,D,D]

        # shape [B,L,L,H,D,D]
        w_att = torch.cat(w_att_batch, dim=0).permute(0, 3, 1, 2, 4, 5)  # -> shape [B,H,L,L,D,D]
        w_msg = torch.cat(w_msg_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2


class CBAM(nn.Module):
    """
    https://github.com/Peachypie98/CBAM
    """

    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                              bias=self.bias)

    def forward(self, x):
        max_out = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max_out, avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        return output


class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.GELU(),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output


class FeatExtract(nn.Module):
    def __init__(self, in_dim, out_dim=None, downsample=False, downsample_sr=2, channel_rate=16):
        super().__init__()
        out_dim = out_dim or in_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=11, padding=5, groups=in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.GELU()
        )
        self.downsample = nn.Maxpool2d(kernel_size=downsample_sr, stride=downsample_sr) if downsample else nn.Identity()
        self.cbam = CBAM(out_dim, channel_rate)

    def forward(self, x):
        out = self.conv(x)
        out = self.downsample(out)
        out = self.cbam(out)
        return out


class EgoQueryGen(nn.Module):
    def __init__(self, dim, output_dim=None):
        super().__init__()
        output_dim = output_dim or dim
        self.query_gen = FeatExtract(dim, out_dim=output_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # shape [B,H,W,C] -> [B,C,H,W]
        ego_query = self.query_gen(x)
        ego_query = ego_query.permute(0, 2, 3, 1)  # shape [B,C,H,W] -> [B,H,W,C]
        return ego_query


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (B, L, 1, 1, 3C)
        batch = x.size(0)
        cav_num = x.size(1)

        if self.radix > 1:
            # x: (B, L, 1, 3, C)
            x = x.view(batch,
                       cav_num,
                       self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=3)
            # B, 3LC
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.rsoftmax = RadixSoftmax(3, 1)

    def forward(self, window_list):
        # window list: [(B, L, H, W, C) * 3]
        assert len(window_list) == 3, 'only 3 windows are supported'
        sw, mw, bw = window_list[0], window_list[1], window_list[2]
        B, L = sw.shape[0], sw.shape[1]

        # global average pooling, B, L, H, W, C
        x_gap = sw + mw + bw
        # B, L, 1, 1, C
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # B, L, 1, 1, 3C
        x_attn = self.fc2(x_gap)
        # B L 1 1 3C
        x_attn = self.rsoftmax(x_attn).view(B, L, 1, 1, -1)

        out = sw * x_attn[:, :, :, :, 0:self.input_dim] + \
              mw * x_attn[:, :, :, :, self.input_dim:2 * self.input_dim] + \
              bw * x_attn[:, :, :, :, self.input_dim * 2:]
        return out


class MultiWinsAttention(nn.Module):
    def __init__(self, dim, args, fuse_type='mean'):
        super().__init__()
        num_blocks = args['num_blocks'] if 'num_blocks' in args else 3
        heads = args['heads'] if 'heads' in args else [2, 4, 8]
        dim_heads = args['dim_heads'] if 'heads' in args else [64, 128, 256]
        window_size_h_list = args['window_size_h_list'] if 'window_size_h_list' in args else [4, 8, 16]
        window_size_w_list = args['window_size_w_list'] if 'window_size_w_list' in args else [4, 8, 16]
        if isinstance(heads, int):
            heads = [heads] * num_blocks
        if isinstance(dim_heads, int):
            dim_heads = [dim_heads] * num_blocks

        window_size_list = list(zip(window_size_h_list, window_size_w_list))
        self.layers = nn.ModuleList([])
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.layers.append(
                BaseWindowAttention(dim, heads[i], dim_head=dim_heads[i], window_size=window_size_list[i]))
        self.fuse_type = fuse_type
        if fuse_type == 'split':
            self.combine_net = SplitAttn(dim)
        elif fuse_type == 'linear':
            self.combine_net = nn.Linear(dim * num_blocks, dim)
        else:
            self.combine_net = None

    def forward(self, x):
        # x shape [B,L,H,W,C]
        win_list = []
        for win_attn in self.layers:
            win_x = win_attn(x)
            win_list.append(win_x)
        if self.num_blocks == 1:
            out = win_list[0]
        else:
            if self.fuse_type == "split":
                out = self.combine_net(win_list)  # win_list shape [(B,L,H,W,C) * 3]
            elif self.fuse_type == "linear":
                win_attn = torch.cat(win_list, dim=-1)
                out = self.combine_net(win_attn)
            else:
                out = torch.mean(torch.stack(win_list, dim=0), dim=0)
        return out


class V2XFusionBlock(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        dropout = args['dropout'] if 'dropout' in args else 0.1
        self.layers = nn.ModuleList([
            PreNormResidual(dim, CAVAttention(dim, args["cav_attn"])),
            PreNormResidual(dim, FeedForward(dim, dropout=dropout)),
            PreNormResidual(dim, MultiWinsAttention(dim, args["WinsAttention"])),
            PreNormResidual(dim, FeedForward(dim, dropout=dropout)),
        ])

    def forward(self, x, prior_encoding, mask, domain_adaptation=False):
        # transform the features to the current timestamp
        # velocity, time_delay, infra
        spatial_feats = []
        patch_feats = []
        [cav_attention, ffn_1, window_attention, ffn_2] = self.layers
        x = cav_attention(x, prior_encoding=prior_encoding, mask=mask)  # (B,L,H,W,C)
        if domain_adaptation:
            spatial_feats.append(x)
        x = ffn_1(x)
        x = window_attention(x)  # shape [B,L,H,W,C]
        if domain_adaptation:
            patch_feats.append(x)
        x = ffn_2(x)
        if domain_adaptation:
            return x, spatial_feats, patch_feats

        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 7, padding=3),
            nn.Conv2d(in_dim // 2, in_dim // 2, 3, padding=1),
            nn.Conv2d(in_dim // 2, 1, 1),
            nn.LeakyReLU(),
            nn.Sigmoid()
        )

    def forward(self, x, record_len):
        # x shape[B,L,H,W,C]
        x = rearrange(x, 'b l h w c -> b l c h w')
        split_x = splitgroup(x, record_len=record_len)  # -> split_x shape [N,C,H,W]
        batch_feats = regroup(split_x, record_len)  # [(2,C,H,W),(3,C,H,W),...]
        all_feats = []
        for b_feat in batch_feats:
            feat = self.conv(b_feat)  # [L,C,H,W]
            weights = F.softmax(feat, dim=0)
            feats = b_feat * weights

            feats = torch.sum(feats, dim=0)
            feats = b_feat[0] + feats  # shape [C,H,W]
            all_feats.append(feats)
        out = torch.stack(all_feats, dim=0)
        # out = rearrange(out,'b c h w -> b h w c')
        return out


class TransAdaNet(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        num_blocks = args['num_blocks'] if 'num_blocks' in args else 3
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(
                V2XFusionBlock(dim, args["fusion_block"])
            )
        self.adaptive_fusion = AdaptiveFusion(dim)

    def forward(self, x, mask, record_len, domain_adaptation=False):
        # transform the features to the current timestamp
        # velocity, time_delay, infra
        # (B,L,H,W,3)
        prior_encoding = x[..., -3:]
        # (B,L,H,W,C)
        x = x[..., :-3]
        global_feats = []
        spatial_feats = []
        patch_feats = []
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        for v2xattn in self.layers:
            if domain_adaptation:
                x, spatial_feat, patch_feat = v2xattn(x, prior_encoding=prior_encoding, mask=com_mask,
                                                      domain_adaptation=domain_adaptation)  # (B,L,H,W,C)
                spatial_feats.append(spatial_feat)
                patch_feats.append(patch_feat)
                global_feats.append(x)
            else:
                x = v2xattn(x, prior_encoding=prior_encoding, mask=com_mask,
                            domain_adaptation=domain_adaptation)  # (B,L,H,W,C)
        out = self.adaptive_fusion(x, record_len=record_len)
        if domain_adaptation:
            return out, spatial_feats, patch_feats, global_feats

        return out
