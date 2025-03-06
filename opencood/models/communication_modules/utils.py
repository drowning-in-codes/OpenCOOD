import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, unpack, rearrange, reduce
from torch import einsum


# helper functions

def exists(v):
    return v is not None


def identity(t):
    return t


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse


def safe_div(num, den, eps=1e-6):
    return num / den.clamp(min=eps)


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


def l2norm(t, dim=-1, eps=1e-6):
    return F.normalize(t, p=2, dim=dim, eps=eps)


# rotation trick related

def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = l2norm(u + q, dim=1).detach()

    return (
            e -
            2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
            2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    )


def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    src, inverse = pack_one(src, '* d')
    tgt, _ = pack_one(tgt, '* d')

    norm_src = src.norm(dim=-1, keepdim=True)
    norm_tgt = tgt.norm(dim=-1, keepdim=True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src),
        safe_div(tgt, norm_tgt),
        src
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return inverse(rotated)


def diversity_loss_fn(t):
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def entropy(prob, eps=1e-5):
    return (-prob * log(prob, eps=eps)).sum(dim=-1)


def cdist(x, y):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min=0).sqrt()


def entropy_loss_fn(dist, temp):
    prob = (-dist * temp).softmax(dim=-1)
    avg_prob = reduce(prob, '... n l -> n l', 'mean')
    return -entropy(avg_prob).mean()


class Channel_Aware_Selection(nn.Module):
    def __init__(self, in_channels, rates=.5, hid_channels=None):
        super().__init__()
        # self.max = nn.AdaptiveMaxPool2d(1)
        # self.avg = nn.AdaptiveAvgPool2d(1)
        # self.shared_mlp = nn.Sequential(
        #     nn.Linear(in_channels*2, in_channels // 8),
        #     nn.GELU(),
        #     nn.Linear(in_channels // 8, in_channels)
        # )

        self.max_channel = nn.AdaptiveMaxPool2d(1)  # global max pooling
        hid_channels = hid_channels or in_channels // 8
        self.rates = rates
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.Linear(hid_channels, hid_channels * 2),
            nn.GELU(),
            nn.Linear(hid_channels * 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, record_len):
        # x shape [N,C,H,W]
        C = x.shape[1]
        outputs = []
        batch_node_features = regroup(x, record_len)
        for neighbor_feature in batch_node_features:
            L = neighbor_feature.shape[0]
            # max_out = self.max(x) # shape [L,C,1,1]
            # avg_out = self.avg(x) # shape [L,C,1,1]
            # out = torch.cat([max_out,avg_out],dim=1) # shape [L,2C,1,1]
            # out = self.shared_mlp(out) # [L,C]
            out = self.max_channel(neighbor_feature)
            out = rearrange(out, 'L C 1 1 -> L 1 C')
            out = self.fc(out).squeeze(dim=1)  # L,1,C
            channel_weights = F.softmax(out, dim=1)  # [L,C]
            # Apply threshold
            selected_channels = int(C * self.rates)
            _, top_k_indices = torch.topk(channel_weights, selected_channels)
            # 创建掩码
            top_k_mask = torch.zeros_like(channel_weights)
            top_k_mask[:, top_k_indices] = 1  # [L,C]
            output = top_k_mask[..., None, None] * neighbor_feature
            outputs.append(output)  # [L,C,H,W]
        return torch.cat(outputs, dim=0)


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class NaiveDecoder(nn.Module):

    def __init__(self, in_dim, h_dim, n_res_layers=4, res_h_dim=None):
        super(NaiveDecoder, self).__init__()
        kernel = 3
        res_h_dim = default(res_h_dim, h_dim // 2)

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim,
                               kernel_size=7, padding=3, groups=h_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
