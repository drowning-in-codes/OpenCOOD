import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, pack, unpack
from math import log2
from typing import Tuple, Callable


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


class ReZero(nn.Module):
    def __init__(self, in_channels: int, res_channels: int):
        super(ReZero, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.GELU(),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, res_channels: int, nb_layers: int):
        super(ResidualStack, self).__init__()
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels)
                                     for _ in range(nb_layers)
                                     ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 res_channels: int, nb_res_layers: int,
                 downscale_factor: int, hidden_channels=None,
                 ):
        super(Encoder, self).__init__()
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        hidden_channels = hidden_channels or out_channels // 2
        downscale_steps = int(log2(downscale_factor))
        c_channel, n_channel = in_channels, hidden_channels
        if downscale_steps == 0:
            n_channel = out_channels
        layers = []
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.GELU(),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 res_channels: int, nb_res_layers: int,
                 upscale_factor: int, hidden_channels=None
                 ):
        super(Decoder, self).__init__()
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        hidden_channels = hidden_channels or out_channels // 2
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(out_channels, res_channels, nb_res_layers))
        c_channel, n_channel = out_channels, hidden_channels
        if upscale_steps == 0:
            n_channel = out_channels
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.GELU(),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class SrvqLayer(nn.Module):
    def __init__(self, in_channels, embed_dim, codebook_size,
                 codebook_transform=None,
                 init_fn: Callable = identity,
                 rotation_trick=True,
                 input_to_quantize_commit_loss_weight=1.,
                 commitment_weight=1.,
                 frozen_codebook_dim=None, ):
        super().__init__()
        self.codebook_size = codebook_size

        frozen_codebook_dim = default(frozen_codebook_dim, embed_dim)
        codebook = torch.randn(codebook_size, frozen_codebook_dim) * (frozen_codebook_dim ** -.5)
        codebook = init_fn(codebook)
        self.pre_conv = nn.Conv2d(in_channels, embed_dim, 1)

        if not exists(codebook_transform):
            codebook_transform = nn.Linear(frozen_codebook_dim, embed_dim, bias=False)
        self.code_transform = codebook_transform
        self.register_buffer("frozen_codebook", codebook)

        self.rotation_trick = rotation_trick
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight
        self.commitment_weight = commitment_weight

    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)

    def indices_to_codes(self, indices):
        frozen_codes = self.frozen_codebook(indices)
        quantized = self.code_transform(frozen_codes)
        quantized = rearrange(quantized, 'b ... d -> b d ...')
        return quantized

    def forward(self, x):
        x = self.pre_conv(x)  # shape: (L,C,H,W)
        x = rearrange(x, 'b c h w -> b h w c')
        x, inverse_pack = pack_one(x, 'b * d')
        implicit_codebook = self.codebook
        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim=-1)

        quantized = implicit_codebook[indices]
        commit_loss = (
                F.mse_loss(x.detach(), quantized) +
                F.mse_loss(x, quantized.detach()) * self.input_to_quantize_commit_loss_weight
        )
        if self.rotation_trick:
            quantized = rotate_to(x, quantized)
        else:
            quantized = (quantized - x).detach() + x
        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, 'b *')
        quantized = rearrange(quantized, 'b ... d-> b d ...')
        quantized.requires_grad_()
        quantized.retain_grad()

        return quantized, commit_loss * self.commitment_weight, indices


class SimVqLayer(nn.Module):
    def __init__(self, in_channels, embed_dim, codebook_size,
                 codebook_transform=None,
                 init_fn: Callable = identity,
                 rotation_trick=True,
                 input_to_quantize_commit_loss_weight=.1,
                 commitment_weight=1.,
                 frozen_codebook_dim=None, ):
        super().__init__()
        self.codebook_size = codebook_size

        frozen_codebook_dim = default(frozen_codebook_dim, embed_dim)
        codebook = torch.randn(codebook_size, frozen_codebook_dim) * (frozen_codebook_dim ** -.5)
        codebook = init_fn(codebook)
        self.pre_conv = nn.Conv2d(in_channels, embed_dim, 1)

        if not exists(codebook_transform):
            codebook_transform = nn.Linear(frozen_codebook_dim, embed_dim, bias=False)
        self.code_transform = codebook_transform
        self.register_buffer("frozen_codebook", codebook)

        self.rotation_trick = rotation_trick
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight
        self.commitment_weight = commitment_weight

    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)

    def indices_to_codes(self, indices):
        frozen_codes = self.frozen_codebook(indices)
        quantized = self.code_transform(frozen_codes)
        quantized = rearrange(quantized, 'b ... d -> b d ...')
        return quantized

    def forward(self, x):
        x = self.pre_conv(x)  # shape: (L,C,H,W)
        x = rearrange(x, 'b c h w -> b h w c')
        x, inverse_pack = pack_one(x, 'b * d')
        implicit_codebook = self.codebook
        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim=-1)

        quantized = implicit_codebook[indices]
        commit_loss = (
                F.mse_loss(x.detach(), quantized) +
                F.mse_loss(x, quantized.detach()) * self.input_to_quantize_commit_loss_weight
        )
        if self.c:
            quantized = rotate_to(x, quantized)
        else:
            quantized = (quantized - x).detach() + x
        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, 'b *')
        quantized = rearrange(quantized, 'b ... d-> b d ...')
        return quantized, commit_loss * self.commitment_weight, indices


"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""


class CodeLayer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, nb_entries: int):
        super(CodeLayer, self).__init__()
        self.pre_conv = nn.Conv2d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5
        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.pre_conv(x).permute(0, 2, 3, 1)  # shape: (L,H,W,C)
        L = x.shape[0]
        # TODO: L cavs should NOT be flattened
        # flatten = x.reshape(L,-1, self.dim)
        flatten = x.reshape(-1, self.dim)

        # TODO: add cos similarity
        # cos_d = F.cosine_similarity(flatten.unsqueeze(1), self.embed.t().unsqueeze(0), dim=2)

        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )  # shape: (flatten.shape,embedding_shape) (H*W,1200)
        # dist = dist + cos_d
        _, embed_ind = (-dist).max(1)  # shape (H*W,1200)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)  # shape (H*W,1200)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)  # shape (1200)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Upscaler(nn.Module):
    def __init__(self, embed_dim: int, scaling_rates):
        super(Upscaler, self).__init__()

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.GELU())
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)


"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels
    TODO: A lot of this class could do with a refactor. It works, but at what cost?
    TODO: Add disrete code decoding function
"""


class MVQVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels=128,
                 res_channels=32,
                 nb_res_layers: int = 2,
                 nb_levels: int = 3,
                 embed_dim: int = 64,
                 nb_entries: int = 512,
                 downscaling_rates=[2, 2, 2],
                 upscaling_rates=[1, 2, 4],
                 ):
        super(MVQVAE, self).__init__()
        self.nb_levels = nb_levels
        if isinstance(out_channels, int):
            out_channels = [out_channels] * nb_levels
        if isinstance(res_channels, int):
            res_channels = [res_channels] * nb_levels
        assert len(downscaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList(
            [Encoder(in_channels, out_channels[0], res_channels[0], nb_res_layers, downscaling_rates[0])])
        for i, sr in enumerate(downscaling_rates[1:]):
            self.encoders.append(Encoder(out_channels[i], out_channels[i + 1], res_channels[i + 1], nb_res_layers, sr))

        self.codebooks = nn.ModuleList()

        for i in range(nb_levels - 1):
            self.codebooks.append(
                SimVqLayer(in_channels=out_channels[i] + embed_dim, codebook_size=nb_entries, embed_dim=embed_dim,
                           frozen_codebook_dim=embed_dim * 2))
        self.codebooks.append(SimVqLayer(in_channels=out_channels[-1], codebook_size=nb_entries, embed_dim=embed_dim,
                                         frozen_codebook_dim=embed_dim * 2))
        self.decoders = nn.ModuleList([Decoder(embed_dim * nb_levels, out_channels=in_channels,
                                               res_channels=res_channels[0], nb_res_layers=nb_res_layers,
                                               upscale_factor=upscaling_rates[0], hidden_channels=out_channels[0])])
        for i, sr in enumerate(upscaling_rates[1:]):
            self.decoders.append(
                Decoder(embed_dim * (nb_levels - 1 - i), out_channels=embed_dim, res_channels=res_channels[i + 1],
                        nb_res_layers=nb_res_layers, upscale_factor=sr, hidden_channels=out_channels[i + 1]))

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, upscaling_rates[1:len(upscaling_rates) - i][::-1]))

    def forward(self, x, verbose=False):
        encoder_outputs = []
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []

        for enc_index, enc in enumerate(self.encoders):
            enc_output = None
            if len(encoder_outputs):
                enc_output = enc(encoder_outputs[-1])
            else:
                enc_output = enc(x)
            encoder_outputs.append(enc_output)
            if verbose:
                print("encode layer", enc_index, "encode shape", enc_output.shape)

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]

            if len(decoder_outputs):  # if we have previous levels to condition on
                code_q, code_d, emb_id = codebook(torch.cat([encoder_outputs[l], decoder_outputs[-1]], axis=1))
            else:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])
            diffs.append(code_d)
            id_outputs.append(emb_id)

            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u + 1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        decode_output = decoder_outputs[-1]  # shape: (L,C,H,W) L is the total number cav in the scene

        extra_loss = {
            "vq loss": diffs,  # include mse
        }
        # return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs
        return extra_loss, decode_output, None

    def decode_codes(self, *cs):
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]
            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u + 1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]

