import torch
import torch.nn as nn
import torch.nn.functional as F
from Cython.Compiler.Options import embed
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torch import einsum
import math


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ReZero(nn.Module):
    def __init__(self, in_channels, res_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, 1, 1),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(),

            nn.Conv2d(res_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.alpha = nn.Parameter(torch.tentor(.0))

    def forward(self, x):
        return x + self.alpha * self.layers(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, res_channels, nb_layers):
        super().__init__()
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) for _ in range(nb_layers)])

    def forward(self, x):
        return self.stack(x)


class StageEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, res_channels, nb_res_layers,
                 downscale_factor=2):
        super().__init__()
        assert math.log2(downscale_factor) % 1 == 0, "downsample ratio must be a power of 2"
        hidden_channels = hidden_channels or res_channels
        downscale_steps = int(math.log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                    LayerNorm(n_channel, eps=1e-6, data_format="channels_first"),
                    nn.GELU())
                # nn.Conv2d(hidden_channels, res_channels, kernel_size=1, stride=1),
                # LayerNorm(res_channels, eps=1e-6, data_format="channels_first"),
                # nn.GELU(),
            )
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, kernel_size=3, stride=1, padding=1))
        layers.append(LayerNorm(n_channel, eps=1e-6, data_format="channels_first"))
        self.layers = nn.Sequential(*layers)
        self.stage = nn.Sequential(*[Block(n_channel, layer_scale_init_value=1e-6) for _ in range(nb_res_layers)])
        self.norm = nn.LayerNorm(n_channel, eps=1e-6)

    def forward(self, x):
        x = self.layers(x)
        x = self.stage(x)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nb_res_layers,
                 upscale_ratio=2):
        super().__init__()
        assert math.log2(upscale_ratio) % 1 == 0, "upscale_ratio must be a power of 2"
        hidden_channels = hidden_channels or out_channels
        upscale_steps = int(math.log2(upscale_ratio))
        layers = [nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)]
        self.layers.append(
            nn.Sequential(*[Block(hidden_channels, layer_scale_init_value=1e-6) for _ in range(nb_res_layers)]))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(n_channel, eps=1e-6)

    def forward(self, x):
        x = self.layers(x)
        x = self.norm(x)
        return x


class CodeLayer(nn.Module):
    def __init__(self, in_channels, embed_dim, codebook_size, decay=0.99, num_codebook=4, temperature=1.0):
        super().__init__()
        assert embed_dim % num_codebook == 0, "embed_dim must be divisible by num_codebook"
        self.embed_dim = embed_dim
        self.n_embed = codebook_size
        self.decay = decay
        self.esp = 1e-5
        self.temp = temperature
        self.num_codebook = num_codebook
        codebook_dim = embed_dim // num_codebook
        self.embed = torch.randn(num_codebook, codebook_size, codebook_dim,
                                 dtype=torch.float32)  # shape [num_codebook, codebook_size, embed_dim]
        self.codebook_dim = codebook_dim
        self.scale = codebook_dim ** -0.5
        self.to_k = nn.Linear(codebook_dim, codebook_dim)
        self.to_v = nn.Linear(codebook_dim, codebook_dim)
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)

    def forward(self, x):
        assert x.shape[1] == self.embed_dim
        H, W = x.shape[2], x.shape[3]
        # shape [1,C,H,W]
        x = self.conv_in(x).permute(0, 2, 3, 1)  # -> [1,H,W,C]
        # flatten_x = x.reshape(-1, self.embed_dim).unsqueeze(1).repeat(1, self.num_codebook, 1) # -> shape [H*W, num_codebook, C]
        q = rearrange(x, 'b h w (t d) -> b t (h w) d', t=self.num_codebook)
        q = q * self.scale
        k, v = self.to_k(self.embed), self.to_v(self.embed)
        # dist = flatten_x.pow(2).sum(1, keepdim=True) - 2 * flatten_x @ self.embed + self.embed.pow(2).sum(0,
        #                                                                                                   keepdim=True)
        logits = einsum('b h i d, h j d -> b h i j', q, k)
        if self.training:
            attn = F.gumbel_softmax(logits, tau=self.temperature, dim=-1, hard=True)
            codebook_indices = attn.argmax(dim=-1)
        else:
            codebook_indices = logits.argmax(dim=-1)
            attn = F.one_hot(codebook_indices, num_classes=self.num_codes).float()

        out = einsum('b h i j, h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b t (h w) d -> b (t d) h w', t=self.num_codebook, h=H, w=W)
        return out, codebook_indices
        # find closest encodings
        # _, embed_ind = (-dist).max(1)
        # embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten_x.dtype)
        # embed_ind = embed_ind.view(*x.shape[:-1])
        # quantize = self.embed_code(embed_ind)
        #
        # if self.training:
        #     embed_onehot_sum = embed_onehot.sum(0)
        #     embed_sum = flatten_x.transpose(0, 1) @ embed_onehot
        #     self.cluster_size.data.mul_(self.decay).add_(
        #         embed_onehot_sum, alpha=1 - self.decay
        #     )
        #     self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        #     n = self.cluster_size.sum()
        #     cluster_size = (
        #             (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
        #     )
        #     embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
        #     self.embed.data.copy_(embed_normalized)
        #
        # diff = (quantize.detach() - x).pow(2).mean()
        # quantize = x + (quantize - x).detach()

        # return quantize.permute(0, 3, 1, 2), diff, embed_ind

    # def embed_code(self, embed_id: torch.LongTensor):
    #     return F.embedding(embed_id, self.embed.transpose(0, 1))


class Upscaler(nn.Module):
    def __init__(self, embed_dim: int, scaling_rates):
        super().__init__()
        assert isinstance(scaling_rates,list),"scaling_rates must be a list"
        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(math.log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.ReLU())
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor,stage:int ) -> torch.FloatTensor:
        return self.stages[stage](x)


class VQComm(nn.Module):
    def __init__(self, in_channels, level_num, codebook_size, scaling_rates, num_codebook=4, embed_dims=None,
                 hidden_channels=None, nb_res_layers=2,
                 temperature=1.0):
        super().__init__()
        assert len(embed_dims) == level_num and len(
            scaling_rates) == level_num, "Number of embed_dims and scaling_list must match level_num"
        self.level_num = level_num
        if hidden_channels is None:
            hidden_channels = 128
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * level_num

        if embed_dims is None:
            embed_dims = 64
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * level_num

        self.encoders = nn.ModuleList(
            [StageEncoder(in_channels, hidden_channels, embed_dims[0], nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:], start=1):
            self.encoders.append(
                StageEncoder(hidden_channels, hidden_channels, embed_dims[i], nb_res_layers, downscale_factor=sr))

        self.codebooks = nn.ModuleList()
        for i in range(level_num - 1):
            self.codebooks.append(
                CodeLayer(hidden_channels + embed_dims[i], embed_dims[i], codebook_size, num_codebook=num_codebook,
                          temperature=temperature))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dims[-1], codebook_size, num_codebook=num_codebook,
                                        temperature=temperature))

        self.decoders = nn.ModuleList([Decoder(embed_dims[0] * level_num, hidden_channels, in_channels, nb_res_layers,
                                               upscale_ratio=scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:], start=1):
            self.decoders.append(
                Decoder(embed_dims[i] * (level_num - 1 - i), hidden_channels, embed_dims[i], nb_res_layers,
                        upscale_ratio=sr))
        self.upscalers = nn.ModuleList()
        for i in range(level_num - 1):
            self.upscalers.append(
                Upscaler(embed_dims[level_num - 1 - i], scaling_rates[1:len(scaling_rates) - i][::-1]))

    def forward(self, x):
        encoder_outputs = []
        code_outputs = []
        decoder_outputs = []
        upscale_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []

        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        for l in range(self.level_num-1,-1,-1):
            codebook,decoder = self.codebooks[l],self.decoders[l]
            if len(decoder_outputs):
                out,emb_id = codebook(torch.cat([encoder_outputs[l],decoder_outputs[-1]],dim=1))
            else:
                out,emb_id = codebook(encoder_outputs[l])

            id_outputs.append(emb_id)
            code_outputs = [self.upscalers[i](c,upscale_counts[i]) for i,c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([out,*code_outputs],dim=1)))
            code_outputs.append(out)
            upscale_counts.append(0)
        return decoder_outputs[-1],encoder_outputs,decoder_outputs,id_outputs



