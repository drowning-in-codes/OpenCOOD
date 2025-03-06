from math import log2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Cython.Compiler.Options import embed
from jsonschema.benchmarks.unused_registry import instance
from torch import einsum
from einops import rearrange,reduce
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
from opencood.models.communication_modules.utils import default, exists, rotate_to, identity, efficient_rotation_trick_transform, \
    pack_one,diversity_loss_fn,entropy_loss_fn


class GlobalDiscriminator(nn.Module):
    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        hidden_channels = in_channels //2
        out_channels = out_channels or hidden_channels //2
        self.c0 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3,padding=1)
        self.c1 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.l0 = nn.Linear(in_channels*2, hidden_channels)
        self.l1 = nn.Linear(hidden_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.l2 = nn.Linear(out_channels, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = self.bn1(h)
        h = self.max_pool(h) # B C 1 1
        h = h.view(y.shape[0], -1) # [B,C]
        h = torch.cat((y, h), dim=1) # [B,2C]
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        h = self.bn2(h)
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self,in_channels,hid_channels=None):
        super().__init__()
        hid_channels = hid_channels or in_channels * 2
        self.c0 = nn.Conv2d(in_channels, hid_channels, kernel_size=1)
        self.c1 = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(hid_channels)
        self.c2 = nn.Conv2d(hid_channels, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        h = self.bn(h)
        return self.c2(h)


class DeepInfoMaxLoss(nn.Module):
    # adopted from https://github.com/DuaneNielsen/DeepInfomaxPytorch/blob/master/train.py
    def __init__(self, in_channels,alpha=0.5, beta=1.0):
        super().__init__()
        self.global_d = GlobalDiscriminator(in_channels)
        self.local_d = LocalDiscriminator(in_channels*2)

        self.maxout = nn.AdaptiveMaxPool2d(1)
        self.alpha = alpha
        self.beta = beta

    def forward(self, y, M, M_prime):
        assert y.shape == M.shape == M_prime.shape,"y shape {y.shape} M shape {M.shape} M_prime shape {M_prime.shape}"
        B,C,H,W = y.shape
        # B C H W
        y_M = torch.cat((M, y), dim=1)
        y_M_prime = torch.cat((M_prime, y), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta
        encoded_y = self.maxout(y).reshape(B,C) # B C

        Ej = -F.softplus(-self.global_d(encoded_y, M)).mean()
        Em = F.softplus(self.global_d(encoded_y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha
        return LOCAL + GLOBAL
# y, M = encoder(x)
# rotate images to create pairs for comparison
# M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
# loss = loss_fn(y, M, M_prime)
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

class SpatialRequestAttention(nn.Module):
    def __init__(self, in_channels, ):
        super(SpatialRequestAttention, self).__init__()
        self.attn = AttentionFusion(in_channels)
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, detector):
        out = detector(x)
        out = self.attn(out)
        avgout = torch.mean(out, dim=1, keepdim=True)
        maxout, _ = torch.max(out, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        return self.sigmoid(self.conv(out))


class EgoFusion(nn.Module):
    def __init__(self, in_channels):
        super(EgoFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
        )
        self.kv = nn.Linear(in_channels, in_channels * 2)

    def forward(self, x, ego_feat):
        # x ego_feat shape [1,C,H,W]
        _, C, H, W = x.shape
        assert x.ndim == 4 and ego_feat.ndim == 4, f"x shape {x.shape}"
        out = self.conv(x)  # shape [1,C,H,W]
        out = rearrange(out, 'L C H W -> L (H W) C')
        ego_feat = rearrange(ego_feat, 'L C H W -> L (H W) C')
        k, v = self.kv(out).chunk(2, dim=-1)  # [1,H*W,C]
        logits = ego_feat.expand(-1, -1, C) @ k.transpose(1, 2)  # [1,H*W,H*W]
        attn = F.softmax(logits, dim=-1)
        out = attn @ v
        out = rearrange(out, 'L (H W) C -> L C H W', H=H, W=W)
        return out

class ReZero(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int):
        super(ReZero, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 3, stride=1, padding=1, bias=False),
            nn.GELU(),

            nn.Conv2d(hid_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, nb_layers: int):
        super(ResidualStack, self).__init__()
        self.stack = nn.Sequential(*[ReZero(in_channels, hid_channels)
                        for _ in range(nb_layers)
                    ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)

class Encoder(nn.Module):
    def __init__(self,
            in_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            downscale_factor: int,
        ):
        super(Encoder, self).__init__()
        assert isinstance(log2(downscale_factor),int) , "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))

        layers = []
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(in_channels //2 , in_channels , 3, stride=1, padding=1),
            ))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        layers.append(ResidualStack(out_channels, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self,
            in_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            upscale_factor: int
        ):
        super(Decoder, self).__init__()
        assert isinstance(log2(upscale_factor),int), "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                  ResidualStack(out_channels, res_channels, nb_res_layers)]
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels // 2, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(out_channels // 2, out_channels , 3, stride=1, padding=1)))
        layers.append(nn.Conv2d(out_channels , out_channels, 3, stride=1, padding=1))
        layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

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
              mw * x_attn[:, :, :, :, self.input_dim:2*self.input_dim] +\
              bw * x_attn[:, :, :, :, self.input_dim*2:]

        return out

class Upscaler(nn.Module):
    def __init__(self,embed_dim: int,out_dim,scaling_rates,nb_res_layers:int=4):
        super(Upscaler, self).__init__()

        self.stages = nn.ModuleList()
        layers = [
            nn.Conv2d(embed_dim, out_dim, 3, stride=1, padding=1),
            ResidualStack(out_dim, embed_dim, nb_res_layers)]
        for _ in range(scaling_rates):
            layers.append(nn.ConvTranspose2d(out_dim, out_dim , 4, stride=2, padding=1))
            layers.append(nn.GELU())
            layers.append(nn.ConvTranspose2d(out_dim , out_dim//2, 7, stride=1, padding=3,groups=out_dim//2))
            layers.append(nn.ConvTranspose2d(out_dim//2, out_dim, 3, stride=1, padding=1))
        self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stages(x)

class SimVQ(nn.Module):
    def __init__(self,heads,dim,codebook_size,rotation_trick=True,
                 input_to_quantize_commit_loss_wight=.25,commitment_weight=1.,
                 entropy_loss_weight=1.,diversity_loss_weight=1.,temperature=10.,
                 frozen_codebook_dim=None):
        super(SimVQ, self).__init__()
        self.dim = dim

        frozen_codebook_dim = default(frozen_codebook_dim, dim)
        self.codebook_size = codebook_size
        self.rotation_trick = rotation_trick
        codebook = torch.randn(heads,codebook_size,frozen_codebook_dim)* (frozen_codebook_dim ** -0.5)

        self.code_transform = nn.Linear(frozen_codebook_dim,dim,bias=False)
        self.register_buffer("frozen_codebook",codebook)
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_wight
        self.commitment_weight = commitment_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.temp = temperature
    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)


    def forward(self,x):
        x,inverse_pack = pack_one(x,'h * d')
        implicit_codebook = self.codebook
        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim = -1)
        quantized = implicit_codebook[indices]
        commit_loss = F.mse_loss(x.detach(),quantized) + F.mse_loss(x,quantized.detach()) * self.input_to_quantize_commit_loss_weight
        if self.rotation_trick:
            quantized = rotate_to(x,quantized)
        else:
            quantized = (quantized - x).detach() + x

        entropy_loss = entropy_loss_fn(dist,self.temp)
        diversity_loss = diversity_loss_fn(implicit_codebook)
        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices,'h *')
        return quantized,indices,commit_loss*self.commitment_weight,entropy_loss*self.entropy_loss_weight,diversity_loss*self.diversity_loss_weight


class MFQCodebook(nn.Module):
    def __int__(self,in_channels,n_embeddings_h,n_embeddings_l,embed_dim_h,embed_dim_l,heads):
        super(MFQCodebook, self).__init__()
        self.heads = heads
        self.h_codebook = SimVQ(in_channels,heads,n_embeddings_h,embed_dim_h)
        self.l_codebook = SimVQ(in_channels,heads,n_embeddings_l,embed_dim_l)
        self.conv = nn.Sequential(
                    nn.Conv2d(embed_dim_h, embed_dim_h , 3, padding=1),
                    nn.Conv2d(embed_dim_h , embed_dim_h * 2, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(embed_dim_h * 2, embed_dim_h, 1, padding=1),
        )
        self.prev_conv_h = nn.Conv2d(in_channels,embed_dim_h,1)
        self.prev_conv_l = nn.Conv2d(in_channels,embed_dim_l,1)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim_l,embed_dim_h),
            nn.Linear(embed_dim_h, embed_dim_h),
        )
    def forward(self,x,detector,threshold=0.1):
        assert x.ndim == 4, "x shape must be [B,C,H,W]."
        h,w = x.shape[2:]
        x_h = self.prev_conv_h(x)
        confidence_spatial_map_h = F.sigmoid(detector(x_h))
        salient_mask = confidence_spatial_map_h >= threshold
        salient_feature = torch.where(salient_mask, x, torch.tensor(1e-5, device=x.device))

        x_l = self.prev_conv_l(x)
        confidence_spatial_map_l = F.sigmoid(detector(x_l))
        background_mask = confidence_spatial_map_l < threshold
        background_feature = torch.where(background_mask, x, torch.tensor(1e-5, device=x.device))

        salient_feature = rearrange(salient_feature,'b c h w -> b (h w) c')
        multi_head_salient_feature = rearrange(salient_feature,'b n (h d) -> h b n d',h=self.heads)
        multi_head_background_feature = rearrange(background_feature,'b n (h d) -> h b n d',h=self.heads)

        # quantized [h, B, N, D], indices [h, B, N]
        quantize_h, embed_ind_h, commit_loss_h,entropy_loss_h,diversity_loss_h = self.h_codebook(multi_head_salient_feature)
        quantize_l, embed_ind_l, commit_loss_l,entropy_loss_l,diversity_loss_l = self.l_codebook(multi_head_background_feature)

        quantize_l = self.proj(quantize_l)
        quantize = quantize_h + quantize_l
        # to feature map
        quantize =  rearrange(quantize,'h b (h w) d -> b (h d) h w',h=h,w=w)
        quantize = self.conv(quantize)
        commit_loss = commit_loss_h + commit_loss_l
        entropy_loss = entropy_loss_h + entropy_loss_l
        diversity_loss = diversity_loss_h + diversity_loss_l
        return quantize,commit_loss, entropy_loss,diversity_loss


class MRFCodeLayer(nn.Module):
    def __init__(self,in_channels,num_quantizers,heads,n_embeddings_h,n_embeddings_l,embed_dim_h,embed_dim_l,drop_rate=.1):
        super().__init__()
        self.num_quantizers = num_quantizers
        codebook_dict = {
            "in_channels": in_channels,
            "n_embeddings_h": n_embeddings_h,
            "n_embeddings_l": n_embeddings_l,
            "embed_dim_h": embed_dim_h,
            "embed_dim_l": embed_dim_l,
            "heads": heads
        }
        self.layers = nn.ModuleList([
            MFQCodebook(**codebook_dict) for _ in range(num_quantizers)
        ])
        self.project_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.Conv2d(in_channels  // 2, in_channels, 3,padding=1),
            nn.GELU()
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self,x):
        quantized_out = 0.
        residual = x

        commitment_loss_list = []
        entropy_loss_list = []
        diversity_loss_list = []
        all_residuals = []
        for quantizer_index,mfq_codebook in enumerate(self.layers):
            all_residuals.append(residual)
            quantized,commitment_loss,entropy_loss,diversity_loss = mfq_codebook(residual)
            residual = residual - quantized.detach()
            quantized_out += quantized
            commitment_loss_list.append(commitment_loss)
            entropy_loss_list.append(entropy_loss)
            diversity_loss_list.append(diversity_loss)

        commitment_loss =  sum(commitment_loss_list)
        entropy_loss =  sum(entropy_loss_list)
        diversity_loss =  sum(diversity_loss_list)
        quantized_out = self.project_out(quantized_out)
        return quantized_out,commitment_loss,entropy_loss,diversity_loss


class MRF_vqvae(nn.Module):
    def __init__(self,args):
        super().__init__()
        in_channels = args["in_dim"]
        out_channels = args["out_dim"]
        levels = args["levels"]
        embed_dim_h = args["embedding_dim_h"]
        embed_dim_l = args["embedding_dim_l"]
        n_embeddings_h = args["n_embeddings_h"]
        n_embeddings_l = args["n_embeddings_l"]
        downscaling_rates = args["downsampling_rates"]
        upscaling_rates = args["upscaling_rates"]
        res_channels = args["res_channels"]
        res_layer_nums = args["res_layers"]
        num_quantizers = args["num_quantizers"]
        heads = args["heads"]
        if isinstance(n_embeddings_h,int):
            n_embeddings_h = [n_embeddings_h]*levels
        if isinstance(n_embeddings_l, int):
            n_embeddings_l = [n_embeddings_l] * levels
        if instance(out_channels,int):
            out_channels = [out_channels]*levels
        assert levels == len(out_channels) == len(downscaling_rates) == len(upscaling_rates), "levels must be equal to len(downscaling_rates) and len(upscaling_rates)"

        dims = [in_channels] + out_channels
        self.encoders = nn.ModuleList([])
        for i,sr in enumerate(downscaling_rates):
            self.encoders.append(Encoder(dims[i],dims[i+1],res_channels,res_layer_nums,sr))
        self.codelayers = nn.ModuleList([])
        for i in range(levels):
            self.codelayers.append(MRFCodeLayer(out_channels[i], num_quantizers=num_quantizers,heads=heads,
                                                n_embeddings_h=n_embeddings_h[i],n_embeddings_l=n_embeddings_l[i],
                                                embed_dim_h=embed_dim_h,embed_dim_l=embed_dim_l))
        self.decoders = nn.ModuleList([])
        # for i,sr in enumerate(upscaling_rates):
        #     self.decoders.append(Decoder(in_channels=embed_dim,out_channels=in_channels,res_channels=res_channels,nb_res_layers=res_layer_nums,upscale_factor=sr))
        self.upscalers = nn.ModuleList()
        for i in range(levels-1,0,-1):
            self.upscalers.append(Upscaler(embed_dim_h,in_channels,upscaling_rates[i]))
        self.adafusion = SplitAttn(in_channels)

    def forward(self,x,detector,verbose=False):
        encoder_outputs = []
        code_outputs = []
        cdiffs = []
        ediffs = []
        ddiffs = []

        for enc_index, enc in enumerate(self.encoders):
            if len(encoder_outputs):
                enc_output = enc(encoder_outputs[-1])
            else:
                enc_output = enc(x)
            encoder_outputs.append(enc_output)
            if verbose:
                print("encode layer", enc_index, "encode shape", enc_output.shape)

        for l in range(self.nb_levels - 1, -1, -1):
            codebook = self.codelayers[l]
            code_q, commitment_loss,entropy_loss,diversity_loss= codebook(encoder_outputs[l],detector)
            cdiffs.append(commitment_loss)
            ediffs.append(entropy_loss)
            ddiffs.append(diversity_loss)
            if l > 0:
                # interpolate to the same size
                code_q = self.upscalers[l-1](code_q)
            code_outputs.append(code_q)

        fuse_output = self.adafusion(code_outputs)
        extra_loss = {
            "vq loss": sum(cdiffs),
            "ent loss":sum(ediffs),
            "div loss":sum(ddiffs),
        }
        # return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs
        return extra_loss, fuse_output


class HMRFQuantizaton(nn.Module):
    def __init__(self,args):
        super().__init__()
        in_channels = args["in_dim"]
        self.vqvae_model = MRF_vqvae(args["vqvae_model"])
        self.mi_loss_module = DeepInfoMaxLoss(in_channels)
        self.ego_attention = SpatialRequestAttention(in_channels)
        self.fusion_net = EgoFusion(in_channels)
        self.beta = args["beta"] if "beta" in args else 1.0
        self.gamma = args["gamma"] if "gamma" in args else 1.0

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len,detector):
        B = x.shape[0]
        # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
        # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
        batch_node_features = self.regroup(x, record_len)
        x_comm = []
        extra_all_loss = {"vq loss": 0,"mi loss":0,"ent loss":0,"div loss":0}
        for b in range(B):
            neighbor_feature = batch_node_features[b] # [L, C, H, W]
            L = neighbor_feature.shape[0]
            ego_feat = neighbor_feature[:1] # [1, C, H, W]
            if L< 2:
                x_comm.append(ego_feat)
                # single agent
                continue
            ego_attn = self.ego_attention(ego_feat, detector) # [1,C,H,W]
            ego_request = 1 - ego_attn
            # communication between ego and neighbors
            communicated_feat = neighbor_feature[1:]
            cavs_feat = []
            for i in range(L-1):
                single_cav_feat = communicated_feat[i:i+1]
                single_cav_feat = self.fusion_net(single_cav_feat,ego_request)
                cavs_feat.append(single_cav_feat)
            cavs_feat = torch.cat(cavs_feat,dim=0) # [L-1,C,H,W]
            enhanced_feat = torch.cat([ego_feat, cavs_feat], dim=0)  # [L,C,H,W]
            # for compute mi loss
            extra_loss, x_hat, _ = self.vqvae_model(enhanced_feat)
            refined_feat = torch.cat([ego_feat, x_hat], dim=0)
            neighbor_prime = torch.cat([neighbor_feature[1:], ego_feat], dim=0)
            mi_loss = self.mi_loss_module(refined_feat,neighbor_feature,neighbor_prime) # y M M_prime
            x_comm.append(refined_feat)
            extra_all_loss["vq loss"] +=  extra_loss["vq loss"]
            extra_all_loss["ent loss"] +=  extra_loss["ent loss"]
            extra_all_loss["div loss"] +=  extra_loss["div loss"]
            extra_all_loss["mi loss"] += mi_loss
        extra_all_loss["vq loss"] *= self.beta
        extra_all_loss["mi loss"] *= self.gamma
        # extra_all_loss["ent loss"] *= self.gamma
        # extra_all_loss["div loss"] *= self.gamma
        x_comm = torch.cat(x_comm,dim=0) # [B,C,H,W]   vq based reconstructed features
        return extra_all_loss, x_comm
