import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
from opencood.models.communication_modules.hier_vq_comm import MVQVAE, SrvqLayer
from opencood.models.communication_modules.utils import NaiveDecoder
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, x, detector=None):
        out = self.attn(x)
        if out.ndim == 3:
            out = out.unsqueeze(0)
        if detector is not None:
            out = detector(out)
        avgout = torch.mean(out, dim=1, keepdim=True)
        maxout, _ = torch.max(out, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        return self.sigmoid(self.conv(out))


class NaiveEgoFusion(nn.Module):
    def __init__(self, in_channels):
        super(NaiveEgoFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x, ego_feat):
        # x ego_feat shape [1,C,H,W]
        x = self.conv(x)  # shape [L,C,H,W]
        out = x * ego_feat
        return out


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


class Channel_Aware_Selection(nn.Module):
    def __init__(self, in_channels, hid_channels=None, act=True):
        super().__init__()
        self.max_channel = nn.AdaptiveMaxPool2d(1)  # global max pooling
        hid_channels = hid_channels or in_channels // 8
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.GELU() if act else nn.Identity(),
            nn.Linear(hid_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, rates=0.5):
        # x shape [1,C,H,W]
        L, C, _, _ = x.shape  # [1,C,H,W]
        out = self.max_channel(x)
        out = out.view(L, C, -1).transpose(1, 2)  # L,1,C
        out = self.fc(out).squeeze(dim=1)  # L,1,C
        channel_weights = F.softmax(out, dim=1)  # [L,C]
        # Apply threshold
        selected_channels = int(C * rates)
        _, top_k_indices = torch.topk(channel_weights, selected_channels)
        # 创建掩码
        top_k_mask = torch.zeros_like(channel_weights)
        top_k_mask[:, top_k_indices] = 1  # [L,C]
        return top_k_mask[..., None, None] * x


class downscale(nn.Module):
    def __init__(self, in_channels, downscale_strides: int, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=downscale_strides, stride=downscale_strides),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.net(x)


class NaiveCommVQ(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model parameters
        self.in_channels = args.get("in_dim")
        self.n_embeddings = args.get("n_embeddings")
        self.embed_dim = args.get("embed_dim")
        self.channel_rate = args.get("channel_rate", 0.5)
        self.beta = args.get("beta", 1.0)
        self.gamma = args.get("gamma", 1.0)
        combine_feat = args.get("combine_feat", False)

        # Submodules
        self.vqvae_model = SrvqLayer(in_channels=self.in_channels,
                                     codebook_size=self.n_embeddings,
                                     embed_dim=self.embed_dim)
        self.combine_channel_feat = self.channel_rate > 0 and combine_feat

        if self.combine_channel_feat:
            self.channel_selection = Channel_Aware_Selection(self.in_channels)

        self.decoder = NaiveDecoder(self.embed_dim, self.in_channels, n_res_layers=4)

    def forward(self, x, record_len):
        """
        Forward pass for NaiveCommVQ.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W].
            record_len (torch.Tensor): Batch record lengths.

        Returns:
            dict: Dictionary containing loss components.
            torch.Tensor: Reconstructed or enhanced feature tensor.
        """
        batch_features = self.regroup(x, record_len)
        B = len(record_len)

        # Initialize outputs and loss
        x_comm_list = []
        channel_feat_list = []
        losses = {
            "vq_loss": torch.tensor(0.0, device=x.device),
            "mse_loss": torch.tensor(0.0, device=x.device)
        }

        # Process each batch element
        x_hats = []
        for b in range(B):
            neighbor_features = batch_features[b]  # Shape: [L, C, H, W]
            L = neighbor_features.shape[0]

            # Ego feature (first element)
            ego_feature = neighbor_features[:1]  # Shape: [1, C, H, W]
            # Channel-aware feature selection
            if self.combine_channel_feat:
                selected_features = self.channel_selection(neighbor_features, rates=self.channel_rate)
                channel_feat_list.append(selected_features)  # Shape: [L, C, H, W]

            if L < 2:  # Only one vehicle, no communication needed
                x_comm_list.append(ego_feature)
                continue

            # VQ-VAE compression and reconstruction
            x_hat, vq_loss, _ = self.vqvae_model(neighbor_features)
            x_hats.append(x_hat)
            reconstructed_features = self.decoder(x_hat)
            x_comm_list.append(reconstructed_features)

            # Compute losses
            mse_loss = F.mse_loss(neighbor_features, reconstructed_features)
            losses["vq_loss"] += vq_loss
            losses["mse_loss"] += mse_loss

        # Scale losses
        losses["vq_loss"] *= self.beta
        losses["mse_loss"] *= self.gamma

        # Combine features
        x_comm = torch.cat(x_comm_list, dim=0)  # Shape: [B, C, H, W]

        if self.combine_channel_feat:
            channel_feat = torch.cat(channel_feat_list, dim=0)  # Shape: [B, C, H, W]
            x_comm += channel_feat
        x_hats = torch.cat(x_hats, dim=0)  # Shape: [B, C, H, W]
        return losses, x_comm,x_hats

    @staticmethod
    def regroup(x, record_len):
        """
        Regroup batch features based on record lengths.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W].
            record_len (torch.Tensor): Record lengths for each batch element.

        Returns:
            list: List of tensors split by record length.
        """
        cum_sum_len = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum_sum_len[:-1].cpu())


class Commvq(nn.Module):
    def __init__(self, args, fuse_ego_attention=True):  # channel_rate=.5
        super(Commvq, self).__init__()
        in_channels = args["in_dim"]
        self.channel_rate = args["channel_rate"] if "channel_rate" in args else 0.5
        self.vqvae_model = MVQVAE(in_channels=in_channels, out_channels=args["out_dim"],
                                  res_channels=args["n_residual_hiddens"], nb_res_layers=args['n_residual_layers'],
                                  nb_levels=args["embedding_level"], embed_dim=args["embedding_dim"],
                                  nb_entries=args['n_embeddings'], downscaling_rates=args["downscaling_rates"],
                                  upscaling_rates=args["upscaling_rates"])
        self.channel_selection = Channel_Aware_Selection(in_channels)
        self.ego_attention = SpatialRequestAttention(in_channels)
        self.fuse_ego_attention = fuse_ego_attention
        self.fusion_net = NaiveEgoFusion(in_channels) if fuse_ego_attention else None
        self.beta = args["beta"] if 'beta' in args else 1.0
        self.gamma = args["gamma"] if 'gamma' in args else 1.0
        self.downscale_strides = args["downscale_strides"]
        if self.downscale_strides > 1:
            self.downscale = downscale(in_channels, self.downscale_strides)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, detector):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """
        B = len(record_len)  # batch size
        # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
        # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
        batch_node_features = self.regroup(x, record_len)
        x_comm = []
        extra_all_loss = {"vq loss": torch.tensor(0.0, device=x.device), "mse loss": torch.tensor(0.0, device=x.device)}
        channel_select_feat = []
        for b in range(B):
            neighbor_feature = batch_node_features[b]  # [L,C,H,W]
            L = neighbor_feature.shape[0]
            # spatial request attention
            ego_feat = neighbor_feature[:1]  # [1,C,H,W]
            if L < 2:  # L == 1, only one cav, no need to communicate
                # enhance ego feature
                x_comm.append(ego_feat)
                channel_select_feat.append(ego_feat)
                continue
            ego_attn = self.ego_attention(ego_feat, detector)  # [1,C,H,W]
            ego_request = 1 - ego_attn
            # Communication between ego and neighbor cavs
            communicated_feat = neighbor_feature[1:]  # [L-1,C,H,W]
            assert communicated_feat.shape[0] != 0 and communicated_feat.ndim == 4
            cav_channel_select_feat = []
            cavs_feat = []
            for i in range(L - 1):
                single_cav_feat = communicated_feat[i].unsqueeze(0)  # [1,C,H,W]
                # communicated_feat  cavs in the same scene [L-1,C,H,W] or [L,C,H,W]
                select_feat = self.channel_selection(single_cav_feat, rates=self.channel_rate)
                cav_channel_select_feat.append(select_feat)  # [1,C,H,W]
                if self.fuse_ego_attention:
                    single_cav_feat = self.fusion_net(single_cav_feat, ego_request)
                else:
                    single_cav_feat = single_cav_feat * ego_request
                cavs_feat.append(single_cav_feat)
            communicated_feat = torch.cat([ego_feat] + cavs_feat, dim=0)  # [L-1,C,H,W]
            cav_channel_select_feat = torch.cat(cav_channel_select_feat, dim=0)  # [L-1,C,H,W]
            channel_select_feat.append(torch.cat([ego_feat, cav_channel_select_feat], dim=0))  # [L,C,H,W]
            # for compute mi loss
            extra_loss, x_hat = self.vqvae_model(communicated_feat)
            mse_loss = F.mse_loss(communicated_feat, x_hat)
            x_comm.append(x_hat)
            extra_all_loss["mse loss"] += mse_loss
            extra_all_loss["vq loss"] += sum(extra_loss["vq loss"]) if isinstance(extra_loss["vq loss"], list) else \
            extra_loss["vq loss"]
        extra_all_loss["vq loss"] *= self.beta
        extra_all_loss["mse loss"] *= self.gamma
        channel_select_feat = torch.cat(channel_select_feat, dim=0)  # [B,C,H,W]
        x_comm = torch.cat(x_comm, dim=0)  # [B,C,H,W]   vq based reconstructed features

        x_comm = x_comm + channel_select_feat
        return extra_all_loss, x_comm
