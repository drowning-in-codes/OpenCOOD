import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.range_fusion_block import LocalAttention,AgentAttention

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self,input_dim,output_dim:int=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim * 2
        self.conv = nn.Conv2d(input_dim,output_dim,kernel_size=3,stride=2,padding=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self,input_dim,output_dim:int=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim // 2
        self.up =  nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = DoubleConv(input_dim, output_dim, input_dim // 2)
        # self.conv = nn.ConvTranspose2d(input_dim,output_dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()
        
    def forward(self,x,prevShape):
        H,W = prevShape
        x = self.up(x)
        x = F.interpolate(x, size=(H,W), mode="bilinear")
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class LayerNorm(nn.Module):
    """
    From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class VL_Blocks(nn.Module):
    def __init__(self,input_dim,num_head) -> None:
        super().__init__()
        self.l_block = Attn_L_Block(input_dim,num_head)
        self.v_block = Attn_V_Block(input_dim,num_head)

    def combine_lg(self):
        pass

    def forward(self,batch_spatial_feature,record_len):
        assert batch_spatial_feature.ndim == 4, "tensor dim not correct" # single batch feature 
        # split_x = self.regroup(batch_spatial_feature, record_len)
        # out = []
        # TODO how to more efficiently combine local and global feaures
        l_feature = self.l_block(batch_spatial_feature,record_len) # [V,C,H,W]
        g_feature = self.v_block(batch_spatial_feature,record_len) # [V,C,H,W]
        out_feature = l_feature + g_feature
        # out.append(out_feature)
        # out = torch.cat(out,dim=0)
        return out_feature
    
    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
class Attn_Block(nn.Module):
    def __init__(self,input_dim,num_head:int=3,layer_num:int=3):
        super().__init__()
        # self.layer_num = layer_num
        # self.VLblocks = nn.ModuleList()
        self.VL_block = VL_Blocks(input_dim,num_head)
        self.conv_attn = nn.Conv2d(input_dim*2,input_dim,1,1)
        # # TODO how two do multi-scale more efficiently
        # for i in range(self.layer_num):
        #     self.VLblocks.append(VL_Blocks(input_dim*pow(2,i),num_head))
        #     self.downsampleList.append(DownSample(input_dim*pow(2,i),input_dim*pow(2,i+1)))
        #     self.upsampleList.append(Upsample(input_dim*pow(2,i+1),input_dim*pow(2,i)))
    
    def forward(self,spatial_features,record_len): # spatial_features [B,C,H,W]
        assert spatial_features.ndim == 4, "tensor dim not correct"
        _,_,H,W = spatial_features.shape
        split_x = self.regroup(spatial_features, record_len) #spatial_features [5,C,H,W]
        out = []
        for batch_spatial_feature in split_x: # [2,C,H,W]
            batch_spatial_feature = self.VL_block(batch_spatial_feature,record_len) # [V,C,H,W]
            # for i in range(self.layer_num):
            #     batch_spatial_feature = self.VLblocks[i](batch_spatial_feature,record_len) # [V,C,H,W]
            #     batch_spatial_feature = self.downsampleList[i](batch_spatial_feature)
            # for i in range(self.layer_num-1,-1,-1):
            #     batch_spatial_feature = self.upsampleList[i](batch_spatial_feature,(H//pow(2,i),W//pow(2,i)))
            #     # agg_feature = self.VLblocks[i](agg_feature,record_len)
            mean_feat = torch.mean(batch_spatial_feature,dim=0,keepdim=True)
            max_feat = torch.max(batch_spatial_feature,dim=0,keepdim=True)[0]
            agg_feat = torch.cat([mean_feat,max_feat],dim=1) # [1,2C,H,W]
            agg_feat = self.conv_attn(agg_feat) # [1,C,H,W]
            out.append(agg_feat)
        out = torch.cat(out,dim=0)
        return out
            
    
    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
class MultiHead_L(nn.Module):
    def __init__(self,input_dim,num_head):
        super().__init__()
        self.num_head = num_head
        self.attention_l = LocalAttention(input_dim//self.num_head)
    
    def forward(self,spatial_feaure):
        _,C,H,W = spatial_feaure.shape
        spatial_feaure = spatial_feaure.reshape(self.num_head,C//self.num_head,H,W)
        multi_head_feats = []
        for i in range(self.num_head):
            multi_head_feats.append(self.attention_l(spatial_feaure[i].unsqueeze(0))) # [1,C//head,H,W]
        fusion_feature = torch.cat(multi_head_feats,dim=0) # [num_head,C//head,H,W]
        fusion_feature = fusion_feature.reshape(1,-1,H,W) # [1,C,H,W]
        return fusion_feature
    
class MultiHead_G(nn.Module):
    def __init__(self,input_dim,num_head):
        super().__init__()
        self.num_head = num_head
        self.attention_g = AgentAttention(input_dim//self.num_head)

    
    def forward(self,spatial_feaure):
        B,C,H,W = spatial_feaure.shape
        spatial_feaure = spatial_feaure.reshape(B,self.num_head,C//self.num_head,H,W)
        multi_head_feats = []
        for i in range(self.num_head):
            multi_head_feats.append(self.attention_g(spatial_feaure[:,i,...])) # [B,C//head,H,W]
        fusion_feature = torch.cat(multi_head_feats,dim=1) # [B,C,H,W]
        return fusion_feature


class Attn_L_Block(nn.Module):
    def __init__(self,input_dim,num_head:int):
        super().__init__()
        assert input_dim % num_head == 0,f"input_dim can't be devided into {num_head} num_head"
        self.ln = LayerNorm(input_dim)
        self.conv1x1 = nn.Conv2d(input_dim,input_dim,1,1)
        # self.mlp = nn.Linear()
        self.mlp = nn.Conv2d(input_dim,input_dim,3,1,padding=1)

        self.mul_attention = MultiHead_L(input_dim,num_head)


    def forward(self,spatial_features,record_len):
        norm_feature = self.ln(spatial_features)
        multi_head_feat = []
        for vehicle_feature in norm_feature:
            multi_head_feature = self.mul_attention(vehicle_feature.unsqueeze(0)) # [1,C,H,W]
            multi_head_feature = self.conv1x1(multi_head_feature) # [1,C,H,W]
            multi_head_feat.append(multi_head_feature) 
        multi_head_feat = torch.cat(multi_head_feat,dim=0)  # [V,C,H,W]
        
        out_feature = multi_head_feat + norm_feature
        identity_feature = out_feature
        out_feature = self.ln(out_feature)
        out_feature = self.mlp(out_feature)
        out_feature = identity_feature+out_feature # [V,C,H,W]

        identity_feature = out_feature
        out_feature = self.mlp(out_feature)
        out_feature = out_feature + identity_feature
        return out_feature

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x


class Attn_V_Block(nn.Module):
    def __init__(self,input_dim,num_head:int):
        super().__init__()
        assert input_dim % num_head == 0,f"input_dim can't be devided into {num_head} num_head"
        self.ln = LayerNorm(input_dim)
        # self.mlp = nn.Linear()
        self.mlp = nn.Conv2d(input_dim,input_dim,kernel_size=3,stride=1,padding=1)
        self.mul_attention = MultiHead_G(input_dim,num_head)

    def forward(self,batch_spatial_feature,record_len):
        norm_feature = self.ln(batch_spatial_feature) # single batch feauture [B,C,H,W]
        # multi_head_feat = []
        # for vehicle_feature in norm_feature:
        #     multi_head_feature = self.mul_attention(vehicle_feature.unsqueeze(0)) 
        #     multi_head_feat.append(multi_head_feature) 
        # multi_head_feat = torch.cat(multi_head_feat,dim=0)  # [V,C,H,W]
        multi_head_feat = self.mul_attention(norm_feature) # [B,C,H,W]
        out_feature = multi_head_feat + norm_feature
        identity_feature = out_feature
        out_feature = self.ln(out_feature)
        out_feature = self.mlp(out_feature)
        out_feature = identity_feature+out_feature # [V,C,H,W]

        identity_feature = out_feature
        out_feature = self.mlp(out_feature)
        out_feature = out_feature + identity_feature
        
        return out_feature

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x