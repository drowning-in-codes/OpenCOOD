import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.range_fusion_block import LocalAttention,AgentAttention


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



class Attn_Block(nn.Module):
    def __init__(self,num_head:int=3):
        self.v_block = Attn_V_Block()
        self.l_block = Attn_L_Block()

    
    def forward(self,spatial_features,record_len):
        pass

    
class MultiHead_L(nn.Module):
    def __init__(self,input_dim,num_head):
        self.num_head = num_head
        self.attention_l = LocalAttention(input_dim//self.num_head)

    
    def forward(self,spatial_feaure):
        _,C,H,W = spatial_feaure.shape
        spatial_feaure = spatial_feaure.reshape(1,self.num_head,C//self.num_head,H,W)
        fusion_feature = self.attention_l(spatial_feaure).reshape(1,-1,H,W)
        return fusion_feature
    
class MultiHead_G(nn.Module):
    def __init__(self,input_dim,num_head):
        self.num_head = num_head
        self.attention_g = AgentAttention(input_dim//self.num_head)

    
    def forward(self,spatial_feaure):
        _,C,H,W = spatial_feaure.shape
        spatial_feaure = spatial_feaure.reshape(1,self.num_head,C//self.num_head,H,W)
        fusion_feature = self.attention_g(spatial_feaure).reshape(1,-1,H,W)

        return fusion_feature

class Attn_L_Block(nn.Module):
    def __init__(self,input_dim,num_head:int):
        super.__init__()
        assert input_dim % num_head == 0,f"input_dim can't be devided into {num_head} num_head"
        self.ln = nn.BatchNorm2d(input_dim)
        self.conv1x1 = nn.Conv2d(input_dim*2,input_dim,1,1)
        # self.mlp = nn.Linear()
        self.mlp = nn.Conv2d(input_dim,input_dim,3,1,padding=1)

        self.mul_attention = MultiHead_L(input_dim,num_head)


    def forward(self,spatial_features,record_len):
        split_x = self.regroup(spatial_features, record_len) #spatial_features [5,C,H,W]
        out = []
        for batch_spatial_feature in split_x: # [2,C,H,W]
            v,c,h,w = batch_spatial_feature.shape
            norm_feature = self.ln(batch_spatial_feature)
            multi_head_feat = []
            for vehicle_feature in norm_feature:
                multi_head_feature = self.mul_attention(vehicle_feature.unsqueeze(0)) # [V,2C,H,W]
                multi_head_feature = self.conv1x1(multi_head_feature) # [V,C,H,W]
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

            out.append(out_feature)
        
        out = torch.cat(out,dim=0)
        return out

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x


class Attn_V_Block(nn.Module):
    def __init__(self,input_dim,num_head:int):
        super.__init__()
        assert input_dim % num_head == 0,f"input_dim can't be devided into {num_head} num_head"
        self.ln = nn.BatchNorm2d(input_dim)
        # self.mlp = nn.Linear()
        self.mlp = nn.Conv2d(input_dim,input_dim,3,1,padding=1)
        self.mul_attention = MultiHead_G(input_dim,num_head)


    def forward(self,spatial_features,record_len):
        split_x = self.regroup(spatial_features, record_len) #spatial_features [5,C,H,W]
        out = []
        for batch_spatial_feature in split_x: # [2,C,H,W]
            v,c,h,w = batch_spatial_feature.shape
            norm_feature = self.ln(batch_spatial_feature)
            # norm_feature = self.ln(batch_spatial_feature)
            multi_head_feat = []
            for vehicle_feature in norm_feature:
                multi_head_feature = self.mul_attention(vehicle_feature.unsqueeze(0)) 
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

            out.append(out_feature)
        
        out = torch.cat(out,dim=0)
        return out

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x