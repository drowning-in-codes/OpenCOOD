from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
 

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


def linear_map(x, range=[-1.0, 1.0]):
    span = range[1] - range[0]
    k = span / (torch.max(x) - torch.min(x))
    b = range[1] - torch.max(x) * k
    return (k * x + b)

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
        

class RangeAttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.agent_att = AgentAttention(input_dim)
        self.local_att = LocalAttention(input_dim)
        # self.conv3x3 = nn.Conv2d(input_dim, input_dim, kernel_size=3,padding=1, bias=True)
        # self.convattn = nn.AdaptiveAvgPool2d
        # self.convattn =  nn.Conv2d(input_dim*2, input_dim, kernel_size=1, bias=True)
        
        self.query = nn.Sequential(
            # nn.Conv2d(input_dim, input_dim*2, kernel_size=3,padding=1, bias=True),
            # nn.Conv2d(input_dim*2, input_dim*2, kernel_size=3,padding=1, bias=True),
            # nn.Conv2d(input_dim*2, input_dim, kernel_size=1,padding=0, bias=True),
            # nn.Conv2d(input_dim, input_dim, kernel_size=1,padding=0, bias=True),
            # LayerNorm(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=3,padding=1, bias=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=3,padding=1, bias=True),
            LayerNorm(input_dim),
            nn.Sigmoid()
        )
        self.key = nn.Sequential(
            # nn.Conv2d(input_dim, input_dim, kernel_size=1,padding=0, bias=True),
            # nn.Conv2d(input_dim, input_dim, kernel_size=1,padding=0, bias=True),
            # nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=3,padding=1, bias=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=3,padding=1, bias=True),
            nn.BatchNorm2d(input_dim),
            nn.GELU()
        )
        self.conv_attn = nn.Sequential(
            # nn.Conv2d(input_dim*2, input_dim*2, kernel_size=1,padding=0, bias=True),
            # nn.Conv2d(input_dim*2, input_dim*2, kernel_size=3,padding=1, bias=True),
            # LayerNorm(input_dim*2),
            # nn.GELU(),
            nn.Conv2d(in_channels=input_dim*2, out_channels=input_dim, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, padding=0),
            LayerNorm(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, 11, padding=5)
        )
        self.gamma = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.skew = nn.Parameter(torch.tensor(0.1,dtype=torch.float32))
    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def combine_att_V2(self,agent_feature,local_feature):
        V,_,_,_ = agent_feature.shape
        # combine local and agent attention 
        out_feat = []
        for vehicle_feature in local_feature:  # [C,H,W]
            q = self.query(vehicle_feature.unsqueeze(0)) # [1,C,H,W]
            g_k = self.key(agent_feature) # [V,C,H,W]
            # interaction between vehicle lcoal feature
            v = vehicle_feature.unsqueeze(0) # [1,C,H,W]
            attn = q * g_k    # [V,C,H,W]
            attn = F.softmax(attn,dim=1)
            g_out = attn * v  # [V,C,H,W]
            g_out = g_out.sum(dim=0) # [C,H,W]
            
            # interaction within vehicle lcoal feature
            l_k = self.key(local_feature)
            l_attn = q * l_k
            l_attn = F.softmax(l_attn,dim=1)
            l_out = l_attn * v
            l_out = l_out.sum(dim=0) # [C,H,W]

            req = 1 - q
            implicit_attn = req*l_k
            implicit_attn = F.softmax(implicit_attn,dim=1)
            implicit_out = implicit_attn * v
            implicit_out = implicit_out.sum(dim=0) # [C,H,W]
 
            l_out = torch.cat([l_out,implicit_out],dim=0) # [2C,H,W]
            l_out = self.conv_attn(l_out)

            out = torch.cat([g_out,l_out],dim=0) # [2C,H,W]
            out = self.conv_attn(out)

            out = out*self.gamma + self.skew + vehicle_feature
            out_feat.append(out)
        
        out = torch.sum(torch.stack(out_feat,dim=0),dim=0,keepdim=True) # [1,C,H,W]
        return out
    
    def combine_att(self,agent_feature,local_feature):
        V,C,H,W = agent_feature.shape
        # combine local and agent attention 
        out_feat = []
        for vehicle_feature in local_feature: 
            v = vehicle_feature.reshape(C,-1).unsqueeze(0) # [1,C,H*W]
            q = self.query(vehicle_feature.unsqueeze(0)) # [1,C,H,W]
            k = self.key(agent_feature) # [V,C,H,W]
            attn = torch.matmul(q.reshape(1,C,-1),k.reshape(V,C,-1).permute(0,2,1)) # [V,C,C]
            attn = F.softmax(attn,dim=-1)
            out = torch.matmul(attn,v) # [V,C,C] @ [1,C,H*W] = [V,C,H*W]
            out = out.reshape(V,C,H,W).sum(dim=0) # [C,H,W]

            req = 1 - q
            implicit_attn = torch.matmul(req.reshape(1,C,-1),k.reshape(V,C,-1).permute(0,2,1)) # [V,C,C]
            implicit_attn = F.softmax(implicit_attn,dim=-1)
            implicit_out = torch.matmul(implicit_attn,v) # [V,C,C] @ [1,C,H*W] = [V,C,H*W]
            implicit_out = implicit_out.reshape(V,C,H,W).sum(dim=0) # [C,H,W]
            
            out = torch.cat([out,implicit_out],dim=0) # [2C,H,W]
            out = self.conv_attn(out)
            out += vehicle_feature # [C,H,W]
            out_feat.append(out)
        out = torch.sum(torch.stack(out_feat,dim=0),dim=0,keepdim=True) # [1,C,H,W]
        return out
    

    def forward(self, spatial_features, record_len):
        assert spatial_features.ndim == 4, "tensor dim not correct"
        split_x = self.regroup(spatial_features, record_len) #spatial_features [5,C,H,W]
        out = []
        # att = []
        for batch_spatial_feature in split_x: # [2,C,H,W]
            """
            combine agent and local attention
            """
            local_feat = []
            agent_feature = self.agent_att(batch_spatial_feature) # [V,C,H,W] V is sum of cavs in the scene which can be detected
            for vehicle_feature in batch_spatial_feature: # [C,H,W]
                local_feature = self.local_att(vehicle_feature.unsqueeze(0)) # [1,2C,H,W]
                # local_feature = self.conv1x1(local_feature) 
                local_feat.append(local_feature)
            local_feat = torch.cat(local_feat,dim=0)  # [V,C,H,W]

            # combine local and agent attention
            # ego_feature = local_feat+agent_feature # [V,C,H,W]
            # # ego_feature = self.conv3x3(ego_feature)
            # out_mean = torch.mean(ego_feature,dim=0,keepdim=True) # [1,C,H,W]
            # out_max = torch.max(ego_feature,dim=0,keepdim=True)[0] # [1,C,H,W]
            # agg_feat = torch.cat([out_mean,out_max],dim=1) # [1,3C,H,W]
            # out_feature = self.convattn(agg_feat) # [1,C,H,W]
            # out.append(out_feature)
            out_feature = self.combine_att_V2(agent_feature,local_feat)
            out.append(out_feature)

        out = torch.cat(out,dim=0) 
        return out


class AgentAttention(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim),
            nn.Sigmoid()
        )
        # self.norm = nn.BatchNorm2d(intput_dim)

        # self.conv1x1 = nn.Conv2d(input_dim, input_dim, 1)

        # self.conv_att = nn.Sequential(
        #     nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(input_dim, eps=1e-5, momentum=0.01, affine=True),
        #     nn.ReLU()
        # )
        self.scale = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))

    def forward(self, batch_spatial_feature):
        # get batch feature
        """
        agent-wise attention
        """
        att_vehicle = self.avg_pool(batch_spatial_feature).squeeze()
        if att_vehicle.ndim == 1:
            att_vehicle = att_vehicle.unsqueeze(0)
        att_vehicle = self.fc(att_vehicle).reshape(att_vehicle.shape[0], att_vehicle.shape[1], 1, 1)
        # identity = att_vehicle
        # att_vehicle = att_vehicle * batch_spatial_feature
        # att_vehicle = identity + batch_spatial_feature
        att_vehicle = (self.scale*att_vehicle+1)* batch_spatial_feature
        # fuse_att = self.conv1x1(att_vehicle)
        # pooling_max = torch.max(fuse_att, dim=0, keepdim=True)[0]
        # pooling_ave = torch.mean(fuse_att, dim=0, keepdim=True)
        # fuse_fea = pooling_max + pooling_ave
        # out = self.conv_att(fuse_fea)
        return att_vehicle

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x


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
    def __init__(self, inp, oup, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)
        # self.scale = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out_attn = identity * a_w * a_h
        # out = (self.scale*out_attn+1)*identity
        return out_attn


class LocalRangeAttention(nn.Module):
    def __init__(self, input_dim,scale:float=1,last_attention:bool=False):
        super().__init__()
        assert input_dim % 2 == 0,"channel must be even number"

        branch_out_channel = input_dim // 2
        self.last_attention = last_attention

        self.conv_a = nn.Sequential(
                nn.Conv2d(input_dim, branch_out_channel, kernel_size=1, bias=True),
                nn.BatchNorm2d(branch_out_channel)
        ) 

        self.conv_b = nn.Sequential(
                nn.Conv2d(input_dim, branch_out_channel, kernel_size=1, bias=True),
                nn.BatchNorm2d(branch_out_channel)
        ) 

        self.conv_attn_a = nn.Sequential(
            nn.Conv2d(in_channels=branch_out_channel + 2, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.conv_attn_b = nn.Sequential(
            nn.Conv2d(in_channels=branch_out_channel + 2, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.conv_attention_a = nn.Sequential(nn.Conv2d(4,1,3,stride=1,padding=1),nn.Sigmoid())
        self.conv_attention_b = nn.Sequential(nn.Conv2d(4,1,3,stride=1,padding=1),nn.Sigmoid())

        self.norm_a = LayerNorm(input_dim)
        self.norm_b = LayerNorm(input_dim)
        # self.lam = nn.Parameter(torch.zeros(scale))
        self.scale_a = nn.Parameter(torch.tensor(scale,dtype=torch.float32))
        self.scale_b = nn.Parameter(torch.tensor(scale,dtype=torch.float32))
        self.init_weight()
        
    def forward(self, spatial_features): # spatial_features [1,C,H,W]

        x_a = self.conv_a(spatial_features) # [1,C//2,H,W]
        x_b = self.conv_b(spatial_features) # [1,C//2,H,W]

        grid_h_batch, grid_w_batch, distance_tensor, = self.range_map(x_a)     

        att_a = self.conv_attn_a(torch.cat([x_a,grid_h_batch,grid_w_batch],dim=1)) # [1,C//2+2,H,W]

        att_b = self.conv_attn_b(torch.cat([x_b,1 - grid_h_batch,1 - grid_w_batch],dim=1)) # [1,C//2+2,H,W]

        max_att_a = torch.max(x_a,1)[0].unsqueeze(1) # [1,1,H,W]
        avg_att_a = torch.mean(x_a,1).unsqueeze(1) # [1,1,H,W]

        att_maps_a = self.conv_attention_a(torch.cat([att_a,max_att_a,avg_att_a,distance_tensor],dim=1))

        max_att_b = torch.max(x_b,1)[0].unsqueeze(1)
        avg_att_b = torch.mean(x_b,1).unsqueeze(1)
        att_maps_b = self.conv_attention_b(torch.cat([att_b,max_att_b,avg_att_b,-1.0*distance_tensor],dim=1))

        if self.last_attention:
            x_a = att_maps_a * x_a
            x_b = att_maps_b * x_b
        else:
            x_a = (1 + self.scale_a*att_maps_a) * x_a
            x_b = (1 + self.scale_b*att_maps_b) * x_b
        out = torch.cat((x_a, x_b), dim=1)
        return out # spatial_features [1,C,H,W]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    

        # feature_out = self.conv1x1(spatial_features)
        # feature_out =         feature_out.permute(0, 2, 3, 1)
        # norm_feature_out = self.norm(feature_out).permute(0, 3, 1, 2)

        # out = torch.cat([norm_feature_out, grid_h_batch, grid_w_batch], dim=1)
        # max_output = torch.max(out, dim=1, keepdim=True)[0]
        # avg_output = torch.mean(out, dim=1, keepdim=True)
        # conv_output = self.conv3x3(out)

        # out = torch.cat([max_output, avg_output, conv_output, distance_tensor], dim=1)
        # attn = self.sigmoid(self.conv2(out))
        # attn = attn * self.lam
        # out = attn * feature_out.permute(0, 3, 1, 2)
        # out = out + feature_out.permute(0, 3, 1, 2)

    def range_map(self, spatial_features,norm:bool=True):
        # (H,W) distance map
        H, W = spatial_features.shape[-2:]

        lin_h = torch.linspace(0, H - 1, H).cuda()
        lin_w = torch.linspace(0, W - 1, W).cuda()
        y, x = torch.meshgrid(lin_h, lin_w)

        y = torch.abs(y - H / 2 + 0.5) if H % 2 == 1 else torch.abs(y - H / 2)
        x = torch.abs(x - W / 2 + 0.5) if W % 2 == 1 else torch.abs(x - W / 2)

        y = y / float(H // 2)
        x = x / float(W // 2)
        distance_tensor = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0).repeat(spatial_features.shape[0], 1, 1,
                                                                                       1)
        if norm:
            distance_tensor = linear_map(distance_tensor)
        grid_h_batch = y.unsqueeze(0).unsqueeze(0).repeat(spatial_features.shape[0], 1, 1, 1)
        grid_w_batch = x.unsqueeze(0).unsqueeze(0).repeat(spatial_features.shape[0], 1, 1, 1)

        return grid_h_batch, grid_w_batch, distance_tensor


class LocalRangeAttention_oldVerion(nn.Module):
    def __init__(self, input_dim,scale:float=1):
        super().__init__()
        self.conv1x1 = nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=True)
        self.conv3x3 = nn.Conv2d(input_dim + 2, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(input_dim)
        # self.lam = nn.Parameter(torch.zeros(scale))
        self.lam = nn.Parameter(torch.tensor(scale,dtype=torch.float32))

    def forward(self, spatial_features):
        grid_h_batch, grid_w_batch, distance_tensor, = self.range_map(spatial_features)
        feature_out = self.conv1x1(spatial_features)
        feature_out = feature_out.permute(0, 2, 3, 1)
        norm_feature_out = self.norm(feature_out).permute(0, 3, 1, 2)

        out = torch.cat([norm_feature_out, grid_h_batch, grid_w_batch], dim=1)
        max_output = torch.max(out, dim=1, keepdim=True)[0]
        avg_output = torch.mean(out, dim=1, keepdim=True)
        conv_output = self.conv3x3(out)

        out = torch.cat([max_output, avg_output, conv_output, distance_tensor], dim=1)
        attn = self.sigmoid(self.conv2(out))
        attn = attn * self.lam
        out = attn * feature_out.permute(0, 3, 1, 2)
        out = out + feature_out.permute(0, 3, 1, 2)
        return out

    def range_map(self, spatial_features,norm:bool=True):
        # (H,W) distance map
        H, W = spatial_features.shape[-2:]

        lin_h = torch.linspace(0, H - 1, H).cuda()
        lin_w = torch.linspace(0, W - 1, W).cuda()
        y, x = torch.meshgrid(lin_h, lin_w)

        y = torch.abs(y - H / 2 + 0.5) if H % 2 == 1 else torch.abs(y - H / 2)
        x = torch.abs(x - W / 2 + 0.5) if W % 2 == 1 else torch.abs(x - W / 2)

        y = y / float(H // 2)
        x = x / float(W // 2)
        distance_tensor = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0).repeat(spatial_features.shape[0], 1, 1,
                                                                                       1).requires_grad_(False)
        if norm:    
            distance_tensor = linear_map(distance_tensor)

        grid_h_batch = y.unsqueeze(0).unsqueeze(0).repeat(spatial_features.shape[0], 1, 1, 1).requires_grad_(False)
        grid_w_batch = x.unsqueeze(0).unsqueeze(0).repeat(spatial_features.shape[0], 1, 1, 1).requires_grad_(False)

        return grid_h_batch, grid_w_batch, distance_tensor


class LocalAttention(nn.Module):
    """
    implement local attention with range encoding
    """

    def __init__(self, input_dim,last_attention:bool=False,scale:float=1):
        super().__init__()

        self.ca = CoordinateAttention(input_dim, input_dim)
        self.lra =  LocalRangeAttention(input_dim,last_attention)
        self.scale_1 = nn.Parameter(torch.tensor(scale,dtype=torch.float32))
        # self.scale_2 = nn.Parameter(torch.tensor(scale,dtype=torch.float32))
        
    def forward(self, spatial_features):  # [1,C,H,W]
        coord_out = self.ca(spatial_features)
        lra_out = self.lra(spatial_features)
        # out = torch.cat([lra_out, coord_out], dim=1)

        out = lra_out+coord_out
        out = torch.sigmoid(out)
        # coord_out = self.ca(lra_out)
        # coord_out = self.scale_2*coord_out * spatial_features
        # lra_out = self.lra(coord_out)
        out = self.scale_1*out * spatial_features

        return out
