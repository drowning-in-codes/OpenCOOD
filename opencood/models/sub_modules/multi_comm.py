# Author: proanimer
# Email: <bukalala174@gmail.com>
# License: MIT

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def linear(x, range=[-1.0, 1.0]):
    span = range[1] - range[0]
    k = span / (torch.max(x) - torch.min(x))
    b = range[1] - torch.max(x) * k
    return (k * x + b)

class Naive_DCN(nn.Module):
    def __init__(self,inc,outc,kernel_size=3,padding=1,stride=1) -> None:
        super().__init__()
        self.ks = kernel_size
        self.pad = padding
        self.stride = stride
        self.conv = nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=kernel_size)

        self.p_conv = nn.Conv2d(inc,2*kernel_size*kernel_size,kernel_size=3,padding=1,stride=stride)


    def forward(self,feat):
        # feat [V,C,H,W]
        offset = self.p_conv(feat) # -> [V,2*ks*ks,H,W]
        ks = self.ks
        N = offset.shape[1] // 2
        p = self.get_p(offset)

        p = p.contiguous().permute(0,2,3,1)

        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, feat.size(2)-1), torch.clamp(q_lt[..., N:], 0, feat.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, feat.size(2)-1), torch.clamp(q_rb[..., N:], 0, feat.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, feat.size(2)-1), torch.clamp(p[..., N:], 0, feat.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self.get_x_q(feat, q_lt, N)
        x_q_rb = self.get_x_q(feat, q_rb, N)
        x_q_lb = self.get_x_q(feat, q_lb, N)
        x_q_rt = self.get_x_q(feat, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt


        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out


    def get_p(self, offset):
            N, h, w = offset.shape[1]//2, offset.size[2], offset.size[3]
            # (1, 2N, 1, 1)
            p_n = self.get_p_n(N)
            # (1, 2N, h, w)
            p_0 = self.get_p_0(h, w, N)
            p = p_0 + p_n + offset
            return p


    def get_p_n(self, N):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1)
        return p_n

    def get_p_0(self, h, w, N):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1)
        return p_0
    
    def get_x_q(self, x, q, N):
            b, h, w, _ = q.size()
            padded_w = x.size(3)
            c = x.size(1)
            # (b, c, h*w)
            x = x.contiguous().view(b, c, -1)

            # (b, h, w, N)
            index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
            # (b, c, h*w*N)
            index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

            x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

            return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class Convrange(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, groups=1,range_type="near",H=None,W=None):
        super(Convrange, self).__init__()
        assert H is not None and W is not None,"H and W must be provided"
        assert range_type in ["near","far"]
        self.H =  H
        self.W = W
        self.range_type = range_type
        if out_channels is None:
            out_channels = in_channels
        
        
        self.convmodule = nn.Sequential(
                nn.Conv2d(in_channels+3, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,groups=groups),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
        )


    def forward(self, x):
        # X [v,c,h,w]
        v = x.shape[0]
        grid_h_batch,grid_w_batch,distance_tensor = self.range_map(range_type=self.range_type)
        grid_h_batch,grid_w_batch,distance_tensor = grid_h_batch.unsqueeze(0).unsqueeze(0).repeat(v,1,1,1),\
            grid_w_batch.unsqueeze(0).unsqueeze(0).repeat(v,1,1,1),distance_tensor.unsqueeze(0).unsqueeze(0).repeat(v,1,1,1)
        
        # identity = x
        # mean_x = torch.mean(torch.cat(v,grid_h_batch,grid_w_batch),dim=1,keepdim=True)
        # max_x = torch.max(torch.cat(v,grid_h_batch,grid_w_batch),dim=1,keepdim=True)[0]

        # out = torch.cat([mean_x,max_x,distance_tensor],dim=1)   
        # attn_out = self.convmodule(out)
        # return attn_out*identity+identity

        x = torch.cat([x, grid_h_batch, grid_w_batch, distance_tensor], dim=1) # TODO: this can be optimized
        out = self.convmodule(x)
        return out
    

    def range_map(self,norm=True,range_type="near"):
        H, W = self.H,self.W
        lin_h = torch.linspace(0, H - 1, H).cuda()
        lin_w = torch.linspace(0, W - 1, W).cuda()
        y, x = torch.meshgrid(lin_h, lin_w)
        y = torch.abs(y - H / 2 + 0.5) if H % 2 == 1 else torch.abs(y - H / 2)
        x = torch.abs(x - W / 2 + 0.5) if W % 2 == 1 else torch.abs(x - W / 2)
        grid_h_batch = y / float(H // 2)
        grid_w_batch = x / float(W // 2)
        distance_tensor = torch.sqrt(x ** 2 + y ** 2)
        if norm:
            distance_tensor = linear(distance_tensor)
        if range_type == "near":
            grid_h_batch = 1 - grid_h_batch
            grid_w_batch = 1 - grid_w_batch
            distance_tensor = -distance_tensor
        return grid_h_batch,grid_w_batch,distance_tensor
    
def linear_map(x, range=[0, 1.0],eps=1e-12):
    span = range[1] - range[0]
    k = span / (torch.max(x) - torch.min(x)+eps)
    b = range[1] - torch.max(x) * k
    return k * x + b

class MultiRangeComm(nn.Module):
    """
    get feature and produce communication map
    """
    def __init__(self, input_dim,cls_head,config,sim_type=0,H=None,W=None):
        super().__init__()
        assert H is not None and W is not None,"H and W must be provided"
        self.dim = input_dim
        strides = config['strides'] if config is not None and 'strides' in config else [1,2,4]
        self.down_convs = nn.ModuleList()
        self.n_convs = nn.ModuleList()
        self.f_convs = nn.ModuleList()
        self.cls_head = cls_head
        self.offset = nn.Sequential(
            nn.Linear(2,2),
        )
        # self.conv_g = nn.Sequential(
        #     nn.Conv2d(len(strides),1,kernel_size=1),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )
        self.conv_g = nn.Conv2d(len(strides),1,kernel_size=1)
        # self.conv_g = nn.Sequential(
        #     nn.Conv2d(len(strides),1,kernel_size=1),
        #     nn.BatchNorm2d(1))
        for stride in strides:
            self.down_convs.append(nn.Conv2d(input_dim,input_dim,kernel_size=stride,stride=stride))
            n_conv = Convrange(input_dim,range_type="near",H=H//stride,W=W//stride)
            f_conv = Convrange(input_dim,range_type="far",H=H//stride,W=W//stride)
            self.n_convs.append(n_conv)
            self.f_convs.append(f_conv)
        self.sim_type = sim_type
        if sim_type == 0:
            assert H is not None and W is not None,"H and W should be provided"
            self.fc = nn.Linear(W,H)
        self.threshold = config['threshold'] if config is not None and 'threshold' in config else 0.01

        if "gaussian_smooth" in config:
            self.smooth = True
            kernel_size = config["gaussian_smooth"]["k_size"]
            c_sigma = config["gaussian_smooth"]["c_sigma"]
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
            )
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def get_sim(self,input_feat,target_feat):
        # find similarity between two agents
        # [V,H,W]
        if self.sim_type == 0:
            tf = self.fc(target_feat)
            sim = torch.matmul(tf,input_feat) # [V,H,W]
        elif self.sim_type == 1:
            sim = input_feat*target_feat
        else:
            sim = torch.matmul(input_feat,target_feat.transpose(1,2))
        return sim

    def gen_comm_map(self,near_range_featm,far_range_feat):
        # params conf_map: torch.Tensor (V,H, W).
        near_conf_map = self.cls_head(near_range_featm) # [V,anchor_num,H,W]
        far_conf_map = self.cls_head(far_range_feat) # [V,anchor_num,H,W]
       
        near_confidence_map, _ = near_conf_map.sigmoid().max(dim=1,keepdim=True) # confidence map # [V,1,H,W]
        far_confidence_map, _ = far_conf_map.sigmoid().max(dim=1,keepdim=True) # confidence 
        if self.smooth:
            near_confidence_map = self.gaussian_filter(near_confidence_map)
            far_confidence_map = self.gaussian_filter(far_confidence_map) # [V,1,H,W]
        req_map = 1 - near_confidence_map  # [V,1,H,W]
        request = req_map[0].unsqueeze(0) # M~j->i~ R~i~ # [1,1,H,W]
        # diff_map = far_confidence_map * request # [V,H,W]
        communication_map = far_confidence_map * request #  [V,1,H,W]
        # sim_map = self.get_sim(conf_map,conf_map) # [V,H,W]
        # communication_map = self.lam*diff_map + (1-self.lam)*sim_map # [V,1,H,W]
        return communication_map

    # @staticmethod
    # def range_map(feature_map,norm=True):
    #     H, W = feature_map.shape[-2:]
    #     lin_h = torch.linspace(0, H - 1, H).to(feature_map)
    #     lin_w = torch.linspace(0, W - 1, W).to(feature_map)
    #     y, x = torch.meshgrid(lin_h, lin_w)
    #     y = torch.abs(y - H / 2 + 0.5) if H % 2 == 1 else torch.abs(y - H / 2)
    #     x = torch.abs(x - W / 2 + 0.5) if W % 2 == 1 else torch.abs(x - W / 2)
    #     y = y / float(H // 2)
    #     x = x / float(W // 2)
    #     distance_tensor = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0).repeat(feature_map.shape[0], 1, 1,1)
    #     if norm:
    #         distance_tensor = linear(distance_tensor)
    #     grid_h_batch = y.unsqueeze(0).unsqueeze(0).repeat(feature_map.shape[0], 1, 1, 1)
    #     grid_w_batch = x.unsqueeze(0).unsqueeze(0).repeat(feature_map.shape[0], 1, 1, 1)
    #     return grid_h_batch,grid_w_batch,distance_tensor


    def sampling_offset(self,x,indices):
        # params x: torch.Tensor (V, C, H, W).
        # params indices: torch.Tensor (V, obj_num).
        # return: torch.Tensor (V, C, H, W).
        def norm_offset(offset,range=(0,1)):
            span = range[1] - range[0]

            k = span / min(H*W,torch.max(offset)) - max(0,torch.min(offset))
            b = range[1] - torch.max(offset) * k
            return (k * offset + b)
        
        H,W = x.shape[-2:]
        h_coor = indices // W
        w_coor = indices - h_coor * W
        h_coor, w_coor = h_coor / H, w_coor / W
        pos_feat = torch.stack([h_coor, w_coor], dim=-1) # [V,obj_num,2]
        offset = self.offset(pos_feat) + pos_feat # [V,obj_num,2]

        offset = norm_offset(offset) # [V,obj_num,2]

        # 从h_coor和w_coor恢复indices
        h_coor = offset[...,0]
        w_coor = offset[...,1]
        h_coor_int = h_coor * H  # 先将h_coor缩放回整数范围
        w_coor_int = w_coor * W  # 先将w_coor缩放回整数范围
        indices_recovered = h_coor_int * W + w_coor_int  # 计算恢复的indices
        return indices_recovered.type(torch.int64)  # [V,obj_num]
    
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = (
                1
                / (2 * np.pi * sigma)
                * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            )
            return g

        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = (
            torch.Tensor(gaussian_kernel)
            .to(self.gaussian_filter.weight.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.gaussian_filter.bias.data.zero_()

    def forward(self,batch_features):
        # params: features: torch.Tensor, shape: [V ,C,H,W]
        V,_,H,W = batch_features.shape
        all_comm_maps = []
        for conv,n_conv,f_conv in zip(self.down_convs,self.n_convs,self.f_convs):
            feat = conv(batch_features) # [V,C,H//stride,W//stride]             
            near_range_feat = n_conv(feat)
            far_range_feat = f_conv(feat)
            comm_map = self.gen_comm_map(near_range_feat,far_range_feat) # [V,H,W]
            h,w = comm_map.shape[-2:]
            if H != h or W != w:
                comm_map = F.interpolate(comm_map, size=(H,W), mode='bilinear', align_corners=False)
            if comm_map.ndim == 4 and comm_map.shape[1] == 1:
                comm_map = comm_map.squeeze(1)
            else:
                raise ValueError("comm_map shape is not correct",f"comm_map shape is {comm_map.shape}")
            all_comm_maps.append(comm_map)
        all_comm_maps = torch.stack(all_comm_maps,dim=1) # [V,multi_scale,H,W]
        # comm_maps = torch.mean(comm_maps, dim=0) # [multi_scale,V,H,W]
        comm_maps = self.conv_g(all_comm_maps) # [V,1,H,W]
        comm_maps = comm_maps.reshape(V,H*W)
        comm_maps = linear_map(comm_maps) # [V,H*W]
        comm_maps = comm_maps.reshape(V,1,H,W)
        # comm_maps = torch.max(comm_maps, dim=0, keepdim=True)[0] # [V,H,W]

        # generate mask for each batch
        if self.training:
            K = int(H*W*random.uniform(0.1,1))
            communication_maps = comm_maps.reshape(-1, H*W)
            _, indices = torch.topk(communication_maps, k=K, sorted=False) # [V,obj_num]
            communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            ones_fill = torch.ones(V,K, dtype=communication_maps.dtype,device=communication_maps.device)
            # deformable position sampling
            # indices = self.sampling_offset(batch_features,indices) # [V,obj_num]
            # print(indices)
            communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(V,1,H,W)
            # print(communication_mask)
        elif self.threshold:
            communication_maps = comm_maps.reshape(-1, H*W)
            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask).reshape(V,1,H,W)
            # obj_num = int(torch.max(torch.sum(communication_mask.reshape(V,-1),dim=-1)).item())
            # _, indices = torch.topk(communication_maps, k=obj_num, sorted=False) # [V,obj_num]
            # deformable position sampling
            # indices = self.sampling_offset(batch_features,indices) # [V,obj_num]
            # communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            # ones_fill = torch.ones(V,obj_num, dtype=communication_maps.dtype,device=communication_maps.device)
            # communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(V,1,H,W)
            # print(communication_mask)
        else:
            communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            
        #  indice [V,H*W*ratio]
        # if self.training or self.threshold:
        #     offset = self.sampling_offset(batch_features,indices) # [V,obj_num]
        #     offset_mask = torch.scatter(communication_mask, -1, offset, ones_fill).reshape(V,1,H,W)
        #     communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(V,1,H,W)
        
        communication_rate = communication_mask.sum() / (V * H * W)
        # Ego
        communication_mask[0] = 1

        batch_features = batch_features*communication_mask # [V,C,H,W]


        return communication_mask,communication_rate.item()


