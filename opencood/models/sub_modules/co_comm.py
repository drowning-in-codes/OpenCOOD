import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.range_attn_fusion import RangeAttentionFusion
from opencood.models.fuse_modules.range_attn_light import RangeAttentionFusion as lightRangeAttentionFusion



class GenConfidence(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    

    def forward(self,x):
        pass

class coComm(nn.Module):
    def __init__(self,H=None,W=None,ratio:float=.9,multi_scale=True,input_dim:int=384,args=None,fusion_args=None) -> None:
        super().__init__()
        self.multi_scale = multi_scale
        self.threshold = 1e-1 if args is None or 'threshold' not in args else args['threshold']
        self.lam = nn.Parameter(torch.tensor(ratio))
        if H is not None and W is not None:
            self.fc = nn.Linear(W,H) # [W,H]
        self.fusion_model = nn.ModuleList()
        if multi_scale:
            self.layer_num = len(args["layer_nums"])
            num_filters = args['num_filters']
            for idx in range(self.layer_num):
                self.fusion_model.append(lightRangeAttentionFusion(num_filters[idx]))
        else:
            self.fusion_model =  RangeAttentionFusion(model_cfg=fusion_args,input_channels=input_dim)
                

    def generate_communication_map(self,conf_map):
        # params conf_map: torch.Tensor (V,H, W).
        req_map = 1 - conf_map # [V,H,W]
        request = req_map[0].unsqueeze(0) # M~j->i~ R~i~ # [1,H,W]
        
        diff_map = conf_map * request # [V,H,W]
        sim_map = self.get_sim(conf_map,conf_map) # [V,H,W]
        communication_map = self.lam*diff_map + (1-self.lam)*sim_map # [V,H,W]
        return communication_map


    def get_sim(self,input_feat,target_feat,type=0):
        # find similarity between two agents
        # [V,H,W]
        if type == 0:
            tf = self.fc(target_feat)
            sim = torch.matmul(tf,input_feat) # [V,H,W]
        elif type == 1:
            sim = input_feat*target_feat
        else:
            sim = torch.matmul(input_feat,target_feat.transpose(1,2))
            
        return sim

    def forward(self, x: torch.Tensor,record_len,conf_map,backbone=None):
        # params x: torch.Tensor (sum(n_cav), C, H, W).
        # params conf_map: torch.Tensor (H, W).
        B = len(record_len)
        if self.multi_scale:
            ups = []
            for idx in range(self.layer_num):
                x = backbone.blocks[idx](x)
                _, _, H, W = x.shape
                if idx == 0:
                    communication_masks = []
                    communication_rates = []
                    # conf_map = cls_head(x) # [B,anchor_num,H, W]
                    # if idx == self.layer_num - 1:
                    #     last_conf_map = conf_map
                    batch_confidence_maps = self.regroup(conf_map, record_len) # [(V,anchor_num,H, W),...]
                    for batch in range(B):
                        V = record_len[batch]
                        confidence_map, _ = batch_confidence_maps[batch].sigmoid().max(dim=1,keepdim=True) # confidence map
                        confidence_map = confidence_map.squeeze() # [V,H, W]
                        if confidence_map.ndim == 2:
                            confidence_map = confidence_map.unsqueeze(0)
                        communication_maps = self.generate_communication_map(confidence_map).to(confidence_map.device) # [V, H, W]
                        # generate mask for each batch
                        if self.training:
                            K = int(H*W*random.uniform(0.1,1))
                            communication_maps = communication_maps.reshape(-1, H*W)
                            _, indices = torch.topk(communication_maps, k=K, sorted=False)
                            communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                            ones_fill = torch.ones(V,K, dtype=communication_maps.dtype,device=communication_maps.device)
                            communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(V,1, H, W)
                        elif self.threshold:
                            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                            communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask).reshape(V,1, H, W)
                        else:
                            communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                            
                        communication_rate = communication_mask.sum() / (V * H * W)
                        # Ego
                        communication_mask[0] = 1

                        communication_masks.append(communication_mask)
                        communication_rates.append(communication_rate)
                    communication_rates = sum(communication_rates) / B
                    communication_masks = torch.cat(communication_masks, dim=0)
                    if x.shape[-1] != communication_masks.shape[-1]:
                        communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                                mode='bilinear', align_corners=False)
                    x = x * communication_masks
                    # communication feat
                    communication_feat = x
                
                x_fuse = self.fusion_model[idx](x,record_len)

                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[idx](x_fuse))
                else:
                    ups.append(x_fuse)
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)

        else:
            communication_masks = []
            communication_rates = []
            # conf_map = cls_head(x) # [B,anchor_num,H, W]
            # last_conf_map = conf_map
            batch_confidence_maps = self.regroup(conf_map, record_len) # [(V,anchor_num,H, W),...]
            _, _, H, W = x.shape

            for batch in range(B):
                V = record_len[batch]
                confidence_map, _ = batch_confidence_maps[batch].sigmoid().max(dim=1,keepdim=True) # confidence map
                confidence_map = confidence_map.squeeze() # [V,H, W]
                if confidence_map.ndim == 2:
                    confidence_map = confidence_map.unsqueeze(0)
                communication_maps = self.generate_communication_map(confidence_map).to(confidence_map.device) # [V, H, W]
                # generate mask for each batch
                if self.training:
                    K = int(H*W*random.uniform(0.1,1))
                    communication_maps = communication_maps.reshape(-1, H*W)
                    _, indices = torch.topk(communication_maps, k=K, sorted=False)
                    communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                    ones_fill = torch.ones(V,K, dtype=communication_maps.dtype,device=communication_maps.device)
                    communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(V,1, H, W)
                elif self.threshold:
                    ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                    zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                    communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask).reshape(V,1, H, W)
                else:
                    communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                    
                communication_rate = communication_mask.sum() / (V * H * W)
                # Ego
                communication_mask[0] = 1

                communication_masks.append(communication_mask)
                communication_rates.append(communication_rate)
            communication_rates = sum(communication_rates) / B
            communication_masks = torch.cat(communication_masks, dim=0)
            if x.shape[-1] != communication_masks.shape[-1]:
                communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                                mode='bilinear', align_corners=False)
            x = x * communication_masks
            # communication feat
            communication_feat = x
            x_fuse = self.fusion_model(x,record_len)
        return x_fuse,communication_rates,communication_feat,0

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
