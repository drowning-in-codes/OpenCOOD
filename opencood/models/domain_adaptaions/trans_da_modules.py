"""
Domain adaption modules
Author: Qiuhao Shu NPU MS
"""
import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.domain_adaptions.da_loss import InstanceAdaptationModule,PatchDomainAdaptationModule,GlobalDomainAdaptationModule,SpatialDomainAdaptationModule,BeforeFusionAdaptation
import torch.nn.functional as F



class TransDomainAdaptaionModule(nn.Module):
    def __init__(self,args):
        super().__init__()
        in_channels = args["in_channels"]
        num_blocks = args["num_blocks"]
        self.num_blocks = num_blocks
        self.bdm = BeforeFusionAdaptation(in_channels)
        self.pdm = PatchDomainAdaptationModule(in_channels)
        self.sdm = SpatialDomainAdaptationModule(in_channels)
        self.gdm = GlobalDomainAdaptationModule(in_channels)
        self.idm =InstanceAdaptationModule(in_channels)

        self.spatial_weight = args['spatial_weight'] if 'spatial_weight' in args else 1.0
        self.patch_weight = args['patch_weight'] if 'patch_weight' in args else 1.0
        self.global_weight = args['global_weight'] if 'global_weight' in args else 1.0
        self.bf_weight = args['bf_weight'] if 'bf_weight' in args else 1.0
        self.instance_weight = args['instance_weight'] if 'instance_weight' in args else 1.0


    def forward(self,output_dict):
        spatial_feats = output_dict['spatial_feats']
        patch_feats = output_dict['patch_feats']
        global_feats = output_dict['global_feats']
        before_fusion_features = output_dict['before_fusion_features']
        src_psm = output_dict['psm']
        tgt_psm = output_dict['target_psm']
        assert len(spatial_feats) == len(patch_feats) == len(global_feats) == len(before_fusion_features) == 2
        win_num = len(patch_feats[0])
        assert win_num == self.num_blocks, f"window number {win_num} is not equal to num_blocks {self.num_blocks}"

        # Loss of DA feature component
        # source domain
        da_spatial_loss_all = []
        da_patch_loss_all = []
        da_global_loss_all = []
        for i in range(self.num_blocks):
            da_spatial_loss = self.sdm(spatial_feats[0][i],spatial_feats[1][i])
            da_patch_loss = self.pdm(patch_feats[0][i],patch_feats[1][i])
            da_global_loss = self.gdm(global_feats[0][i],global_feats[1][i])
            da_before_loss = self.bdm(before_fusion_features[0][i],before_fusion_features[1][i])

            da_spatial_loss_all.append(da_spatial_loss)
            da_patch_loss_all.append(da_patch_loss)
            da_global_loss_all.append(da_global_loss)
        da_spatial_loss = sum(da_spatial_loss_all)
        da_patch_loss = sum(da_patch_loss_all)
        da_global_loss = sum(da_global_loss_all)
        da_instance_loss = self.idm(src_psm,tgt_psm)
        losses = {}
        if self.spatial_weight > 0:
            losses['spa_loss'] = da_spatial_loss * self.fea_weight
        if self.global_weight > 0:
            losses['global_loss'] = da_patch_loss * self.global_weight
        if self.patch_weight > 0:
            losses['patch_loss'] = da_global_loss * self.patch_weight
        if self.bf_weight > 0:
            losses['bf_loss'] = da_before_loss * self.bf_weight
        if self.instance_weight > 0:
            losses['instance_loss'] = da_instance_loss * self.instance_weight
        # TODO: add instance loss or target MI estimation loss(from TVT)?
        return losses



