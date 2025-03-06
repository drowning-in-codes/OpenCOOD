import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import torch.nn.functional as F

""""
domain adaptation
可能的改进方法:
1. 多尺度,不同尺度的特征上计算loss loss增加category-level?
2. soft label??
3. combined with other adversarial learning or GAN??  数年之前的da方法汇总?? 结合object detection e.g. Domain Adaptive Faster R-CNN for Object Detection in the Wild
4. 其他协同感知算法代码?
在模型上
1. 使用transformer e.g.cdtrans

v2x和位置噪声、延迟的情况
可能改进的方法:
1. 基于transformer继续改进
2. 看看最新论文?
"""

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight * grad_input, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class DA_feature_Head(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=0)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        for l in [self.conv1_da, self.conv2_da]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        t = F.relu(self.conv1_da(features))
        img_features = self.conv2_da(t)
        return img_features


class DA_instance_Head(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)
        self.in_channels = in_channels

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, 0.5)
        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.feature_head = DA_feature_Head()
        self.instance_head = DA_instance_Head()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fea_weight = args["DA_feature_weight"]
        self.ins_weight = args["DA_instance_weight"]
        self.grl_features = GradientReverseLayer(-1.0 * args["grl_feature_weight"])
        self.grl_instance = GradientReverseLayer(-1.0 * args["grl_instance_weight"])

    def __call__(self, output_dict):
        source_fea = output_dict["source_feature"]
        target_fea = output_dict["target_feature"]
        source_psm = output_dict['psm']
        target_psm = output_dict["target_psm"]

        # Loss of DA feature component
        source_grl_feature = self.grl_features(source_fea)
        target_grl_feature = self.grl_features(target_fea) # 使用grl 梯度反向,反向改进的是之前的提取器

        da_source_feature = self.feature_head(source_grl_feature) # 使用同一个feature特征提取器
        da_target_feature = self.feature_head(target_grl_feature)
        # label
        da_souce_fea_label = torch.ones_like(da_source_feature, dtype=torch.float32)
        da_target_fea_label = torch.zeros_like(da_target_feature, dtype=torch.float32)

        # [B, C*H*W]
        da_souce_fea_level = da_source_feature.reshape(da_source_feature.shape[0], -1)
        da_target_fea_level = da_target_feature.reshape(da_target_feature.shape[0], -1)
        da_fea = torch.cat([da_souce_fea_level, da_target_fea_level], dim=0)  # [B*2, C*H*W]
        da_fea_label = torch.cat([da_souce_fea_label, da_target_fea_label], dim=0)  # [B*2, C*H*W]

        # feature loss,通过跟domain label计算bce loss
        da_fea_loss = F.binary_cross_entropy_with_logits(da_fea, da_fea_label) # 通过新的feature_head得到的0,1进行对齐

        source_psm_grl = self.grl_instance(source_psm) # 使用grl,反向梯度改进得到psm的提取器 source_psm [B,]
        target_psm_grl = self.grl_instance(target_psm)

        # refer to PointPillarLoss------->[B, 48, 176, 2]
        cls_preds_source = source_psm_grl.permute(0, 2, 3, 1).contiguous()
        # refer to  PointPillarLoss------->[B, 48, 176, 2]
        cls_preds_target = target_psm_grl.permute(0, 2, 3, 1).contiguous()
        cls_preds_source = self.avgpool(cls_preds_source)
        cls_preds_target = self.avgpool(cls_preds_target)  ##[B, 48, 88, 1]
        # [B*H, C *W]====[B*H, 88*1]
        cls_preds_source = cls_preds_source.view(
            source_psm.shape[0] * source_psm.shape[2],
            -1)
        # [B*H, C *W]====[B*H, 88*1]
        cls_preds_target = cls_preds_target.view(
            target_psm.shape[0] * source_psm.shape[2],
            -1)

        da_ins_source = self.instance_head(
            cls_preds_source)  # [B*H, 2* 88]----> [B*H, 1]
        da_ins_target = self.instance_head(
            cls_preds_target)  # [B*H, 2* 88]----> [B*H, 1]
        # source label is 1
        da_source_ins_label = torch.ones_like(da_ins_source,
                                              dtype=torch.float32)
        # target label is 0
        da_target_ins_label = torch.zeros_like(da_ins_target,
                                               dtype=torch.float32)

        da_ins = torch.cat([da_ins_source, da_ins_target],
                           dim=0)
        da_ins_label = torch.cat([da_source_ins_label, da_target_ins_label],
                                 dim=0)

        # da instance loss
        da_ins_loss = F.binary_cross_entropy_with_logits(da_ins, da_ins_label) # instace loss 将得到的类图通过网络得到[B*H,1]这种shape计算BCE

        losses = {}
        if self.fea_weight > 0:
            losses['fea_loss'] = da_fea_loss * self.fea_weight
        if self.ins_weight > 0:
            losses['ins_loss'] = da_ins_loss * self.ins_weight

        return losses
