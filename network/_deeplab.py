import torch
from torch import nn
from torch.nn import functional as F

from .utils import _Segmentation


__all__ = ["DeepLabV3"]


class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels=2049, out_channels=2048, reduction=8):
        super(AdaptiveFusion, self).__init__()

        # --- 通道注意力 (SE Block) ---
        self.se_fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.se_fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # --- 卷积层 (保持通道数为 2048) ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, feature, feature_seg):
        """
        feature: [B, C, H, W] 主干网络特征
        feature_seg: [B, 1, Hf, Wf] 显著性分割图
        return: 融合后的特征
        """
        B, C, H, W = feature.size()

        # 1. 将 feature_seg 上采样到与 feature 相同的空间尺寸
        feature_seg_resized = F.interpolate(feature_seg, size=(H, W), mode="bilinear", align_corners=False)

        # 2. 拼接 feature 和 feature_seg
        fusion_input = torch.cat([feature, feature_seg_resized], dim=1)  # [B, C+1, H, W]

        # 3. 通道注意力（SE Block）
        se_vec = F.adaptive_avg_pool2d(fusion_input, 1).view(B, -1)  # [B, C+1]
        se_weight = self.se_fc1(se_vec)
        se_weight = self.relu(se_weight)
        se_weight = self.se_fc2(se_weight)
        se_weight = self.sigmoid(se_weight).view(B, -1, 1, 1)  # [B, C+1, 1, 1]

        # 4. 加权融合
        fusion_input = fusion_input * se_weight  # [B, C+1, H, W]

        # 5. 卷积处理 (保持通道数为2048)
        fusion_output = self.conv1(fusion_input)  # [B, C, H, W] (C=2048)

        return fusion_output


class DeepLabV3(_Segmentation):
    pass


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        # self.fusion = nn.Conv2d(2049, 2048, kernel_size=3, stride=1, padding=1)
        self.fusion = AdaptiveFusion()
        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        # feature['out'] = self.fusion(feature['out'], feature_seg)
        output_feature = self.aspp(feature['out'])

        # feat_concat = torch.cat([feature['out'], F.interpolate(feature_seg, size=feature['out'].shape[2:])], dim=1)  # [B, C+1, Hf, Wf]
        # feat_guided = self.fusion(feat_concat)
        # feature['out'] = self.aspp(feature['out'])
        # feat_concat = torch.cat([feature['out'], F.interpolate(feature_seg, size=feature['out'].shape[2:])],
        #                         dim=1)  # [B, C+1, Hf, Wf]
        # feat_guided = self.fusion(feature['out'],feature_seg)
        # feat_guided = self.fusion(output_feature,feature_seg)
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
    # def forward(self, feature):
    #     low_level_feature = self.project(feature['low_level'])
    #     output_feature = self.aspp(feature['out'])
    #     output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
    #                                    align_corners=False)
    #     return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))




    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module