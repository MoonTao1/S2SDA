import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import *
from network.code.network.MINet import MINet_Res50  # 你已有的实现


class _Segmentation(nn.Module):
    def __init__(self, backbone, classifier):
        super(_Segmentation, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

        # 显著性分割模型（冻结参数）
        self.segmention = MINet_Res50()
        state_dict = torch.load("/MINet_Res50.pth", map_location="cpu")
        self.segmention.load_state_dict(state_dict)
        self.segmention.eval()
        for param in self.segmention.parameters():
            param.requires_grad = False

    def forward(self, x, mu_t_f1=None, std_t_f1=None, transfer=False, mix=False, activation=None):
        device = x.device
        input_shape = x.shape[-2:]
        features = {}

        # backbone 第一层特征
        features['low_level'] = self.backbone(
            x, trunc1=False, trunc2=False, trunc3=False, trunc4=False,
            get1=True, get2=False, get3=False, get4=False
        )

        # 显著性分割特征
        # with torch.no_grad():
        #     features_seg = self.segmention(x)

        # 处理默认 mu/std
        if mu_t_f1 is None:
            mu_t_f1 = torch.zeros((x.size(0), features['low_level'].size(1), 1, 1), device=device)
        else:
            mu_t_f1 = mu_t_f1.to(device)

        if std_t_f1 is None:
            std_t_f1 = torch.ones((x.size(0), features['low_level'].size(1), 1, 1), device=device)
        else:
            std_t_f1 = std_t_f1.to(device)

        # 特征归一化 / 转移
        if transfer:
            mean_f1, std_f1 = calc_mean_std(features['low_level'])
            self.size = features['low_level'].size()
            features_low_norm = (features['low_level'] - mean_f1.expand(self.size)) / std_f1.expand(self.size)

            # if mix:
            #     s = torch.rand((mean_f1.shape[0], mean_f1.shape[1]), device=device).unsqueeze(-1).unsqueeze(-1)
            #     mu_mix = s * mean_f1 + (1 - s) * mu_t_f1
            #     std_mix = s * std_f1 + (1 - s) * std_t_f1
            #     features['low_level'] = (std_mix.expand(self.size) * features_low_norm + mu_mix.expand(self.size))
            # else:
            features['low_level'] = (std_t_f1.expand(self.size) * features_low_norm + mu_t_f1.expand(self.size))

            if activation is not None:
                features['low_level'] = activation(features['low_level'])

        # backbone 最后一层特征
        features['out'] = self.backbone(
            features['low_level'], trunc1=True, trunc2=False, trunc3=False, trunc4=False,
            get1=False, get2=False, get3=False, get4=True)
        # 分类器融合
        x = self.classifier(features)

        # x = self.classifier(features)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return torch.sigmoid(output), features
