import torch
from torch.nn import functional as F


# def calc_mean_std(feat, eps=1e-5):
#     # eps is a small value added to the variance to avoid divide-by-zero.
#     size = feat.size()
#     assert (len(size) == 4)
#     N, C = size[:2]
#     feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
#     feat_std = feat_var.sqrt().view(N, C, 1, 1)
#     feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
#     return feat_mean, feat_std


def calc_mean_std(feat, eps=1e-5, seg_mask=None):
    """
    feat: [B, C, H, W]
    seg_mask: [B, 1, H, W] 显著性掩码（0~1），可选
    """
    size = feat.size()
    N, C = size[:2]

    feat_reshaped = feat.reshape(N, C, -1)  # [B, C, HW]

    if seg_mask is not None:
        seg_mask = F.interpolate(seg_mask, size=feat.shape[2:], mode="bilinear", align_corners=False)
        seg_mask = seg_mask.reshape(N, 1, -1)  # [B, 1, HW]
        seg_mask = seg_mask / (seg_mask.sum(dim=2, keepdim=True) + eps)  # 归一化

        # 加权均值
        feat_mean = (feat_reshaped * seg_mask).sum(dim=2).view(N, C, 1, 1)
        # 加权方差
        feat_var = ((feat_reshaped - feat_mean.view(N, C, 1))**2 * seg_mask).sum(dim=2).view(N, C, 1, 1)
    else:
        feat_var = feat_reshaped.var(dim=2) + eps
        feat_mean = feat_reshaped.mean(dim=2)

        feat_var = feat_var.view(N, C, 1, 1)
        feat_mean = feat_mean.view(N, C, 1, 1)

    feat_std = (feat_var + eps).sqrt()
    return feat_mean, feat_std
