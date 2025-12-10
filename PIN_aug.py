import pickle
import os
import clip
import torch
import network
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import calc_mean_std  # 仍保留，但我们在 PIN 中使用局部计算
import argparse
from torch.utils import data
import numpy as np
import random
from utils_.build_datasets import build_dataset
from utils_.options import parser
args = parser.parse_args()
from utils_.mid_metrics import cc, sim, kldiv
from torch.utils.tensorboard import SummaryWriter
from network.code.network.MINet import MINet_Res50  # 你的显著性模型

# --- 模板和一些 util ---
def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

# ------------------------------
# 我们实现一个 PIN 类，带 seg2style adapter
# ------------------------------
class PIN(nn.Module):
    def __init__(self, content_feat, seg2style_hidden=32):
        super().__init__()
        # content_feat: [B,C,H,W]
        self.content_feat = content_feat.detach().clone()
        self.content_mean, self.content_std = calc_mean_std(self.content_feat)  # [B,C,1,1]
        self.size = self.content_feat.size()
        self.content_feat_norm = (self.content_feat - self.content_mean.expand(self.size)) / (self.content_std.expand(self.size) + 1e-8)

        C = self.content_mean.shape[1]
        # learnable base params (initialized to content stats)
        self.style_mean = nn.Parameter(self.content_mean.clone().detach(), requires_grad=True)   # [B,C,1,1]
        self.style_std = nn.Parameter(self.content_std.clone().detach(), requires_grad=True)     # [B,C,1,1]

        # small seg->style adapter
        self.seg2style = nn.Sequential(
            nn.Conv2d(1, seg2style_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),    # -> [B,hidden,1,1]
            nn.Flatten(),               # -> [B,hidden]
            nn.Linear(seg2style_hidden, C*2)  # -> [B, 2*C]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, seg_mask=None, mu_scale=0.1, std_scale=0.1):
        # clamp std to be positive (softplus is more stable)
        style_std_pos = F.softplus(self.style_std)
        style_mean = self.style_mean

        mu_delta = 0.0
        std_delta = 0.0
        if seg_mask is not None:
            seg = seg_mask.to(self.style_mean.device)
            seg_feat = self.seg2style(seg)  # [B, 2C]
            mu_d, std_d = seg_feat.chunk(2, dim=1)  # [B,C], [B,C]
            mu_d = mu_d.view(-1, style_mean.shape[1], 1, 1)
            std_d = std_d.view(-1, style_mean.shape[1], 1, 1)
            # apply *small* scaling factors to prevent large jumps
            mu_delta = mu_scale * torch.tanh(mu_d)      # 限幅
            std_delta = std_scale * torch.tanh(std_d)   # 限幅
            style_mean = style_mean + mu_delta
            style_std_pos = style_std_pos * (1.0 + std_delta)  # multiplicative

        target_feat = self.content_feat_norm * style_std_pos.expand(self.size) + style_mean.expand(self.size)
        return self.relu(target_feat), mu_delta, std_delta, style_mean, style_std_pos

# --------------------------------
# 主函数 main（修改后）
# --------------------------------
def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--save_dir", type=str,default='/data9102/workspace/mwt/Experiment/DADA/PODA/pins/rainy/',
                        help="path for learnt parameters saving")
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')

    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default='RN50',
                        help="backbone name")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--total_it", type=int, default=100,
                        help="total number of optimization iterations")
    # learn statistics
    parser.add_argument("--resize_feat", action='store_true', default=True,
                        help="resize the features map to the dimension corresponding to CLIP")
    # random seed
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    # target domain description
    parser.add_argument("--domain_desc", type=str, default="driving under rain.",
                        help="description of the target domain")

    return parser

def main():
    opts = get_argparser().parse_args()

    # Set visible GPU (this affects CUDA_VISIBLE_DEVICES)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print(opts.domain_desc)
    # INIT seeds
    torch.manual_seed(opts.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # dataset
    # args.category = 'TrafficGaze'
    # args.root = '/data/workspace/mwt/traffic_dataset/'
    args.w = 'sunny'
    train_loader, valid_loader, _ = build_dataset(args=args)
    print("train len:", len(train_loader),args.w)

    # segmentation model (main task)
    model = network.modeling.__dict__[opts.model](num_classes=1, BB=opts.BB,
                                                  replace_stride_with_dilation=[False, False, False])
    # freeze backbone of main model
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()
    model.to(device)

    # load and prepare MINet (saliency model) once
    seg_model = MINet_Res50()
    # load seg checkpoint (确保路径正确)
    seg_ckpt = "/data9102/workspace/mwt/PODA/MINet_Res50.pth"
    if os.path.isfile(seg_ckpt):
        sd = torch.load(seg_ckpt, map_location='cpu')
        # 如果保存的是 state_dict 直接 load
        try:
            seg_model.load_state_dict(sd)
        except Exception:
            # 如果 sd 里包含键名不同，尝试直接拿 'state_dict'
            if 'state_dict' in sd:
                seg_model.load_state_dict(sd['state_dict'])
            else:
                seg_model.load_state_dict(sd)
        print("Loaded seg model weights from", seg_ckpt)
    else:
        print("[Warning] seg ckpt not found at", seg_ckpt, "using random init (not recommended)")

    seg_model.to(device)
    seg_model.eval()
    for p in seg_model.parameters():
        p.requires_grad = False

    # CLIP 模型
    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()

    if not os.path.isdir(opts.save_dir):
        os.makedirs(opts.save_dir, exist_ok=True)

    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56, 56))
    else:
        t1 = lambda x: x

    # prepare text target
    target = compose_text_with_templates(opts.domain_desc, imagenet_templates)
    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size, 1).type(torch.float32)  # (B,1024)

    # main loop: 对每个批次生成 PIN 并优化 style params
    for i, (images, labels, indices) in enumerate(train_loader):
        cur_itrs = 0
        # ensure images on device for seg_model inference & backbone
        images = images.to(device)
        # optional resize for clip (你原来用 224/256)
        from torchvision.transforms import Resize
        torch_resize = Resize([256, 256])
        img_for_clip = torch_resize(images)  # 用于 CLIP/backbone consistent

        # backbone low-level feature f1
        f1 = model.backbone(img_for_clip, trunc1=False, trunc2=False,
                            trunc3=False, trunc4=False, get1=True, get2=False, get3=False, get4=False)  # (B,C1,H1,W1)
        f1 = f1.to(device)

        # get seg_mask from seg_model (no grad)
        with torch.no_grad():
            seg_mask = seg_model(img_for_clip)  # shape [B,1,Hseg,Wseg], e.g. [B,1,256,256]
            # ensure in range [0,1] (MINet should output sigmoid already)
            seg_mask = seg_mask.clamp(0.0, 1.0)

        # instantiate PIN for this batch (content_feat = f1)
        model_pin_1 = PIN(f1, seg2style_hidden=32).to(device) # seg2style inside PIN will be trained
        model_pin_1.to(device)
        lambda_reg = 1e-3
        # optimizer will optimize style_mean/style_std and seg2style params
        optimizer_pin_1 = torch.optim.SGD(model_pin_1.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        # adjust text_target size in final batch if needed
        if i == len(train_loader) - 1 and f1.shape[0] < opts.batch_size:
            text_target = text_target[:f1.shape[0]]

        print("batch", i, "/", len(train_loader), "f1.shape", f1.shape, "seg_mask.shape", seg_mask.shape)

        # optimize style params with CLIP objective
        while cur_itrs < opts.total_it:
            cur_itrs += 1
            optimizer_pin_1.zero_grad()
            f1_hal, mu_delta, std_delta, _, _ = model_pin_1(seg_mask)

            # transform feature map to CLIP-compatible size
            f1_hal_trans = t1(f1_hal)

            # push through backbone tail to get global features for CLIP
            target_features_from_f1 = model.backbone(
                f1_hal_trans, trunc1=True, trunc2=False, trunc3=False, trunc4=False,
                get1=False, get2=False, get3=False, get4=False
            )
            # normalize and compute similarity loss with text_target
            target_features_from_f1 = target_features_from_f1 / (target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach() + 1e-8)
            loss_CLIP1 = (1 - torch.cosine_similarity(text_target, target_features_from_f1, dim=1)).mean()
            reg = lambda_reg * (mu_delta.pow(2).mean() + (std_delta).pow(2).mean())
            writer.add_scalar("loss_CLIP_f1_batch{}".format(i), loss_CLIP1.item(), cur_itrs)
            loss = loss_CLIP1 + reg
            loss.backward(retain_graph=True)
            optimizer_pin_1.step()

        # extract learned params and save per-sample stats to .pkl like原来
        learnt_mu_f1 = model_pin_1.style_mean.detach().cpu()
        learnt_std_f1 = model_pin_1.style_std.detach().cpu()

        for k in range(learnt_mu_f1.shape[0]):
            stats = {'mu_f1': learnt_mu_f1[k].clone(), 'std_f1': learnt_std_f1[k].clone()}
            index_id = indices[k].item() if isinstance(indices[k], torch.Tensor) else indices[k]
            save_path = os.path.join(opts.save_dir, f"{index_id}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(stats, f)

    print("Done generating PIN stats.")

if __name__ == "__main__":
    main()
