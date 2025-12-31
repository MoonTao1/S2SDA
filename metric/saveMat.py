import os
import time
import sys
import os

# 将项目根目录加入 Python 搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import random
import warnings
import logging
import numpy as np
import json
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from utils_.build_datasets import build_dataset
from utils_.options import parser
args = parser.parse_args()
import network
import random
import argparse
# 把上一级目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

warnings.simplefilter("ignore")
os.environ['NUMEXPR_MAX_THREADS'] = '64'
args = parser.parse_args()
torch.cuda.set_device(int(args.gpu))

def get_argparser():
    parser = argparse.ArgumentParser()
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type = str, default = "RN50",
                        help = "backbone of the segmentation network")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    # parser.add_argument("--total_itrs", type=int, default=200,
    #                     help="epoch number (default: 200k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)

    parser.add_argument("--ckpt", default='', type=str,
                        help="restore from checkpoint")

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--forward_pass",action='store_true',default=False,
                        help="forward pass to update BN statistics")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--freeze_BB", action='store_true',default=True,
                        help="Freeze the backbone when training")
    parser.add_argument("--ckpts_path", type = str ,default='',
                        help="path for checkpoints saving")
    parser.add_argument("--data_aug", action='store_true', default=True)
    #validation
    parser.add_argument("--val_results_dir", type=str,help="Folder name for validation results saving")
    #Augmented features
    parser.add_argument("--train_aug",action='store_true',default=False,
                        help="train on augmented features using CLIP")
    parser.add_argument("--path_mu_sig", type=str,default='/data9102/workspace/mwt/PODA/pins')
    parser.add_argument("--mix", action='store_true',default=False,
                        help="mix statistics")

    return parser

opts = get_argparser().parse_args()
writer = SummaryWriter()
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

model_root = 'models'
cuda = True

def predict_mat(test_loader, model):
    import scipy.io as sio

    test_imgs = [json.loads(line) for line in open(args.root + f'{args.w}_test.json')]

    # switch to evaluate mode
    model.eval()
    print(args.w)
    save_path = f"{ckpts}{args.w}_outputMat/"

    total_samples = len(test_imgs)

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="matFile Saving", ncols=90)

    for i, (input, target,_) in progress_bar:
        target = target.squeeze(1)
        input = input.cuda()

        # compute output
        output,_ = model(input)
        output = output.squeeze(1)


        target = target.detach().cpu().numpy()
        output = output.detach().cpu().numpy()

        batch = output.shape[0]

        for j in range(batch):
            index = i * test_loader.batch_size + j
            save_name = test_imgs[index].replace('png', 'mat')


            tar = target[j]
            pre = output[j]

            tar_mat_dict = {'tar': tar}
            pre_mat_dict = {'pre': pre}

            # create directories if not exist
            # print(save_name)
            path = save_path + 'tar/' + save_name
            path2 = path.replace('tar', 'pre')

            path_ = os.path.dirname(path)
            path2_ = os.path.dirname(path2)

            # print(path, path2)
            # exit(0)
            os.makedirs(path_, exist_ok=True)
            os.makedirs(path2_, exist_ok=True)

            sio.savemat(path, tar_mat_dict)
            sio.savemat(path2, pre_mat_dict)
        
            # update
            # progress_bar.set_postfix(Iter="{:03d}/{:03d}".format(i + 1, len(valid_loader)))

if __name__ == '__main__':
    # save------>next
    _, _, test_loader = build_dataset(args=args)

    print(args.category)
    # ckpts = f'ckpts/{args.category}/{args.test_weight}/'
    ckpts = '/data9102/workspace/mwt/Experiment/DADA/PODA/abalation/DADA/ori+seg+fusion/snowy/'
    model = network.modeling.__dict__[opts.model](num_classes=1, BB=opts.BB,
                                                  replace_stride_with_dilation=[False, False, True])
    model.backbone.attnpool = nn.Identity()

    checkpoint = torch.load(
        '/data9102/workspace/mwt/Experiment/DADA/PODA/abalation/DADA/ori+seg+fusion/snowy/best__deeplabv3plus_resnet_clip_DADA.pth',
        map_location="cuda:0")
    model.load_state_dict(checkpoint["model_state"])


    model = model.cuda()
    model.eval()

    if any(key.startswith('module.') for key in checkpoint['model_state'].keys()):
        # multi-GPU: move 'module.'
        print("[Start Testing] Detected multi-GPU training. Loading")
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state'].items()})
    else:
        # single-GPU：
        print("[Start Testing] Detected single-GPU training. Loading")
        model.load_state_dict(checkpoint['model_state'])

    predict_mat(test_loader, model)
