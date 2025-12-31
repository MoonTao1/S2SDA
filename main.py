from re import L
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import time
from torch.utils import data
from datasets import Cityscapes, gta5
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import logging
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pickle
from utils.utils import denormalize
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils_.build_datasets import build_dataset
from utils_.options import parser
args = parser.parse_args()
from utils_.mid_metrics import cc, sim, kldiv
from utils.stats import calc_mean_std
writer = SummaryWriter()
import math

def get_argparser():
    parser = argparse.ArgumentParser()

    
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default="RN50",
                        help="backbone of the segmentation network")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=50000,
                        help="epoch number (default: 200k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)

    parser.add_argument("--ckpt",
                        default='',
                        type=str,
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
    parser.add_argument("--val_interval", type=int, default=5000,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--forward_pass", action='store_true', default=True,
                        help="forward pass to update BN statistics")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--freeze_BB", action='store_true', default=True,
                        help="Freeze the backbone when training")
    parser.add_argument("--ckpts_path", type=str, default='',
                        help="path for checkpoints saving")
    parser.add_argument("--data_aug", action='store_true', default=False)
    # validation
    parser.add_argument("--val_results_dir", type=str, help="Folder name for validation results saving")
    # Augmented features
    parser.add_argument("--train_aug", action='store_true', default=True,
                        help="train on augmented features using CLIP")
    parser.add_argument("--path_mu_sig", type=str, default='')
    parser.add_argument("--mix", action='store_true', default=False,
                        help="mix statistics")
    return parser


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(opts, model, loader,device):
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    start = time.time()
    metrics = [0, 0, 0]
    with torch.no_grad():
        for i, (input, target,_) in enumerate(loader):

            input = input.to(device)
            target = target.to(device)
            # compute output
            model = model.to(device)
            output, features = model(input)

            criterion = nn.BCELoss(reduction='mean')
            loss = criterion(output, target)

            # measure accuracy and record loss

            losses.update(loss.data, target.size(0))
            # valid metrics printing
            output = output.squeeze(1)
            target = target.squeeze(1)
            val_cc = cc(output, target).item()
            val_sim = sim(output, target).item()
            val_kld = kldiv(output, target).item()

            metrics[0] += 0 if math.isnan(val_cc) else val_cc
            metrics[1] += 0 if math.isnan(val_sim) else val_sim
            metrics[2] += 0 if math.isnan(val_kld) else val_kld


            msg = 'Validating Iter {:03d} Loss {:.6f} || CC {:4f}  SIM {:4f}  KLD {:4f} in {:.3f}s'.format(i + 1,
                                                                                                           losses.avg,
                                                                                                           metrics[
                                                                                                               0] / (
                                                                                                                   i + 1),
                                                                                                           metrics[
                                                                                                               1] / (
                                                                                                                   i + 1),
                                                                                                           metrics[
                                                                                                               2] / (
                                                                                                                   i + 1),
                                                                                                           time.time() - start)
            print(msg)
            # logging.info(msg)
            start = time.time()

            del input, target, output
            # gc.collect()

            interval = 5
            if (i + 1) % interval == 0:
                logging.info(msg)
    model.train()

    return losses.avg, metrics

def main():
    opts = get_argparser().parse_args()
    print('begin')
    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    # _,val_loader,_ = build_dataset(args=args)
    # args.category = 'TrafficGaze'
    # args.root = '/'
    print(args.category)
    print(args.w)
    _, val_loader, _ = build_dataset(args=args)
    args.w = 'sunny'
    # args.category = 'TrafficGaze'
    # args.root = ''

    train_loader, _,_ = build_dataset(args=args)
    # train_loader = data.DataLoader(
    #     train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
    #     drop_last=True)  # drop_last=True to ignore single-image batches.

    # val_loader = data.DataLoader(
    #     val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)

    print("Dataset: %s, Train set: %d, Val set: %d" %
          (args.category, len(train_loader), len(val_loader)))

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=1, BB=opts.BB,
                                                  replace_stride_with_dilation=[False, False, True])
    model.backbone.attnpool = nn.Identity()

    # fix the backbone
    if opts.freeze_BB:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()

    if opts.freeze_BB:
        optimizer = torch.optim.SGD(params=[
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.001 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.9)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.BCELoss(reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    if not opts.test_only:
        utils.mkdir(opts.ckpts_path)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):

        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model.to(device)

    interval_loss = 0

    if opts.train_aug:
        files = [f for f in os.listdir(opts.path_mu_sig + '/')]

    relu = nn.ReLU(inplace=True)

    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====

        if opts.freeze_BB:
            model.classifier.train()
        else:
            model.train()

        cur_epochs += 1
        for (images, labels,im_id) in train_loader:
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            if opts.train_aug:
                mu_t_f1 = torch.zeros([opts.batch_size, 256, 1, 1])
                std_t_f1 = torch.zeros([opts.batch_size, 256, 1, 1])
                mu_t_f1 = mu_t_f1.cuda()
                std_t_f1 = std_t_f1.cuda()
                for k in range(opts.batch_size):
                    with open(opts.path_mu_sig + '/' + random.choice(files), 'rb') as f:
                        loaded_dict = pickle.load(f)
                        mu_t_f1[k] = loaded_dict['mu_f1']
                        std_t_f1[k] = loaded_dict['std_f1']

                outputs, features = model(images, mu_t_f1.to(device), std_t_f1.to(device),
                                          transfer=True, mix=opts.mix, activation=relu)

            else:
                outputs, features = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss", loss, cur_itrs)
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if cur_itrs % opts.val_interval == 0:
                print('valid')
                val_loss, val_metrics = validate(opts, model, val_loader, device)
                print(
                    f"Validation | Loss: {val_loss:.6f} | CC: {val_metrics[0]:.4f} | SIM: {val_metrics[1]:.4f} | KLD: {val_metrics[2]:.4f}")

                # 保存 best model 依据 CC 最大
                if val_metrics[0] > best_score:
                    best_score = val_metrics[0]
                    save_ckpt(opts.ckpts_path + '/best__%s_%s.pth' %
                              (opts.model, args.category))
                    # torch.save(model.state_dict(),
                    #            os.path.join(opts.ckpts_path, f'best_{opts.model}_{args.category}.pth'))

                if opts.freeze_BB:
                    model.classifier.train()
                else:
                    model.train()

            if opts.train_aug and cur_itrs == opts.total_itrs:
                save_ckpt(opts.ckpts_path + '/adapted_%s_%s_fin.pth' %
                          (opts.model, args.category))

            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
