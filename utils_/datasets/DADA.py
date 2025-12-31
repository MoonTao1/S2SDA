from torch.utils.data import Dataset
import imageio as io
import cv2
import torch
from numpy import *
import os
import numpy as np



class ImageList_DADA(Dataset):
    def __init__(self, args, imgs, for_train=False):
        self.root = args.root
        self.label_root = args.root
        self.imgs = imgs
        self.img_shape = args.img_shape
        self.for_train = for_train

        # 用于备份缺失帧
        self.last_valid_frame = None

    def __getitem__(self, index):
        img_rel_path = self.imgs[index]  # 例如 "1/001/images/0049.png"
        vid_cls, vid_id, _, frame_file = img_rel_path.split('/')
        frame_index = int(frame_file[:-4])

        # 构建图像路径
        img_name = os.path.join(self.root, f'{vid_cls}/{vid_id}/images/{frame_index:04d}.png')
        if not os.path.exists(img_name):
            if self.last_valid_frame is None:
                raise FileNotFoundError(f"No valid image for index {index}: {img_name}")
            img_name = self.last_valid_frame
        else:
            self.last_valid_frame = img_name

        # 读入图像
        img = io.imread(img_name)
        img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32') / 255.0

        # 构建标签路径
        lab_name = os.path.join(self.label_root, f'{vid_cls}/{vid_id}/maps/{frame_index:04d}.png')
        if not os.path.exists(lab_name):
            raise FileNotFoundError(f"Label not found: {lab_name}")

        lab_img = cv2.imread(lab_name, 0)
        lab_img = cv2.resize(lab_img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        lab_img = lab_img.astype('float32') / 255.0

        # 数据增强
        if self.for_train:
            img, lab_img = transform(img, lab_img)

        # 转换通道顺序
        img = img.transpose(2, 0, 1)
        lab_img = lab_img[None, ...]  # (1,H,W)

        img = np.ascontiguousarray(img)
        lab_img = np.ascontiguousarray(lab_img)

        return torch.from_numpy(img), torch.from_numpy(lab_img),index

    def __len__(self):
        return len(self.imgs)



def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y
