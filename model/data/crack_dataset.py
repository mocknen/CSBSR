# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Toyota Technological Institute
# Author: Yuki Kondo
# Copyright (c) 2024
# yuki.kondo.ab@gmail.com
#
# This source code is licensed under the Apache License license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from copy import copy
import os
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

from model.data.blur.blur import set_blur, conv_kernel2d
from model.data.samplers.patch_sampler import SplitPatch


class CrackDataSet(Dataset):
    def __init__(self, cfg, image_dir, mask_dir,
                 transforms=None, sr_transforms=None):
        print('Instantiating CrackDataSet')
        print(f'CrackDataSet {image_dir=}')
        print(f'CrackDataSet {mask_dir=}')

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.fnames = [p.name for p in self.image_dir.glob('*.jpg')]
        self.fnames += [p.name for p in self.image_dir.glob('*.png')]
        self.img_transforms = transforms
        self.sr_transforms = sr_transforms
        self.blur_flag = cfg.BLUR.FLAG
        self.blur_kernel_size = cfg.BLUR.KERNEL_SIZE_OUTPUT
        self.blur_isotropic = cfg.BLUR.ISOTROPIC

    def __getitem__(self, i):
        fname = self.fnames[i]
        img, seg_target = self.img_transforms(
            np.array(Image.open(self.image_dir / fname)),
            np.array(Image.open(self.mask_dir / fname))[:, :, np.newaxis]
        )
        sr_target = copy(img)

        if self.blur_flag:
            blur_kernel = set_blur(self.blur_kernel_size, mode='gaus',
                                   isotropic=self.blur_isotropic).to('cpu')
            img = conv_kernel2d(img, blur_kernel).to('cpu')
            blur_kernel = blur_kernel.view(1, *blur_kernel.shape)
        else:
            blur_kernel = torch.zeros((1, self.blur_kernel_size,
                                       self.blur_kernel_size))
            center = self.blur_kernel_size // 2
            blur_kernel[0, center, center] = 1

        # print('dataset', blur_kernel.size())

        img = self.sr_transforms(img)

        return img, sr_target, seg_target, blur_kernel

    def __len__(self):
        # print(self.fnames)
        return len(self.fnames)


class CrackDataSetTest(Dataset):
    def __init__(self, cfg, image_dir, mask_dir,
                 blur_dir, blur_name, batch_size,
                 transforms=None, sr_transforms=None):
        print('Instantiating CrackDataSetTest')

        print(f'{image_dir=}')
        self.img_paths = sorted(Path(image_dir).glob('*.jpg'))
        self.img_paths += sorted(Path(image_dir).glob('*.png'))

        print(f'{mask_dir=}')
        self.mask_dir = Path(mask_dir)

        self.kern_dir = Path(blur_dir) / blur_name / 'kernels'
        self.lrimg_dir = Path(blur_dir) / blur_name / 'lr_images'

        self.img_transforms = transforms
        self.sr_transforms = sr_transforms
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        patch_sizeh, patch_sizew = [int(i / self.scale_factor)
                                    for i in cfg.INPUT.IMAGE_SIZE]
        self.split_img_patch = SplitPatch(batch_size, 3, patch_sizeh,
                                          patch_sizew)
        self.seg_ch = cfg.MODEL.NUM_CLASSES
        self.blur_kernel_size = cfg.BLUR.KERNEL_SIZE_OUTPUT

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        sr_target = np.array(Image.open(img_path))  # Image.open(fpath)
        mask_path = self.mask_dir / img_path.name
        seg_target = (
            np.array(Image.open(mask_path))[:, :, np.newaxis] if mask_path.is_file()
            else None
        )
        sr_target, mask = self.img_transforms(sr_target, seg_target)

        kern = (lambda x: x / torch.sum(x))(
            self.img_transforms(np.array(
                Image.open(self.kern_dir / img_path.with_suffix('.png').name)
            )[:, :, np.newaxis])[0]
        )
        # print(kern.shape)

        if self.scale_factor != 1:
            img = self.img_transforms(np.array(
                Image.open(self.lrimg_dir / img_path.with_suffix('.png').name)
            ))[0]
        else:
            img = sr_target.detach()

        img, img_unfold_shape = self.split_img_patch(img)  # img.shape = [patches, ch, H, W]
        # Considering the effect of upsampling when combining patch images
        img_unfold_shape[[5, 6]] = img_unfold_shape[[5, 6]] * self.scale_factor
        seg_unfold_shape = img_unfold_shape.copy()
        # initialize segmentation channel
        seg_unfold_shape[[1, 4]] = self.seg_ch

        num_patch = img_unfold_shape[2] * img_unfold_shape[3]
        kern = kern.expand(num_patch, *kern.shape[1:])

        return img, sr_target, mask, kern, img_path.name, img_unfold_shape, seg_unfold_shape

    def __len__(self):
        return len(self.img_paths)


class TTICrackDataSetTest(Dataset):
    def __init__(self, cfg, image_dir, batch_size, transforms=None):
        self.image_dir = image_dir
        self.fnames = [path.name for path in Path(image_dir).glob('*.png')]
        self.img_transforms = transforms
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.split_img_patch = SplitPatch(batch_size, 3, *cfg.INPUT.IMAGE_SIZE)
        self.seg_ch = cfg.MODEL.NUM_CLASSES

    def __getitem__(self, i):
        fname = self.fnames[i]
        fpath = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(fpath))  # Image.open(fpath)
        img, _ = self.img_transforms(img, None)
        # print(img.shape)
        # print(blur_kernel.shape)

        # img.shape = [patches, ch, H, W]
        img, img_unfold_shape = self.split_img_patch(img)
        # Considering the effect of upsampling when combining patch images
        img_unfold_shape[[5, 6]] = img_unfold_shape[[5, 6]] * self.scale_factor
        # print(img_unfold_shape)
        seg_unfold_shape = img_unfold_shape.copy()
        # initialize segmentation channel
        seg_unfold_shape[[1, 4]] = self.seg_ch

        return img, fname, img_unfold_shape, seg_unfold_shape

    def __len__(self):
        return len(self.fnames)


class SRPretrainDataSet(Dataset):
    def __init__(self, cfg, image_dir, transforms=None, sr_transforms=None):
        print('image_dir', image_dir)
        self.image_dir = image_dir
        self.fnames = [path.name for path in Path(image_dir).glob('*.png')]
        self.img_transforms = transforms
        self.sr_transforms = sr_transforms
        self.blur_flag = cfg.BLUR.FLAG
        self.blur_kernel_size = cfg.BLUR.KERNEL_SIZE_OUTPUT
        self.blur_isotropic = cfg.BLUR.ISOTROPIC

    def __getitem__(self, i):
        fname = self.fnames[i]
        fpath = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(fpath))   # Image.open(fpath)

        img, _ = self.img_transforms(img, None)
        sr_target = copy(img)

        # range_gaus_deterioration_ratio=0.2
        if self.blur_flag:
            blur_kernel = set_blur(self.blur_kernel_size, mode='gaus',
                                   isotropic=self.blur_isotropic).to('cpu')
            # print(blur_kernel.shape)
            img = conv_kernel2d(img, blur_kernel).to('cpu')
            blur_kernel = blur_kernel.view(1, *blur_kernel.shape)

        else:
            blur_kernel = torch.zeros((1, self.blur_kernel_size,
                                       self.blur_kernel_size))
            center = self.blur_kernel_size // 2
            blur_kernel[0, center, center] = 1

        img = self.sr_transforms(img)
        # print('fin dataset')
        return img, sr_target, blur_kernel

    def __len__(self):
        return len(self.fnames)
