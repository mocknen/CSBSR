# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Toyota Technological Institute
# Author: Yuki Kondo
# Copyright (c) 2024
# yuki.kondo.ab@gmail.com
#
# This source code is licensed under the Apache License license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os

import torchvision.transforms as transforms
import torch


def save_img(out_dir, sr_preds, names):
    for batch_num in range(sr_preds.size()[0]):
        if sr_preds.shape[1] == 3:
            sr_img = transforms.ToPILImage(mode='RGB')(sr_preds[batch_num])
        elif sr_preds.shape[1] == 1:
            sr_img = transforms.ToPILImage(mode='L')(sr_preds[batch_num])
        dest_path = out_dir / names[batch_num]
        os.makedirs(dest_path.parent, exist_ok=True)
        sr_img.save(dest_path)


def save_mask(out_dir, segment_preds, names, iou_th):
    for batch_num in range(segment_preds.size()[0]):
        mask_img = transforms.ToPILImage()(segment_preds[batch_num])
        dest_path = out_dir / f'th_{iou_th:.2f}' / names[batch_num]
        os.makedirs(dest_path.parent, exist_ok=True)
        mask_img.save(dest_path)


def save_kernel(args, kernel_preds, fname, num_batch, add_path=''):
    # print(segment_predss.shape)
    # print(type(segment_preds))
    # segment_preds = segment_preds.to("cpu")
    num_patch = kernel_preds.shape[0] // num_batch
    for i in range(num_batch):
        fname_ = f'{fname[i]}'.replace('.png', '')
        for j in range(num_patch):
            idx = i * num_patch + j

            kernel_pred = kernel_preds[idx] / torch.max(kernel_preds[idx])
            kernel_pred = transforms.ToPILImage()(kernel_pred)

            outdir = args.output_dirname / f'kernels{add_path}'
            os.makedirs(outdir, exist_ok=True)
            fname_j = fname_ + f'_{j}' + '.png'
            mpath = outdir / fname_j
            # print(mpath)

            kernel_pred.save(mpath)

            # print(torch.max(kernel_preds[idx]))
            kernel_pred_origin = kernel_preds[idx] / torch.sum(kernel_preds[idx])
            kernel_pred_origin = transforms.ToPILImage()(kernel_pred_origin)

            outdir = args.output_dirname / f'kernels{add_path}_origin'
            os.makedirs(outdir, exist_ok=True)
            fname_j = fname_ + f'_{j}_origin' + '.png'
            mpath = outdir / fname_j
            kernel_pred_origin.save(mpath)
