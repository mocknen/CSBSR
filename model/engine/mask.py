from datetime import datetime
from pathlib import Path

import torch

from model.data.samplers.patch_sampler import JointPatch
from model.utils.save_output import save_img, save_mask  # , save_kernel


def generate_mask(cfg, model, loader):
    iter_n = len(loader)
    assert iter_n, 'Dataset size is 0!!'

    joint_patch = JointPatch()

    print('Initializing the model')
    model.eval()

    for iter, (imgs, sr_targets, names, img_unfold_shape,
               seg_unfold_shape) in enumerate(loader, 1):
        print(f'{datetime.now()}: Iteration {iter}/{iter_n}')

        imgs = imgs.view(-1, *imgs.shape[2:])
        num_batch = len(imgs)
        num_patch = img_unfold_shape[0][2] * img_unfold_shape[0][3]
        kerns = torch.zeros((num_batch * num_patch, 1,
                             cfg.BLUR.KERNEL_SIZE, cfg.BLUR.KERNEL_SIZE))

        sr_preds, segment_preds, kernel_preds = model(
            imgs, kerns, sr_targets=sr_targets)

        sr_preds = joint_patch(sr_preds, img_unfold_shape[0])
        segment_preds = joint_patch(segment_preds, seg_unfold_shape[0])

        if not cfg.MODEL.SR_SEG_INV and cfg.MODEL.SCALE_FACTOR != 1:
            sr_preds[sr_preds > 1] = 1
            sr_preds[sr_preds < 0] = 0
            save_img(Path(cfg.OUTPUT_DIR) / 'images', sr_preds, names)

            # if cfg.MODEL.SR == 'KBPN':
            #     kernel_preds[kernel_preds > 1] = 1
            #     kernel_preds[kernel_preds < 0] = 0
            #     save_kernel(args, kernel_preds, fname, num_batch)

        save_mask(Path(cfg.OUTPUT_DIR) / 'masks', segment_preds, names, -1)
