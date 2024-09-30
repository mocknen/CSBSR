from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import re

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler

from model.config import get_cfg_defaults
from model.modeling.build_model import JointModel, JointInvModel
from model.data.mask_dataset import GenMaskDataset
from model.engine.mask import generate_mask
from model.utils.misc import fix_model_state_dict


def get_args():
    p = ArgumentParser()
    p.add_argument('base_dir', type=Path,
                   default='weights/CSBSR_w_PSPNet_beta03')
    p.add_argument('iter_or_weight_name', type=str, default='latest')
    p.add_argument('input_dir', type=Path,
                   default='datasets/crack_segmentation_dataset')
    p.add_argument('--config-file',
                   type=Path, default=None, metavar='FILE')
    p.add_argument('--origin-img-size',
                   action=BooleanOptionalAction,
                   default=True)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--test-blured', type=str, default=None)
    p.add_argument('--trained-model', type=Path, default=None)

    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--num-gpus', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--save-image',
                   action=BooleanOptionalAction,
                   default=True)
    return p.parse_args()


def main():
    args = get_args()

    if re.match(r'^\d+$', args.iter_or_weight_name):
        default_model = f'iteration_{args.iter_or_weight_name}'
    else:
        default_model = args.iter_or_weight_name

    default_out_dir = args.base_dir / 'out' / default_model
    default_model_path = args.base_dir / 'model' / f'{default_model}.pth'

    cfg = get_cfg_defaults()
    print(f'cfg: {cfg}')
    config_file = args.config_file or args.base_dir / 'config.yaml'
    print(f'Configration file: {config_file}')
    cfg.merge_from_file(config_file)

    if args.test_blured:
        cfg.DATASET.TEST_BLURED_NAME = args.test_blured
        default_out_dir = default_out_dir.parent / f'blured_{args.test_blured}'

    print(f'Size of input image is {cfg.INPUT.IMAGE_SIZE}.')

    cfg.OUTPUT_DIR = str(args.output_dir or default_out_dir)
    cfg.freeze()

    print(f'config:\n{cfg}')

    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    device = torch.device(cfg.DEVICE)
    model = (JointInvModel(cfg).to(device) if cfg.MODEL.SR_SEG_INV
             else JointModel(cfg).to(device))

    model.load_state_dict(fix_model_state_dict(torch.load(
        args.trained_model or default_model_path,
        map_location=lambda x, _: x)))

    model.eval()

    print('Loading Datasets...')
    # sr_transforms = FactorResize(cfg.MODEL.SCALE_FACTOR,
    #                              cfg.SOLVER.DOWNSCALE_INTERPOLATION)
    dataset = GenMaskDataset(
        args.input_dir,
        args.batch_size,
        cfg.INPUT.IMAGE_SIZE,
        cfg.MODEL.NUM_CLASSES,
        cfg.MODEL.SCALE_FACTOR)
    batch_sampler = BatchSampler(
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size, drop_last=False)
    loader = DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

    if args.num_gpus > 1:
        device_ids = list(range(args.num_gpus))
        print(f'{device_ids=}')
        model = DataParallel(model, device_ids=device_ids)

    with torch.no_grad():
        generate_mask(cfg, model, loader)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
