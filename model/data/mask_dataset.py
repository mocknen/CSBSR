from pathlib import Path

from PIL import Image
import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

from model.data.samplers.patch_sampler import SplitPatch


class GenMaskDataset(Dataset):
    def __init__(self,
                 image_dir,
                 batch_size,
                 input_image_size,
                 model_class_n,
                 scale_factor):
        print('Instantiating GenMaskDataset')
        print(f'{image_dir=}')
        self.img_paths = sorted(Path(image_dir).glob('*.jpg'))
        self.img_paths += sorted(Path(image_dir).glob('*.png'))
        self.scale_factor = scale_factor
        patch_size = [int(i / self.scale_factor) for i in input_image_size]
        self.split_img_patch = SplitPatch(batch_size, 3, *patch_size)
        self.seg_ch = model_class_n

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        sr_target = from_numpy(
            np.array(Image.open(img_path)).astype(np.float32)
        ).permute(2, 0, 1) / 255

        height, width = sr_target.shape[-2:]
        sr_transform = Resize(
            (int(height / self.scale_factor), int(width / self.scale_factor)),
            InterpolationMode.BICUBIC)
        img = sr_transform(sr_target)

        img, img_unfold_shape = self.split_img_patch(img)  # img.shape = [patches, ch, H, W]

        # Considering the effect of upsampling when combining patch images
        img_unfold_shape[[5, 6]] = img_unfold_shape[[5, 6]] * self.scale_factor

        seg_unfold_shape = img_unfold_shape.copy()

        # initialize segmentation channel
        seg_unfold_shape[[1, 4]] = self.seg_ch

        return img, sr_target, img_path.name, img_unfold_shape, seg_unfold_shape

    def __len__(self):
        return len(self.img_paths)
