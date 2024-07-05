import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf

from PIL import Image
from typing import Tuple


class DINOTransform(nn.Module):
    """
        The DINO data augmentation, using only random crops. Mostly taken from
        https://github.com/facebookresearch/dino/blob/main/main_dino.py#L419.
    """
    def __init__(
        self,
        global_crops_scale: Tuple[int, int] = (0.25, 1.0),
        local_crops_scale: Tuple[int, int] = (0.05, 0.25),
        local_crops_number: int = 10,
        crops_size: int = 224
    ):
        super().__init__()

        self.local_crops_number = local_crops_number

        normalize = tf.Compose([
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.first_global_crop = tf.Compose([
            tf.RandomResizedCrop(crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            normalize
        ])

        self.second_global_crop = tf.Compose([
            tf.RandomResizedCrop(crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            normalize
        ])

        self.local_crop = tf.Compose([
            tf.RandomResizedCrop(crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            normalize
        ])

    def __call__(self, image: Image):
        crops = [self.first_global_crop(image), self.second_global_crop(image)]
        crops += [self.local_crop(image) for _ in range(self.local_crops_number)]

        # Crops is now a 12-items array (assuming the regular amount
        # of local crops is ued). We return a [12, 3, 224, 224] tensor
        return torch.cat([x.unsqueeze(dim=0) for x in crops], dim=0)
