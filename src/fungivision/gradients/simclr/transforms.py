import math
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf

from PIL import Image


class Patchify(nn.Module):
    def __init__(self, num_patches: int, image_size: int = 224, stride_scale: int = 4, **kwargs) -> None:
        """
            Data augmentation strategy that splits an image into equal-size
            patches. The data augmentation works as follows:

            1. Resize the input image to (image_size, image_size)
            2. Split the image into num_patches patches. The square root
               of the number of patches must be an integer!
            3. Resize each patch to (image_size, image_size) and apply the
               ImageNet normalization

            Args:
                num_patches (int): number of patches to extract if the stride
                    scale was 1. Must be a perfect square.
                image_size (int): the input size required by the model.
                stride_scale (int): the factor (1/x) used to calculate the stride
                    step relative to the patch size.
        """
        super().__init__()

        self.num_patches = num_patches
        self.patch_size = image_size / math.sqrt(num_patches)
        self.stride_scale = stride_scale

        assert self.patch_size.is_integer(), f"the image size must be divisible by the square root of the number of patches! image size is {image_size}, num patches is {num_patches}"

        # The patch size must be an integer to work with unfold
        self.patch_size = int(self.patch_size)

        # Resize the image to (image_size, image_size) and
        # convert it into a tensor
        self.in_transform = tf.Compose([
            tf.Resize((image_size, image_size), interpolation=Image.BICUBIC, antialias=True),
            tf.ToTensor()
        ])

        # Normalize the output patches and resize them to (image_size, image_size) 
        self.out_transform = tf.Compose([
            tf.Resize((image_size, image_size), interpolation=Image.BICUBIC, antialias=True),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
            Patchify an image.

            Args:
                img (PIL.Image): a PIL image.

            Returns:
                torch.Tensor: a tensor of shape [V, C, H, W] where V
                    is the total number of patches.
        """
        img = self.in_transform(img)
        stride, kernel_h, kernel_w = int(self.patch_size / self.stride_scale), self.patch_size, self.patch_size

        # Based on this answer
        # https://discuss.pytorch.org/t/efficiently-slicing-tensor-like-a-convolution/44840
        patches = img.unfold(1, kernel_h, stride).unfold(2, kernel_w, stride)

        # N is the number of patches for every individual dimension
        C, N, _, H, W = patches.shape

        # Transform the patches tensor into [patches, channels, height, width]
        patches = patches.reshape(C, N ** 2, H, W).permute(1, 0, 2, 3)
        # Patches are now (patch_size, patch_size). Normalize and resize them
        # to have shape (image_size, image_size)
        return torch.cat(
            [self.out_transform(x).unsqueeze(dim=0) for x in patches],
            dim=0
        )
