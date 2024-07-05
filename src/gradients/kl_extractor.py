import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf

from .base_extractor import BaseGradientExtractor
from torchvision.transforms import InterpolationMode
from torch.nn.functional import log_softmax, softmax, kl_div


class KLGradientsExtractor(BaseGradientExtractor):
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: torch.device,
        projection: torch.Tensor,
        projection_scaling: float,
        embeddings_dim: int,
        latent_dim: int,
        input_dim: int = 224,
        use_fp16: bool = False,
        fp16_dtype: torch.dtype = torch.bfloat16,
        temperature: float = 1,
        **kwargs
    ) -> None:
        super().__init__(
            model,
            target_layer,
            device,
            projection,
            projection_scaling,
            embeddings_dim,
            latent_dim,
            input_dim,
            use_fp16,
            fp16_dtype,
            **kwargs
        )

        self.temperature = temperature

    def input_transform(self, input_dim: int) -> nn.Module:
        """
            Returns the transform applied to input images, which
            resizes the input to (256, 256), crops it in the center
            so that the input has shape (input_dim, input_dim) and
            normalizes it using the ImageNet statistics.

            Args:
                input_dim (int): the input dimensionality of the model.

            Returns:
                nn.Module: a transform that returns a [C, H, W] tensor.
        """
        return tf.Compose([
            tf.ToImage(),
            tf.Resize((256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True),
            tf.CenterCrop(input_dim),
            tf.ToDtype(torch.float32, scale=True),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def compute_loss(self, latents: torch.Tensor, views_per_sample: int, **kwargs) -> torch.Tensor:
        """
            Compute the KL divergence between the softmaxed latents and an
            uniform distribution.

            Args:
                latents (torch.Tensor): the encoded latents for all batch
                    items and their views.
                views_per_sample (int): the number of views generated for
                    each original image.

            Returns:
                torch.Tensor: the loss tensor, as a single number averaged
                    over the batch dimension.
        """
        latent_dim = latents.shape[1]

        uniform = (torch.ones(latent_dim) / latent_dim).to(self.device)

        softmax_uniform = softmax(uniform / self.temperature, dim=0)
        softmax_uniform = softmax_uniform.unsqueeze(dim=0).repeat(latents.shape[0], 1)

        softmax_latents = log_softmax(latents / self.temperature, dim=1)

        return kl_div(softmax_latents, softmax_uniform, reduction="mean")
