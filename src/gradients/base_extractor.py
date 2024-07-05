import torch
import torch.nn as nn
import src.utils.autograd_hacks as autograd_hacks

from PIL import Image
from typing import List
from abc import abstractmethod, ABC
from src.utils.misc import rgetattr
from torch.utils.data import Dataset


class BaseGradientExtractor(ABC):
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: torch.device,
        projection: torch.Tensor,
        projection_scaling: float,
        input_dim: int = 224,
        use_fp16: bool = False,
        fp16_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ) -> None:
        """Initialize a gredient extractor."""
        # FIXME: Expand on this
        super().__init__()

        assert model is not None, f"a valid model must be provided! {model} given"
        assert target_layer is not None, f"a valid target layer must be provided! {target_layer} given"

        # FIXME: We assume the model is already wrapped by autograd hacks? I guess yes, but we need
        # to add the projection head before passing it to this function

        self.model = model
        self.device = device
        self.projection = projection.to(device)
        self.projection_scaling = projection_scaling
        self.target_layer = target_layer

        # FP16 configuration. Using fp16 assumes we are running on a CUDA device
        self.use_fp16 = use_fp16
        self.fp16_dtype = fp16_dtype

        self.transform = self.input_transform(input_dim=input_dim, **kwargs)

        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    @abstractmethod
    def input_transform(self, input_dim: int) -> nn.Module:
        """
            Returns the transform applied to input images. The transform may
            return one or more images, but their shape must be [V, C, H, W],
            where V indicates the number of views generated for a specific
            image
        """
        raise NotImplementedError

    def setup(self, dataset: Dataset) -> None:
        """
            Generate any data required for the loss computation,
            given a dataset. This is useful for e.g. contrastive
            learning, where a negatives batch is needed to compute
            the loss
        """
        pass

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """Extracts the gradients for a list of images"""
        # Clear tensors used for the per-sample gradient computation
        autograd_hacks.clear_backprops(self.model)

        # Transform the input images and generate the views for the SSL loss
        views = [self.transform(x).unsqueeze(dim=0) for x in images]

        batch_size = len(images)

        # Encode the views into latents
        views = torch.cat(views, dim=0).to(self.device)

        if len(views.shape) == 5:
            # Remove the view dimension when the samples are being encoded
            B, V, C, H, W = views.shape
            views = views.reshape(B * V, C, H, W)
        else:
            V = 1

        # Compute the latents, loss, backpropagate and calculate the per-view gradients
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_fp16):
            latents = self.model(views)
            loss = self.compute_loss(latents=latents, views_per_sample=V)

        return self.backprop(
            loss=loss,
            batch_size=batch_size,
            views_per_sample=V
        )

    def backprop(self, loss: torch.Tensor, batch_size: int, views_per_sample: int) -> torch.Tensor:
        if self.use_fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        autograd_hacks.compute_grad1(self.model, layer_paths=[self.target_layer])

        # Extract the per-sample gradients
        layer = rgetattr(self.model, self.target_layer)
        gradients = self.extract_gradients(layer=layer, batch_size=batch_size, views_per_sample=views_per_sample)

        # Project the gradients
        if self.use_fp16:
            gradients = gradients.to(self.fp16_dtype)

        gradients = gradients.view(batch_size, -1)
        gradients = self.projection_scaling * (self.projection @ gradients.T).permute(1, 0)

        return gradients

    def extract_gradients(self, layer: nn.Module, batch_size: int, views_per_sample: int) -> torch.Tensor:
        """Extract per-sample gradients for a given linear layer"""
        H, W = layer.weight.shape[:2]

        weight_gradient = layer.weight.grad1.reshape(batch_size, views_per_sample, H, W)
        weight_gradient = weight_gradient.sum(dim=1)

        if hasattr(layer, "bias") and layer.bias is not None:
            # Extract the gradient for the bias vector as well
            bias_gradient = layer.bias.grad1.sum(dim=1).reshape(batch_size, views_per_sample, -1)
            bias_gradient = bias_gradient.sum(dim=1).unsqueeze(dim=-1)

            gradients = torch.cat([weight_gradient, bias_gradient], dim=-1)
        else:
            gradients = weight_gradient

        return gradients

    @abstractmethod
    def compute_loss(self, latents: torch.Tensor, views_per_sample: int, **kwargs) -> torch.Tensor:
        """
            Given a batch of latents computes a per-sample loss,
            where every sample may have one or more associated views
        """
        raise NotImplementedError
