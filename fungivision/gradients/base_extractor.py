import torch
import torch.nn as nn
import fungivision.utils.autograd_hacks as autograd_hacks

from PIL import Image
from typing import List
from abc import abstractmethod, ABC
from torch.utils.data import Dataset
from fungivision.utils.misc import rgetattr


class BaseGradientExtractor(ABC):
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
        **kwargs
    ) -> None:
        """
            Initialize a gradient extractor.

            Args:
                model (nn.Module): the model to extract gradients from.
                target_layer (str): path to the layer to extract gradients from,
                    e.g. blocks.11.attn.proj.
                device (torch.device): the device used for the forward and
                    backward passes.
                projection (torch.Tensor): the projection for downsampling gradients.
                projection_scaling (float): scaling factor for the projected gradients.
                embeddings_dim (int): dimensionality of the model embeddings.
                latent_dim (int): output dimensionality of the projection head.
                input_dim (int): the expected nodel input dimensions.
                use_fp16 (bool): whether to run the model in float16.
                fp16_dtype (torch.dtype): which float16 data type to use
                    if float16 is enabled.
        """
        super().__init__()

        assert model is not None, f"a valid model must be provided! {model} given"
        assert target_layer is not None, f"a valid target layer must be provided! {target_layer} given"

        self.model = model
        self.device = device
        self.projection = projection.to(device)
        self.projection_scaling = projection_scaling
        self.target_layer = target_layer

        # FP16 configuration. Using fp16 assumes we are running on a CUDA device
        self.use_fp16 = use_fp16
        self.fp16_dtype = fp16_dtype

        self.projection_head = nn.Linear(embeddings_dim, latent_dim).to(self.device)
        self.transform = self.input_transform(input_dim=input_dim, **kwargs)

        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    @abstractmethod
    def input_transform(self, input_dim: int) -> nn.Module:
        """
            Returns the transform applied to input images.

            Args:
                input_dim (int): the input dimensionality of the model.

            Returns:
                nn.Module: a transform that returns a [C, H, W] or [V, C, H, W]
                    tensor, where V is the views dimension.
        """
        raise NotImplementedError

    def setup(self, dataset: Dataset) -> None:
        """
            Generate any supporting data required for the loss computation,
            e.g. the negatives batch for contrastive learning.

            Args:
                dataset (torch.utils.data.Dataset): the dataset used to generate
                    the supporting data, generally the training split of a benchmark
                    dataset
        """
        pass

    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract gradient features for a list of images. See forward()."""
        return self.forward(images=images)

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """
            Extract gradient features for a list of images.

            Args:
                images (PIL.Image): a list of images to be encoded.

            Returns:
                torch.Tensor: a tensor of size [B, E], where B is the number of images,
                    containing the per-image gradients.
        """
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
            latents = self.encode(views=views)
            loss = self.compute_loss(latents=latents, views_per_sample=V)

        return self.backprop(
            loss=loss,
            batch_size=batch_size,
            views_per_sample=V
        )

    def encode(self, views: torch.Tensor) -> torch.Tensor:
        """
            Encodes a set of views into latents.

            Args:
                views (torch.Tensor): a tensor of shape [B, C, H, W] of images
                    to be encoded.

            Returns:
                torch.Tensor: the encoded images as a [B, L] tensor.
        """
        embeddings = self.model(views)
        embeddings = nn.functional.normalize(embeddings, dim=-1, p=2)

        return self.projection_head(embeddings)

    def backprop(self, loss: torch.Tensor, batch_size: int, views_per_sample: int) -> torch.Tensor:
        """
            Runs backpropagation, extracts and projects gradients.

            Args:
                loss (torch.Tensor): loss tensor.
                batch_size (int): the batch size used in the forward pass.
                views_per_sample (int): the number of views generated for
                    each original image.

            Returns:
                torch.Tensor: the projected gradients as a [B, E] tensor, where
                    B is the batch size and E the model embeddings dim.
        """
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
        """
            Extract per-sample gradients for a given linear layer.

            Args:
                layer (nn.Linear): the linear layer to extract gradients from. The
                    layer is assumed to have been wrapper by autograd_hacks.
                batch_size (int): the batch size used in the forward pass.
                views_per_sample (int): the number of views generated for
                    each original image.

            Returns:
                torch.Tensor: the gradients as a [B, H * (W + 1)] tensor, where B
                    is the batch size and H, and W the dimensions of the gradient
                    matrix. If the layer had no bias vector the gradients shape
                    will be [B, H * W]. The per-view gradients are summed over the
                    view dimension, i.e. [B, V, H, W] -> [B, H, W] -> [B, H * W]
        """
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
            Given a batch of latents computes a per-sample loss, where
            every sample may have one or more associated views.

            Args:
                latents (torch.Tensor): the encoded latents for all batch
                    items and their views.
                views_per_sample (int): the number of views generated for
                    each original image.

            Returns:
                torch.Tensor: the loss tensor, as a single number averaged
                    over the batch dimension.
        """
        raise NotImplementedError
