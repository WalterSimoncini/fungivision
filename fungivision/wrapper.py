import torch
import logging
import torch.nn as nn

import fungivision.utils.autograd_hacks as autograd_hacks

from PIL import Image
from typing import List, Optional
from torch.utils.data import Dataset
from fungivision.utils.misc import model_feature_dim, rgetattr, freeze_model
from fungivision.utils.compression import suggested_scaling_factor, generate_projection_matrix


class FUNGIWrapper():
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: torch.device,
        projection: Optional[torch.Tensor] = None,
        input_dim: int = 224,
        use_fp16: bool = False,
        fp16_dtype: torch.dtype = torch.bfloat16,
        extractor_configs: List = []
    ) -> None:
        """
            Initialize a FUNGI features extractor.

            Args:
                model (nn.Module): the model to extract gradients from.
                target_layer (str): path to the layer to extract gradients from,
                    e.g. blocks.11.attn.proj.
                device (torch.device): the device used for the forward and
                    backward passes.
                projection (torch.Tensor): the projection for downsampling gradients.
                input_dim (int): the expected nodel input dimensions.
                use_fp16 (bool): whether to run the model in float16.
                fp16_dtype (torch.dtype): which float16 data type to use
                    if float16 is enabled.
                extractor_configs (List): the configurations for each gradients
                    extractor to use in the forward method.
        """
        assert model is not None, f"a valid model must be provided! {model} given"
        assert target_layer is not None, f"a valid target layer must be provided! {target_layer} given"
        assert len(extractor_configs) > 0, "you must provide at least one extractor config!"

        self.target_layer = target_layer
        self.model = model.to(device)

        logging.info(f"initializing FUNGI wrapper...")
        logging.info(f"estimating the model output dimensionality...")

        self.embeddings_dim = model_feature_dim(model, device, image_size=input_dim)
        self.model = model

        # Freeze the model, except the target layer
        # FIXME: We should also freeze batch norm layers if any
        freeze_model(self.model, exclusions=[self.target_layer])

        # Add hooks to compute per-sample gradients
        autograd_hacks.add_hooks(self.model, layer_paths=[self.target_layer])

        # Generate the random projection if needed
        if projection is None:
            logging.info(f"generating the projection matrix...")

            layer = rgetattr(self.model, self.target_layer)
            projection = generate_projection_matrix(
                dims=(self.embeddings_dim, self.gradients_dim(layer=layer)),
                device=device
            )

            scaling = suggested_scaling_factor(projection.shape[1])

        if use_fp16:
            projection = projection.to(fp16_dtype)

        # Configure the gradients extractors
        extractor_params = {
            "model": self.model,
            "target_layer": self.target_layer,
            "device": device,
            "projection": projection,
            "projection_scaling": scaling,
            "embeddings_dim": self.embeddings_dim,
            "input_dim": input_dim,
            "use_fp16": use_fp16,
            "fp16_dtype": fp16_dtype
        }

        self.extractors = [
            cfg.get_extractor(base_params=extractor_params) for cfg in extractor_configs
        ]

    def setup(self, dataset: Dataset) -> None:
        """
            Run the setup method for every FUNGI feature extractor.

            Args:
                dataset (torch.utils.data.Dataset): the dataset used to generate
                    the supporting data for each gradients extractor, generally
                    using the training split.
        """
        for extractor in self.extractors:
            logging.info(f"running setup for extractor {type(extractor).__name__}")

            extractor.setup(dataset=dataset)

    def gradients_dim(self, layer: nn.Module) -> int:
        """
            Returns the dimensionality of the gradient of a linear layer

            Args:
                layer (nn.Linear): the layer to extract gradients from

            Returns:
                int: the total dimensionality of gradients for the given layer
        """
        H, W = layer.weight.shape[:2]

        if hasattr(layer, "bias") and layer.bias is not None:
            W += 1

        return H * W

    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        return self.forward(images=images)

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """
            Extract gradient features for a list of images.

            Args:
                images (List[PIL.Image]): list of imafes to extract gradients from.

            Returns:
                torch.Tensor: tensor of shape [B, E * N] where B is the number
                    of images, E the model embeddings dim and N the number of
                    gradients configured for the FUNGI wrapper. Each gradient
                    is L2-normalized independently of the others.
        """
        gradients = []

        for extractor in self.extractors:
            extractor_gradients = extractor(images=images)
            gradients.append(
                nn.functional.normalize(extractor_gradients, dim=-1, p=2)
            )

        return torch.cat(gradients, dim=-1)
