import torch
import logging
import torch.nn as nn

import src.utils.autograd_hacks as autograd_hacks

from PIL import Image
from typing import List
from torch.utils.data import Dataset
from src.utils.head import FUNGIHead
from src.gradients.dino import DINOGradientsExtractor
from src.gradients.simclr import SimCLRGradientsExtractor
from src.gradients.kl_extractor import KLGradientsExtractor
from src.utils.misc import model_feature_dim, rgetattr, freeze_model
from src.utils.compression import suggested_scaling_factor, generate_projection_matrix


class FUNGIWrapper():
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: torch.device,
        latent_dim: int = 768,
        projection: torch.Tensor = None,
        input_dim: int = 224,
        use_fp16: bool = False,
        fp16_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        assert model is not None, f"a valid model must be provided! {model} given"
        assert target_layer is not None, f"a valid target layer must be provided! {target_layer} given"

        self.target_layer = f"backbone.{target_layer}"
        self.model = model.to(device)

        logging.info(f"initializing FUNGI wrapper...")
        logging.info(f"estimating the model output dimensionality...")
        self.embeddings_dim = model_feature_dim(model, device, image_size=input_dim)

        self.model = FUNGIHead(
            backbone=model,
            embeddings_dim=self.embeddings_dim,
            latent_dim=latent_dim
        ).to(device)

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
            "input_dim": input_dim,
            "use_fp16": use_fp16,
            "fp16_dtype": fp16_dtype,
        }

        self.kl_gradients_extractor = KLGradientsExtractor(**extractor_params)
        self.dino_gradients_extractor = DINOGradientsExtractor(**extractor_params)
        self.simclr_gradients_extractor = SimCLRGradientsExtractor(**extractor_params)

        self.extractors = [
            self.kl_gradients_extractor,
            self.dino_gradients_extractor,
            self.simclr_gradients_extractor
        ]

    def setup(self, dataset: Dataset) -> None:
        """Run the setup method for every FUNGI feature extractor"""
        for extractor in self.extractors:
            logging.info(f"running setup for extractor {extractor}")

            extractor.setup(dataset=dataset)

    def gradients_dim(self, layer: nn.Module) -> int:
        """Returns the dimensionality of the gradient of a linear layer"""
        H, W = layer.weight.shape[:2]

        if hasattr(layer, "bias") and layer.bias is not None:
            W += 1

        return H * W

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        gradients = []

        # FIXME: Also extract embeddings?    

        for extractor in self.extractors:
            extractor_gradients = extractor.forward(images=images)
            gradients.append(
                nn.functional.normalize(extractor_gradients)
            )

        return torch.cat(gradients, dim=-1)
