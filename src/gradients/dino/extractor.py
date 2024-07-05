import torch
import torch.nn as nn
import src.utils.autograd_hacks as autograd_hacks

from typing import List
from PIL.Image import Image
from src.gradients.base_extractor import BaseGradientExtractor

from .transforms import DINOTransform


class DINOGradientsExtractor(BaseGradientExtractor):
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
        teacher_temperature: float = 0.07,
        student_temperature: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(
            model,
            target_layer,
            device,
            projection,
            projection_scaling,
            input_dim,
            use_fp16,
            fp16_dtype,
            **kwargs
        )

        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature

        self.teacher_projection = nn.Linear(
            self.model.embeddings_dim,
            self.model.latent_dim
        ).to(self.device)

        # Disable the gradients calculation for the teacher
        # projection, as we do not backpropagate through it
        self.teacher_projection.weight.requires_grad = False
        self.teacher_projection.bias.requires_grad = False

    def input_transform(self, input_dim: int) -> nn.Module:
        return DINOTransform(crops_size=input_dim)

    def forward(self, images: List[Image]) -> torch.Tensor:
        # Clear tensors used for the per-sample gradient computation
        autograd_hacks.clear_backprops(self.model)

        # Transform the input images and generate the views for the SSL loss
        views = [self.transform(x).unsqueeze(dim=0) for x in images]
        views = torch.cat(views, dim=0)

        # Batch, crops, channels, height and width
        B, CR, C, H, W = views.shape

        views = views.reshape(-1, C, H, W).to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_fp16):
            # Encode global and local views using the backbone
            latents = self.model.backbone(views)
            latents = latents.reshape(B, CR, -1)
            latents = nn.functional.normalize(latents, dim=-1, p=2)

            # Project the global views with the teacher projection,
            # without registering gradients
            with torch.no_grad():
                teacher_latents = latents[:, :2].reshape(B * 2, -1).detach()
                teacher_latents = self.teacher_projection(teacher_latents)

            # Project the student gradients
            student_latents = latents.reshape(B * CR, -1)
            student_latents = self.model.projection(student_latents)

            loss = self.compute_loss(
                latents=student_latents,
                views_per_sample=CR,
                teacher_latents=teacher_latents
            )

        return self.backprop(loss=loss, batch_size=B, views_per_sample=CR)

    def compute_loss(self, latents: torch.Tensor, views_per_sample: int, teacher_latents: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the DINO loss"""
        batch_size = latents.shape[0] // views_per_sample

        student_emb = latents.reshape(batch_size, views_per_sample, -1)
        teacher_emb = teacher_latents.reshape(batch_size, 2, -1)

        # Calculate the DINO loss
        teacher_emb = nn.functional.softmax(teacher_emb / self.teacher_temperature, dim=-1)
        student_emb = nn.functional.log_softmax(student_emb / self.student_temperature, dim=-1)

        # Convert the teacher and student embeddings in [CR, B, E]
        teacher_emb = teacher_emb.permute(1, 0, 2)
        student_emb = student_emb.permute(1, 0, 2)

        losses = torch.zeros(batch_size).to(self.device)

        for _, tv in enumerate(teacher_emb):
            for _, sv in enumerate(student_emb):
                # The original loss does not operate on the same views
                # between student and teacher, but we found that doing
                # so produces more predictive gradients
                losses += (-tv * sv).sum(dim=-1)

        return losses.mean()
