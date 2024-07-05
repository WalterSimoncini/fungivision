import torch.nn as nn


class FUNGIHead(nn.Module):
    def __init__(self, backbone: nn.Module, embeddings_dim: int, latent_dim: int):
        super().__init__()

        self.backbone = backbone
        self.projection = nn.Linear(embeddings_dim, latent_dim)

        self.embeddings_dim = embeddings_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=-1, p=2)

        return self.projection(x)
