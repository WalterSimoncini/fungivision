import math
import torch
import logging
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.functional import cosine_similarity

from .transforms import Patchify
from src.fungivision.gradients.base_extractor import BaseGradientExtractor


class SimCLRGradientsExtractor(BaseGradientExtractor):
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
        temperature: float = 0.07,
        self_similarity_constant: float = -10e3,
        comparison_batch_size: int = 64,
        comparison_batch_encoding_batch_size: int = 32,
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
        self.self_similarity_constant = self_similarity_constant
        self.comparison_batch_size = comparison_batch_size
        self.comparison_batch_encoding_batch_size = comparison_batch_encoding_batch_size

    def input_transform(self, input_dim: int) -> nn.Module:
        return Patchify(num_patches=4, stride_scale=6, input_dim=input_dim)

    def setup(self, dataset: Dataset) -> None:
        """
            Build the negative batch by sampling images from the given dataset.

            Args:
                dataset (torch.utils.Dataset): the dataset to sample the negative
                    batch from. It's recommended to use the training split.
        """
        logging.info("computing the simclr comparison batch")

        sample_indices = torch.randperm(len(dataset))
        sample_indices = sample_indices[:self.comparison_batch_size]

        encoded_samples = []
        samples = [self.transform(dataset[i][0]) for i in sample_indices]
        samples = torch.cat(samples, dim=0)

        num_batches = math.ceil(len(samples) / self.comparison_batch_encoding_batch_size)

        logging.info(f"encoding {len(samples)} samples...")

        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                start_index = i * self.comparison_batch_encoding_batch_size
                end_index = (i + 1) * self.comparison_batch_encoding_batch_size

                minibatch = samples[start_index:end_index].to(self.device)
                embeddings = self.model(minibatch)

                encoded_samples.append(self.projection_head(embeddings).detach())

        self.comparison_batch = torch.cat(encoded_samples, dim=0)

    def encode(self, views: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.model(views))

    def compute_loss(self, latents: torch.Tensor, views_per_sample: int, **kwargs) -> torch.Tensor:
        """
            Compute the InfoNCE loss for each batch item individually,
            and return the mean loss.
        """
        batch_size = latents.shape[0] // views_per_sample
        latents = latents.reshape(batch_size, views_per_sample, -1)

        losses = torch.zeros(batch_size).to(self.device)

        for i, sample_latents in enumerate(latents):
            losses[i] = self.info_nce_loss(latents=sample_latents)

        return losses.mean()

    def info_nce_loss(self, latents: torch.Tensor):
        """
            Calculates the InfoNCE loss for a pair of positive
            examples (test_views) against a batch of negative
            examples (batch).

            :param batch:
                The negative examples for the loss calculation
            :param test_views:
                The positive examples for the loss calculation
        """
        # Concatenate the batch samples and the test views in a single tensor
        features = torch.cat([self.comparison_batch, latents], dim=0)

        if self.use_fp16:
            features = features.to(self.fp16_dtype)

        n_positive_views = latents.shape[0]
        features = nn.functional.normalize(features, dim=-1, p=2)

        cosine_sim = self.masked_cosine_similarity(
            features=features,
            device=self.device,
            n_positive_views=n_positive_views
        )

        # Select only the bottom-right corner of the cosine similarity matrix, i.e.
        # the similarities between the positive views
        positive_cosine_sim = cosine_sim[-n_positive_views:, -n_positive_views:] / self.temperature

        # Zero out the diagonal to remove the effect of self-similarities and calculate the
        # mean self-similarities by averaging over columns. We divide by positives - 1 as
        # one element will always be zero
        nll = -(positive_cosine_sim - torch.diag(torch.diag(positive_cosine_sim))).sum(dim=1) / (n_positive_views - 1)
        nll += torch.logsumexp(cosine_sim[-n_positive_views:, :], dim=-1)

        return nll.mean()

    def masked_cosine_similarity(self, features: torch.Tensor, device: torch.device, n_positive_views: int) -> torch.Tensor:
        # We are only interested in the cosine similarity of the positive views
        # against all other views, so we only compute the bottom rectangle of
        # the cosine similarity matrix, selecting only rows that belong to
        # positive views
        cosine_sim = cosine_similarity(
            features[-n_positive_views:, None, :],
            features[None, :, :],
            dim=-1
        )

        # We will always have more columns than rows here
        # as the columns include the negatives, while the
        # rows only include the positive views
        n_rows, n_cols = cosine_sim.shape

        # Fill the "rightmost" diagonal with the self-similarity
        # constant, as this diagonal represents the matches of
        # each positive view with itself
        return torch.diagonal_scatter(
            cosine_sim,
            torch.ones(n_rows) * self.self_similarity_constant,
            n_cols - n_rows
        )
