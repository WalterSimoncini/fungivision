from dataclasses import dataclass, asdict

from fungivision.gradients.simclr import SimCLRGradientsExtractor


@dataclass
class SimCLRConfig:
    """
        Configuration for the SimCLR objective.

        Args:
            temperature (float): the temperature used to scale the cosine similarities.
            self_similarity_constant (float): highly negative constant used to replace
                the similarity between a view and itself in the loss computation. The
                default value is -10e3, which is negative enough and fits in the fp16
                range.
            comparison_batch_size (int): the size (in original images) of the negative
                batch. If this value is N, the final batch will have NxV samples, where
                V is the number of views generated for each individual sample.
            comparison_batch_encoding_batch_size (int): batch size used when encoding
                the negatives batch.
            latent_dim (int): output size of the linear projection head.
            num_patches (int): number of patches to extract if the stride
                scale was 1. Must be a perfect square.
            stride_scale (int): the factor (1/x) used to calculate the stride
                step relative to the patch size.
    """
    temperature: float = 0.07
    self_similarity_constant: float = -10e3
    comparison_batch_size: int = 64
    comparison_batch_encoding_batch_size: int = 32
    latent_dim: int = 96
    num_patches: int = 4
    stride_scale: int = 6

    def get_extractor(self, base_params: dict) -> SimCLRGradientsExtractor:
        """
            Create an instance of SimCLRGradientsExtractor configured using
            the given base parameters (common to all extractors) and the
            extractor-specific parameters defined in this class.

            Args:
                base_params (dict): base parameters (i.e. parameters common to all
                    extractors) used to initialize the extractor.
        """
        params = base_params | asdict(self)

        return SimCLRGradientsExtractor(**params)
