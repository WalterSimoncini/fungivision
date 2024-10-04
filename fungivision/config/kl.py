from dataclasses import dataclass, asdict

from fungivision.gradients.kl_extractor import KLGradientsExtractor


@dataclass
class KLConfig:
    """
        Configuration for the KL objective.

        Args:
            temperature (float): softmax temperature for the latents and uniform
            latent_dim (int): output size of the linear projection head.
    """
    temperature: float = 1
    latent_dim: int = 768

    def get_extractor(self, base_params: dict) -> KLGradientsExtractor:
        """
            Create an instance of KLGradientsExtractor configured using
            the given base parameters (common to all extractors) and the
            extractor-specific parameters defined in this class.

            Args:
                base_params (dict): base parameters (i.e. parameters common to all
                    extractors) used to initialize the extractor.
        """
        params = base_params | asdict(self)

        return KLGradientsExtractor(**params)
