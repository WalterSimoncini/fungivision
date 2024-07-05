from dataclasses import dataclass, asdict

from src.gradients.simclr import SimCLRGradientsExtractor


@dataclass
class SimCLRConfig:
    temperature: float = 0.07
    self_similarity_constant: float = -10e3
    comparison_batch_size: int = 64
    comparison_batch_encoding_batch_size: int = 32

    def get_extractor(self, base_params: dict) -> SimCLRGradientsExtractor:
        """
            Create an instance of SimCLRGradientsExtractor configured using
            the given base parameters (common to all extractors) and the
            extractor-specific parameters defined in this class
        """
        params = base_params | asdict(self)

        return SimCLRGradientsExtractor(**params)
