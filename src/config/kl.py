from dataclasses import dataclass, asdict

from src.gradients.kl_extractor import KLGradientsExtractor


@dataclass
class KLConfig:
    temperature: float = 1

    def get_extractor(self, base_params: dict) -> KLGradientsExtractor:
        """
            Create an instance of KLGradientsExtractor configured using
            the given base parameters (common to all extractors) and the
            extractor-specific parameters defined in this class
        """
        params = base_params | asdict(self)

        return KLGradientsExtractor(**params)
