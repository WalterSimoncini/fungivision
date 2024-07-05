from dataclasses import dataclass, asdict

from src.gradients.dino import DINOGradientsExtractor


@dataclass
class DINOConfig:
    teacher_temperature: float = 0.07
    student_temperature: float = 0.1

    def get_extractor(self, base_params: dict) -> DINOGradientsExtractor:
        """
            Create an instance of DINOGradientsExtractor configured using
            the given base parameters (common to all extractors) and the
            extractor-specific parameters defined in this class
        """
        params = base_params | asdict(self)

        return DINOGradientsExtractor(**params)
