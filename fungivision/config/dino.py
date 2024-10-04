from dataclasses import dataclass, asdict

from fungivision.gradients.dino import DINOGradientsExtractor


@dataclass
class DINOConfig:
    """
        Configuration for the DINO objective.

        Args:
            teacher_temperature (float): softmax temperature for the teacher latents.
            student_temperature (float): softmax temperature for the student latents.
            latent_dim (int): output size of the linear projection head, for both the
                teacher and student models.
    """
    teacher_temperature: float = 0.07
    student_temperature: float = 0.1
    latent_dim: int = 2048

    def get_extractor(self, base_params: dict) -> DINOGradientsExtractor:
        """
            Create an instance of DINOGradientsExtractor configured using
            the given base parameters (common to all extractors) and the
            extractor-specific parameters defined in this class.

            Args:
                base_params (dict): base parameters (i.e. parameters common to all
                    extractors) used to initialize the extractor.
        """
        params = base_params | asdict(self)

        return DINOGradientsExtractor(**params)
