from dataclasses import dataclass


@dataclass
class DINOConfig:
    temperature: float = 0.07
    self_similarity_constant: float = -10e3
    comparison_batch_size: int = 64
    comparison_batch_encoding_batch_size: int = 32
