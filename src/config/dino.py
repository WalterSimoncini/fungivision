from dataclasses import dataclass


@dataclass
class DINOConfig:
    teacher_temperature: float = 0.07
    student_temperature: float = 0.1
