"""Configuration dataclasses used across the Modern LLM project."""

from .model_config import ModernLLMConfig, MoEConfig
from .train_config import TrainingConfig

__all__ = ["ModernLLMConfig", "MoEConfig", "TrainingConfig"]

