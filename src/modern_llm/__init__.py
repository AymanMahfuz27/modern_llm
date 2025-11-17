"""Public interface for the Modern LLM package."""

from modern_llm.config.model_config import ModernLLMConfig, MoEConfig
from modern_llm.config.train_config import TrainingConfig

try:
    from modern_llm.models.transformer import ModernDecoderLM
except ModuleNotFoundError:  # pragma: no cover - torch might be missing during initial setup.
    ModernDecoderLM = None  # type: ignore[assignment]

__all__ = [
    "ModernLLMConfig",
    "MoEConfig",
    "TrainingConfig",
    "ModernDecoderLM",
]

