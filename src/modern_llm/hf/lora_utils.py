"""Minimal PEFT/LoRA helpers (Hu et al., 2021)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class LoraConfig:
    """LoRA hyperparameters mirroring Hu et al. (2021, Eq. 5)."""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Sequence[str] = field(default_factory=lambda: ("q_proj", "v_proj"))
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("LoRA rank r must be positive.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not self.target_modules:
            raise ValueError("At least one target module name must be provided.")


def prepare_lora_model(model, config: LoraConfig):
    """Inject low-rank adapters as described by Hu et al. (2021)."""

    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Install `peft` to use LoRA utilities: pip install peft") from exc

    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias=config.bias,
        task_type=config.task_type,
    )
    return get_peft_model(model, peft_config)

