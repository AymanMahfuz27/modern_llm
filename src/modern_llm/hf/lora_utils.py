"""Minimal PEFT/LoRA helpers (Hu et al., 2021)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class LoraConfig:
    """LoRA hyperparameters mirroring Hu et al. (2021, Eq. 5).

    The `task_type` field should match `peft.TaskType` enum names, e.g.:
    - \"CAUSAL_LM\" for decoder-only LMs.
    - \"SEQ_CLS\" for sequence classification.
    """

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
        if not self.task_type:
            raise ValueError("task_type must be a non-empty string.")


def prepare_lora_model(model, config: LoraConfig):
    """Inject low-rank adapters as described by Hu et al. (2021).

    Pre:
        - `config.task_type` matches a member of `peft.TaskType`, e.g. \"CAUSAL_LM\" or \"SEQ_CLS\".
    """

    try:
        from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("Install `peft` to use LoRA utilities: pip install peft") from exc

    try:
        task_type_enum = getattr(TaskType, config.task_type)
    except AttributeError as exc:
        raise ValueError(f"Invalid LoRA task_type '{config.task_type}' for peft.TaskType") from exc

    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias=config.bias,
        task_type=task_type_enum,
    )
    return get_peft_model(model, peft_config)

