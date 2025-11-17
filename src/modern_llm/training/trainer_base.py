"""Minimal training loop scaffold grounded in cross-entropy LM training (e.g., GPT family)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from modern_llm.config.train_config import TrainingConfig
from modern_llm.utils.checkpointing import save_checkpoint
from modern_llm.utils.logging_utils import create_logger


@dataclass
class Trainer:
    """Causal LM trainer with gradient accumulation and AMP support."""

    model: nn.Module
    optimizer: Optimizer
    train_dataloader: Iterable
    config: TrainingConfig
    eval_dataloader: Optional[Iterable] = None
    lr_scheduler: Optional[_LRScheduler] = None

    device: torch.device = field(init=False)
    logger: logging.Logger = field(init=False)
    use_amp: bool = field(init=False)
    scaler: Optional[GradScaler] = field(init=False, default=None)
    global_step: int = field(init=False, default=0)
    micro_step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = create_logger(f"trainer.{self.config.run_name}")
        self.model.to(self.device)
        if self.config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]
        self.use_amp = self.config.mixed_precision in {"fp16", "bf16"} and self.device.type == "cuda"
        if self.config.mixed_precision == "fp16" and self.device.type == "cuda":
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def train(self) -> None:
        """Run the optimization loop.

        Pre:
            - model, optimizer, dataloaders initialized.
        Post:
            - checkpoints and logs emitted per configuration.
        """

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation_steps = self.config.gradient_accumulation_steps
        max_steps = self.config.max_steps

        while self.global_step < max_steps:
            for batch in self.train_dataloader:
                loss = self._training_step(batch, accumulation_steps)
                if self.global_step >= max_steps:
                    break
                if self.config.log_every > 0 and self.global_step % self.config.log_every == 0:
                    self.logger.info(
                        "step=%d loss=%.4f lr=%.3e",
                        self.global_step,
                        loss,
                        self.optimizer.param_groups[0]["lr"],
                    )
                if (
                    self.config.eval_every > 0
                    and self.eval_dataloader
                    and self.global_step % self.config.eval_every == 0
                ):
                    metrics = self.evaluate()
                    self.logger.info(
                        "eval step=%d loss=%.4f ppl=%.2f",
                        self.global_step,
                        metrics["loss"],
                        metrics["perplexity"],
                    )
                if self.config.save_every > 0 and self.global_step % self.config.save_every == 0:
                    self._save_checkpoint()
                if self.global_step >= max_steps:
                    break
            if self.global_step >= max_steps:
                break

        self._save_checkpoint(suffix="final")

    def _training_step(self, batch: Dict[str, Tensor], accumulation_steps: int) -> float:
        batch = self._move_batch_to_device(batch)
        micro_loss = self._forward_loss(batch) / accumulation_steps

        if self.use_amp and self.scaler is not None:
            self.scaler.scale(micro_loss).backward()
        else:
            micro_loss.backward()

        self.micro_step += 1
        step_completed = self.micro_step % accumulation_steps == 0
        if step_completed:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.config.max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        return float(micro_loss.detach().cpu())

    def _forward_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        autocast_dtype = None
        if self.use_amp:
            autocast_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        with autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            raise ValueError("Model must return a loss when labels are provided.")
        return loss

    def evaluate(self) -> Dict[str, float]:
        if not self.eval_dataloader:
            return {"loss": float("nan"), "perplexity": float("nan")}
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._move_batch_to_device(batch)
                loss = self._forward_loss(batch)
                total_loss += loss.item()
                total_batches += 1
        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(avg_loss)
        self.model.train()
        return {"loss": avg_loss, "perplexity": perplexity}

    def _move_batch_to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        tag = suffix or f"step{self.global_step}"
        path = self.config.output_dir / f"{self.config.run_name}_{tag}.pt"
        save_checkpoint(
            path,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            step=self.global_step,
            run_name=self.config.run_name,
        )

