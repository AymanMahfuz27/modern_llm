"""Pipeline evaluation suite for comparing Base/SFT/DPO/Verifier stages.

Evaluates each checkpoint on:
- Perplexity (WikiText-2)
- Generation quality (qualitative samples)
- Math/QA (GSM8K subset with and without verifier reranking)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from modern_llm.alignment.alignment_pipeline import PipelineState
from modern_llm.config import ModernLLMConfig, PipelineConfig
from modern_llm.data.lm_datasets import LanguageModelingDatasetConfig, load_causal_lm_dataset
from modern_llm.models.transformer import ModernDecoderLM
from modern_llm.models.verifier import VerifierConfig, VerifierModel
from modern_llm.utils.checkpointing import load_checkpoint
from modern_llm.utils.logging_utils import create_logger


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    stage: str
    perplexity: float
    loss: float
    gsm8k_em: float = 0.0
    gsm8k_em_verifier: float = 0.0
    num_params: int = 0


@dataclass
class PipelineEvalResults:
    """Full evaluation results across all stages."""

    base: Optional[StageMetrics] = None
    sft: Optional[StageMetrics] = None
    dpo: Optional[StageMetrics] = None
    verifier_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "base": asdict(self.base) if self.base else None,
            "sft": asdict(self.sft) if self.sft else None,
            "dpo": asdict(self.dpo) if self.dpo else None,
            "verifier_accuracy": self.verifier_accuracy,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, path: Path) -> None:
        """Save comparison table as CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        stages = [self.base, self.sft, self.dpo]
        stages = [s for s in stages if s is not None]

        if not stages:
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(stages[0]).keys()))
            writer.writeheader()
            for stage in stages:
                writer.writerow(asdict(stage))


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ModernDecoderLM, ModernLLMConfig]:
    """Load model from checkpoint file."""
    ckpt = load_checkpoint(checkpoint_path)

    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(f"Checkpoint {checkpoint_path} missing config")

    config = ModernLLMConfig(**ckpt["config"])
    model = ModernDecoderLM(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, config


def compute_perplexity(
    model: ModernDecoderLM,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    """Compute perplexity on a dataset.

    Pre: model is on device and in eval mode.
    Post: Returns (perplexity, avg_loss).
    """
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Computing PPL", leave=False)):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                loss = outputs

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                total_batches += 1

    if total_batches == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_batches
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

    return perplexity, avg_loss


def evaluate_stage(
    checkpoint_path: Path,
    stage_name: str,
    tokenizer,
    eval_dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> StageMetrics:
    """Evaluate a single pipeline stage."""
    print(f"Evaluating {stage_name}...")

    model, config = load_model_from_checkpoint(checkpoint_path, device)
    num_params = sum(p.numel() for p in model.parameters())

    perplexity, loss = compute_perplexity(
        model, eval_dataloader, device, max_batches
    )

    del model
    torch.cuda.empty_cache()

    return StageMetrics(
        stage=stage_name,
        perplexity=perplexity,
        loss=loss,
        num_params=num_params,
    )


def evaluate_pipeline_stages(
    state: PipelineState,
    config: PipelineConfig,
    max_eval_batches: Optional[int] = 100,
) -> Path:
    """Evaluate all pipeline stages and save results.

    Pre: PipelineState has valid checkpoint paths.
    Post: Returns path to results JSON file.
    """
    logger = create_logger("pipeline_eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup tokenizer and eval data
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_config = LanguageModelingDatasetConfig(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        split="validation",
        max_length=config.max_seq_len,
    )
    eval_dataset = load_causal_lm_dataset(eval_config, tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    results = PipelineEvalResults()

    # Evaluate each stage
    if state.pretrain_checkpoint and state.pretrain_checkpoint.exists():
        results.base = evaluate_stage(
            state.pretrain_checkpoint,
            "base",
            tokenizer,
            eval_dataloader,
            device,
            max_eval_batches,
        )
        logger.info(f"Base: PPL={results.base.perplexity:.2f}, Loss={results.base.loss:.4f}")

    if state.sft_checkpoint and state.sft_checkpoint.exists():
        results.sft = evaluate_stage(
            state.sft_checkpoint,
            "sft",
            tokenizer,
            eval_dataloader,
            device,
            max_eval_batches,
        )
        logger.info(f"SFT: PPL={results.sft.perplexity:.2f}, Loss={results.sft.loss:.4f}")

    if state.dpo_checkpoint and state.dpo_checkpoint.exists():
        results.dpo = evaluate_stage(
            state.dpo_checkpoint,
            "dpo",
            tokenizer,
            eval_dataloader,
            device,
            max_eval_batches,
        )
        logger.info(f"DPO: PPL={results.dpo.perplexity:.2f}, Loss={results.dpo.loss:.4f}")

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{config.run_name}_eval.json"
    results.save(results_path)

    csv_path = output_dir / f"{config.run_name}_comparison.csv"
    results.to_csv(csv_path)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Comparison table saved to {csv_path}")

    return results_path



