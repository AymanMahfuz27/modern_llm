"""Train scratch LM from a JSON config file.

Loads a configuration from configs/ and runs the full LM training loop,
enabling one-button max-size model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig, TrainingConfig
from modern_llm.data import LanguageModelingDatasetConfig, load_causal_lm_dataset
from modern_llm.models import ModernDecoderLM
from modern_llm.training.trainer_base import Trainer
from modern_llm.training.train_lm import generate_text


def main() -> None:
    """Train ModernDecoderLM from a JSON config and optionally generate samples."""

    parser = argparse.ArgumentParser(description="Train scratch LM from JSON config.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--output_dir", type=str, default="experiments/runs")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--gen_prompt", type=str, default="The meaning of life is")
    parser.add_argument("--gen_max_new_tokens", type=int, default=80)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_k", type=int, default=50)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        cfg = json.load(f)

    run_name = cfg["run_name"]
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg["max_seq_len"]

    train_dataset = load_causal_lm_dataset(
        LanguageModelingDatasetConfig(
            dataset_name=cfg["dataset_name"],
            dataset_config_name=cfg["dataset_config_name"],
            split="train",
            max_length=cfg["max_seq_len"],
            num_proc=args.num_proc,
        ),
        tokenizer,
    )
    eval_dataset = load_causal_lm_dataset(
        LanguageModelingDatasetConfig(
            dataset_name=cfg["dataset_name"],
            dataset_config_name=cfg["dataset_config_name"],
            split="validation",
            max_length=cfg["max_seq_len"],
            num_proc=args.num_proc,
        ),
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["micro_batch_size"],
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg["micro_batch_size"],
        shuffle=False,
    )

    model_config = ModernLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ffn_hidden_size=cfg["ffn_hidden_size"],
        max_seq_len=cfg["max_seq_len"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        dropout=cfg.get("dropout", 0.0),
        use_rope=cfg.get("use_rope", True),
        use_attention_sinks=cfg.get("use_attention_sinks", True),
        num_attention_sinks=cfg.get("num_attention_sinks", 2),
        use_swiglu=cfg.get("use_swiglu", True),
        tie_embeddings=cfg.get("tie_embeddings", True),
    )
    model = ModernDecoderLM(model_config)

    training_config = TrainingConfig(
        run_name=run_name,
        dataset_name=cfg["dataset_name"],
        tokenizer_name=cfg["tokenizer_name"],
        output_dir=output_dir,
        batch_size=cfg["batch_size"],
        micro_batch_size=cfg["micro_batch_size"],
        gradient_accumulation_steps=cfg["batch_size"] // cfg["micro_batch_size"],
        learning_rate=cfg["learning_rate"],
        max_steps=cfg["max_steps"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        eval_every=cfg.get("eval_every", 200),
        save_every=cfg.get("save_every", 500),
        log_every=cfg.get("log_every", 50),
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        compile_model=cfg.get("compile_model", False),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    scheduler = None
    if training_config.warmup_steps > 0:
        def lr_lambda(step: int) -> float:
            if step < training_config.warmup_steps:
                return float(step + 1) / float(training_config.warmup_steps)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=training_config,
        lr_scheduler=scheduler,
    )
    trainer.train()

    if args.gen_max_new_tokens > 0:
        print(f"\nGenerating sample from final checkpoint with prompt: '{args.gen_prompt}'")
        output = generate_text(
            model,
            tokenizer,
            args.gen_prompt,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            top_k=args.gen_top_k,
        )
        print(f"\nGenerated text:\n{output}\n")


if __name__ == "__main__":
    main()



