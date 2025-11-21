"""Math reasoning finetuning helpers (GSM8K, Cobbe et al., 2021).

GSM8K contains grade-school math word problems with chain-of-thought answers.
We finetune decoder-only models (GPT-2, TinyLlama) on a subset and evaluate EM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from modern_llm.config import TrainingConfig
from modern_llm.data import TaskDatasetConfig, load_supervised_text_dataset
from modern_llm.hf import LoraConfig, prepare_lora_model
from modern_llm.training import Trainer


def _build_dataloaders(
    tokenizer,
    max_length: int,
    micro_batch_size: int,
    num_proc: int,
):
    """Build GSM8K train/validation loaders.

    GSM8K samples have 'question' and 'answer' fields. We concatenate them
    as a single autoregressive sequence: "Q: {question}\nA: {answer}".

    Pre:
        - tokenizer supports causal LM.
    Post:
        - returns (train_loader, eval_loader) with tokenized QA pairs.
    """
    train_config = TaskDatasetConfig(
        dataset_name="gsm8k",
        dataset_config_name="main",
        split="train",
        text_fields=("question",),
        label_field="answer",
        task_type="seq2seq",
        max_source_length=max_length,
        max_target_length=max_length,
        padding="max_length",
        num_proc=num_proc,
    )
    eval_config = TaskDatasetConfig(
        dataset_name="gsm8k",
        dataset_config_name="main",
        split="test",
        text_fields=("question",),
        label_field="answer",
        task_type="seq2seq",
        max_source_length=max_length,
        max_target_length=max_length,
        padding="max_length",
        num_proc=num_proc,
    )
    train_dataset = load_supervised_text_dataset(train_config, tokenizer, target_tokenizer=tokenizer)
    eval_dataset = load_supervised_text_dataset(eval_config, tokenizer, target_tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
    )
    return train_loader, eval_loader


def main() -> None:
    """Finetune GPT-2 or TinyLlama on GSM8K with optional LoRA/QLoRA."""

    parser = argparse.ArgumentParser(description="LoRA/QLoRA finetuning for GSM8K math reasoning.")
    parser.add_argument("--run_name", type=str, default="gpt2-gsm8k-lora")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--eval_every", type=int, default=300)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="experiments/runs")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adapters.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="c_attn,c_proj",
        help="Comma-separated list of module names for LoRA.",
    )
    args = parser.parse_args()

    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size.")

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    train_loader, eval_loader = _build_dataloaders(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        micro_batch_size=args.micro_batch_size,
        num_proc=args.num_proc,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora:
        target_modules = tuple(m.strip() for m in args.lora_target_modules.split(",") if m.strip())
        lora_config = LoraConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = prepare_lora_model(model, lora_config)

    training_config = TrainingConfig(
        run_name=args.run_name,
        dataset_name="gsm8k",
        tokenizer_name=args.model_name,
        output_dir=output_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        mixed_precision=args.mixed_precision,
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


if __name__ == "__main__":
    main()

