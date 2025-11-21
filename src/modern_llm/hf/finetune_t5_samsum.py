"""Summarization finetuning for T5-small / FLAN-T5 on SAMSum (Gliwa et al., 2019).

SAMSum is a corpus of messenger-style conversations annotated with concise summaries.
We finetune encoder-decoder models (T5, FLAN-T5) with LoRA and evaluate via ROUGE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from modern_llm.config import TrainingConfig
from modern_llm.data import TaskDatasetConfig, load_supervised_text_dataset
from modern_llm.hf import LoraConfig, prepare_lora_model
from modern_llm.training import Trainer


def _build_dataloaders(
    tokenizer,
    max_source_length: int,
    max_target_length: int,
    micro_batch_size: int,
    num_proc: int,
    use_dummy_data: bool,
):
    """Build SAMSum train/validation loaders.

    When `use_dummy_data` is True this constructs a tiny synthetic dataset
    in-memory, which is useful for smoke tests or offline environments
    where the SAMSum dataset is not reachable from the Hugging Face Hub.

    Pre:
        - tokenizer is an encoder-decoder tokenizer (e.g., T5).
    Post:
        - returns (train_loader, eval_loader) with tokenized dialogues and summaries.
    """

    if use_dummy_data:
        # Small synthetic dialogue/summary pairs for quick smoke tests.
        base_dialogues = [
            "A: Hey, are we still on for tonight?\nB: Yes, see you at 8.",
            "A: Did you finish the report?\nB: Almost, I'll send it in an hour.",
            "A: I'm running late.\nB: No worries, take your time.",
        ]
        base_summaries = [
            "They confirm evening plans.",
            "They discuss finishing and sending a report.",
            "One person is late and the other is understanding.",
        ]

        if len(base_dialogues) != len(base_summaries):
            raise ValueError("Dummy dialogues and summaries must have the same length")

        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, received {micro_batch_size}")

        # Repeat base examples so that dataset length >= micro_batch_size, ensuring
        # at least one full batch when drop_last=True in the DataLoader.
        base_len = len(base_dialogues)
        repeats = (micro_batch_size + base_len - 1) // base_len
        dialogues = base_dialogues * repeats
        summaries = base_summaries * repeats

        model_inputs = tokenizer(
            dialogues,
            truncation=True,
            max_length=max_source_length,
            padding="max_length",
        )
        labels = tokenizer(
            summaries,
            truncation=True,
            max_length=max_target_length,
            padding="max_length",
        )

        class _Seq2SeqDataset(Dataset):
            """Minimal seq2seq dataset for synthetic SAMSum-style examples."""

            def __init__(self) -> None:
                self.input_ids = torch.tensor(model_inputs["input_ids"], dtype=torch.long)
                self.attention_mask = torch.tensor(
                    model_inputs["attention_mask"], dtype=torch.long
                )
                self.labels = torch.tensor(labels["input_ids"], dtype=torch.long)

            def __len__(self) -> int:
                return self.input_ids.size(0)

            def __getitem__(self, idx: int):
                if idx < 0 or idx >= self.input_ids.size(0):
                    raise IndexError(f"Index {idx} is out of bounds for dataset")
                return {
                    "input_ids": self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels": self.labels[idx],
                }

        train_dataset = _Seq2SeqDataset()
        eval_dataset = _Seq2SeqDataset()

    else:
        train_config = TaskDatasetConfig(
            dataset_name="samsum",
            dataset_config_name=None,
            split="train",
            text_fields=("dialogue",),
            label_field="summary",
            task_type="seq2seq",
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding="max_length",
            num_proc=num_proc,
        )
        eval_config = TaskDatasetConfig(
            dataset_name="samsum",
            dataset_config_name=None,
            split="test",
            text_fields=("dialogue",),
            label_field="summary",
            task_type="seq2seq",
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding="max_length",
            num_proc=num_proc,
        )
        train_dataset = load_supervised_text_dataset(train_config, tokenizer)
        eval_dataset = load_supervised_text_dataset(eval_config, tokenizer)

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
    """Finetune T5/FLAN-T5 on SAMSum summarization with optional LoRA."""

    parser = argparse.ArgumentParser(description="LoRA finetuning of T5 on SAMSum.")
    parser.add_argument("--run_name", type=str, default="t5-samsum-lora")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
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
        default="q,v",
        help="Comma-separated list of module substrings for LoRA.",
    )
    parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        help="Use small synthetic SAMSum-style examples instead of loading the dataset "
        "from the Hugging Face Hub (useful for offline smoke tests).",
    )
    args = parser.parse_args()

    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size.")

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_source_length

    train_loader, eval_loader = _build_dataloaders(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        micro_batch_size=args.micro_batch_size,
        num_proc=args.num_proc,
        use_dummy_data=args.use_dummy_data,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.use_lora:
        target_modules = tuple(m.strip() for m in args.lora_target_modules.split(",") if m.strip())
        lora_config = LoraConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="SEQ_2_SEQ_LM",
        )
        model = prepare_lora_model(model, lora_config)

    training_config = TrainingConfig(
        run_name=args.run_name,
        dataset_name="samsum",
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

