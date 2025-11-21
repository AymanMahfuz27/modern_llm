"""Evaluate a finetuned T5/FLAN-T5 model on SAMSum and report ROUGE scores.

Loads a seq2seq checkpoint and computes ROUGE-1, ROUGE-2, ROUGE-L on validation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from modern_llm.data import TaskDatasetConfig, load_supervised_text_dataset
from modern_llm.evaluation.metrics import compute_metrics


def main() -> None:
    """Evaluate HF SAMSum checkpoint and write ROUGE metrics."""

    parser = argparse.ArgumentParser(description="Evaluate HF SAMSum checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_csv", type=str, default="experiments/samsum_eval_metrics.csv")
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_source_length

    eval_config = TaskDatasetConfig(
        dataset_name="samsum",
        dataset_config_name=None,
        split="test",
        text_fields=("dialogue",),
        label_field="summary",
        task_type="seq2seq",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        padding="max_length",
        num_proc=args.num_proc,
    )
    eval_dataset = load_supervised_text_dataset(eval_config, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["labels"]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_target_length,
                num_beams=4,
                early_stopping=True,
            )
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * args.batch_size} samples...")

    result = compute_metrics("samsum", predictions, references)
    print(f"\nROUGE-1: {result.metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {result.metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {result.metrics['rougeL']:.4f}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "rouge1", "rouge2", "rougeL"])
        writer.writeheader()
        writer.writerow({
            "checkpoint": checkpoint_path.name,
            "rouge1": result.metrics["rouge1"],
            "rouge2": result.metrics["rouge2"],
            "rougeL": result.metrics["rougeL"],
        })
    print(f"Wrote ROUGE metrics to {output_path}")


if __name__ == "__main__":
    main()

