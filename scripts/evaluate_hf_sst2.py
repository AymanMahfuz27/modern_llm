"""Evaluate a finetuned HF model on SST-2 and report accuracy + examples.

Loads a checkpoint from GPT-2/DistilGPT2 LoRA finetuning and computes
validation accuracy, F1, and saves misclassified examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from modern_llm.data import TaskDatasetConfig, load_supervised_text_dataset
from modern_llm.evaluation.metrics import compute_metrics, compute_f1


def main() -> None:
    """Evaluate HF SST-2 checkpoint and write accuracy + error examples."""

    parser = argparse.ArgumentParser(description="Evaluate HF SST-2 checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_metrics_csv", type=str, default="experiments/sst2_eval_metrics.csv")
    parser.add_argument("--output_errors_json", type=str, default="experiments/sst2_misclassified.json")
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    eval_config = TaskDatasetConfig(
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        text_fields=("sentence",),
        label_field="label",
        task_type="classification",
        max_source_length=args.max_seq_len,
        padding="max_length",
        num_proc=args.num_proc,
    )
    eval_dataset = load_supervised_text_dataset(eval_config, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    misclassified = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    misclassified.append({
                        "sentence": sentence,
                        "predicted": int(preds[i]),
                        "true_label": int(labels[i]),
                    })

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * args.batch_size} samples...")

    result = compute_metrics("sst-2", all_preds, all_labels)
    f1_result = compute_f1(all_preds, all_labels, num_classes=2)
    print(f"\nAccuracy: {result.metrics['accuracy']:.4f}")
    print(f"Macro F1: {f1_result['macro_f1']:.4f}")

    metrics_path = Path(args.output_metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerow({
            "checkpoint": checkpoint_path.name,
            "accuracy": result.metrics["accuracy"],
            "macro_f1": f1_result["macro_f1"],
        })
    print(f"Wrote metrics to {metrics_path}")

    errors_path = Path(args.output_errors_json)
    with errors_path.open("w") as f:
        json.dump(misclassified[:50], f, indent=2)
    print(f"Wrote {len(misclassified)} misclassified examples (showing first 50) to {errors_path}")


if __name__ == "__main__":
    main()



