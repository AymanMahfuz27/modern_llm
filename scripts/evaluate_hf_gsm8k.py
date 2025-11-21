"""Evaluate a finetuned HF causal LM on GSM8K and report exact match.

Loads a decoder checkpoint (GPT-2, TinyLlama) and measures EM on GSM8K test.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from modern_llm.evaluation.metrics import compute_metrics


def _extract_answer_number(text: str) -> str:
    """Extract the final numeric answer from GSM8K-style chain-of-thought.

    GSM8K answers end with '#### {number}'. We look for that pattern or
    fallback to the last number in the text.

    Pre:
        - text is a non-empty string.
    Post:
        - returns the extracted number as a string, or empty if none found.
    """
    if not text:
        return ""
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


def main() -> None:
    """Evaluate HF GSM8K checkpoint and write exact match metrics."""

    parser = argparse.ArgumentParser(description="Evaluate HF GSM8K checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_csv", type=str, default="experiments/gsm8k_eval_metrics.csv")
    parser.add_argument("--output_errors_json", type=str, default="experiments/gsm8k_errors.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit test samples for speed.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    dataset = load_dataset("gsm8k", "main", split="test")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    predictions = []
    references = []
    errors = []

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            question = sample["question"]
            answer = sample["answer"]
            ref_number = _extract_answer_number(answer)

            prompt = f"Q: {question}\nA:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated[len(prompt):].strip()
            pred_number = _extract_answer_number(continuation)

            predictions.append(pred_number)
            references.append(ref_number)

            if pred_number != ref_number:
                errors.append({
                    "question": question,
                    "predicted_answer": continuation,
                    "predicted_number": pred_number,
                    "true_answer": answer,
                    "true_number": ref_number,
                })

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(dataset)} samples...")

    result = compute_metrics("gsm8k", predictions, references)
    print(f"\nExact Match: {result.metrics['exact_match']:.4f}")

    metrics_path = Path(args.output_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "exact_match", "num_errors"])
        writer.writeheader()
        writer.writerow({
            "checkpoint": checkpoint_path.name,
            "exact_match": result.metrics["exact_match"],
            "num_errors": len(errors),
        })
    print(f"Wrote metrics to {metrics_path}")

    errors_path = Path(args.output_errors_json)
    with errors_path.open("w") as f:
        json.dump(errors[:50], f, indent=2)
    print(f"Wrote first 50 errors to {errors_path}")


if __name__ == "__main__":
    main()



