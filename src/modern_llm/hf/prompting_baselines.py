"""Prompting baselines (zero/few-shot per Brown et al., 2020).

Runs zero-shot and few-shot in-context learning on SST-2, SAMSum, and GSM8K
using HF models without any parameter updates, matching GPT-3 style evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from modern_llm.evaluation.metrics import compute_metrics


def _prompt_sst2_zeroshot(sentence: str) -> str:
    """Build zero-shot SST-2 classification prompt.

    Pre:
        - sentence is a non-empty string.
    Post:
        - returns prompt string requesting sentiment classification.
    """
    if not sentence:
        raise ValueError("sentence must be non-empty")
    return f"Classify the sentiment of this sentence as positive or negative.\n\nSentence: {sentence}\nSentiment:"


def _prompt_sst2_fewshot(sentence: str, examples: List[tuple]) -> str:
    """Build few-shot SST-2 classification prompt.

    Pre:
        - examples is a list of (sentence, label_str) tuples.
    Post:
        - returns prompt with examples followed by the test sentence.
    """
    if not sentence:
        raise ValueError("sentence must be non-empty")
    prompt = "Classify the sentiment as positive or negative.\n\n"
    for ex_sent, ex_label in examples:
        prompt += f"Sentence: {ex_sent}\nSentiment: {ex_label}\n\n"
    prompt += f"Sentence: {sentence}\nSentiment:"
    return prompt


def _parse_sentiment(text: str) -> int:
    """Parse generated text to extract positive=1 or negative=0.

    Pre:
        - text is the model's continuation after the prompt.
    Post:
        - returns 1 if 'positive' appears first, else 0.
    """
    lower = text.lower()
    if "positive" in lower and "negative" not in lower:
        return 1
    if "negative" in lower and "positive" not in lower:
        return 0
    if "positive" in lower and "negative" in lower:
        pos_idx = lower.index("positive")
        neg_idx = lower.index("negative")
        return 1 if pos_idx < neg_idx else 0
    return 0


def _run_sst2_prompting(
    model,
    tokenizer,
    device: torch.device,
    num_shots: int = 0,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run SST-2 prompting baseline.

    Pre:
        - num_shots >= 0.
    Post:
        - returns dict with accuracy.
    """
    dataset = load_dataset("glue", "sst2", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    few_shot_examples = [
        ("This movie was fantastic!", "positive"),
        ("I hated every minute of it.", "negative"),
        ("A masterpiece of cinema.", "positive"),
    ]

    predictions = []
    references = []

    for i, sample in enumerate(dataset):
        sentence = sample["sentence"]
        label = sample["label"]

        if num_shots == 0:
            prompt = _prompt_sst2_zeroshot(sentence)
        else:
            prompt = _prompt_sst2_fewshot(sentence, few_shot_examples[:num_shots])

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt):].strip()
        pred = _parse_sentiment(continuation)
        predictions.append(pred)
        references.append(label)

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(dataset)} SST-2 samples...")

    result = compute_metrics("sst-2", predictions, references)
    return result.metrics


def _run_samsum_prompting(
    model,
    tokenizer,
    device: torch.device,
    num_shots: int = 0,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run SAMSum prompting baseline for summarization.

    Pre:
        - model is seq2seq (T5-style) or decoder-only.
    Post:
        - returns ROUGE metrics.
    """
    dataset = load_dataset("samsum", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    predictions = []
    references = []

    for i, sample in enumerate(dataset):
        dialogue = sample["dialogue"]
        summary = sample["summary"]

        if num_shots == 0:
            prompt = f"Summarize the following conversation:\n\n{dialogue}\n\nSummary:"
        else:
            prompt = "Summarize the conversation.\n\n"
            for j in range(min(num_shots, len(dataset))):
                ex = dataset[j]
                prompt += f"Conversation: {ex['dialogue']}\nSummary: {ex['summary']}\n\n"
            prompt += f"Conversation: {dialogue}\nSummary:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt):].strip() if isinstance(model, AutoModelForCausalLM) else generated
        predictions.append(continuation)
        references.append(summary)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(dataset)} SAMSum samples...")

    result = compute_metrics("samsum", predictions, references)
    return result.metrics


def _run_gsm8k_prompting(
    model,
    tokenizer,
    device: torch.device,
    num_shots: int = 0,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run GSM8K prompting baseline for math reasoning.

    Pre:
        - num_shots >= 0.
    Post:
        - returns dict with exact_match.
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    predictions = []
    references = []

    for i, sample in enumerate(dataset):
        question = sample["question"]
        answer = sample["answer"]

        if num_shots == 0:
            prompt = f"Q: {question}\nA:"
        else:
            prompt = ""
            for j in range(min(num_shots, len(dataset))):
                ex = dataset[j]
                prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
            prompt += f"Q: {question}\nA:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt):].strip()
        predictions.append(continuation)
        references.append(answer)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(dataset)} GSM8K samples...")

    result = compute_metrics("gsm8k", predictions, references)
    return result.metrics


def main() -> None:
    """Run prompting baselines for SST-2, SAMSum, and GSM8K."""

    parser = argparse.ArgumentParser(description="Zero/few-shot prompting baselines.")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "samsum", "gsm8k"])
    parser.add_argument("--num_shots", type=int, default=0, help="Number of few-shot examples (0 for zero-shot).")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit eval samples for speed.")
    parser.add_argument("--output_csv", type=str, default="experiments/prompting_baseline_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in args.model_name.lower() or "flan" in args.model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    results = []
    for task in args.tasks:
        task_lower = task.lower()
        print(f"\n--- Running {args.num_shots}-shot prompting on {task} ---")
        if task_lower == "sst2":
            metrics = _run_sst2_prompting(model, tokenizer, device, args.num_shots, args.max_samples)
        elif task_lower == "samsum":
            metrics = _run_samsum_prompting(model, tokenizer, device, args.num_shots, args.max_samples)
        elif task_lower == "gsm8k":
            metrics = _run_gsm8k_prompting(model, tokenizer, device, args.num_shots, args.max_samples)
        else:
            print(f"Unknown task: {task}, skipping...")
            continue

        results.append({
            "task": task,
            "model": args.model_name,
            "num_shots": args.num_shots,
            **metrics,
        })
        print(f"Results: {metrics}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote prompting baseline results to {output_path}")


if __name__ == "__main__":
    main()

