"""Generate text samples from LM checkpoints for qualitative analysis.

Loads scratch LM checkpoints and samples from a fixed set of prompts,
saving outputs under experiments/ for side-by-side comparison in the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
from modern_llm.training.train_lm import generate_text


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> ModernDecoderLM:
    """Load a trained ModernDecoderLM from checkpoint.

    Pre:
        - checkpoint_path is a .pt file with 'model_state' and 'config'.
    Post:
        - returns model in eval mode on device.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" not in state or "config" not in state:
        raise ValueError(f"Checkpoint {checkpoint_path} missing 'model_state' or 'config'.")

    config = ModernLLMConfig(**state["config"])
    model = ModernDecoderLM(config)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    """Generate from scratch LM checkpoints and save outputs."""

    parser = argparse.ArgumentParser(description="Generate text from scratch LM checkpoints.")
    parser.add_argument("--runs_dir", type=str, default="experiments/runs")
    parser.add_argument("--output_json", type=str, default="experiments/lm_generation_samples.json")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["The meaning of life is", "In a football match,", "Scientists have discovered"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoint_paths = sorted(runs_dir.rglob("*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {runs_dir}")

    results = []
    for ckpt_path in checkpoint_paths:
        run_name = ckpt_path.parent.name
        ckpt_name = ckpt_path.stem
        print(f"Generating from {run_name}/{ckpt_name}...")

        try:
            model = _load_checkpoint(ckpt_path, device)
            samples = {}
            for prompt in args.prompts:
                output = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                samples[prompt] = output
            results.append({
                "run_name": run_name,
                "checkpoint": ckpt_name,
                "samples": samples,
            })
        except Exception as exc:
            print(f"  Skipping {ckpt_path}: {exc}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} checkpoint generation samples to {output_path}")


if __name__ == "__main__":
    main()



