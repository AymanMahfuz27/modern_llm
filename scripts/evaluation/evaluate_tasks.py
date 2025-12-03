#!/usr/bin/env python3
"""Unified task evaluation runner.

Evaluates models on all tasks and generates comparison tables.
Supports both scratch models and HuggingFace baselines.

Usage:
    # Evaluate scratch model on all tasks
    python scripts/evaluation/evaluate_tasks.py --checkpoint path/to/model.pt

    # Evaluate HF baseline
    python scripts/evaluation/evaluate_tasks.py --hf-model gpt2

    # Compare all stage checkpoints
    python scripts/evaluation/evaluate_tasks.py --stage-checkpoints experiments/runs/gpu-full/

    # Full comparison (scratch + HF baselines)
    python scripts/evaluation/evaluate_tasks.py --checkpoint model.pt --include-baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

# Ensure both the project root and `src` are on sys.path so that we can import
# local packages (e.g., `modern_llm`) and the `scripts` package when this file
# is executed as a script (python scripts/.../evaluate_tasks.py) from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def find_stage_checkpoints(run_dir: Path) -> dict[str, Path]:
    """Find checkpoints for each training stage in a run directory.
    
    Note: Excludes verifier since it's a classification model, not an LM.
    """
    stages = ["pretrain", "sft", "dpo"]  # Verifier uses different model architecture
    checkpoints = {}
    
    for stage in stages:
        # Look for *_<stage>_final.pt or checkpoints under a stage-specific
        # subdirectory. We search recursively so that layouts like:
        #   <run_root>/<run_name>-<stage>/<run_name>-<stage>_final.pt
        # are discovered correctly.
        patterns = [
            f"*-{stage}_final.pt",          # e.g., gpu-full-dpo_final.pt
            f"*-{stage}/checkpoint_*.pt",   # e.g., gpu-full-dpo/checkpoint_*.pt
            f"{stage}_final.pt",
        ]
        for pattern in patterns:
            matches = sorted(run_dir.rglob(pattern))
            if matches:
                checkpoints[stage] = matches[-1]  # latest match
                break
    
    return checkpoints


def evaluate_single_model(
    model_path: str = None,
    hf_model: str = None,
    device: str = "cuda",
    max_sst2: int = 200,
    max_gsm8k: int = 50,
) -> dict:
    """Evaluate a single model on all tasks."""
    # Import task-specific evaluators from the local scripts package.
    # The project root was added to sys.path above so `scripts` is importable.
    from scripts.evaluation.eval_sst2 import (
        evaluate_sst2,
        load_hf_model,
        load_scratch_model,
    )

    if hf_model:
        model, tokenizer = load_hf_model(hf_model, device)
        model_name = hf_model
        is_hf = True
    else:
        model, tokenizer = load_scratch_model(model_path, device)
        model_name = Path(model_path).stem
        is_hf = False

    results = {
        "model": model_name,
        "is_hf_baseline": is_hf,
        "timestamp": datetime.now().isoformat(),
    }

    # SST-2
    print(f"  Evaluating SST-2...")
    sst2_results = evaluate_sst2(model, tokenizer, device, max_sst2, is_hf)
    results["sst2_accuracy"] = sst2_results["accuracy"]

    # GSM8K (skip for HF models for now - they need different prompting)
    if not is_hf and model_path:
        print(f"  Evaluating GSM8K...")
        from scripts.evaluation.eval_gsm8k import evaluate_gsm8k
        gsm8k_results = evaluate_gsm8k(model, tokenizer, device, max_samples=max_gsm8k)
        results["gsm8k_em"] = gsm8k_results["exact_match_no_verifier"]
        results["gsm8k_errors"] = gsm8k_results["error_taxonomy"]

    return results


def generate_comparison_table(all_results: list[dict]) -> str:
    """Generate markdown comparison table from results."""
    lines = [
        "| Model | SST-2 Acc | GSM8K EM | Notes |",
        "|-------|-----------|----------|-------|",
    ]
    
    for r in all_results:
        sst2 = f"{r.get('sst2_accuracy', 0):.1%}" if r.get('sst2_accuracy') else "N/A"
        gsm8k = f"{r.get('gsm8k_em', 0):.1%}" if r.get('gsm8k_em') else "N/A"
        notes = "HF Baseline" if r.get("is_hf_baseline") else r.get("stage", "")
        lines.append(f"| {r['model']} | {sst2} | {gsm8k} | {notes} |")
    
    return "\n".join(lines)


def generate_stage_gains_table(stage_results: dict) -> str:
    """Generate stage-wise gains table."""
    lines = [
        "| Stage | SST-2 Acc | GSM8K EM | Δ SST-2 | Δ GSM8K |",
        "|-------|-----------|----------|---------|---------|",
    ]
    
    prev_sst2, prev_gsm8k = 0, 0
    for stage in ["pretrain", "sft", "dpo"]:
        if stage not in stage_results:
            continue
        r = stage_results[stage]
        sst2 = r.get("sst2_accuracy", 0)
        gsm8k = r.get("gsm8k_em", 0)
        delta_sst2 = sst2 - prev_sst2 if prev_sst2 else 0
        delta_gsm8k = gsm8k - prev_gsm8k if prev_gsm8k else 0
        
        lines.append(
            f"| {stage.upper()} | {sst2:.1%} | {gsm8k:.1%} | "
            f"{delta_sst2:+.1%} | {delta_gsm8k:+.1%} |"
        )
        prev_sst2, prev_gsm8k = sst2, gsm8k
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Unified task evaluation")
    parser.add_argument("--checkpoint", type=str, help="Single model checkpoint")
    parser.add_argument("--hf-model", type=str, help="HuggingFace model name")
    parser.add_argument("--stage-checkpoints", type=str, help="Directory with stage checkpoints")
    parser.add_argument("--include-baselines", action="store_true", help="Include GPT-2 baselines")
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    parser.add_argument("--max-sst2", type=int, default=200)
    parser.add_argument("--max-gsm8k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Evaluate HF baselines if requested
    if args.include_baselines:
        for hf_name in ["gpt2", "distilgpt2"]:
            print(f"\nEvaluating HF baseline: {hf_name}")
            results = evaluate_single_model(
                hf_model=hf_name,
                device=args.device,
                max_sst2=args.max_sst2,
            )
            all_results.append(results)

    # Evaluate single checkpoint
    if args.checkpoint:
        print(f"\nEvaluating: {args.checkpoint}")
        results = evaluate_single_model(
            model_path=args.checkpoint,
            device=args.device,
            max_sst2=args.max_sst2,
            max_gsm8k=args.max_gsm8k,
        )
        all_results.append(results)

    # Evaluate single HF model
    if args.hf_model:
        print(f"\nEvaluating HF model: {args.hf_model}")
        results = evaluate_single_model(
            hf_model=args.hf_model,
            device=args.device,
            max_sst2=args.max_sst2,
        )
        all_results.append(results)

    # Evaluate stage checkpoints
    stage_results = {}
    if args.stage_checkpoints:
        run_dir = Path(args.stage_checkpoints)
        checkpoints = find_stage_checkpoints(run_dir)
        
        for stage, ckpt_path in checkpoints.items():
            print(f"\nEvaluating stage: {stage} ({ckpt_path})")
            results = evaluate_single_model(
                model_path=str(ckpt_path),
                device=args.device,
                max_sst2=args.max_sst2,
                max_gsm8k=args.max_gsm8k,
            )
            results["stage"] = stage
            all_results.append(results)
            stage_results[stage] = results

    # Save raw results
    with open(output_dir / "task_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate and save comparison table
    comparison_md = generate_comparison_table(all_results)
    with open(output_dir / "baseline_comparison.md", "w") as f:
        f.write("# Model Comparison\n\n")
        f.write(comparison_md)

    # Generate stage gains table if applicable
    if stage_results:
        gains_md = generate_stage_gains_table(stage_results)
        with open(output_dir / "stage_gains.md", "w") as f:
            f.write("# Stage-wise Gains\n\n")
            f.write(gains_md)

    print("\n" + "=" * 50)
    print("Evaluation Complete")
    print("=" * 50)
    print(f"\nResults saved to {output_dir}/")
    print("\nComparison Table:")
    print(comparison_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())

