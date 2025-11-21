"""Python wrapper to run Phase 1 & 2 experiments without bash.

Usage:
    python scripts/run_experiments.py --phase 1        # Phase 1 only
    python scripts/run_experiments.py --phase 2        # Phase 2 only
    python scripts/run_experiments.py --phase all      # Both phases
    python scripts/run_experiments.py --phase eval     # Evaluations only
    python scripts/run_experiments.py --phase smoke    # Quick smoke test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> None:
    """Execute a shell command and handle errors.

    Pre:
        - cmd is a list of command arguments.
    Post:
        - prints description and runs command, raising on failure.
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    project_root = Path(__file__).parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root / 'src'}:{env.get('PYTHONPATH', '')}"
    
    result = subprocess.run(cmd, check=False, cwd=project_root, env=env)
    if result.returncode != 0:
        print(f"\nERROR: Command failed with exit code {result.returncode}")
        print(f"Command: {' '.join(cmd)}")
        sys.exit(result.returncode)


def run_phase1() -> None:
    """Execute all Phase 1 experiments."""
    run_command(
        ["python", "scripts/evaluate_lm_checkpoints.py"],
        "Phase 1.1: Evaluating existing LM checkpoints",
    )
    run_command(
        ["python", "scripts/generate_from_checkpoints.py"],
        "Phase 1.2: Generating text samples",
    )
    run_command(
        ["python", "scripts/train_lm_from_config.py", "--config", "configs/lm_max_rtx3060.json"],
        "Phase 1.3: Training MAX-SIZE scratch LM (LONG RUN)",
    )
    run_command(
        ["python", "scripts/experiment_attention_sinks.py"],
        "Phase 1.4: Attention sinks long-context experiment",
    )
    run_command(
        ["python", "notebooks/visualize_lm_results.py"],
        "Phase 1.5: Visualizing LM results",
    )


def run_phase2() -> None:
    """Execute all Phase 2 experiments."""
    run_command(
        [
            "python", "src/modern_llm/hf/finetune_gpt2_sst2.py",
            "--run_name", "gpt2-sst2-lora-main",
            "--max_steps", "2000",
            "--use_lora",
        ],
        "Phase 2.1: GPT-2 + LoRA on SST-2",
    )
    run_command(
        [
            "python", "scripts/evaluate_hf_sst2.py",
            "--checkpoint_path", "experiments/runs/gpt2-sst2-lora-main/gpt2-sst2-lora-main_final.pt",
            "--model_name", "gpt2",
        ],
        "Phase 2.2: Evaluating GPT-2 SST-2",
    )
    run_command(
        [
            "python", "src/modern_llm/hf/finetune_t5_samsum.py",
            "--run_name", "t5-samsum-lora-main",
            "--max_steps", "2000",
            "--use_lora",
        ],
        "Phase 2.3: T5 + LoRA on SAMSum",
    )
    run_command(
        [
            "python", "scripts/evaluate_hf_samsum.py",
            "--checkpoint_path", "experiments/runs/t5-samsum-lora-main/t5-samsum-lora-main_final.pt",
            "--model_name", "t5-small",
        ],
        "Phase 2.4: Evaluating T5 SAMSum",
    )
    run_command(
        [
            "python", "src/modern_llm/hf/finetune_math_gsm8k.py",
            "--run_name", "gpt2-gsm8k-lora-main",
            "--max_steps", "3000",
            "--use_lora",
        ],
        "Phase 2.5: GPT-2 + LoRA on GSM8K",
    )
    run_command(
        [
            "python", "scripts/evaluate_hf_gsm8k.py",
            "--checkpoint_path", "experiments/runs/gpt2-gsm8k-lora-main/gpt2-gsm8k-lora-main_final.pt",
            "--model_name", "gpt2",
        ],
        "Phase 2.6: Evaluating GPT-2 GSM8K",
    )
    run_command(
        [
            "python", "-m", "modern_llm.hf.prompting_baselines",
            "--tasks", "sst2", "samsum", "gsm8k",
            "--max_samples", "500",
        ],
        "Phase 2.7: Prompting baselines",
    )


def run_evaluations_only() -> None:
    """Run all evaluations assuming checkpoints exist."""
    run_command(
        ["python", "scripts/evaluate_lm_checkpoints.py"],
        "Evaluating LM checkpoints",
    )
    run_command(
        ["python", "scripts/generate_from_checkpoints.py"],
        "Generating text samples",
    )
    run_command(
        ["python", "notebooks/visualize_lm_results.py"],
        "Visualizing LM results",
    )
    
    sst2_ckpt = Path("experiments/runs/gpt2-sst2-lora-main/gpt2-sst2-lora-main_final.pt")
    if sst2_ckpt.exists():
        run_command(
            ["python", "scripts/evaluate_hf_sst2.py", "--checkpoint_path", str(sst2_ckpt), "--model_name", "gpt2"],
            "Evaluating SST-2 checkpoint",
        )
    
    samsum_ckpt = Path("experiments/runs/t5-samsum-lora-main/t5-samsum-lora-main_final.pt")
    if samsum_ckpt.exists():
        run_command(
            ["python", "scripts/evaluate_hf_samsum.py", "--checkpoint_path", str(samsum_ckpt), "--model_name", "t5-small"],
            "Evaluating SAMSum checkpoint",
        )
    
    gsm8k_ckpt = Path("experiments/runs/gpt2-gsm8k-lora-main/gpt2-gsm8k-lora-main_final.pt")
    if gsm8k_ckpt.exists():
        run_command(
            ["python", "scripts/evaluate_hf_gsm8k.py", "--checkpoint_path", str(gsm8k_ckpt), "--model_name", "gpt2"],
            "Evaluating GSM8K checkpoint",
        )
    
    run_command(
        ["python", "src/modern_llm/hf/prompting_baselines.py", "--max_samples", "500"],
        "Running prompting baselines",
    )


def run_smoke_test() -> None:
    """Run minimal smoke test to verify components."""
    run_command(
        [
            "python", "src/modern_llm/training/train_lm.py",
            "--run_name", "smoke-test-lm",
            "--max_steps", "50",
            "--d_model", "256",
            "--n_layers", "4",
            "--batch_size", "16",
            "--micro_batch_size", "4",
        ],
        "Smoke: Small scratch LM",
    )
    run_command(
        [
            "python", "src/modern_llm/hf/finetune_gpt2_sst2.py",
            "--run_name", "smoke-test-sst2",
            "--max_steps", "50",
            "--use_lora",
        ],
        "Smoke: GPT-2 SST-2",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 & 2 experiments.")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "all", "eval", "smoke"],
        default="all",
        help="Which phase to run: 1, 2, all, eval (evaluations only), or smoke (quick test)",
    )
    args = parser.parse_args()

    if args.phase == "1":
        run_phase1()
    elif args.phase == "2":
        run_phase2()
    elif args.phase == "all":
        run_phase1()
        run_phase2()
    elif args.phase == "eval":
        run_evaluations_only()
    elif args.phase == "smoke":
        run_smoke_test()

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

