#!/usr/bin/env python3
"""Evaluate all pipeline stages and generate comparison metrics.

Usage:
    python scripts/evaluate_pipeline.py --state experiments/runs/pipeline/pipeline_state.json
    python scripts/evaluate_pipeline.py --config local --run-name my-pipeline

Evaluates Base/SFT/DPO stages on perplexity and generates a comparison table.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modern_llm.alignment.alignment_pipeline import PipelineState
from modern_llm.config import PipelineConfig, get_pipeline_preset
from modern_llm.evaluation.pipeline_eval import evaluate_pipeline_stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pipeline Stages")
    parser.add_argument(
        "--state",
        type=Path,
        help="Path to pipeline_state.json file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="local",
        help="Config preset or JSON path",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name to look for state file",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum batches for evaluation",
    )

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config = PipelineConfig.load(args.config)
    else:
        config = get_pipeline_preset(args.config)

    # Find or load state
    if args.state:
        state = PipelineState.load(args.state)
    elif args.run_name:
        state_path = Path("experiments/runs") / args.run_name / "pipeline_state.json"
        if state_path.exists():
            state = PipelineState.load(state_path)
        else:
            print(f"State file not found: {state_path}")
            sys.exit(1)
    else:
        print("Must provide --state or --run-name")
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print("Pipeline Evaluation")
    print(f"{'=' * 50}\n")

    results_path = evaluate_pipeline_stages(state, config, args.max_batches)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()


