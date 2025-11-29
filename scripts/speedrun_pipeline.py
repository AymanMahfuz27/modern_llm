#!/usr/bin/env python3
"""Main pipeline orchestration script for Modern LLM.

This is the Python entrypoint called by speedrun.sh. It runs the full
alignment pipeline: Pretrain -> SFT -> DPO -> Verifier -> Evaluate -> Report.

Usage:
    python scripts/speedrun_pipeline.py --config local
    python scripts/speedrun_pipeline.py --config gpu --checkpoint-dir /path/to/checkpoints
    python scripts/speedrun_pipeline.py --config configs/custom.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modern_llm.alignment.alignment_pipeline import AlignmentPipeline, run_alignment_pipeline
from modern_llm.config import PipelineConfig, get_pipeline_preset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modern LLM Speedrun Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local smoke test (quick)
  python scripts/speedrun_pipeline.py --config local-smoke

  # Full local training on RTX 3060
  python scripts/speedrun_pipeline.py --config local

  # GPU smoke test
  python scripts/speedrun_pipeline.py --config gpu-smoke

  # Full GPU training
  python scripts/speedrun_pipeline.py --config gpu --checkpoint-dir /path/to/checkpoints

  # Custom config file
  python scripts/speedrun_pipeline.py --config configs/custom.json

Pipeline Stages:
  1. Pretrain  - Language model pretraining on text corpora
  2. SFT       - Supervised fine-tuning on instruction data
  3. DPO       - Direct preference optimization for alignment
  4. Verifier  - Train answer correctness model
  5. Evaluate  - Compare all stages on benchmarks
  6. Report    - Generate markdown report with metrics
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config preset (local, local-smoke, gpu, gpu-smoke) or path to JSON file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints (default: experiments/runs/<run_name>)",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Path to existing pretrain checkpoint (skips pretraining)",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining stage (requires --pretrain-checkpoint or existing state)",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT stage",
    )
    parser.add_argument(
        "--skip-dpo",
        action="store_true",
        help="Skip DPO stage",
    )
    parser.add_argument(
        "--skip-verifier",
        action="store_true",
        help="Skip verifier training",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation stage",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing reports (default: append timestamp to new reports)",
    )

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        print(f"Loading config from file: {args.config}")
        config = PipelineConfig.load(args.config)
    else:
        print(f"Using config preset: {args.config}")
        config = get_pipeline_preset(args.config)

    print(f"\n{'=' * 60}")
    print(f"Modern LLM Speedrun Pipeline")
    print(f"{'=' * 60}")
    print(f"Run name:     {config.run_name}")
    print(f"Model size:   d={config.d_model}, L={config.n_layers}, H={config.n_heads}")
    print(f"Hardware:     {config.hardware_preset}")
    print(f"Data scale:   {config.data_preset}")
    print(f"{'=' * 60}\n")

    # Run alignment pipeline
    state = run_alignment_pipeline(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        pretrain_checkpoint=args.pretrain_checkpoint,
        skip_pretrain=args.skip_pretrain,
        skip_sft=args.skip_sft,
        skip_dpo=args.skip_dpo,
        skip_verifier=args.skip_verifier,
    )

    # Run evaluation
    if not args.skip_eval:
        print("\n[Evaluation] Running evaluation suite...")
        try:
            from modern_llm.evaluation.pipeline_eval import evaluate_pipeline_stages
            results = evaluate_pipeline_stages(state, config)
            print(f"Evaluation results saved to: {results}")
        except ImportError:
            print("Evaluation module not ready, skipping...")

    # Generate report
    if not args.skip_report:
        print("\n[Report] Generating report...")
        try:
            from modern_llm.report import generate_report
            from datetime import datetime

            report_dir = Path("report")
            report_path = report_dir / f"{config.run_name}_report.md"

            # Protect existing reports unless --force is set
            if report_path.exists() and not args.force:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = report_dir / f"{config.run_name}_report_{timestamp}.md"
                print(f"Existing report found, saving to: {report_path}")

            report_path = generate_report(state, config, output_path=report_path)
            print(f"Report saved to: {report_path}")
        except ImportError:
            print("Report module not ready, skipping...")

    print("\n" + "=" * 60)
    print("Speedrun complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()



