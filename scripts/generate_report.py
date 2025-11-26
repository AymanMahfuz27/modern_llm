#!/usr/bin/env python3
"""Generate markdown report for pipeline run.

Usage:
    python scripts/generate_report.py --state experiments/runs/pipeline/pipeline_state.json
    python scripts/generate_report.py --config local --run-name my-pipeline

Generates a comprehensive report with architecture, metrics, and samples.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modern_llm.alignment.alignment_pipeline import PipelineState
from modern_llm.config import PipelineConfig, get_pipeline_preset
from modern_llm.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pipeline Report")
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
        "--output-dir",
        type=Path,
        default=Path("report"),
        help="Output directory for report",
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
            # Create empty state
            state = PipelineState()
    else:
        # Create empty state
        state = PipelineState()

    print(f"\n{'=' * 50}")
    print("Report Generation")
    print(f"{'=' * 50}\n")

    report_path = generate_report(state, config, args.output_dir)

    print(f"\nReport generated: {report_path}")


if __name__ == "__main__":
    main()



