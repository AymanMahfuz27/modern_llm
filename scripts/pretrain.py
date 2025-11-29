#!/usr/bin/env python3
"""Pretraining script for Modern LLM.

Usage:
    python scripts/pretrain.py --config local
    python scripts/pretrain.py --config configs/custom.json

This trains YOUR scratch model from random initialization on language modeling.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modern_llm.config import PipelineConfig, get_pipeline_preset
from modern_llm.training.train_lm import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Modern LLM")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config preset (local, gpu, etc.) or path to JSON file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory",
    )

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        print(f"Loading config from: {args.config}")
        config = PipelineConfig.load(args.config)
    else:
        print(f"Using preset: {args.config}")
        config = get_pipeline_preset(args.config)

    model_config = config.get_model_config()
    train_config = config.get_pretrain_config()

    if args.checkpoint_dir:
        train_config.output_dir = args.checkpoint_dir / train_config.run_name

    print(f"\n{'=' * 50}")
    print(f"Pretraining: {config.run_name}")
    print(f"{'=' * 50}")
    print(f"Model: d={model_config.d_model}, L={model_config.n_layers}, H={model_config.n_heads}")
    print(f"Steps: {train_config.max_steps}")
    print(f"Batch: {train_config.batch_size} (micro={train_config.micro_batch_size})")
    print(f"LR: {train_config.learning_rate}")
    print(f"{'=' * 50}\n")

    checkpoint = run_training(model_config, train_config)
    print(f"\nPretraining complete: {checkpoint}")


if __name__ == "__main__":
    main()



