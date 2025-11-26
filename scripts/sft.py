#!/usr/bin/env python3
"""Supervised Fine-Tuning script for Modern LLM.

Usage:
    python scripts/sft.py --pretrain-checkpoint PATH [--config CONFIG]
    python scripts/sft.py --pretrain-checkpoint experiments/runs/pretrain/final.pt --config local

This script instruction-tunes YOUR pretrained model on dialog/instruction data.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modern_llm.training.train_sft import main

if __name__ == "__main__":
    main()



