#!/usr/bin/env python3
"""Direct Preference Optimization script for Modern LLM.

Usage:
    python scripts/dpo.py --sft-checkpoint PATH [--config CONFIG]
    python scripts/dpo.py --sft-checkpoint experiments/runs/sft/final.pt --config local

This script aligns YOUR SFT model using preference data (chosen vs rejected).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modern_llm.training.train_dpo import main

if __name__ == "__main__":
    main()


