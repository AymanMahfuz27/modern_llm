#!/usr/bin/env python3
"""Verifier training script for Modern LLM.

Usage:
    python scripts/train_verifier.py [--config CONFIG]
    python scripts/train_verifier.py --config local

Trains a small encoder model to score answer correctness for math/QA problems.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modern_llm.training.train_verifier import main

if __name__ == "__main__":
    main()


