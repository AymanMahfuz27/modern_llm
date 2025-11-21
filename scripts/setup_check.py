"""Verify that all required dependencies are installed."""

from __future__ import annotations

import sys


def check_imports() -> bool:
    """Check if all required packages can be imported.
    
    Post:
        - returns True if all imports succeed, False otherwise.
    """
    required = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "evaluate",
        "rouge_score",
        "tqdm",
        "matplotlib",
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} - MISSING")
            missing.append(pkg)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)



