#!/usr/bin/env python3
"""Verify all datasets in DATASET_REGISTRY load correctly.

Pre: HuggingFace datasets library installed, network access available.
Post: Prints status report for each dataset; exits 0 if all critical ones load.

Usage:
    python scripts/verify_datasets.py
    python scripts/verify_datasets.py --quick  # Only test first 10 rows
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

# Critical datasets for pipeline (must work)
CRITICAL_DATASETS = ["wikitext-103-raw-v1", "roneneldan/TinyStories"]

# Optional datasets (nice to have)
OPTIONAL_DATASETS = ["wikitext-2-raw-v1", "openwebtext", "bookcorpus"]


@dataclass
class DatasetStatus:
    name: str
    success: bool
    num_rows: Optional[int] = None
    error: Optional[str] = None
    text_sample: Optional[str] = None


def test_dataset(
    name: str,
    hf_name: str,
    hf_config: Optional[str],
    text_field: str,
    quick: bool = False,
) -> DatasetStatus:
    """Test loading a single dataset."""
    try:
        from datasets import load_dataset

        # Load with streaming for quick mode
        if quick:
            ds = load_dataset(
                hf_name,
                hf_config,
                split="train",
                streaming=True,
            )
            # Take first 10 rows
            rows = list(ds.take(10))
            num_rows = len(rows)
            sample = rows[0].get(text_field, "")[:100] if rows else ""
        else:
            ds = load_dataset(
                hf_name,
                hf_config,
                split="train",
            )
            num_rows = len(ds)
            sample = ds[0].get(text_field, "")[:100] if num_rows > 0 else ""

        return DatasetStatus(
            name=name,
            success=True,
            num_rows=num_rows,
            text_sample=sample,
        )

    except Exception as e:
        return DatasetStatus(
            name=name,
            success=False,
            error=str(e),
        )


def main():
    parser = argparse.ArgumentParser(description="Verify dataset loading")
    parser.add_argument("--quick", action="store_true", help="Quick check (streaming, 10 rows)")
    args = parser.parse_args()

    # Import registry
    sys.path.insert(0, str(__file__).replace("/scripts/verify_datasets.py", "/src"))
    from modern_llm.data.lm_datasets import DATASET_REGISTRY

    print("=" * 60)
    print("Dataset Verification Report")
    print("=" * 60)
    print(f"Mode: {'Quick (streaming)' if args.quick else 'Full (download)'}\n")

    results = []
    all_critical_ok = True

    # Test all datasets in registry
    for name, (hf_name, hf_config, text_field) in DATASET_REGISTRY.items():
        print(f"Testing: {name}...", end=" ", flush=True)
        status = test_dataset(name, hf_name, hf_config, text_field, quick=args.quick)
        results.append(status)

        if status.success:
            print(f"OK ({status.num_rows:,} rows)")
            if status.text_sample:
                print(f"  Sample: {status.text_sample!r}...")
        else:
            print(f"FAILED")
            print(f"  Error: {status.error}")

            if name in CRITICAL_DATASETS:
                all_critical_ok = False

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r.success)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    print("\nCritical datasets status:")
    for name in CRITICAL_DATASETS:
        status = next((r for r in results if r.name == name), None)
        if status and status.success:
            print(f"  [OK] {name}")
        else:
            print(f"  [FAIL] {name}")

    if all_critical_ok:
        print("\n✓ All critical datasets loaded successfully!")
        print("  Safe to proceed with training.")
        return 0
    else:
        print("\n✗ Some critical datasets failed to load.")
        print("  Check network access and dataset availability.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

