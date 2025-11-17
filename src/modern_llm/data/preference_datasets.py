"""Preference pair datasets for DPO or similar alignment procedures.

Typical sources include Anthropic HH-RLHF (Bai et al., 2022) or synthetic judge
labels derived from instruction-following corpora.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PreferenceDatasetConfig:
    """Configuration describing a pairwise preference dataset."""

    dataset_name: str
    dataset_config_name: Optional[str] = None
    split: str = "train"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    prompt_field: Optional[str] = "prompt"

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name is required for preference loading.")
        if self.chosen_field == self.rejected_field:
            raise ValueError("chosen_field and rejected_field must differ.")


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore

        return load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required. Install it with `pip install datasets`."
        ) from exc


def load_preference_dataset(config: PreferenceDatasetConfig):
    """Return a dataset that yields prompt, chosen, rejected triples.

    Pre:
        - dataset exposes the configured fields.
    Post:
        - returns Hugging Face Dataset ready for DPO batching.
    Complexity:
        - O(num_examples) to scan column names and instantiate the dataset.
    Invariants:
        - Each row keeps prompt/response ordering intact.
    """

    load_dataset = _require_datasets()
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        split=config.split,
    )
    column_names = dataset.column_names
    for field in filter(None, [config.prompt_field, config.chosen_field, config.rejected_field]):
        if field not in column_names:
            raise ValueError(f"Field '{field}' missing from dataset columns: {column_names}")
    return dataset

