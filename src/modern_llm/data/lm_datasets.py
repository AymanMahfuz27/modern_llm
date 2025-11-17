"""Causal language modeling data prep (e.g., WikiText-2, TinyStories).

WikiText-2 (Merity et al., 2016) and TinyStories (Gao et al., 2023) are the
primary corpora; this module standardizes how we fetch and tokenize them so the
training scripts can assume reproducible, research-grade preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class LanguageModelingDatasetConfig:
    """Configure a Hugging Face dataset for causal LM use."""

    dataset_name: str
    dataset_config_name: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    max_length: int = 1024
    num_proc: Optional[int] = None
    streaming: bool = False

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be a non-empty string")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, received {self.max_length}")


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore

        return load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required. Install it with `pip install datasets`."
        ) from exc


def load_causal_lm_dataset(
    config: LanguageModelingDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
):
    """Load and tokenize a dataset for causal language modeling.

    Pre:
        - `tokenizer` is a causal LM tokenizer with `pad_token_id` defined.
        - the Hugging Face dataset specified in `config` is reachable.
    Post:
        - returns a tokenized `datasets.Dataset` with `input_ids` and `attention_mask` tensors.
    Complexity:
        - O(num_examples · max_length) due to tokenization work.
    Invariants:
        - Output dataset always exposes `input_ids`, `attention_mask`, `labels`.
    """

    load_dataset = _require_datasets()
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        split=config.split,
        streaming=config.streaming,
    )

    if config.streaming:
        raise NotImplementedError("Streaming datasets are not yet supported in Phase 0 scaffolding.")

    column_names = dataset.column_names
    if config.text_field not in column_names:
        raise ValueError(
            f"text_field '{config.text_field}' not present in dataset columns: {column_names}"
        )

    def _tokenize(batch):
        texts = batch[config.text_field]
        outputs = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors=None,
        )
        labels = []
        for ids, mask in zip(outputs["input_ids"], outputs["attention_mask"]):
            label_row = [token if mask[idx] == 1 else -100 for idx, token in enumerate(ids)]
            labels.append(label_row)
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": labels,
        }

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=column_names,
        num_proc=config.num_proc,
        desc=f"Tokenizing {config.dataset_name}",
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

