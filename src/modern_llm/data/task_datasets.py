"""Task dataset helpers for GLUE/SAMSum/GSM8K style benchmarks.

References:
- SST-2 classification from GLUE (Wang et al., 2019).
- SAMSum summarization corpus (Gliwa et al., 2019).
- GSM8K math word problems (Cobbe et al., 2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class TaskDatasetConfig:
    """Describe supervised datasets (classification or seq2seq)."""

    dataset_name: str
    dataset_config_name: Optional[str] = None
    split: str = "train"
    text_fields: Sequence[str]
    label_field: Optional[str] = "label"
    task_type: str = "classification"
    max_source_length: int = 512
    max_target_length: Optional[int] = None
    padding: str = "max_length"
    num_proc: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be provided")
        if not self.text_fields:
            raise ValueError("text_fields must contain at least one column name")
        if self.max_source_length <= 0:
            raise ValueError(f"max_source_length must be positive, received {self.max_source_length}")
        if self.task_type not in {"classification", "seq2seq"}:
            raise ValueError(f"task_type must be 'classification' or 'seq2seq', received {self.task_type}")
        if self.task_type == "seq2seq" and (self.max_target_length is None or self.max_target_length <= 0):
            raise ValueError("seq2seq tasks require max_target_length > 0")


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore

        return load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required. Install it with `pip install datasets`."
        ) from exc


def load_supervised_text_dataset(
    config: TaskDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Load and tokenize a supervised dataset for classification or seq2seq tasks.

    Pre:
        - tokenizer vocab matches the model we plan to finetune.
        - dataset columns listed in `config.text_fields` exist.
    Post:
        - returns a `datasets.Dataset` with `input_ids`, `attention_mask`, `labels`.
    Complexity:
        - O(num_examples · max_length).
    Invariants:
        - Labels remain aligned with their original sample order.
    """

    load_dataset = _require_datasets()
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        split=config.split,
    )

    column_names = dataset.column_names
    for field in config.text_fields:
        if field not in column_names:
            raise ValueError(f"text field '{field}' is missing from dataset columns: {column_names}")

    label_field = config.label_field or "label"
    if config.task_type == "classification" and label_field not in column_names:
        raise ValueError(f"label field '{label_field}' missing from dataset columns: {column_names}")

    def _concat_text(batch):
        parts = ["\n".join(str(batch[field][i]) for field in config.text_fields) for i in range(len(batch[config.text_fields[0]]))]
        return parts

    def _preprocess_classification(batch):
        inputs = _concat_text(batch)
        encoded = tokenizer(
            inputs,
            truncation=True,
            max_length=config.max_source_length,
            padding=config.padding,
        )
        labels = batch[label_field]
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    def _preprocess_seq2seq(batch):
        if target_tokenizer is None:
            target_tok = tokenizer
        else:
            target_tok = target_tokenizer

        inputs = _concat_text(batch)
        targets = batch[label_field]
        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=config.max_source_length,
            padding=config.padding,
        )
        labels = target_tok(
            targets,
            truncation=True,
            max_length=config.max_target_length,
            padding=config.padding,
        )
        labels_ids = labels["input_ids"]
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels_ids,
        }

    preprocess_fn = _preprocess_classification if config.task_type == "classification" else _preprocess_seq2seq

    processed = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
        num_proc=config.num_proc,
        desc=f"Tokenizing {config.dataset_name} ({config.task_type})",
    )
    processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return processed

