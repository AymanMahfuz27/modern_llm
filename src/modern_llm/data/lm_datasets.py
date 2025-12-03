"""Causal language modeling data prep (e.g., WikiText-2, TinyStories, OpenWebText).

WikiText-2 (Merity et al., 2016) and TinyStories (Gao et al., 2023) are the
primary corpora; this module standardizes how we fetch and tokenize them so the
training scripts can assume reproducible, research-grade preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from transformers import PreTrainedTokenizerBase


# Dataset name -> (hf_name, hf_config, text_field) mapping
DATASET_REGISTRY = {
    "wikitext-2-raw-v1": ("wikitext", "wikitext-2-raw-v1", "text"),
    "wikitext-103-raw-v1": ("wikitext", "wikitext-103-raw-v1", "text"),
    "roneneldan/TinyStories": ("roneneldan/TinyStories", None, "text"),
    "openwebtext": ("Skylion007/openwebtext", None, "text"),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "text"),
}


@dataclass
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


def _parse_dataset_spec(spec: str) -> tuple:
    """Parse dataset spec like 'name:100000' into (name, max_samples).
    
    Examples:
        'wikitext-103-raw-v1' -> ('wikitext-103-raw-v1', None)
        'roneneldan/TinyStories:100000' -> ('roneneldan/TinyStories', 100000)
    """
    if ":" in spec:
        parts = spec.rsplit(":", 1)
        name = parts[0]
        try:
            max_samples = int(parts[1])
        except ValueError:
            # Not a number, treat whole thing as name
            return spec, None
        return name, max_samples
    return spec, None


def load_multi_dataset(
    dataset_names: List[str],
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
    max_length: int = 1024,
    max_samples_per_dataset: Optional[int] = None,
):
    """Load and concatenate multiple datasets for pretraining.
    
    Pre: dataset_names are keys in DATASET_REGISTRY or valid HF dataset paths.
         Supports 'name:N' syntax to cap individual datasets (e.g. 'TinyStories:100000').
    Post: Returns concatenated tokenized dataset.
    
    Args:
        dataset_names: List of dataset identifiers (optionally with :N suffix)
        tokenizer: Tokenizer to use
        split: Dataset split (train/validation)
        max_length: Max sequence length
        max_samples_per_dataset: Global cap for all datasets (per-dataset :N takes precedence)
    """
    from datasets import concatenate_datasets
    
    all_datasets = []
    
    for spec in dataset_names:
        name, per_dataset_cap = _parse_dataset_spec(spec)
        print(f"Loading dataset: {name}" + (f" (capped to {per_dataset_cap})" if per_dataset_cap else ""))
        
        # Look up in registry or use as-is
        if name in DATASET_REGISTRY:
            hf_name, hf_config, text_field = DATASET_REGISTRY[name]
        else:
            # Assume it's a direct HF path
            hf_name = name
            hf_config = None
            text_field = "text"
        
        try:
            config = LanguageModelingDatasetConfig(
                dataset_name=hf_name,
                dataset_config_name=hf_config,
                split=split,
                text_field=text_field,
                max_length=max_length,
            )
            dataset = load_causal_lm_dataset(config, tokenizer)
            
            # Apply per-dataset cap first, then global cap
            cap = per_dataset_cap or max_samples_per_dataset
            if cap and len(dataset) > cap:
                dataset = dataset.select(range(cap))
                print(f"  Capped to {cap} samples")
            
            print(f"  Loaded {len(dataset)} samples from {name}")
            all_datasets.append(dataset)
            
        except Exception as e:
            print(f"  WARNING: Failed to load {name}: {e}")
            continue
    
    if not all_datasets:
        raise ValueError("No datasets were successfully loaded")
    
    # Concatenate all datasets
    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)
    # Re-apply torch format (lost after concatenate/shuffle)
    combined.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print(f"Combined dataset: {len(combined)} total samples")
    
    return combined

