"""Checkpoint management helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _strip_orig_mod_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize state dict keys that come from compiled models.

    When using ``torch.compile``, PyTorch wraps the original module and
    prefixes all parameter keys with ``\"_orig_mod.\"``. We want checkpoints
    to be loadable into the plain, non-compiled model, so we strip this
    prefix when present.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict

    needs_strip = any(isinstance(k, str) and k.startswith("_orig_mod.") for k in state_dict.keys())
    if not needs_strip:
        return state_dict

    normalized: Dict[str, Any] = {}
    prefix = "_orig_mod."
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith(prefix):
            normalized[key[len(prefix) :]] = value
        else:
            normalized[key] = value
    return normalized


def save_checkpoint(
    path: Path,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    **metadata: Any,
) -> None:
    """Persist model/optimizer state along with optional metadata.

    Pre:
        - path points to desired checkpoint file location.
        - model_state contains tensors on CPU or GPU-resident (torch.save handles both).
    Post:
        - file written to disk with model_state, optimizer, and metadata.
    Complexity:
        - O(total_parameter_count) due to serialization.
    """

    if not isinstance(path, Path):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model_state, "metadata": metadata}
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    for k, v in metadata.items():
        if k not in payload:
            payload[k] = v
    torch.save(payload, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a checkpoint from disk.

    Pre:
        - path exists and was created via `save_checkpoint`.
    Post:
        - returns dict containing `model_state`, optional `optimizer`, and metadata.
        - if the checkpoint was saved from a compiled model, parameter keys are
          normalized so they can be loaded into a non-compiled module.
    Complexity:
        - O(file_size) for deserialization.
    """

    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    payload = torch.load(path, map_location="cpu")

    # Normalize compiled-model checkpoints so they can be loaded into plain modules.
    if isinstance(payload, dict) and "model_state" in payload and isinstance(payload["model_state"], dict):
        payload["model_state"] = _strip_orig_mod_prefix(payload["model_state"])

    return payload

