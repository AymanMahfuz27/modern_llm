"""Checkpoint management helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


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
        - returns dict containing `model`, optional `optimizer`, and metadata.
    Complexity:
        - O(file_size) for deserialization.
    """

    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(path, map_location="cpu")

