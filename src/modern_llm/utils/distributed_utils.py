"""Helpers for safe distributed initialization."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed_mode() -> Tuple[int, int]:
    """Initialize torch.distributed if environment variables are set.

    Pre:
        - Environment variables RANK and WORLD_SIZE optionally defined.
    Post:
        - Returns (rank, world_size); initializes process group if needed.
    Complexity:
        - O(1) aside from torch.distributed initialization handshake.
    """

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return rank, world_size

