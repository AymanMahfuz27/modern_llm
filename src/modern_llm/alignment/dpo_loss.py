"""Direct Preference Optimization loss helper.

Implements the DPO objective from Rafailov et al. (2023), Eq. 2:

    L = -log σ(β · (log πθ(y⁺|x) - log πθ(y⁻|x) - log π_ref(y⁺|x) + log π_ref(y⁻|x)))

In the single-model setting we omit the reference policy subtraction and keep the
β temperature, which still encourages higher log-prob for preferred responses.
"""

from __future__ import annotations

import torch
from torch import Tensor


def dpo_loss(
    chosen_logprobs: Tensor,
    rejected_logprobs: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Compute L = -log σ(β · (Δ log π)) with Δ log π = log π(y⁺) - log π(y⁻).

    Pre:
        - chosen_logprobs and rejected_logprobs share identical shapes.
        - beta > 0 to keep the temperature meaningful.
    Post:
        - returns a scalar suitable for `loss.backward()`.
    Complexity:
        - O(N) where N is the number of preference pairs.
    Invariants:
        - No grad flows through beta (scalar hyperparameter).
    """

    if chosen_logprobs.shape != rejected_logprobs.shape:
        raise ValueError(
            "chosen_logprobs and rejected_logprobs must have the same shape "
            f"(got {chosen_logprobs.shape} vs {rejected_logprobs.shape})"
        )
    if beta <= 0:
        raise ValueError(f"beta must be positive, received {beta}")
    preference_margins = chosen_logprobs - rejected_logprobs
    losses = -torch.nn.functional.logsigmoid(beta * preference_margins)
    return losses.mean()

