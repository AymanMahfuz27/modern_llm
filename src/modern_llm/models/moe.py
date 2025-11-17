"""Mixture-of-Experts components following Shazeer et al. (2017).

We will eventually match Switch Transformer / GLaM style routing where the
router predicts expert weights and dispatches each token to the top-k experts.
Documenting the math now keeps the file aligned with the research literature.
"""

from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn

from modern_llm.config.model_config import MoEConfig


class TopKRouter(nn.Module):
    """Top-k gating distribution (Shazeer et al., 2017, Eq. 1)."""

    def __init__(self, dim: int, config: MoEConfig) -> None:
        super().__init__()
        self.dim = dim
        self.config = config
        self.router = nn.Linear(dim, config.num_experts, bias=False)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (topk_scores, topk_indices) once routing is implemented.

        Pre:
            - hidden_states shape: (batch, seq, dim)
        Post:
            - scores shape: (batch, seq, top_k)
            - indices shape: (batch, seq, top_k)
        Complexity:
            - O(dim * num_experts) per token for the linear projection.
        """

        raise NotImplementedError("TopKRouter forward will be implemented in the advanced phase.")


class MixtureOfExperts(nn.Module):
    """Expert network container (Fedus et al., 2021 Switch Transformer).

    Pre:
        - hidden_states.shape[-1] == dim.
    Post:
        - returns tensor with same shape after routing/aggregation.
    Complexity:
        - O(top_k · dim · 4dim) per token once routing is active.
    Invariants:
        - Expert weights are shared across tokens.
    """

    def __init__(self, dim: int, moe_config: MoEConfig) -> None:
        super().__init__()
        self.dim = dim
        self.moe_config = moe_config
        self.router = TopKRouter(dim, moe_config)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
            for _ in range(moe_config.num_experts)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        raise NotImplementedError("MixtureOfExperts forward pass will be implemented with routing logic later.")

