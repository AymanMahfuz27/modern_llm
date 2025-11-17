"""Verifier network for post-hoc scoring.

The verifier follows the trend of training lightweight judges for math/QA
(e.g., Cobbe et al., 2021; Lightman et al., 2023). It is a Transformer encoder
that predicts correctness logits for generated solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor, nn


@dataclass(slots=True)
class VerifierConfig:
    """Hyperparameters for the verifier encoder."""
    vocab_size: int
    d_model: int = 512
    num_layers: int = 4
    n_heads: int = 8
    max_position_embeddings: int = 512
    num_classes: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >= 2.")


class VerifierModel(nn.Module):
    """Encoder-only verifier producing correctness logits.

    Math:
        h₀ = token_embed(input_ids) + positional_embed
        h_L = TransformerEncoder(h₀)
        logits = W h_cls
        Loss = cross_entropy(logits, labels)

    Pre:
        - input_ids shape: (batch, seq)
    Post:
        - logits shape: (batch, num_classes)
    Complexity:
        - O(num_layers · seq² · d_model / n_heads).
    Invariants:
        - Classifier weights share d_model dimension with encoder outputs.

    This mirrors encoder stacks from BERT/RoBERTa but with a smaller footprint.
    """

    def __init__(self, config: VerifierConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Placeholder until the tokenizer wiring is implemented.

        Pre:
            - attention_mask is broadcast-compatible or None.
        Post:
            - logits tensor once implemented.
        """

        raise NotImplementedError("Verifier forward pass will be implemented once tokenizer wiring is ready.")

