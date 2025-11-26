"""Verifier network for post-hoc scoring.

The verifier follows the trend of training lightweight judges for math/QA
(e.g., Cobbe et al., 2021; Lightman et al., 2023). It is a Transformer encoder
that predicts correctness logits for generated solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
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
    dropout: float = 0.1

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
        self.position_embed = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small variance."""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.position_embed.weight, std=0.02)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Forward pass for verifier.

        Pre:
            - input_ids: (batch, seq) token indices
            - attention_mask: (batch, seq) with 1 for real tokens, 0 for padding
            - labels: (batch,) with class indices (0=incorrect, 1=correct)
        Post:
            - Returns dict with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape

        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        # Embeddings
        token_embeds = self.token_embed(input_ids)
        position_embeds = self.position_embed(positions)
        hidden = self.embed_dropout(token_embeds + position_embeds)

        # Create attention mask for transformer (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        # Encoder forward
        encoded = self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)

        # Pool: use CLS token (first position) or mean pooling
        # We use mean pooling over non-padded tokens for robustness
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        logits = self.classifier(pooled)

        result = {"logits": logits}

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            result["loss"] = loss

        return result

    def score(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return probability of correctness.

        Pre: input_ids is a (batch, seq) tensor.
        Post: Returns (batch,) tensor of correctness probabilities in [0, 1].
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
            # Return probability of class 1 (correct)
            return probs[:, 1]

    def predict(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return predicted class (0=incorrect, 1=correct).

        Pre: input_ids is a (batch, seq) tensor.
        Post: Returns (batch,) tensor of class predictions.
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs["logits"].argmax(dim=-1)
