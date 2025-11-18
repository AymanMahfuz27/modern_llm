import pytest
import torch

from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
from modern_llm.training.train_lm import generate_text


class DummyTokenizer:
    """Minimal tokenizer stub for generation tests."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, return_tensors: str = "pt"):
        if not text:
            return torch.empty(1, 0, dtype=torch.long)
        # Encode each whitespace-separated token as a simple index.
        token_ids = [2 + (i % 3) for i, _ in enumerate(text.split())]
        ids = torch.tensor([token_ids], dtype=torch.long)
        if return_tensors == "pt":
            return ids
        return ids.tolist()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # Map back to a fixed token to avoid depending on specific ids.
        return " ".join("tok" for _ in ids)


def _make_small_model() -> ModernDecoderLM:
    config = ModernLLMConfig(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=16,
    )
    return ModernDecoderLM(config)


def test_generate_text_basic() -> None:
    model = _make_small_model()
    tokenizer = DummyTokenizer()
    text = generate_text(model, tokenizer, prompt="hello world", max_new_tokens=4)
    assert isinstance(text, str)
    assert text  # non-empty


def test_generate_text_rejects_non_positive_max_tokens() -> None:
    model = _make_small_model()
    tokenizer = DummyTokenizer()
    with pytest.raises(ValueError):
        generate_text(model, tokenizer, prompt="hello", max_new_tokens=0)


