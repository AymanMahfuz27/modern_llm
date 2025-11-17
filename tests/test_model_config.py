import pytest

from modern_llm.config import ModernLLMConfig, MoEConfig


def test_invalid_head_configuration_raises() -> None:
    with pytest.raises(ValueError):
        ModernLLMConfig(
            vocab_size=100,
            d_model=63,
            n_layers=2,
            n_heads=8,
            ffn_hidden_size=256,
            max_seq_len=128,
        )


def test_moe_requires_config_when_enabled() -> None:
    with pytest.raises(ValueError):
        ModernLLMConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=8,
            ffn_hidden_size=256,
            max_seq_len=128,
            use_moe=True,
        )


def test_valid_moe_configuration_passes() -> None:
    config = ModernLLMConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=8,
        ffn_hidden_size=256,
        max_seq_len=128,
        use_moe=True,
        moe_config=MoEConfig(num_experts=2, top_k=1),
    )
    assert config.use_moe is True

