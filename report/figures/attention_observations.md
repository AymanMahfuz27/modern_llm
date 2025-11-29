# Attention Pattern Observations

Model: experiments/runs/smoke-test/smoke-test-dpo/smoke-test-dpo_final.pt
Layer visualized: -1 (negative = from end)

## Summary Statistics

| Example | Self-Attn | Local-Attn | First-Token | Entropy |
|---------|-----------|------------|-------------|--------|
| sentiment_positive | 0.275 | 0.119 | 0.275 | 1.591 |
| sentiment_negative | 0.275 | 0.119 | 0.275 | 1.591 |
| entity | 0.259 | 0.112 | 0.259 | 1.666 |
| math | 0.202 | 0.088 | 0.202 | 1.971 |

## Observations

- Higher entropy indicates more distributed attention
- Self-attention measures diagonal concentration
- Local attention measures nearby token focus
- First-token attention often reflects BOS/sink behavior
