# Attention Pattern Observations

Model: /work/09999/aymanmahfuz/ls6/modern_llm_runs/checkpoints/gpu-full-dpo/gpu-full-dpo_final.pt
Layer visualized: -1 (negative = from end)

## Summary Statistics

| Example | Self-Attn | Local-Attn | First-Token | Entropy |
|---------|-----------|------------|-------------|--------|
| sentiment_positive | 0.276 | 0.118 | 0.286 | 1.589 |
| sentiment_negative | 0.285 | 0.118 | 0.267 | 1.588 |
| entity | 0.235 | 0.107 | 0.319 | 1.649 |
| math | 0.187 | 0.082 | 0.232 | 1.957 |

## Observations

- Higher entropy indicates more distributed attention
- Self-attention measures diagonal concentration
- Local attention measures nearby token focus
- First-token attention often reflects BOS/sink behavior
