# Attention Pattern Analysis

**Model**: ModernDecoderLM (253M params, d=1024, L=12, H=16)  
**Checkpoint**: DPO-aligned model  
**Layer Visualized**: Final layer (-1)

---

## Summary Statistics

| Example | Self-Attn | Local-Attn | First-Token | Entropy |
|---------|-----------|------------|-------------|---------|
| sentiment_positive | 0.276 | 0.118 | 0.286 | 1.589 |
| sentiment_negative | 0.285 | 0.118 | 0.267 | 1.588 |
| entity | 0.235 | 0.107 | 0.319 | 1.649 |
| math | 0.187 | 0.082 | 0.232 | 1.957 |

---

## Key Observations

### 1. Math Reasoning Requires Distributed Attention

The math example ("If John has 5 apples...") shows the **highest entropy (1.957)** and **lowest self-attention (0.187)**. This indicates the model distributes attention broadly across tokens when processing multi-step reasoning, rather than focusing narrowly. This aligns with findings from chain-of-thought literature showing that reasoning tasks require integrating information across the full context.

Conversely, the **lowest local attention (0.082)** for math suggests the model isn't just looking at adjacent tokens but rather distant dependencies (e.g., connecting "5 apples" early in the sentence to "how many" at the end).

### 2. Sentiment Examples Show Consistent Patterns

Both sentiment examples (positive and negative) exhibit nearly identical attention statistics:
- Self-attention: 0.276 vs 0.285 (Δ=0.009)
- Local-attention: 0.118 vs 0.118 (Δ=0.000)
- Entropy: 1.589 vs 1.588 (Δ=0.001)

This suggests the model processes sentiment polarity through **semantic content** rather than **structural attention patterns**. The model doesn't need to attend differently for positive vs. negative—it reads similarly and decides based on the actual tokens ("love" vs "terrible").

### 3. Entity Examples Show Strong First-Token Attention

The entity example ("The quick brown fox...") has the **highest first-token attention (0.319)**. In causal language models, the first token often serves as a "sink" that aggregates global context (Xiao et al., 2023). The strong first-token attention here suggests the model uses this mechanism to track entity references across the sentence.

The relatively **lower self-attention (0.235)** for entities compared to sentiment indicates the model looks beyond the current position to resolve references—consistent with how coreference resolution works.

### 4. Entropy as a Complexity Indicator

The entropy values form a clear ordering:
1. **Math (1.957)** - Most complex, broadest attention
2. **Entity (1.649)** - Medium complexity, reference tracking
3. **Sentiment (1.589)** - Simplest, focused pattern

This ordering matches intuitive task complexity: math requires multi-step reasoning, entities require tracking across positions, and sentiment primarily requires recognizing key words.

---

## Implications for Model Behavior

1. **Why SST-2 accuracy is near-random**: The identical attention patterns for positive/negative sentiment suggest the final layer representations are similar. The model may lack discriminative features at the embedding level, causing few-shot classification to struggle.

2. **Why GSM8K fails**: Despite broad attention (high entropy), the 253M model lacks the capacity for multi-step arithmetic reasoning. The attention mechanism is working correctly (distributing broadly), but the representations aren't rich enough for computation.

3. **First-token sink behavior**: The consistently high first-token attention (0.23-0.32) indicates our RoPE + causal masking setup correctly implements attention sink patterns, even with sinks disabled for Flash Attention compatibility.

---

## Methodology Notes

- **Self-attention**: Average of diagonal elements (token attending to itself)
- **Local-attention**: Average attention within 3-token window
- **First-token attention**: Average attention to position 0
- **Entropy**: Shannon entropy of attention distribution (higher = more uniform)

All statistics computed from the final transformer layer, averaged across attention heads.
