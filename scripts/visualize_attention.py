#!/usr/bin/env python3
"""Attention visualization for interpretability.

Generates heatmap plots showing attention patterns on short examples.
Focus on sentiment words and entity attention patterns.

Usage:
    python scripts/visualize_attention.py --checkpoint path/to/model.pt
    python scripts/visualize_attention.py --checkpoint model.pt --examples custom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Example sentences for visualization
EXAMPLES = {
    "sentiment_positive": "I absolutely love this amazing movie, it's fantastic!",
    "sentiment_negative": "This terrible film was boring and completely unwatchable.",
    "entity": "The quick brown fox jumps over the lazy dog near Paris.",
    "math": "If John has 5 apples and gives 2 to Mary, how many does John have?",
}


def load_model_with_attention(checkpoint_path: str, device: str):
    """Load model configured to output attention weights."""
    from modern_llm.config.model_config import ModernLLMConfig
    from modern_llm.models.transformer import ModernDecoderLM

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        config = ModernLLMConfig(**checkpoint["config"])
    else:
        config = ModernLLMConfig()

    model = ModernDecoderLM(config)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, config


def extract_attention_weights(
    model,
    tokenizer,
    text: str,
    device: str,
    layer_idx: int = -1,
) -> tuple[np.ndarray, list[str]]:
    """Extract attention weights for a given text.
    
    Returns attention matrix and token strings.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Hook to capture attention
    attention_weights = []
    
    def attention_hook(module, input, output):
        # Try to capture attention from the output
        # This depends on how the model returns attention
        if hasattr(output, 'attentions') and output.attentions is not None:
            attention_weights.append(output.attentions)
    
    # For our custom model, we need to extract attention differently
    # We'll do a forward pass and manually compute attention
    with torch.no_grad():
        # Get embeddings
        x = model.token_embed(inputs["input_ids"])
        
        # Collect attention from each layer
        all_attentions = []
        for layer in model.blocks:
            # Access the attention module
            attn = layer.attn
            
            # Get Q, K, V
            seq_len = x.size(1)
            
            # Apply attention and capture weights
            # This requires modifying the forward to return attention
            # For now, we'll compute attention manually
            if hasattr(attn, 'q_proj'):
                Q = attn.q_proj(x)
                K = attn.k_proj(x)
                
                # Reshape for multi-head
                batch, seq, d_model = Q.shape
                n_heads = attn.num_q_heads
                head_dim = attn.head_dim
                
                Q = Q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
                K = K.view(batch, seq, n_heads, head_dim).transpose(1, 2)
                
                # Compute attention scores
                scale = head_dim ** -0.5
                scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                
                # Apply causal mask
                mask = torch.triu(torch.ones(seq, seq, device=device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))
                
                attn_weights = torch.softmax(scores, dim=-1)
                all_attentions.append(attn_weights.cpu().numpy())
            
            # Forward through layer for next iteration
            x = layer(x)
        
        if all_attentions:
            # Use specified layer (default: last)
            attn_matrix = all_attentions[layer_idx]
            # Average across heads
            attn_matrix = attn_matrix.mean(axis=1)[0]  # Remove batch dim, average heads
        else:
            # Fallback: return uniform attention
            n_tokens = len(tokens)
            attn_matrix = np.ones((n_tokens, n_tokens)) / n_tokens
    
    return attn_matrix, tokens


def plot_attention_heatmap(
    attention: np.ndarray,
    tokens: list[str],
    title: str,
    output_path: Path,
    highlight_tokens: Optional[list[str]] = None,
):
    """Generate and save attention heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Clean up token strings for display
    clean_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in tokens]
    
    # Create heatmap
    sns.heatmap(
        attention,
        xticklabels=clean_tokens,
        yticklabels=clean_tokens,
        cmap="Blues",
        ax=ax,
        square=True,
        cbar_kws={"label": "Attention Weight"},
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Key Tokens", fontsize=12)
    ax.set_ylabel("Query Tokens", fontsize=12)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def compute_attention_summary(attention: np.ndarray, tokens: list[str]) -> dict:
    """Compute summary statistics about attention patterns."""
    n_tokens = len(tokens)
    
    # Self-attention (diagonal)
    self_attn = np.diag(attention).mean()
    
    # Local attention (within 3 tokens)
    local_mask = np.abs(np.arange(n_tokens)[:, None] - np.arange(n_tokens)) <= 3
    local_attn = attention[local_mask].mean()
    
    # Attention to first token (often special in transformer)
    first_token_attn = attention[:, 0].mean()
    
    # Entropy (how spread out is attention)
    entropy = -np.sum(attention * np.log(attention + 1e-10), axis=-1).mean()
    
    return {
        "self_attention_avg": float(self_attn),
        "local_attention_avg": float(local_attn),
        "first_token_attention": float(first_token_attn),
        "attention_entropy": float(entropy),
    }


def main():
    parser = argparse.ArgumentParser(description="Attention visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="report/figures")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to visualize (-1 for last)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer, config = load_model_with_attention(args.checkpoint, args.device)
    print(f"  Model: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")

    all_summaries = {}

    for name, text in EXAMPLES.items():
        print(f"\nProcessing: {name}")
        print(f"  Text: {text[:50]}...")
        
        attention, tokens = extract_attention_weights(
            model, tokenizer, text, args.device, args.layer
        )
        
        # Plot heatmap
        title = f"Attention Pattern: {name.replace('_', ' ').title()}"
        output_path = output_dir / f"attention_{name}.png"
        plot_attention_heatmap(attention, tokens, title, output_path)
        
        # Compute summary
        summary = compute_attention_summary(attention, tokens)
        all_summaries[name] = summary
        print(f"  Self-attn: {summary['self_attention_avg']:.3f}, "
              f"Local: {summary['local_attention_avg']:.3f}, "
              f"Entropy: {summary['attention_entropy']:.3f}")

    # Save summary
    import json
    summary_path = output_dir / "attention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Generate markdown observations
    obs_path = output_dir / "attention_observations.md"
    with open(obs_path, "w") as f:
        f.write("# Attention Pattern Observations\n\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Layer visualized: {args.layer} (negative = from end)\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("| Example | Self-Attn | Local-Attn | First-Token | Entropy |\n")
        f.write("|---------|-----------|------------|-------------|--------|\n")
        for name, s in all_summaries.items():
            f.write(f"| {name} | {s['self_attention_avg']:.3f} | "
                    f"{s['local_attention_avg']:.3f} | {s['first_token_attention']:.3f} | "
                    f"{s['attention_entropy']:.3f} |\n")
        
        f.write("\n## Observations\n\n")
        f.write("- Higher entropy indicates more distributed attention\n")
        f.write("- Self-attention measures diagonal concentration\n")
        f.write("- Local attention measures nearby token focus\n")
        f.write("- First-token attention often reflects BOS/sink behavior\n")
    
    print(f"Observations saved to {obs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

