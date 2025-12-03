#!/bin/bash
#SBATCH -J llm-diagnose
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --exclude=c318-001
#SBATCH -A ASC25078

# =============================================================================
# Diagnostic Script - Check model quality and training
# =============================================================================

set -e

export PROJECT_DIR="${HOME}/modern_llm"
export WORK_DIR="${WORK}/modern_llm_runs"

cd "${PROJECT_DIR}"
module load cuda/12.2
source .venv/bin/activate
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

echo "=============================================="
echo "Model Diagnostics"
echo "=============================================="
echo "Time: $(date)"
echo ""

# Run diagnostic script
python - << 'DIAG_PY'
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from modern_llm.config.model_config import ModernLLMConfig
from modern_llm.models.transformer import ModernDecoderLM

WORK_DIR = Path("/work/09999/aymanmahfuz/ls6/modern_llm_runs")
CKPT_DIR = WORK_DIR / "checkpoints"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def load_model(ckpt_path):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).copy()
    
    # Normalize keys
    if "num_layers" in cfg and "n_layers" not in cfg:
        cfg["n_layers"] = cfg.pop("num_layers")
    if "max_position_embeddings" in cfg and "max_seq_len" not in cfg:
        cfg["max_seq_len"] = cfg.pop("max_position_embeddings")
    
    valid_keys = {
        "vocab_size", "d_model", "n_layers", "n_heads", "ffn_hidden_size",
        "max_seq_len", "rmsnorm_eps", "dropout", "initializer_range",
        "rope_theta", "rope_scaling", "use_rope", "use_attention_sinks",
        "num_attention_sinks", "use_swiglu", "swiglu_multiplier", "use_gqa",
        "gqa_groups", "use_moe", "moe_config", "tie_embeddings",
    }
    cfg = {k: v for k, v in cfg.items() if k in valid_keys}
    
    config = ModernLLMConfig(**cfg)
    model = ModernDecoderLM(config)
    
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    # Strip _orig_mod prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def generate_text(model, prompt, max_tokens=100, temperature=0.7):
    """Generate text from prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def evaluate_sst2_multi_prompt(model, max_samples=100):
    """Try multiple SST-2 prompt formats."""
    from datasets import load_dataset
    
    ds = load_dataset("glue", "sst2", split="validation")
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    
    prompts = {
        "original": lambda t: f'Review: "{t}"\nSentiment:',
        "simple": lambda t: f"{t}\nThis is",
        "question": lambda t: f'Is this review positive or negative? "{t}"\nAnswer:',
        "completion": lambda t: f'The sentiment of "{t}" is',
        "binary": lambda t: f"{t}\n\nPositive or Negative?",
    }
    
    results = {}
    
    for name, prompt_fn in prompts.items():
        correct = 0
        pos_tokens = tokenizer.encode(" positive", add_special_tokens=False)
        neg_tokens = tokenizer.encode(" negative", add_special_tokens=False)
        pos_id = pos_tokens[0] if pos_tokens else tokenizer.encode("positive")[0]
        neg_id = neg_tokens[0] if neg_tokens else tokenizer.encode("negative")[0]
        
        for ex in ds:
            text = ex["sentence"]
            label = ex["label"]  # 0=neg, 1=pos
            
            prompt = prompt_fn(text)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                next_logits = logits[0, -1, :]
            
            pos_prob = next_logits[pos_id].item()
            neg_prob = next_logits[neg_id].item()
            pred = 1 if pos_prob > neg_prob else 0
            
            if pred == label:
                correct += 1
        
        acc = correct / len(ds)
        results[name] = acc
        print(f"  {name}: {acc:.1%}")
    
    return results


print("=" * 60)
print("1. SAMPLE GENERATIONS")
print("=" * 60)

checkpoints = {
    "pretrain": CKPT_DIR / "gpu-full-pretrain/gpu-full-pretrain_final.pt",
    "sft": CKPT_DIR / "gpu-full-sft/gpu-full-sft_final.pt",
    "dpo": CKPT_DIR / "gpu-full-dpo/gpu-full-dpo_final.pt",
}

test_prompts = [
    "Once upon a time",
    "The capital of France is",
    "In machine learning, a neural network",
    "Question: What is 2 + 2?\nAnswer:",
]

for stage, ckpt_path in checkpoints.items():
    if not ckpt_path.exists():
        print(f"\n[{stage}] Checkpoint not found: {ckpt_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Stage: {stage.upper()}")
    print(f"{'='*60}")
    
    model = load_model(ckpt_path)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        try:
            output = generate_text(model, prompt, max_tokens=50, temperature=0.7)
            # Show just the generated part
            generated = output[len(prompt):].strip()
            print(f"Output: {generated[:200]}")
        except Exception as e:
            print(f"Error: {e}")
    
    del model
    torch.cuda.empty_cache()


print("\n")
print("=" * 60)
print("2. TRAINING LOSS ANALYSIS")
print("=" * 60)

# Check for training logs
log_dir = WORK_DIR / "logs"
for log_file in sorted(log_dir.glob("pipeline_*.log")):
    print(f"\nAnalyzing: {log_file.name}")
    
    losses = {"pretrain": [], "sft": [], "dpo": []}
    current_stage = None
    
    with open(log_file) as f:
        for line in f:
            if "pretrain" in line.lower() and "stage" in line.lower():
                current_stage = "pretrain"
            elif "sft" in line.lower() and "stage" in line.lower():
                current_stage = "sft"
            elif "dpo" in line.lower() and "stage" in line.lower():
                current_stage = "dpo"
            
            # Look for loss patterns like "loss: 3.456" or "Loss: 3.456"
            loss_match = re.search(r"[Ll]oss[:\s]+(\d+\.?\d*)", line)
            if loss_match and current_stage:
                try:
                    loss = float(loss_match.group(1))
                    if 0 < loss < 100:  # sanity check
                        losses[current_stage].append(loss)
                except ValueError:
                    pass
    
    for stage, loss_list in losses.items():
        if loss_list:
            print(f"\n  {stage.upper()}:")
            print(f"    Start loss: {loss_list[0]:.4f}")
            print(f"    End loss:   {loss_list[-1]:.4f}")
            print(f"    Min loss:   {min(loss_list):.4f}")
            print(f"    Samples:    {len(loss_list)}")
            if len(loss_list) > 1:
                delta = loss_list[-1] - loss_list[0]
                print(f"    Delta:      {delta:+.4f} ({'improved' if delta < 0 else 'WORSENED'})")


print("\n")
print("=" * 60)
print("3. SST-2 WITH MULTIPLE PROMPTS")
print("=" * 60)

# Test with DPO model (final)
dpo_path = CKPT_DIR / "gpu-full-dpo/gpu-full-dpo_final.pt"
if dpo_path.exists():
    print("\nLoading DPO model for SST-2 evaluation...")
    model = load_model(dpo_path)
    
    print("\nTrying different prompt formats (100 samples):\n")
    results = evaluate_sst2_multi_prompt(model, max_samples=100)
    
    best_prompt = max(results, key=results.get)
    print(f"\nBest prompt format: '{best_prompt}' with {results[best_prompt]:.1%} accuracy")
    
    del model
    torch.cuda.empty_cache()


print("\n")
print("=" * 60)
print("4. CHINCHILLA ANALYSIS")
print("=" * 60)

# Chinchilla optimal: N ≈ 20 * D (params ≈ 20 * tokens)
# Our model: ~253M params
# Chinchilla optimal tokens: 253M / 20 = 12.6M tokens (way less than we used!)
# We used: 40K steps * 128 batch * 1024 seq = 5.2B tokens

model_params = 253_000_000
tokens_seen = 40_000 * 128 * 1024  # ~5.2B

chinchilla_optimal_tokens = model_params * 20
chinchilla_optimal_params = tokens_seen / 20

print(f"Our model parameters:     {model_params/1e6:.0f}M")
print(f"Tokens seen:              {tokens_seen/1e9:.1f}B")
print(f"")
print(f"Chinchilla analysis:")
print(f"  For {model_params/1e6:.0f}M params, optimal tokens: {chinchilla_optimal_tokens/1e9:.2f}B")
print(f"  For {tokens_seen/1e9:.1f}B tokens, optimal params:  {chinchilla_optimal_params/1e9:.1f}B")
print(f"")
print(f"Ratio (actual/optimal tokens): {tokens_seen/chinchilla_optimal_tokens:.1f}x")
print(f"")

if tokens_seen > chinchilla_optimal_tokens * 2:
    print("⚠️  OVERTRAINED: Model saw way more tokens than Chinchilla suggests.")
    print("   This can lead to diminishing returns but shouldn't cause forgetting.")
    print("   More likely issues: learning rate decay, data quality, or architecture.")
elif tokens_seen < chinchilla_optimal_tokens * 0.5:
    print("⚠️  UNDERTRAINED: Model needs more data for its size.")
else:
    print("✓  Training budget is roughly Chinchilla-optimal.")


print("\n")
print("=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
print(f"Time: {__import__('datetime').datetime.now()}")
DIAG_PY

echo ""
echo "Diagnostics complete!"

