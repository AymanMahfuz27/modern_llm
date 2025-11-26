#!/bin/bash
# Compare YOUR scratch LM vs GPT-2 (same size ~125M params)
# This is what you actually want - direct comparison on the same benchmarks
# Run: bash scripts/compare_scratch_vs_gpt2.sh

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
source .venv/bin/activate

echo "============================================================"
echo "YOUR MODEL vs GPT-2 COMPARISON"
echo "============================================================"
echo ""

# Find your best checkpoint
CKPT_DIR="experiments/runs/scratch-wikitext2-max-rtx3060"
BEST_CKPT=$(ls -t "$CKPT_DIR"/*.pt 2>/dev/null | head -1)
echo "Your checkpoint: $BEST_CKPT"
echo ""

python << 'PYTHON_SCRIPT'
import torch
import math
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
from modern_llm.training.train_lm import generate_text
import os

CKPT_PATH = os.environ.get('BEST_CKPT', 'experiments/runs/scratch-wikitext2-max-rtx3060/scratch-wikitext2-max-rtx3060_step16000.pt')

# ============================================================
# Load YOUR scratch model
# ============================================================
print("Loading YOUR scratch model...")
config = ModernLLMConfig(
    vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
    ffn_hidden_size=3072, max_seq_len=1024, rope_theta=10000.0,
    dropout=0.0, use_rope=True, use_attention_sinks=True,
    num_attention_sinks=4, use_swiglu=True, tie_embeddings=True,
)
ckpt = torch.load(CKPT_PATH, map_location='cuda', weights_only=False)
scratch_model = ModernDecoderLM(config)
scratch_model.load_state_dict(ckpt['model_state'])
scratch_model.to('cuda').eval()

scratch_params = sum(p.numel() for p in scratch_model.parameters())
print(f"  Parameters: {scratch_params:,} ({scratch_params/1e6:.1f}M)")

# ============================================================
# Load GPT-2 (similar size baseline)
# ============================================================
print("\nLoading GPT-2 (baseline)...")
gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda').eval()
gpt2_params = sum(p.numel() for p in gpt2_model.parameters())
print(f"  Parameters: {gpt2_params:,} ({gpt2_params/1e6:.1f}M)")

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 1. PERPLEXITY COMPARISON (WikiText-2)
# ============================================================
print("\n" + "="*60)
print("1. PERPLEXITY on WikiText-2 Validation")
print("="*60)

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
text = ' '.join([x for x in dataset['text'] if x.strip()])
encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=50000)
input_ids = encodings.input_ids[0]

def compute_perplexity(model, input_ids, max_len=1024, stride=512, is_scratch=False):
    nlls = []
    seq_len = min(input_ids.size(0), 20000)  # Limit for speed
    
    for i in range(0, seq_len - max_len, stride):
        chunk = input_ids[i:i+max_len].unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            if is_scratch:
                outputs = model(chunk)
                logits = outputs['logits']
            else:
                outputs = model(chunk)
                logits = outputs.logits
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
            
            loss = torch.nn.CrossEntropyLoss()(
                shift_logits.view(-1, logits.size(-1)), 
                shift_labels.view(-1)
            )
            nlls.append(loss.item())
    
    return math.exp(sum(nlls) / len(nlls))

print("\nComputing perplexity (this takes a minute)...")
scratch_ppl = compute_perplexity(scratch_model, input_ids, is_scratch=True)
gpt2_ppl = compute_perplexity(gpt2_model, input_ids, is_scratch=False)

print(f"\n  YOUR MODEL:  {scratch_ppl:.2f}")
print(f"  GPT-2:       {gpt2_ppl:.2f}")

# ============================================================
# 2. GENERATION QUALITY COMPARISON
# ============================================================
print("\n" + "="*60)
print("2. GENERATION QUALITY")
print("="*60)

prompts = [
    "The history of artificial intelligence began",
    "In a groundbreaking scientific discovery,",
    "The economic implications of",
]

def generate_gpt2(model, tokenizer, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

generations = []
for prompt in prompts:
    print(f"\nPROMPT: {prompt}")
    print("-" * 40)
    
    scratch_out = generate_text(scratch_model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=50)
    gpt2_out = generate_gpt2(gpt2_model, tokenizer, prompt)
    
    print(f"YOUR MODEL: {scratch_out[:200]}...")
    print(f"GPT-2:      {gpt2_out[:200]}...")
    
    generations.append({
        'prompt': prompt,
        'scratch': scratch_out,
        'gpt2': gpt2_out,
    })

# ============================================================
# 3. SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"""
┌─────────────────────┬──────────────────┬──────────────────┐
│ Metric              │ YOUR MODEL       │ GPT-2 (baseline) │
├─────────────────────┼──────────────────┼──────────────────┤
│ Parameters          │ {scratch_params/1e6:>13.1f}M   │ {gpt2_params/1e6:>13.1f}M   │
│ WikiText-2 PPL      │ {scratch_ppl:>16.2f} │ {gpt2_ppl:>16.2f} │
│ Architecture        │ Modern (RoPE,    │ Original 2019    │
│                     │ RMSNorm, SwiGLU, │ (sinusoidal pos, │
│                     │ Attn Sinks, GQA) │ LayerNorm, GELU) │
└─────────────────────┴──────────────────┴──────────────────┘
""")

# Save results
results = {
    'scratch_model': {
        'checkpoint': CKPT_PATH,
        'parameters': scratch_params,
        'perplexity': scratch_ppl,
        'architecture': ['RoPE', 'RMSNorm', 'SwiGLU', 'Attention Sinks', 'GQA-ready'],
    },
    'gpt2_baseline': {
        'parameters': gpt2_params,
        'perplexity': gpt2_ppl,
        'architecture': ['Sinusoidal Pos', 'LayerNorm', 'GELU'],
    },
    'generations': generations,
}

with open('experiments/model_comparison.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("Results saved to experiments/model_comparison.json")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "COMPARISON COMPLETE!"
echo "============================================================"

