#!/bin/bash
# Evaluate scratch LM: perplexity, generation quality, attention sinks
# Run from project root: bash scripts/evaluate_scratch_lm.sh

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
source .venv/bin/activate

echo "============================================================"
echo "Scratch LM Evaluation Pipeline"
echo "============================================================"
echo ""

# Find the best checkpoint (max-size model)
CKPT_DIR="experiments/runs/scratch-wikitext2-max-rtx3060"
if [ -d "$CKPT_DIR" ]; then
    BEST_CKPT=$(ls -t "$CKPT_DIR"/*.pt 2>/dev/null | head -1)
    echo "Found checkpoint: $BEST_CKPT"
else
    echo "ERROR: No scratch LM checkpoint found at $CKPT_DIR"
    exit 1
fi

# ============================================================
# 1. Perplexity Evaluation on WikiText-2 validation
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] 1. Evaluating perplexity on WikiText-2 validation..."
python -c "
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
import math

# Config matching lm_max_rtx3060.json
config = ModernLLMConfig(
    vocab_size=50257,
    d_model=768,
    n_layers=12,
    n_heads=12,
    ffn_hidden_size=3072,
    max_seq_len=1024,
    rope_theta=10000.0,
    dropout=0.0,  # No dropout for eval
    use_rope=True,
    use_attention_sinks=True,
    num_attention_sinks=4,
    use_swiglu=True,
    tie_embeddings=True,
)

print('Loading checkpoint...')
ckpt = torch.load('$BEST_CKPT', map_location='cuda', weights_only=False)
model = ModernDecoderLM(config)
model.load_state_dict(ckpt['model_state'])
model.to('cuda').eval()

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print('Loading WikiText-2 validation set...')
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
text = ' '.join([x for x in dataset['text'] if x.strip()])

# Tokenize and compute perplexity
encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=config.max_seq_len * 100)
input_ids = encodings.input_ids[0]

# Sliding window perplexity
stride = 512
seq_len = input_ids.size(0)
nlls = []

print(f'Computing perplexity over {seq_len} tokens...')
for i in range(0, seq_len - config.max_seq_len, stride):
    end = min(i + config.max_seq_len, seq_len)
    chunk = input_ids[i:end].unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        outputs = model(chunk)
        logits = outputs['logits']
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
        nlls.append(loss.item())

ppl = math.exp(sum(nlls) / len(nlls))
print(f'')
print(f'=== PERPLEXITY RESULTS ===')
print(f'WikiText-2 Validation Perplexity: {ppl:.2f}')
print(f'Average NLL: {sum(nlls)/len(nlls):.4f}')
print(f'Evaluated on {len(nlls)} chunks of {config.max_seq_len} tokens')

# Save results
import json
results = {
    'checkpoint': '$BEST_CKPT',
    'perplexity': ppl,
    'avg_nll': sum(nlls)/len(nlls),
    'num_chunks': len(nlls),
    'seq_len': config.max_seq_len,
}
with open('experiments/scratch_lm_perplexity.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to experiments/scratch_lm_perplexity.json')
"

# ============================================================
# 2. Text Generation Samples
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] 2. Generating text samples..."
python -c "
import torch
from transformers import AutoTokenizer
from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
from modern_llm.training.train_lm import generate_text
import json

config = ModernLLMConfig(
    vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
    ffn_hidden_size=3072, max_seq_len=1024, rope_theta=10000.0,
    dropout=0.0, use_rope=True, use_attention_sinks=True,
    num_attention_sinks=4, use_swiglu=True, tie_embeddings=True,
)

ckpt = torch.load('$BEST_CKPT', map_location='cuda', weights_only=False)
model = ModernDecoderLM(config)
model.load_state_dict(ckpt['model_state'])
model.to('cuda').eval()

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    'The history of artificial intelligence',
    'In the year 2050, humanity',
    'The scientific method involves',
    'Once upon a time in a distant kingdom',
    'The economic impact of technology',
]

print('=== GENERATION SAMPLES ===')
print('')
samples = []
for prompt in prompts:
    output = generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50)
    print(f'PROMPT: {prompt}')
    print(f'OUTPUT: {output}')
    print('-' * 60)
    samples.append({'prompt': prompt, 'output': output})

with open('experiments/scratch_lm_generations.json', 'w') as f:
    json.dump(samples, f, indent=2)
print(f'Saved to experiments/scratch_lm_generations.json')
"

# ============================================================
# 3. Attention Sinks Comparison (short context vs long)
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] 3. Testing attention sinks (long context stability)..."
python -c "
import torch
from transformers import AutoTokenizer
from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
from modern_llm.training.train_lm import generate_text
import json

config = ModernLLMConfig(
    vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
    ffn_hidden_size=3072, max_seq_len=1024, rope_theta=10000.0,
    dropout=0.0, use_rope=True, use_attention_sinks=True,
    num_attention_sinks=4, use_swiglu=True, tie_embeddings=True,
)

ckpt = torch.load('$BEST_CKPT', map_location='cuda', weights_only=False)
model = ModernDecoderLM(config)
model.load_state_dict(ckpt['model_state'])
model.to('cuda').eval()

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Test: Generate with increasingly long context
print('=== ATTENTION SINKS TEST ===')
print('Testing generation stability with increasing context length...')
print('')

base_prompt = 'The development of modern computing began with'
results = []

for context_len in [50, 200, 500, 800]:
    # Generate some context first, then continue
    output = generate_text(model, tokenizer, base_prompt, max_new_tokens=context_len, temperature=0.7, top_k=50)
    
    # Check for repetition (sign of instability)
    words = output.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    
    print(f'Context length ~{context_len} tokens:')
    print(f'  Unique word ratio: {unique_ratio:.2%}')
    print(f'  Last 50 chars: ...{output[-50:]}')
    print('')
    
    results.append({
        'context_length': context_len,
        'unique_word_ratio': unique_ratio,
        'output_preview': output[-100:],
    })

with open('experiments/scratch_lm_attention_sinks.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to experiments/scratch_lm_attention_sinks.json')
"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Scratch LM Evaluation COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - experiments/scratch_lm_perplexity.json"
echo "  - experiments/scratch_lm_generations.json"
echo "  - experiments/scratch_lm_attention_sinks.json"

