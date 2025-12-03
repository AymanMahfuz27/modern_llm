#!/bin/bash
#SBATCH -J test-datasets
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --exclude=c318-001
#SBATCH -A ASC25078

# =============================================================================
# Test Script - Verify all datasets load correctly before big training run
# =============================================================================

set -e

export PROJECT_DIR="${HOME}/modern_llm"
export WORK_DIR="${WORK}/modern_llm_runs"

cd "${PROJECT_DIR}"
module load cuda/12.2
source .venv/bin/activate
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

echo "=============================================="
echo "Dataset & Evaluation Test"
echo "=============================================="
echo "Time: $(date)"
echo ""

python - << 'TEST_PY'
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoTokenizer

print("=" * 60)
print("1. PRETRAINING DATASETS")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Test datasets from the new config
pretrain_datasets = [
    "wikitext-103-raw-v1",
    "openwebtext",
    "wikipedia",
    "roneneldan/TinyStories:100000",  # Downsampled
]

from modern_llm.data.lm_datasets import load_multi_dataset, _parse_dataset_spec

print("\nTesting dataset spec parsing:")
for spec in pretrain_datasets:
    name, cap = _parse_dataset_spec(spec)
    print(f"  '{spec}' -> name='{name}', cap={cap}")

print("\nLoading pretrain datasets (this may take a few minutes)...")
print("-" * 60)

try:
    combined = load_multi_dataset(
        pretrain_datasets,
        tokenizer,
        split="train",
        max_length=1024,
    )
    print(f"\n✓ Pretrain datasets loaded successfully!")
    print(f"  Total samples: {len(combined)}")
    
    # Show a sample
    sample = combined[0]
    decoded = tokenizer.decode(sample["input_ids"][:50])
    print(f"  Sample text: {decoded[:100]}...")
    
except Exception as e:
    print(f"\n✗ FAILED to load pretrain datasets: {e}")
    import traceback
    traceback.print_exc()


print("\n")
print("=" * 60)
print("2. SFT DATASETS")
print("=" * 60)

sft_datasets = [
    "tatsu-lab/alpaca",
    "databricks/databricks-dolly-15k",
    "Open-Orca/OpenOrca:50000",
]

from datasets import load_dataset

for spec in sft_datasets:
    name, cap = _parse_dataset_spec(spec)
    print(f"\nLoading: {name}" + (f" (cap: {cap})" if cap else ""))
    
    try:
        # Load with streaming first to check availability
        ds = load_dataset(name, split="train", streaming=True)
        # Take a small sample to verify
        sample = next(iter(ds))
        print(f"  ✓ Available. Sample keys: {list(sample.keys())}")
        
        # Now load for real (non-streaming) with cap
        ds = load_dataset(name, split="train")
        total = len(ds)
        if cap and total > cap:
            ds = ds.select(range(cap))
        print(f"  ✓ Loaded {len(ds)} samples (total available: {total})")
        
        # Show text field
        text_fields = ["text", "instruction", "input", "output", "response", "question", "answer"]
        found = [f for f in text_fields if f in sample]
        print(f"  Text fields found: {found}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


print("\n")
print("=" * 60)
print("3. EVALUATION PROMPT FORMAT TEST")
print("=" * 60)

# Test the new SST-2 prompt format
NEW_PROMPT = '''Is this review positive or negative?

Review: "I love this movie, it's fantastic!"
Answer: positive

Review: "This was terrible and boring."
Answer: negative

Review: "A wonderful experience from start to finish."
Answer: positive

Review: "{text}"
Answer:'''

test_texts = [
    "This movie was absolutely wonderful!",
    "I hated every minute of this terrible film.",
    "It was okay, nothing special.",
]

print("\nNew prompt format test:")
for text in test_texts:
    prompt = NEW_PROMPT.format(text=text)
    tokens = tokenizer.encode(prompt)
    print(f"\n  Text: '{text[:50]}...'")
    print(f"  Prompt length: {len(tokens)} tokens")
    
    # Check that we can get positive/negative token IDs
    pos_tokens = tokenizer.encode(" positive", add_special_tokens=False)
    neg_tokens = tokenizer.encode(" negative", add_special_tokens=False)
    print(f"  'positive' token ID: {pos_tokens[0] if pos_tokens else 'N/A'}")
    print(f"  'negative' token ID: {neg_tokens[0] if neg_tokens else 'N/A'}")

print("\n✓ Prompt format is valid")


print("\n")
print("=" * 60)
print("4. CHINCHILLA CHECK FOR NEW CONFIG")
print("=" * 60)

# New config: 80K steps, batch 128, seq 1024
model_params = 253_000_000
steps = 80_000
batch_size = 128
seq_len = 1024
tokens_seen = steps * batch_size * seq_len

chinchilla_optimal_tokens = model_params * 20

print(f"Model parameters:     {model_params/1e6:.0f}M")
print(f"Training steps:       {steps:,}")
print(f"Batch size:           {batch_size}")
print(f"Sequence length:      {seq_len}")
print(f"Tokens to see:        {tokens_seen/1e9:.1f}B")
print(f"")
print(f"Chinchilla optimal:   {chinchilla_optimal_tokens/1e9:.2f}B tokens")
print(f"Ratio (actual/opt):   {tokens_seen/chinchilla_optimal_tokens:.1f}x")
print(f"")

if tokens_seen > chinchilla_optimal_tokens * 3:
    print("⚠️  WARNING: Significantly overtrained for model size")
elif tokens_seen > chinchilla_optimal_tokens * 1.5:
    print("ℹ️  Slightly overtrained, but acceptable for diverse data")
else:
    print("✓  Training budget is Chinchilla-optimal or undertrained")


print("\n")
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print(f"Time: {__import__('datetime').datetime.now()}")
print("")
print("If all tests passed, you can proceed with:")
print("  1. Remove old checkpoints: rm -rf ${WORK}/modern_llm_runs/checkpoints/*")
print("  2. Remove old logs: rm -rf ${WORK}/modern_llm_runs/logs/*")
print("  3. Dispatch training: sbatch scripts/tacc_pipeline.slurm")
TEST_PY

echo ""
echo "Test complete!"


