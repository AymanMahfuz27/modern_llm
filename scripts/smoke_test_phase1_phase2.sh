#!/bin/bash
# Quick smoke test for Phase 1 & 2 components
# Runs minimal versions to verify everything is wired correctly

set -e

# Ensure we're in the project root and add src to PYTHONPATH
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

echo "========================================="
echo "Phase 1 & 2 Smoke Test"
echo "========================================="

echo ""
echo "=== Smoke 1: Small scratch LM (50 steps) ==="
python src/modern_llm/training/train_lm.py \
    --run_name smoke-test-lm \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tokenizer_name gpt2 \
    --max_seq_len 256 \
    --d_model 256 \
    --n_heads 4 \
    --n_layers 4 \
    --ffn_hidden_size 1024 \
    --batch_size 16 \
    --micro_batch_size 4 \
    --learning_rate 3e-4 \
    --max_steps 50 \
    --warmup_steps 10 \
    --eval_every 25 \
    --save_every 50 \
    --log_every 10 \
    --mixed_precision bf16 \
    --gen_max_new_tokens 30

echo ""
echo "=== Smoke 2: GPT-2 LoRA SST-2 (50 steps) ==="
python src/modern_llm/hf/finetune_gpt2_sst2.py \
    --run_name smoke-test-sst2 \
    --model_name gpt2 \
    --max_seq_len 128 \
    --batch_size 16 \
    --micro_batch_size 4 \
    --max_steps 50 \
    --warmup_steps 10 \
    --log_every 10 \
    --use_lora

echo ""
echo "=== Smoke 3: T5 LoRA SAMSum (50 steps) ==="
python src/modern_llm/hf/finetune_t5_samsum.py \
    --run_name smoke-test-samsum \
    --model_name t5-small \
    --max_source_length 256 \
    --max_target_length 32 \
    --batch_size 16 \
    --micro_batch_size 4 \
    --max_steps 50 \
    --warmup_steps 10 \
    --log_every 10 \
    --use_lora \
    --use_dummy_data

echo ""
echo "=== Smoke 4: Evaluate existing checkpoints ==="
if [ -d "experiments/runs" ]; then
    python scripts/evaluate_lm_checkpoints.py \
        --runs_dir experiments/runs \
        --output_csv experiments/smoke_lm_metrics.csv \
        --batch_size 4
    
    python scripts/generate_from_checkpoints.py \
        --runs_dir experiments/runs \
        --output_json experiments/smoke_generations.json \
        --max_new_tokens 30
fi

echo ""
echo "========================================="
echo "Smoke Test Complete!"
echo "========================================="
echo "If no errors, all components are wired correctly."

