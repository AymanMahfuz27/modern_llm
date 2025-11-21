#!/bin/bash
# Phase 2 only: HF Finetuning and Prompting Baselines
set -e

# Ensure we're in the project root and add src to PYTHONPATH
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

echo "========================================="
echo "Phase 2: HF Finetuning & Prompting"
echo "========================================="

echo ""
echo "=== Step 1: GPT-2 + LoRA on SST-2 ==="
python src/modern_llm/hf/finetune_gpt2_sst2.py \
    --run_name gpt2-sst2-lora-main \
    --model_name gpt2 \
    --max_seq_len 128 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 2e-4 \
    --max_steps 2000 \
    --use_lora \
    --lora_r 16

echo ""
echo "=== Step 2: Evaluate GPT-2 SST-2 ==="
python scripts/evaluate_hf_sst2.py \
    --checkpoint_path experiments/runs/gpt2-sst2-lora-main/gpt2-sst2-lora-main_final.pt \
    --model_name gpt2

echo ""
echo "=== Step 3: T5 + LoRA on SAMSum ==="
python src/modern_llm/hf/finetune_t5_samsum.py \
    --run_name t5-samsum-lora-main \
    --model_name t5-small \
    --max_source_length 512 \
    --max_target_length 64 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 5e-4 \
    --max_steps 2000 \
    --use_lora \
    --lora_r 16

echo ""
echo "=== Step 4: Evaluate T5 SAMSum ==="
python scripts/evaluate_hf_samsum.py \
    --checkpoint_path experiments/runs/t5-samsum-lora-main/t5-samsum-lora-main_final.pt \
    --model_name t5-small

echo ""
echo "=== Step 5: GPT-2 + LoRA on GSM8K ==="
python src/modern_llm/hf/finetune_math_gsm8k.py \
    --run_name gpt2-gsm8k-lora-main \
    --model_name gpt2 \
    --max_seq_len 512 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 3e-4 \
    --max_steps 3000 \
    --use_lora \
    --lora_r 16

echo ""
echo "=== Step 6: Evaluate GPT-2 GSM8K ==="
python scripts/evaluate_hf_gsm8k.py \
    --checkpoint_path experiments/runs/gpt2-gsm8k-lora-main/gpt2-gsm8k-lora-main_final.pt \
    --model_name gpt2

echo ""
echo "=== Step 7: Prompting baselines ==="
python src/modern_llm/hf/prompting_baselines.py \
    --model_name gpt2 \
    --tasks sst2 samsum gsm8k \
    --num_shots 0 \
    --max_samples 500

echo ""
echo "Phase 2 Complete! Check experiments/ for results."

