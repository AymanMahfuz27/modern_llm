#!/bin/bash
# Master orchestration script for Phase 1 and Phase 2
# Runs all baseline experiments, evaluations, and visualizations

set -e

# Ensure we're in the project root and add src to PYTHONPATH
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

echo "========================================="
echo "Phase 1 & 2 Execution Pipeline"
echo "========================================="

# Phase 1: Scratch LM experiments
echo ""
echo "=== Phase 1.1: Evaluating existing LM checkpoints ==="
python scripts/evaluate_lm_checkpoints.py \
    --runs_dir experiments/runs \
    --output_csv experiments/lm_checkpoint_metrics.csv \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tokenizer_name gpt2 \
    --max_seq_len 512 \
    --batch_size 8

echo ""
echo "=== Phase 1.2: Generating text samples from checkpoints ==="
python scripts/generate_from_checkpoints.py \
    --runs_dir experiments/runs \
    --output_json experiments/lm_generation_samples.json \
    --tokenizer_name gpt2 \
    --max_new_tokens 80 \
    --temperature 1.0

echo ""
echo "=== Phase 1.3: Training MAX-SIZE scratch LM (RTX 3060) ==="
echo "This will run for many hours. Consider running in tmux/screen."
python scripts/train_lm_from_config.py \
    --config configs/lm_max_rtx3060.json \
    --output_dir experiments/runs \
    --num_proc 4 \
    --gen_prompt "The meaning of life is" \
    --gen_max_new_tokens 100

echo ""
echo "=== Phase 1.4: Attention sinks long-context experiment ==="
python scripts/experiment_attention_sinks.py \
    --output_dir experiments/runs \
    --output_json experiments/attention_sinks_results.json \
    --train_context 512 \
    --gen_context_multiplier 3 \
    --max_steps 500

echo ""
echo "=== Phase 1.5: Visualizing LM results ==="
python notebooks/visualize_lm_results.py \
    --metrics_csv experiments/lm_checkpoint_metrics.csv \
    --output_dir experiments/plots

# Phase 2: HF Finetuning
echo ""
echo "=== Phase 2.1: GPT-2 + LoRA on SST-2 (full run) ==="
python src/modern_llm/hf/finetune_gpt2_sst2.py \
    --run_name gpt2-sst2-lora-main \
    --model_name gpt2 \
    --max_seq_len 128 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 2e-4 \
    --max_steps 2000 \
    --warmup_steps 100 \
    --eval_every 200 \
    --save_every 500 \
    --log_every 50 \
    --mixed_precision bf16 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32

echo ""
echo "=== Phase 2.2: Evaluating GPT-2 SST-2 checkpoint ==="
python scripts/evaluate_hf_sst2.py \
    --checkpoint_path experiments/runs/gpt2-sst2-lora-main/gpt2-sst2-lora-main_final.pt \
    --model_name gpt2 \
    --max_seq_len 128 \
    --output_metrics_csv experiments/sst2_eval_metrics.csv \
    --output_errors_json experiments/sst2_misclassified.json

echo ""
echo "=== Phase 2.3: T5 + LoRA on SAMSum ==="
python src/modern_llm/hf/finetune_t5_samsum.py \
    --run_name t5-samsum-lora-main \
    --model_name t5-small \
    --max_source_length 512 \
    --max_target_length 64 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 5e-4 \
    --max_steps 2000 \
    --warmup_steps 100 \
    --eval_every 200 \
    --save_every 500 \
    --log_every 50 \
    --mixed_precision bf16 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32

echo ""
echo "=== Phase 2.4: Evaluating T5 SAMSum checkpoint ==="
python scripts/evaluate_hf_samsum.py \
    --checkpoint_path experiments/runs/t5-samsum-lora-main/t5-samsum-lora-main_final.pt \
    --model_name t5-small \
    --max_source_length 512 \
    --max_target_length 64 \
    --output_csv experiments/samsum_eval_metrics.csv

echo ""
echo "=== Phase 2.5: GPT-2 + LoRA on GSM8K ==="
python src/modern_llm/hf/finetune_math_gsm8k.py \
    --run_name gpt2-gsm8k-lora-main \
    --model_name gpt2 \
    --max_seq_len 512 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 3e-4 \
    --max_steps 3000 \
    --warmup_steps 150 \
    --eval_every 300 \
    --save_every 1000 \
    --log_every 50 \
    --mixed_precision bf16 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32

echo ""
echo "=== Phase 2.6: Evaluating GPT-2 GSM8K checkpoint ==="
python scripts/evaluate_hf_gsm8k.py \
    --checkpoint_path experiments/runs/gpt2-gsm8k-lora-main/gpt2-gsm8k-lora-main_final.pt \
    --model_name gpt2 \
    --max_seq_len 512 \
    --max_new_tokens 128 \
    --output_csv experiments/gsm8k_eval_metrics.csv \
    --output_errors_json experiments/gsm8k_errors.json

echo ""
echo "=== Phase 2.7: Running prompting baselines (zero-shot) ==="
python src/modern_llm/hf/prompting_baselines.py \
    --model_name gpt2 \
    --tasks sst2 samsum gsm8k \
    --num_shots 0 \
    --max_samples 500 \
    --output_csv experiments/prompting_baseline_results.csv

echo ""
echo "========================================="
echo "Phase 1 & 2 Complete!"
echo "========================================="
echo "Results in experiments/"
echo "  - lm_checkpoint_metrics.csv"
echo "  - lm_generation_samples.json"
echo "  - attention_sinks_results.json"
echo "  - sst2_eval_metrics.csv"
echo "  - samsum_eval_metrics.csv"
echo "  - gsm8k_eval_metrics.csv"
echo "  - prompting_baseline_results.csv"
echo "  - plots/lm_metrics.png"

