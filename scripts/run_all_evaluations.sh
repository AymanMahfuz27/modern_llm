#!/bin/bash
# Run all evaluation scripts assuming checkpoints already exist
# Useful for re-evaluating after training is complete

set -e

# Ensure we're in the project root and add src to PYTHONPATH
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

echo "========================================="
echo "Running All Evaluations"
echo "========================================="

echo ""
echo "=== Evaluating scratch LM checkpoints ==="
python scripts/evaluate_lm_checkpoints.py \
    --runs_dir experiments/runs \
    --output_csv experiments/lm_checkpoint_metrics.csv

echo ""
echo "=== Generating text samples ==="
python scripts/generate_from_checkpoints.py \
    --runs_dir experiments/runs \
    --output_json experiments/lm_generation_samples.json

echo ""
echo "=== Visualizing LM results ==="
python notebooks/visualize_lm_results.py \
    --metrics_csv experiments/lm_checkpoint_metrics.csv \
    --output_dir experiments/plots

echo ""
echo "=== Evaluating GPT-2 SST-2 checkpoint ==="
if [ -f "experiments/runs/gpt2-sst2-lora-main/gpt2-sst2-lora-main_final.pt" ]; then
    python scripts/evaluate_hf_sst2.py \
        --checkpoint_path experiments/runs/gpt2-sst2-lora-main/gpt2-sst2-lora-main_final.pt \
        --model_name gpt2
else
    echo "  SST-2 checkpoint not found, skipping..."
fi

echo ""
echo "=== Evaluating T5 SAMSum checkpoint ==="
if [ -f "experiments/runs/t5-samsum-lora-main/t5-samsum-lora-main_final.pt" ]; then
    python scripts/evaluate_hf_samsum.py \
        --checkpoint_path experiments/runs/t5-samsum-lora-main/t5-samsum-lora-main_final.pt \
        --model_name t5-small
else
    echo "  SAMSum checkpoint not found, skipping..."
fi

echo ""
echo "=== Evaluating GPT-2 GSM8K checkpoint ==="
if [ -f "experiments/runs/gpt2-gsm8k-lora-main/gpt2-gsm8k-lora-main_final.pt" ]; then
    python scripts/evaluate_hf_gsm8k.py \
        --checkpoint_path experiments/runs/gpt2-gsm8k-lora-main/gpt2-gsm8k-lora-main_final.pt \
        --model_name gpt2
else
    echo "  GSM8K checkpoint not found, skipping..."
fi

echo ""
echo "=== Running prompting baselines ==="
python src/modern_llm/hf/prompting_baselines.py \
    --model_name gpt2 \
    --tasks sst2 samsum gsm8k \
    --num_shots 0 \
    --max_samples 500 \
    --output_csv experiments/prompting_baseline_results.csv

echo ""
echo "========================================="
echo "All Evaluations Complete!"
echo "========================================="

