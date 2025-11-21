#!/bin/bash
# Phase 1 only: Scratch LM experiments
set -e

# Ensure we're in the project root and add src to PYTHONPATH
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

echo "========================================="
echo "Phase 1: Scratch LM Experiments"
echo "========================================="

echo ""
echo "=== Step 1: Evaluating existing LM checkpoints ==="
python scripts/evaluate_lm_checkpoints.py \
    --runs_dir experiments/runs \
    --output_csv experiments/lm_checkpoint_metrics.csv

echo ""
echo "=== Step 2: Generating text samples from checkpoints ==="
python scripts/generate_from_checkpoints.py \
    --runs_dir experiments/runs \
    --output_json experiments/lm_generation_samples.json

echo ""
echo "=== Step 3: Training MAX-SIZE scratch LM ==="
echo "WARNING: This will take many hours!"
python scripts/train_lm_from_config.py \
    --config configs/lm_max_rtx3060.json \
    --output_dir experiments/runs \
    --num_proc 4

echo ""
echo "=== Step 4: Attention sinks experiment ==="
python scripts/experiment_attention_sinks.py \
    --output_dir experiments/runs \
    --output_json experiments/attention_sinks_results.json \
    --train_context 512 \
    --max_steps 500

echo ""
echo "=== Step 5: Visualizing results ==="
python notebooks/visualize_lm_results.py \
    --metrics_csv experiments/lm_checkpoint_metrics.csv \
    --output_dir experiments/plots

echo ""
echo "Phase 1 Complete! Check experiments/ for results."

