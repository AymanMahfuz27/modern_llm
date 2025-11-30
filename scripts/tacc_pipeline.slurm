#!/bin/bash
#SBATCH -J modern-llm-h100
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --exclude=c318-001
#SBATCH -A ASC25078

# =============================================================================
# Modern LLM Pipeline - TACC H100 Training Script
# =============================================================================
# This script runs the full pipeline: Pretrain -> SFT -> DPO -> Verifier -> Eval
# The pipeline handles checkpointing and resume internally.
#
# Usage:
#   sbatch scripts/tacc_pipeline.slurm
# =============================================================================

set -e  # Exit on error

# ---- Configuration ----
export PROJECT_DIR="${HOME}/modern_llm"
export WORK_DIR="${WORK}/modern_llm_runs"

# Create directories
mkdir -p "${WORK_DIR}/checkpoints"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/results"
mkdir -p "${WORK_DIR}/figures"
mkdir -p "${PROJECT_DIR}/logs"

# ---- Environment Setup ----
echo "=============================================="
echo "Modern LLM Pipeline - Starting"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Time: $(date)"
echo "Project Dir: ${PROJECT_DIR}"
echo "Work Dir: ${WORK_DIR}"
echo ""

# Load modules (keep default TACC modules, just add cuda)
module load cuda/12.2

# Activate virtual environment
cd "${PROJECT_DIR}"
source .venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

# Verify GPU (don't fail if nvidia-smi has issues)
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "Warning: nvidia-smi failed, but continuing anyway"
echo ""

# ---- Run Full Pipeline ----
echo "=============================================="
echo "Running Full Pipeline (Pretrain -> SFT -> DPO -> Verifier)"
echo "=============================================="

python -u scripts/run_pipeline.py \
    --config gpu \
    --stage all \
    --output-dir "${WORK_DIR}/checkpoints" \
    | tee "${WORK_DIR}/logs/pipeline_${SLURM_JOB_ID}.log"

echo ""
echo "=============================================="
echo "Pipeline Training Complete!"
echo "=============================================="

# ---- Run Evaluation ----
echo ""
echo "=============================================="
echo "Running Task Evaluations"
echo "=============================================="

# Find final DPO checkpoint for evaluation
FINAL_CKPT=$(ls -t ${WORK_DIR}/checkpoints/*dpo*.pt 2>/dev/null | head -1 || echo "")

if [ -n "$FINAL_CKPT" ]; then
    echo "Found checkpoint: ${FINAL_CKPT}"
    
    # Run task evaluations
    python scripts/evaluation/evaluate_tasks.py \
        --stage-checkpoints "${WORK_DIR}/checkpoints" \
        --include-baselines \
        --output-dir "${WORK_DIR}/results" \
        | tee "${WORK_DIR}/logs/eval_${SLURM_JOB_ID}.log" || echo "Evaluation failed, continuing..."

    # Run attention visualization
    python scripts/visualize_attention.py \
        --checkpoint "${FINAL_CKPT}" \
        --output-dir "${WORK_DIR}/figures" \
        | tee -a "${WORK_DIR}/logs/eval_${SLURM_JOB_ID}.log" || echo "Visualization failed, continuing..."
else
    echo "No DPO checkpoint found, skipping evaluation"
fi

echo ""
echo "=============================================="
echo "All Done!"
echo "=============================================="
echo "Checkpoints: ${WORK_DIR}/checkpoints/"
echo "Results: ${WORK_DIR}/results/"
echo "Figures: ${WORK_DIR}/figures/"
echo "Time: $(date)"
echo ""

# Copy results back to project dir
cp -r "${WORK_DIR}/results" "${PROJECT_DIR}/experiments/" 2>/dev/null || true
cp -r "${WORK_DIR}/figures" "${PROJECT_DIR}/report/" 2>/dev/null || true

echo "Results copied to ${PROJECT_DIR}/experiments/"
echo "Figures copied to ${PROJECT_DIR}/report/"
