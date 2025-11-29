#!/bin/bash
#SBATCH --job-name=modern-llm-pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# =============================================================================
# Modern LLM Pipeline - TACC H100 Training Script
# =============================================================================
# This script runs the full pipeline: Pretrain -> SFT -> DPO -> Verifier
# with auto-checkpointing and resume capability.
#
# Usage:
#   sbatch scripts/tacc_pipeline.slurm
#
# To resume a failed job:
#   sbatch scripts/tacc_pipeline.slurm  (auto-resumes from last checkpoint)
# =============================================================================

set -e  # Exit on error

# ---- Configuration ----
# IMPORTANT: Change these paths for your TACC setup
export PROJECT_DIR="${HOME}/modern_llm"
export WORK_DIR="${WORK}/modern_llm_runs"  # Use $WORK for checkpoints (not $HOME)
export RUN_NAME="gpu-full-$(date +%Y%m%d)"

# Create directories
mkdir -p "${WORK_DIR}/checkpoints"
mkdir -p "${WORK_DIR}/logs"
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

# Load modules (adjust for your TACC system)
module purge
module load cuda/12.2
module load python3/3.11

# Activate virtual environment
cd "${PROJECT_DIR}"
source .venv/bin/activate

# Verify GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# ---- Stage Tracking ----
STAGE_FILE="${WORK_DIR}/checkpoints/.pipeline_stage"
PRETRAIN_DONE="${WORK_DIR}/checkpoints/.pretrain_done"
SFT_DONE="${WORK_DIR}/checkpoints/.sft_done"
DPO_DONE="${WORK_DIR}/checkpoints/.dpo_done"
VERIFIER_DONE="${WORK_DIR}/checkpoints/.verifier_done"

# ---- Helper Functions ----
mark_done() {
    touch "$1"
    echo "$(date)" >> "$1"
}

check_done() {
    [ -f "$1" ]
}

# ---- Run Pipeline ----
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

# Override output directory to use WORK
export PIPELINE_OUTPUT_DIR="${WORK_DIR}/checkpoints"

echo "=============================================="
echo "Stage 1: Pretraining"
echo "=============================================="

if check_done "$PRETRAIN_DONE"; then
    echo "Pretraining already complete, skipping..."
else
    echo "Starting pretraining..."
    
    python -u scripts/speedrun_pipeline.py \
        --config gpu \
        --stage pretrain \
        --output-dir "${WORK_DIR}/checkpoints" \
        --run-name "${RUN_NAME}" \
        2>&1 | tee "${WORK_DIR}/logs/pretrain_${SLURM_JOB_ID}.log"
    
    mark_done "$PRETRAIN_DONE"
    echo "Pretraining complete!"
fi

echo ""
echo "=============================================="
echo "Stage 2: Supervised Fine-Tuning (SFT)"
echo "=============================================="

if check_done "$SFT_DONE"; then
    echo "SFT already complete, skipping..."
else
    echo "Starting SFT..."
    
    # Find pretrain checkpoint
    PRETRAIN_CKPT=$(ls -t ${WORK_DIR}/checkpoints/*pretrain*.pt 2>/dev/null | head -1)
    if [ -z "$PRETRAIN_CKPT" ]; then
        echo "ERROR: No pretrain checkpoint found!"
        exit 1
    fi
    echo "Loading from: ${PRETRAIN_CKPT}"
    
    python -u scripts/speedrun_pipeline.py \
        --config gpu \
        --stage sft \
        --checkpoint "${PRETRAIN_CKPT}" \
        --output-dir "${WORK_DIR}/checkpoints" \
        --run-name "${RUN_NAME}" \
        2>&1 | tee "${WORK_DIR}/logs/sft_${SLURM_JOB_ID}.log"
    
    mark_done "$SFT_DONE"
    echo "SFT complete!"
fi

echo ""
echo "=============================================="
echo "Stage 3: Direct Preference Optimization (DPO)"
echo "=============================================="

if check_done "$DPO_DONE"; then
    echo "DPO already complete, skipping..."
else
    echo "Starting DPO..."
    
    # Find SFT checkpoint
    SFT_CKPT=$(ls -t ${WORK_DIR}/checkpoints/*sft*.pt 2>/dev/null | head -1)
    if [ -z "$SFT_CKPT" ]; then
        echo "ERROR: No SFT checkpoint found!"
        exit 1
    fi
    echo "Loading from: ${SFT_CKPT}"
    
    python -u scripts/speedrun_pipeline.py \
        --config gpu \
        --stage dpo \
        --checkpoint "${SFT_CKPT}" \
        --output-dir "${WORK_DIR}/checkpoints" \
        --run-name "${RUN_NAME}" \
        2>&1 | tee "${WORK_DIR}/logs/dpo_${SLURM_JOB_ID}.log"
    
    mark_done "$DPO_DONE"
    echo "DPO complete!"
fi

echo ""
echo "=============================================="
echo "Stage 4: Verifier Training"
echo "=============================================="

if check_done "$VERIFIER_DONE"; then
    echo "Verifier already complete, skipping..."
else
    echo "Starting verifier training..."
    
    # Find DPO checkpoint
    DPO_CKPT=$(ls -t ${WORK_DIR}/checkpoints/*dpo*.pt 2>/dev/null | head -1)
    if [ -z "$DPO_CKPT" ]; then
        echo "ERROR: No DPO checkpoint found!"
        exit 1
    fi
    echo "Loading from: ${DPO_CKPT}"
    
    python -u scripts/speedrun_pipeline.py \
        --config gpu \
        --stage verifier \
        --checkpoint "${DPO_CKPT}" \
        --output-dir "${WORK_DIR}/checkpoints" \
        --run-name "${RUN_NAME}" \
        2>&1 | tee "${WORK_DIR}/logs/verifier_${SLURM_JOB_ID}.log"
    
    mark_done "$VERIFIER_DONE"
    echo "Verifier complete!"
fi

echo ""
echo "=============================================="
echo "Stage 5: Evaluation"
echo "=============================================="

echo "Running task evaluations..."

# Find final checkpoints
FINAL_CKPT=$(ls -t ${WORK_DIR}/checkpoints/*dpo*.pt 2>/dev/null | head -1)
VERIFIER_CKPT=$(ls -t ${WORK_DIR}/checkpoints/*verifier*.pt 2>/dev/null | head -1)

# Run evaluations
python scripts/evaluation/evaluate_tasks.py \
    --stage-checkpoints "${WORK_DIR}/checkpoints" \
    --include-baselines \
    --output-dir "${WORK_DIR}/results" \
    2>&1 | tee "${WORK_DIR}/logs/eval_${SLURM_JOB_ID}.log"

# Run attention visualization
python scripts/visualize_attention.py \
    --checkpoint "${FINAL_CKPT}" \
    --output-dir "${WORK_DIR}/figures" \
    2>&1 | tee -a "${WORK_DIR}/logs/eval_${SLURM_JOB_ID}.log"

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Checkpoints: ${WORK_DIR}/checkpoints/"
echo "Results: ${WORK_DIR}/results/"
echo "Figures: ${WORK_DIR}/figures/"
echo "Time: $(date)"
echo ""

# Copy important results back to project dir for easy access
cp -r "${WORK_DIR}/results" "${PROJECT_DIR}/experiments/" 2>/dev/null || true
cp -r "${WORK_DIR}/figures" "${PROJECT_DIR}/report/" 2>/dev/null || true

echo "Results copied to ${PROJECT_DIR}/experiments/results/"
echo "Figures copied to ${PROJECT_DIR}/report/figures/"

