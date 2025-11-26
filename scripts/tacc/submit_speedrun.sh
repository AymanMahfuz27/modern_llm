#!/bin/bash
#SBATCH -J modern-llm-speedrun
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus-per-node=1
#SBATCH -t 48:00:00
#SBATCH -A YOUR_ALLOCATION

# Modern LLM: Full pipeline SLURM job for TACC
#
# Usage:
#   sbatch submit_speedrun.sh [config]
#
# Arguments:
#   config: One of "tacc", "tacc-smoke", or path to JSON config file
#           Defaults to "tacc" if not specified
#
# Example:
#   sbatch submit_speedrun.sh tacc-smoke   # Quick smoke test
#   sbatch submit_speedrun.sh tacc         # Full training
#   sbatch submit_speedrun.sh configs/tacc_h100.json  # Custom config

set -euo pipefail

# Configuration
CONFIG="${1:-tacc}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRATCH_DIR="${SCRATCH:-/scratch/${USER}}/modern_llm"
VENV_PATH="${PROJECT_DIR}/.venv"

echo "========================================"
echo "Modern LLM Speedrun Pipeline"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG}"
echo "Project dir: ${PROJECT_DIR}"
echo "Scratch dir: ${SCRATCH_DIR}"
echo "========================================"

# Create necessary directories
mkdir -p "${SCRATCH_DIR}/checkpoints"
mkdir -p "${SCRATCH_DIR}/logs"
mkdir -p logs

# Load TACC modules
module purge
module load gcc/11.2.0
module load cuda/12.0
module load python3/3.11.1

# Activate virtual environment (create if needed)
if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

# Set environment variables
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Symlink scratch checkpoints to project dir for easier access
ln -sf "${SCRATCH_DIR}/checkpoints" "${PROJECT_DIR}/experiments/runs/scratch_checkpoints" 2>/dev/null || true

# Run the full pipeline
echo ""
echo "Starting speedrun pipeline..."
echo ""

cd "${PROJECT_DIR}"

# Use torchrun for potential multi-GPU (even single GPU benefits from its error handling)
torchrun --standalone --nproc_per_node=1 \
    scripts/speedrun_pipeline.py \
    --config "${CONFIG}" \
    --checkpoint-dir "${SCRATCH_DIR}/checkpoints"

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo "Checkpoints saved to: ${SCRATCH_DIR}/checkpoints"
echo "Logs saved to: logs/${SLURM_JOB_ID:-local}"


