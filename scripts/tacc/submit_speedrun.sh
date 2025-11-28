#!/bin/bash
#SBATCH -J modern-llm-speedrun
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=aymanmahfuz27@utexas.edu  # Update with your email
#SBATCH --mail-type=all             # Email notifications for job status
#SBATCH -t 48:00:00
#SBATCH -A ASC25078

# Modern LLM: Full pipeline SLURM job for TACC Lonestar6
# Reference: https://docs.tacc.utexas.edu/hpc/lonestar6/
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

set -euo pipefail

# Configuration
CONFIG="${1:-tacc}"

# Get project directory robustly
# SLURM copies scripts to /var/spool, so BASH_SOURCE[0] won't work.
# We find the project root by looking for the marker file (requirements.txt).
find_project_root() {
    local search_dir="${1:-$(pwd)}"
    while [ "$search_dir" != "/" ]; do
        if [ -f "$search_dir/requirements.txt" ] && [ -d "$search_dir/src/modern_llm" ]; then
            echo "$search_dir"
            return 0
        fi
        search_dir="$(dirname "$search_dir")"
    done
    return 1
}

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$(find_project_root "${SLURM_SUBMIT_DIR}")" || {
        echo "ERROR: Could not find project root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
        echo "Please run sbatch from within the modern_llm project directory."
        exit 1
    }
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(find_project_root "${SCRIPT_DIR}")" || {
        echo "ERROR: Could not find project root from script dir"
        exit 1
    }
fi

# Use $WORK for persistent storage (not $SCRATCH which gets purged)
WORK_DIR="${WORK}/modern_llm"
VENV_PATH="${PROJECT_DIR}/.venv"

echo "========================================"
echo "Modern LLM Speedrun Pipeline"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG}"
echo "Project dir: ${PROJECT_DIR}"
echo "Work dir: ${WORK_DIR}"
echo "========================================"

# Create necessary directories in $WORK
mkdir -p "${WORK_DIR}/checkpoints"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${PROJECT_DIR}/logs"

# Load TACC modules for Lonestar6
# Reference: https://docs.tacc.utexas.edu/hpc/lonestar6/
module purge
module load gcc/11.2.0
module load cuda/12.0
# Use default python3 - specific versions may not exist
module load python3

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
# Use segmented CUDA allocator to reduce fragmentation for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Symlink work checkpoints to project dir for easier access
ln -sf "${WORK_DIR}/checkpoints" "${PROJECT_DIR}/experiments/runs/work_checkpoints" 2>/dev/null || true

# Run the full pipeline
echo ""
echo "Starting speedrun pipeline..."
echo ""

cd "${PROJECT_DIR}"

# Run the pipeline script (path relative to project root)
python "${PROJECT_DIR}/scripts/speedrun_pipeline.py" \
    --config "${CONFIG}" \
    --checkpoint-dir "${WORK_DIR}/checkpoints"

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo "Checkpoints saved to: ${WORK_DIR}/checkpoints"
echo "Logs saved to: ${PROJECT_DIR}/logs"
