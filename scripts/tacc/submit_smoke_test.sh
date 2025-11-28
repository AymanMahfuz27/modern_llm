#!/bin/bash
#SBATCH -J modern-llm-smoke
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100-dev		
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ASC25078

# Modern LLM: Quick smoke test on TACC Lonestar6
# Reference: https://docs.tacc.utexas.edu/hpc/lonestar6/
#
# This runs a minimal config to verify everything works before
# committing to a long training run.
#
# Usage:
#   sbatch submit_smoke_test.sh

set -euo pipefail

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
WORK_DIR="${WORK}/modern_llm"
VENV_PATH="${PROJECT_DIR}/.venv"

echo "========================================"
echo "Modern LLM Smoke Test"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "Work dir: ${WORK_DIR}"
echo "========================================"

mkdir -p "${WORK_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/logs"

# Load TACC modules for Lonestar6
# Reference: https://docs.tacc.utexas.edu/hpc/lonestar6/
module purge
module load gcc/11.2.0
module load cuda/12.0
# Use default python3 - specific versions may not exist
module load python3

# Setup venv
if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"

echo ""
echo "Running smoke test pipeline..."
echo ""

# Run with tacc-smoke config (minimal steps)
python "${PROJECT_DIR}/scripts/speedrun_pipeline.py" \
    --config "tacc-smoke" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/smoke-test"

echo ""
echo "========================================"
echo "Smoke test complete!"
echo "========================================"
echo "If no errors occurred, you can run the full pipeline:"
echo "  cd ${PROJECT_DIR} && sbatch scripts/tacc/submit_speedrun.sh tacc"

