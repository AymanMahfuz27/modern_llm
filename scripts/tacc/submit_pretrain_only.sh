#!/bin/bash
#SBATCH -J modern-llm-pretrain
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A ASC25078

# Modern LLM: Pretraining only SLURM job for TACC Lonestar6
# Reference: https://docs.tacc.utexas.edu/hpc/lonestar6/
#
# Usage:
#   sbatch submit_pretrain_only.sh [config]

set -euo pipefail

CONFIG="${1:-tacc}"

# Get project directory robustly
# SLURM copies scripts to /var/spool, so BASH_SOURCE[0] won't work.
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
        exit 1
    }
else
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(find_project_root "${SCRIPT_DIR}")" || {
        echo "ERROR: Could not find project root"
        exit 1
    }
fi

WORK_DIR="${WORK}/modern_llm"
VENV_PATH="${PROJECT_DIR}/.venv"

echo "Modern LLM Pretraining"
echo "Config: ${CONFIG}"
echo "Project dir: ${PROJECT_DIR}"
echo "Work dir: ${WORK_DIR}"

mkdir -p "${WORK_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/logs"

# Load TACC modules for Lonestar6
module purge
module load gcc/11.2.0
module load cuda/12.0
module load python3

if [ ! -d "${VENV_PATH}" ]; then
    python3 -m venv "${VENV_PATH}"
fi
source "${VENV_PATH}/bin/activate"

pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"

python "${PROJECT_DIR}/scripts/pretrain.py" \
    --config "${CONFIG}" \
    --checkpoint-dir "${WORK_DIR}/checkpoints"

echo "Pretraining complete!"
echo "Checkpoint: ${WORK_DIR}/checkpoints"
