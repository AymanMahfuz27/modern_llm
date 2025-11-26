#!/bin/bash
#SBATCH -J modern-llm-pretrain
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus-per-node=1
#SBATCH -t 24:00:00
#SBATCH -A YOUR_ALLOCATION

# Modern LLM: Pretraining only SLURM job for TACC
#
# Usage:
#   sbatch submit_pretrain_only.sh [config]

set -euo pipefail

CONFIG="${1:-tacc}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRATCH_DIR="${SCRATCH:-/scratch/${USER}}/modern_llm"
VENV_PATH="${PROJECT_DIR}/.venv"

echo "Modern LLM Pretraining"
echo "Config: ${CONFIG}"

mkdir -p "${SCRATCH_DIR}/checkpoints"
mkdir -p logs

module purge
module load gcc/11.2.0
module load cuda/12.0
module load python3/3.11.1

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

torchrun --standalone --nproc_per_node=1 \
    scripts/pretrain.py \
    --config "${CONFIG}" \
    --checkpoint-dir "${SCRATCH_DIR}/checkpoints"

echo "Pretraining complete!"
echo "Checkpoint: ${SCRATCH_DIR}/checkpoints"


