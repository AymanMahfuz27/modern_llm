#!/bin/bash
#SBATCH -J modern-llm-align
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A YOUR_ALLOCATION

# Modern LLM: Alignment pipeline (SFT -> DPO -> Verifier) SLURM job
# Reference: https://docs.tacc.utexas.edu/hpc/lonestar6/
#
# Requires pretrained checkpoint from submit_pretrain_only.sh
#
# Usage:
#   sbatch submit_alignment.sh [config] [pretrain_checkpoint]

set -euo pipefail

CONFIG="${1:-tacc}"
PRETRAIN_CKPT="${2:-}"

# Get paths - script is in scripts/tacc/, project root is 2 levels up
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_DIR="${WORK}/modern_llm"
VENV_PATH="${PROJECT_DIR}/.venv"

if [ -z "${PRETRAIN_CKPT}" ]; then
    # Auto-detect latest pretrain checkpoint in $WORK
    PRETRAIN_CKPT=$(ls -t "${WORK_DIR}/checkpoints"/*pretrain*final*.pt 2>/dev/null | head -1)
    if [ -z "${PRETRAIN_CKPT}" ]; then
        echo "ERROR: No pretrain checkpoint found. Run submit_pretrain_only.sh first."
        exit 1
    fi
fi

echo "Modern LLM Alignment Pipeline"
echo "Config: ${CONFIG}"
echo "Pretrain checkpoint: ${PRETRAIN_CKPT}"
echo "Project dir: ${PROJECT_DIR}"
echo "Work dir: ${WORK_DIR}"

mkdir -p "${WORK_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/logs"

module purge
module load gcc/11.2.0
module load cuda/12.0
module load python3/3.11.1

source "${VENV_PATH}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"

# Run SFT
echo "Stage 1: SFT..."
python "${PROJECT_DIR}/scripts/sft.py" \
    --config "${CONFIG}" \
    --pretrain-checkpoint "${PRETRAIN_CKPT}" \
    --output-dir "${WORK_DIR}/checkpoints"

SFT_CKPT=$(ls -t "${WORK_DIR}/checkpoints"/*sft*final*.pt | head -1)

# Run DPO
echo "Stage 2: DPO..."
python "${PROJECT_DIR}/scripts/dpo.py" \
    --config "${CONFIG}" \
    --sft-checkpoint "${SFT_CKPT}" \
    --output-dir "${WORK_DIR}/checkpoints"

DPO_CKPT=$(ls -t "${WORK_DIR}/checkpoints"/*dpo*final*.pt | head -1)

# Train Verifier
echo "Stage 3: Verifier..."
python "${PROJECT_DIR}/scripts/train_verifier.py" \
    --config "${CONFIG}" \
    --output-dir "${WORK_DIR}/checkpoints"

echo "Alignment complete!"
echo "Final DPO checkpoint: ${DPO_CKPT}"
