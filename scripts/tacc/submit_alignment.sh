#!/bin/bash
#SBATCH -J modern-llm-alignment
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=aymanmahfuz27@utexas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -A ASC25078

set -euo pipefail

CHECKPOINT="${1:-/work/09999/aymanmahfuz/ls6/modern_llm/checkpoints/tacc-full-pretrain/tacc-full-pretrain_step30000.pt}"
CHECKPOINT_DIR="/work/09999/aymanmahfuz/ls6/modern_llm/checkpoints"

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
    PROJECT_DIR="$(find_project_root "${SLURM_SUBMIT_DIR}")" || exit 1
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(find_project_root "${SCRIPT_DIR}")" || exit 1
fi

cd "$PROJECT_DIR"

echo "=============================================="
echo "Modern LLM: Alignment Pipeline"
echo "Pretrain checkpoint: ${CHECKPOINT}"
echo "Start time: $(date)"
echo "=============================================="

[ -f "$CHECKPOINT" ] || { echo "ERROR: Checkpoint not found"; exit 1; }

module load cuda/12.8
module load python3

# Use existing venv - PyTorch was built with CUDA 12.8
VENV_DIR="${WORK}/modern_llm_venv"
source "${VENV_DIR}/bin/activate"
pip install --quiet -e .

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs experiments/results report

# Verify CUDA is working
echo "Checking CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('ERROR: CUDA not available!')
    import sys
    sys.exit(1)
"

# Run all stages via Python
python << 'PYTHON_SCRIPT'
import torch
from pathlib import Path

from modern_llm.config import get_pipeline_preset
from modern_llm.data.instruction_datasets import InstructionDatasetConfig
from modern_llm.data.preference_datasets import PreferenceDatasetConfig
from modern_llm.training.train_sft import run_sft
from modern_llm.training.train_dpo import DPOConfig, run_dpo
from modern_llm.training.train_verifier import VerifierConfig, VerifierDatasetConfig, run_verifier_training

# Config
config = get_pipeline_preset('tacc')
checkpoint_dir = Path('/work/09999/aymanmahfuz/ls6/modern_llm/checkpoints')
pretrain_ckpt = Path('/work/09999/aymanmahfuz/ls6/modern_llm/checkpoints/tacc-full-pretrain/tacc-full-pretrain_step30000.pt')

print("\n" + "="*50)
print("STAGE 1: Supervised Fine-Tuning (SFT)")
print("="*50)

sft_train_config = config.get_sft_config()
sft_train_config.output_dir = checkpoint_dir / 'tacc-full-sft-v2'
sft_train_config.output_dir.mkdir(parents=True, exist_ok=True)

sft_dataset_config = InstructionDatasetConfig(
    dataset_name=config.sft_dataset,
    max_length=config.max_seq_len,
)

sft_ckpt = run_sft(
    pretrain_checkpoint=pretrain_ckpt,
    train_config=sft_train_config,
    dataset_config=sft_dataset_config,
    tokenizer_name=config.tokenizer_name,
)
print(f"SFT complete: {sft_ckpt}")

print("\n" + "="*50)
print("STAGE 2: Direct Preference Optimization (DPO)")
print("="*50)

dpo_train_config = config.get_dpo_config()
dpo_train_config.output_dir = checkpoint_dir / 'tacc-full-dpo-v2'
dpo_train_config.output_dir.mkdir(parents=True, exist_ok=True)

dpo_config = DPOConfig(
    beta=config.dpo_beta,
    max_length=config.max_seq_len,
)
preference_config = PreferenceDatasetConfig(
    dataset_name=config.dpo_dataset,
)

dpo_ckpt = run_dpo(
    sft_checkpoint=sft_ckpt,
    train_config=dpo_train_config,
    dpo_config=dpo_config,
    preference_config=preference_config,
    tokenizer_name=config.tokenizer_name,
)
print(f"DPO complete: {dpo_ckpt}")

print("\n" + "="*50)
print("STAGE 3: Verifier Training")
print("="*50)

verifier_train_config = config.get_verifier_config()
verifier_train_config.output_dir = checkpoint_dir / 'tacc-full-verifier-v2'
verifier_train_config.output_dir.mkdir(parents=True, exist_ok=True)

verifier_config = VerifierConfig(
    vocab_size=50257,
    d_model=512,
    num_layers=4,
    n_heads=8,
    max_position_embeddings=config.max_seq_len,
)
verifier_dataset_config = VerifierDatasetConfig(
    max_length=config.max_seq_len,
)

verifier_ckpt = run_verifier_training(
    train_config=verifier_train_config,
    verifier_config=verifier_config,
    dataset_config=verifier_dataset_config,
    tokenizer_name=config.tokenizer_name,
)
print(f"Verifier complete: {verifier_ckpt}")

print("\n" + "="*50)
print("ALL ALIGNMENT STAGES COMPLETE!")
print("="*50)
print(f"Checkpoints saved:")
print(f"  SFT:      {sft_ckpt}")
print(f"  DPO:      {dpo_ckpt}")
print(f"  Verifier: {verifier_ckpt}")
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Pipeline finished at $(date)"
echo "=============================================="

