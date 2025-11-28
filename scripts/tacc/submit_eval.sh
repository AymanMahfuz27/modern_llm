#!/bin/bash
#SBATCH -J modern-llm-eval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ASC25078

set -euo pipefail

cd /home1/09999/aymanmahfuz/modern_llm

module load cuda/12.8
module load python3

source "${WORK}/modern_llm_venv/bin/activate"

echo "Running evaluation and comparison..."
python scripts/evaluate_and_compare.py

echo "Done!"

