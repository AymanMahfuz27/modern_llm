#!/bin/bash
#SBATCH -J modern-llm-h100
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
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

# ---- Run Full Pipeline (with resume) ----
STATE_FILE="${WORK_DIR}/checkpoints/pipeline_state.json"
DPO_CKPT=""

if [ -f "${STATE_FILE}" ]; then
    echo "Existing pipeline state found at ${STATE_FILE}, checking for completed DPO stage..."
    DPO_CKPT=$(python - "${STATE_FILE}" << 'PY'
import json, sys
from pathlib import Path

state_path = Path(sys.argv[1])
try:
    with state_path.open() as f:
        data = json.load(f)
except Exception:
    print("")
    sys.exit(0)

ckpt = data.get("dpo_checkpoint") or ""
if ckpt and Path(ckpt).exists():
    print(ckpt)
else:
    print("")
PY
)
fi

if [ -n "${DPO_CKPT}" ]; then
    echo "Full pipeline already completed. Reusing DPO checkpoint: ${DPO_CKPT}"
else
    echo "=============================================="
    echo "Running Full Pipeline (Pretrain -> SFT -> DPO -> Verifier)"
    echo "=============================================="

    # Extended training: 80K pretrain, 10K SFT, 6K DPO
    # Diverse pretraining with downsampled TinyStories (100K samples to avoid story-mode collapse)
    python -u scripts/run_pipeline.py \
        --config gpu \
        --stage all \
        --output-dir "${WORK_DIR}/checkpoints" \
        --pretrain-steps 80000 \
        --sft-steps 10000 \
        --dpo-steps 6000 \
        --pretrain-datasets "wikitext-103-raw-v1,openwebtext,wikipedia,roneneldan/TinyStories:100000" \
        | tee "${WORK_DIR}/logs/pipeline_${SLURM_JOB_ID}.log"

    echo ""
    echo "=============================================="
    echo "Pipeline Training Complete!"
    echo "=============================================="

    # Reload state to find DPO checkpoint after training
    if [ -f "${STATE_FILE}" ]; then
        DPO_CKPT=$(python - "${STATE_FILE}" << 'PY'
import json, sys
from pathlib import Path

state_path = Path(sys.argv[1])
try:
    with state_path.open() as f:
        data = json.load(f)
except Exception:
    print("")
    sys.exit(0)

ckpt = data.get("dpo_checkpoint") or ""
if ckpt and Path(ckpt).exists():
    print(ckpt)
else:
    print("")
PY
)
    fi
fi

# ---- Run Evaluation ----
echo ""
echo "=============================================="
echo "Running Task Evaluations"
echo "=============================================="

# Use DPO checkpoint from pipeline state if available
FINAL_CKPT="${DPO_CKPT}"

if [ -n "${FINAL_CKPT}" ]; then
    echo "Found DPO checkpoint for evaluation: ${FINAL_CKPT}"
    
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

# ---- Generate Final Report ----
echo ""
echo "=============================================="
echo "Generating Final Report"
echo "=============================================="

python - "${STATE_FILE}" "${WORK_DIR}/results" "${WORK_DIR}/figures" "${PROJECT_DIR}/report/final_tacc_report.md" << 'REPORT_PY'
import json
import sys
from pathlib import Path
from datetime import datetime

state_path = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
figures_dir = Path(sys.argv[3])
output_path = Path(sys.argv[4])

# Load pipeline state
state = {}
if state_path.exists():
    with open(state_path) as f:
        state = json.load(f)

# Load task metrics
task_metrics = []
metrics_path = results_dir / "task_metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        task_metrics = json.load(f)

# Load attention summary
attention_summary = {}
attn_path = figures_dir / "attention_summary.json"
if attn_path.exists():
    with open(attn_path) as f:
        attention_summary = json.load(f)

# Build metrics lookup
stage_metrics = {}
baseline_metrics = {}
for m in task_metrics:
    if m.get("is_hf_baseline"):
        baseline_metrics[m["model"]] = m
    elif m.get("stage"):
        stage_metrics[m["stage"]] = m

# Generate report
lines = [
    "# Modern LLM Final Report (TACC H100 Run)",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "---",
    "",
    "## Architecture",
    "",
    "- **Model type**: Decoder-only Transformer (GPT-style)",
    "- **Parameters**: ~253M (d_model=1024, n_layers=12, n_heads=16, ffn=4096)",
    "- **Vocab size**: 50,257 (GPT-2 tokenizer)",
    "- **Max sequence length**: 1024 tokens",
    "- **Features**: RoPE, RMSNorm, SwiGLU, Flash Attention (SDPA)",
    "",
    "---",
    "",
    "## Training Pipeline",
    "",
    "### Stage 1: Pretraining",
    "- **Datasets**: WikiText-2 + WikiText-103 + TinyStories",
    "- **Steps**: 40,000",
    "- **Batch size**: 128 (micro=32)",
    "- **Learning rate**: 3e-4",
    f"- **Checkpoint**: `{state.get('pretrain_checkpoint', 'N/A')}`",
    "",
    "### Stage 2: SFT",
    "- **Dataset**: tatsu-lab/alpaca (52K)",
    "- **Steps**: 5,000",
    f"- **Checkpoint**: `{state.get('sft_checkpoint', 'N/A')}`",
    "",
    "### Stage 3: DPO",
    "- **Dataset**: Anthropic/hh-rlhf",
    "- **Steps**: 3,000",
    "- **Beta**: 0.1",
    f"- **Checkpoint**: `{state.get('dpo_checkpoint', 'N/A')}`",
    "",
    "### Stage 4: Verifier",
    "- **Dataset**: GSM8K",
    "- **Steps**: 3,000",
    f"- **Checkpoint**: `{state.get('verifier_checkpoint', 'N/A')}`",
    "",
    "---",
    "",
    "## Results",
    "",
    "### Task Metrics",
    "",
    "| Model | SST-2 Acc | GSM8K EM | Notes |",
    "|-------|-----------|----------|-------|",
]

# Add baselines
for name in ["gpt2", "distilgpt2"]:
    m = baseline_metrics.get(name, {})
    sst2 = f"{m.get('sst2_accuracy', 0):.1%}" if m.get('sst2_accuracy') else "N/A"
    lines.append(f"| {name} | {sst2} | N/A | HF Baseline |")

# Add stages
for stage in ["pretrain", "sft", "dpo"]:
    m = stage_metrics.get(stage, {})
    sst2 = f"{m.get('sst2_accuracy', 0):.1%}" if m.get('sst2_accuracy') else "N/A"
    gsm8k = f"{m.get('gsm8k_em', 0):.1%}" if m.get('gsm8k_em') else "N/A"
    lines.append(f"| Our {stage.upper()} | {sst2} | {gsm8k} | |")

lines.extend([
    "",
    "### Stage-wise Gains",
    "",
    "| Stage | SST-2 Acc | GSM8K EM | Δ SST-2 | Δ GSM8K |",
    "|-------|-----------|----------|---------|---------|",
])

prev_sst2, prev_gsm8k = 0, 0
for stage in ["pretrain", "sft", "dpo"]:
    m = stage_metrics.get(stage, {})
    sst2 = m.get("sst2_accuracy", 0)
    gsm8k = m.get("gsm8k_em", 0)
    d_sst2 = sst2 - prev_sst2 if prev_sst2 else 0
    d_gsm8k = gsm8k - prev_gsm8k if prev_gsm8k else 0
    lines.append(f"| {stage.upper()} | {sst2:.1%} | {gsm8k:.1%} | {d_sst2:+.1%} | {d_gsm8k:+.1%} |")
    prev_sst2, prev_gsm8k = sst2, gsm8k

lines.extend([
    "",
    "---",
    "",
    "## Interpretability",
    "",
    "### Attention Patterns",
    "",
])

if attention_summary:
    for ex in attention_summary.get("examples", []):
        lines.append(f"- **{ex.get('name', 'Unknown')}**: self-attn={ex.get('self_attention_ratio', 0):.3f}, local={ex.get('local_attention_ratio', 0):.3f}, entropy={ex.get('attention_entropy', 0):.3f}")
    lines.append("")
    obs_path = figures_dir / "attention_observations.md"
    if obs_path.exists():
        lines.append("**Observations** (see `report/figures/attention_observations.md`):")
        with open(obs_path) as f:
            for line in f.readlines()[:10]:
                lines.append(f"  {line.rstrip()}")
else:
    lines.append("*Attention analysis pending*")

lines.extend([
    "",
    "---",
    "",
    "## Figures",
    "",
    "- `report/figures/attention_sentiment_positive.png`",
    "- `report/figures/attention_sentiment_negative.png`",
    "- `report/figures/attention_entity.png`",
    "- `report/figures/attention_math.png`",
    "",
    "---",
    "",
    "## Reproducibility",
    "",
    "```bash",
    "# Full pipeline",
    "sbatch scripts/tacc_pipeline.slurm",
    "",
    "# Local smoke test",
    "python scripts/run_pipeline.py --config local-smoke --stage all",
    "```",
    "",
    "- **Seed**: 42",
    "- **Hardware**: NVIDIA H100 PCIe 80GB",
    "- **CUDA**: 12.2",
    "",
])

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"Final report saved to: {output_path}")
REPORT_PY

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
mkdir -p "${PROJECT_DIR}/experiments/results"
mkdir -p "${PROJECT_DIR}/report/figures"
cp -r "${WORK_DIR}/results/"* "${PROJECT_DIR}/experiments/results/" 2>/dev/null || true
cp -r "${WORK_DIR}/figures/"* "${PROJECT_DIR}/report/figures/" 2>/dev/null || true
cp "${WORK_DIR}/checkpoints/pipeline_state.json" "${PROJECT_DIR}/experiments/" 2>/dev/null || true

echo "Results copied to ${PROJECT_DIR}/experiments/results/"
echo "Figures copied to ${PROJECT_DIR}/report/figures/"
echo "Final report at ${PROJECT_DIR}/report/final_tacc_report.md"
