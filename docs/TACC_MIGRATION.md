# TACC Migration Guide

What worked on local 3060 and what changes for TACC H100.

---

## Verified Working Locally (RTX 3060)

### Full Pipeline Smoke Test ✅
- **Command**: `python scripts/speedrun_pipeline.py --config local-smoke`
- **Stages completed**:
  - Pretrain: 100 steps (WikiText-2, ~6 min)
  - SFT: 50 steps (Alpaca, ~12 sec)
  - DPO: 50 steps (HH-RLHF, ~12 sec)
  - Verifier: 50 steps (GSM8K, ~6 sec)
  - Evaluation: PPL computed for all stages
- **Checkpoints saved**: `experiments/runs/smoke-test/*/`
- **Total time**: ~10 minutes

### Evaluation Scripts ✅
- `scripts/evaluation/eval_sst2.py` - SST-2 few-shot classification
- `scripts/evaluation/eval_gsm8k.py` - GSM8K with verifier analysis
- `scripts/evaluation/evaluate_tasks.py` - Unified task runner
- GPT-2 baseline comparison works

### Visualization ✅
- `scripts/visualize_attention.py` - Generates attention heatmaps
- Output saved to `report/figures/`

### Dataset Loading ✅
- WikiText-2, WikiText-103: ✅
- TinyStories: ✅
- OpenWebText, BookCorpus: ❌ (legacy scripts not supported)
- Alpaca, HH-RLHF, GSM8K: ✅

---

## Changes for TACC


### 0. Update heading for tacc_pipeline.slurm to look like this instead


### 1. Checkpoint Directory
**Local**:
```bash
experiments/runs/<run_name>/
```

**TACC** (use $WORK, not $HOME - home has quota limits):
```bash
$WORK/modern_llm_runs/checkpoints/
```

The SLURM script (`scripts/tacc_pipeline.slurm`) already handles this.

### 2. Config Preset
**Local**:
```bash
python scripts/speedrun_pipeline.py --config local-smoke
```

**TACC**:
```bash
python scripts/speedrun_pipeline.py --config gpu
```

Key differences in `gpu` config:
- 40K pretrain steps (vs 100 for smoke)
- d_model=1024, n_layers=12 (vs 256, 4)
- batch_size=128 (vs 64)
- attention_sinks=False (for Flash Attention)

### 3. Virtual Environment Path
**Local**:
```bash
source .venv/bin/activate
```

**TACC**:
```bash
source .venv/bin/activate  # Same, just verify it exists
```

If not present, create:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Module Loading
TACC requires loading modules before running:
```bash
module purge
module load cuda/12.2
module load python3/3.11
```

This is handled in `scripts/tacc_pipeline.slurm`.

### 5. Job Submission
```bash
# Submit the full pipeline
sbatch scripts/tacc_pipeline.slurm

# Check status
squeue -u $USER

# View logs
tail -f logs/pipeline_*.out
```

### 6. Resume on Failure
The SLURM script creates marker files for each completed stage:
- `$WORK/modern_llm_runs/checkpoints/.pretrain_done`
- `$WORK/modern_llm_runs/checkpoints/.sft_done`
- `$WORK/modern_llm_runs/checkpoints/.dpo_done`
- `$WORK/modern_llm_runs/checkpoints/.verifier_done`

To resume after a failure, just resubmit:
```bash
sbatch scripts/tacc_pipeline.slurm
```

### 7. Time Estimates (H100)

| Stage | Steps | Estimated Time |
|-------|-------|----------------|
| Pretrain | 40,000 | ~17 hours |
| SFT | 5,000 | ~3 hours |
| DPO | 3,000 | ~2 hours |
| Verifier | 3,000 | ~2 hours |
| Evaluation | - | ~30 min |
| **Total** | - | **~25 hours** |

Well under the 48h limit.

ALSO ENSURE THAT THE report/final_report.md FILE IS UPDATED WITH A final_tacc_report.md SO WE HAVE BETTER REPORTING ON THE BIGGEST MODEL WE CAN AFFORD.
---

## Pre-flight Checklist for TACC

1. [ ] Transfer code to TACC:
   ```bash
   rsync -avz --exclude '.venv' --exclude 'experiments' . tacc:~/modern_llm/
   ```

2. [ ] Create venv on TACC:
   ```bash
   cd ~/modern_llm
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. [ ] Verify datasets load:
   ```bash
   python scripts/verify_datasets.py --quick
   ```

4. [ ] Create work directory:
   ```bash
   mkdir -p $WORK/modern_llm_runs/{checkpoints,logs,results,figures}
   ```

5. [ ] Submit job:
   ```bash
   sbatch scripts/tacc_pipeline.slurm
   ```

---

## Troubleshooting

### OOM on H100
Reduce batch size in `gpu_full_config()`:
```python
pretrain_batch_size=64,  # was 128
pretrain_micro_batch_size=4,  # was 8
```

### Training slowdown after 31K steps
This was observed previously. The `gpu` config now caps at 40K steps to avoid it.
If still slow, check:
- GPU utilization: `nvidia-smi`
- Memory fragmentation: restart job

### Checkpoint not found
The SLURM script looks for `*pretrain*.pt`, `*sft*.pt`, etc.
Verify files exist:
```bash
ls -la $WORK/modern_llm_runs/checkpoints/
```

---

## Results Location After Run

**TACC ($WORK)**:
```
$WORK/modern_llm_runs/
├── checkpoints/
│   ├── gpu-full-*-pretrain_final.pt
│   ├── gpu-full-*-sft_final.pt
│   ├── gpu-full-*-dpo_final.pt
│   └── gpu-full-*-verifier_final.pt
├── results/
│   ├── task_metrics.json
│   ├── baseline_comparison.md
│   └── stage_gains.md
└── figures/
    ├── attention_*.png
    └── attention_summary.json
```

The SLURM script copies results back to project directory for easy access.

