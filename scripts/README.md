# Scripts Directory

CLI entrypoints for training, evaluation, and orchestration.

---

## Quick Start

```bash
# Smoke test (5 min)
python scripts/run_pipeline.py --config local-smoke --stage all

# Full local training (RTX 3060)
python scripts/run_pipeline.py --config local --stage all

# TACC (submit SLURM job)
sbatch scripts/tacc/submit_speedrun.sh
```

---

## Main Entry Points

| Script | Description |
|--------|-------------|
| `run_pipeline.py` | **Unified entry point** - run any stage or full pipeline |
| `speedrun_pipeline.py` | Full pipeline orchestrator (called by `speedrun.sh`) |

### `run_pipeline.py` Usage

```bash
# Run individual stages
python scripts/run_pipeline.py --config local --stage pretrain
python scripts/run_pipeline.py --config local --stage sft --checkpoint path/to/pretrain.pt
python scripts/run_pipeline.py --config local --stage dpo --checkpoint path/to/sft.pt
python scripts/run_pipeline.py --config local --stage verifier

# Run full pipeline
python scripts/run_pipeline.py --config local --stage all

# Evaluate existing checkpoints
python scripts/run_pipeline.py --config local --stage eval
```

**Options:**
- `--config`: Config preset (`local-smoke`, `local`, `tacc-smoke`, `tacc`) or path to JSON
- `--stage`: Pipeline stage (`pretrain`, `sft`, `dpo`, `verifier`, `eval`, `all`)
- `--checkpoint`: Resume from checkpoint (required for SFT/DPO stages)
- `--output-dir`: Custom output directory
- `--max-steps`: Override training steps
- `--force`: Overwrite existing results

---

## Standalone Stage Scripts

For running individual stages with full control:

| Script | Description |
|--------|-------------|
| `pretrain.py` | Pretrain language model from random init |
| `sft.py` | Supervised fine-tuning on instructions |
| `dpo.py` | Direct preference optimization |
| `train_verifier.py` | Train answer correctness verifier |

---

## Evaluation Scripts

| Script | Description |
|--------|-------------|
| `evaluate_pipeline.py` | Evaluate all pipeline stages on metrics |
| `evaluate_and_compare.py` | Compare your model vs GPT-2 baseline |
| `evaluate_lm_checkpoints.py` | Compute perplexity for all checkpoints |
| `generate_from_checkpoints.py` | Generate text samples from checkpoints |
| `generate_report.py` | Generate markdown report |

### Comparing Against GPT-2

```bash
python scripts/evaluate_and_compare.py
```

This generates:
- `experiments/comparison_log_v2.txt` - Summary
- `experiments/results/<run>_eval.json` - Detailed metrics
- `report/<run>_report.md` - Markdown report

---

## Experiments

| Script | Description |
|--------|-------------|
| `experiment_attention_sinks.py` | Compare generation stability with/without attention sinks |

---

## Utilities

| Script | Description |
|--------|-------------|
| `setup_check.py` | Verify all dependencies are installed |

---

## TACC SLURM Scripts

Located in `scripts/tacc/`:

| Script | Description |
|--------|-------------|
| `submit_speedrun.sh` | Full pipeline (pretrain + SFT + DPO + verifier) |
| `submit_pretrain_only.sh` | Pretrain stage only |
| `submit_alignment.sh` | Alignment stages only (SFT + DPO + verifier) |
| `submit_smoke_test.sh` | Quick smoke test |
| `submit_eval.sh` | Evaluation only |

### Usage

```bash
cd scripts/tacc
sbatch submit_speedrun.sh        # Full pipeline
sbatch submit_pretrain_only.sh   # Then:
sbatch submit_alignment.sh       # After pretrain completes
```

---

## Config Presets

| Preset | Hardware | Duration | Description |
|--------|----------|----------|-------------|
| `local-smoke` | Any | ~5 min | Quick sanity check |
| `local` | RTX 3060 | ~24 hours | Full training |
| `tacc-smoke` | A100/H100 | ~10 min | TACC quick test |
| `tacc` | H100 | ~48 hours | Full TACC training |
