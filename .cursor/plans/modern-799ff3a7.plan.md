<!-- 799ff3a7-ec85-4f4b-85e3-a0794f3edb4f 1ab92e73-0bb2-42a6-b0c7-deb4b34a057c -->
# Phase 3: Frontier Alignment Pipeline

## 1. Codebase Cleanup

### Files to DELETE (Phase 2 HF training clutter)

These trained external HF models, not YOUR model - remove them:

**Scripts:**

- `scripts/finetune_sst2.py`, `scripts/finetune_samsum.py`, `scripts/finetune_gsm8k.py`
- `scripts/evaluate_hf_sst2.py`, `scripts/evaluate_hf_samsum.py`, `scripts/evaluate_hf_gsm8k.py`
- `scripts/run_phase1_only.sh`, `scripts/run_phase2_only.sh`, `scripts/run_phase1_phase2.sh`
- `scripts/run_phase2_full.sh`, `scripts/smoke_test_phase1_phase2.sh`, `scripts/run_all_evaluations.sh`

**Source (HF finetuning - not your model):**

- `src/modern_llm/hf/finetune_gpt2_sst2.py`
- `src/modern_llm/hf/finetune_t5_samsum.py`
- `src/modern_llm/hf/finetune_math_gsm8k.py`
- `src/modern_llm/hf/prompting_baselines.py`

**Root clutter:**

- `PHASE1_PHASE2_COMMANDS.md`, `PHASE1_PHASE2_IMPLEMENTATION_SUMMARY.md`

**Experiments (HF runs - keep scratch runs):**

- `experiments/runs/gpt2-*`, `experiments/runs/t5-*`, `experiments/runs/smoke-test-samsum`

### Files to KEEP

- All `src/modern_llm/models/` (your architecture)
- All `src/modern_llm/training/` (trainer, `train_lm.py`)
- All `src/modern_llm/alignment/` (DPO loss, pipeline stubs)
- All `src/modern_llm/data/` (dataset loaders)
- `src/modern_llm/hf/lora_utils.py` (useful for PEFT on your model if ever needed)
- `scripts/train_lm.py`, `scripts/train_lm_from_config.py`
- `scripts/compare_scratch_vs_gpt2.sh`, `scripts/evaluate_scratch_lm.sh`
- `experiments/runs/scratch-wikitext2-*` (your model checkpoints)

---

## 2. Multi-Hardware Training Support

### New config system in `src/modern_llm/config/hardware_config.py`

```python
@dataclass
class HardwareConfig:
    """Auto-detect or specify hardware for training."""
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_gpus: int = 1
    gpu_memory_gb: int = 12  # RTX 3060 default
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    
    # TACC-specific
    is_distributed: bool = False
    world_size: int = 1
```

### Preset configs in `configs/`

- `configs/local_rtx3060.json` - Current max config for local training
- `configs/tacc_a100.json` - Optimized for A100 (40GB/80GB)
- `configs/tacc_h100.json` - Optimized for H100 (80GB)

These configs also point to **data scale presets** (see Section 7): e.g., `tokens_target`, `dataset_mix` (WikiText-2 only vs. WikiText-2 + TinyStories + extra HF corpora), so you can run a small local pretrain or a large TACC-scale pretrain with the same code.

### TACC job script template: `scripts/tacc/submit_pretrain.sh`

SLURM script that:

- Loads TACC modules (CUDA, Python)
- Activates venv
- Runs distributed training with `torchrun` using the chosen hardware config
- Saves checkpoints to `$SCRATCH` so long runs are durable

Later sections (speedrun) will reuse the same hardware + data configs so **one codepath** works on both your RTX 3060 and TACC.

---

## 3. Phase 3: Alignment Pipeline Implementation (Your Model Only)

### 3.1 Supervised Fine-Tuning (SFT)

**File:** `src/modern_llm/training/train_sft.py`

Purpose: Instruction-tune **your scratch model** on QA/dialog data.

Key components:

- Load your pretrained scratch checkpoint (either local run or TACC run)
- Use instruction dataset (Alpaca-style or OpenAssistant subset)
- Reuse `Trainer` abstraction, just a different dataset + loss wiring
- Save SFT checkpoint (e.g., `scratch-sft_stepXXXX.pt`)

**Dataset:** `src/modern_llm/data/instruction_datasets.py`

- Load from HF: `tatsu-lab/alpaca` or `OpenAssistant/oasst1`
- Normalize to a unified format: `{"instruction": ..., "input": ..., "output": ...}`
- Tokenize with a simple chat template so future prompts are consistent

### 3.2 Direct Preference Optimization (DPO)

**File:** `src/modern_llm/training/train_dpo.py` (currently stub)

Implement full DPO training for **your SFT model**:

- Load SFT checkpoint as policy model
- Load preference pairs from `preference_datasets.py`
- Use existing `alignment/dpo_loss.py` (already implemented correctly)
- Compute log-probs for chosen/rejected responses
- Optimize with DPO objective, log loss and simple preference metrics

**Dataset options:**

- `Anthropic/hh-rlhf` (human preference data subset)
- Synthetic: generate prompt/answer pairs and label them with a stronger judge (e.g., GPT-4) if you want

### 3.3 Verifier Model

**File:** `src/modern_llm/models/verifier.py` (partially implemented)

Complete implementation:

- Small encoder (reuse your architecture ideas or use `nn.TransformerEncoder`)
- Binary classifier: correct/incorrect
- Scoring API: `score(problem: str, answer: str) -> float`

**Training:** `src/modern_llm/training/train_verifier.py`

- Dataset: GSM8K with correct/incorrect labels (generate incorrect variants)
- Train to predict answer correctness
- Save verifier checkpoint and expose a simple Python API for scoring

### 3.4 Alignment Pipeline Orchestration

**File:** `src/modern_llm/alignment/alignment_pipeline.py`

Implement end-to-end:

```text
Base LM → SFT → DPO → +Verifier Reranking
```

For each stage:

- Load checkpoint
- Evaluate on GSM8K/math subset
- Log metrics (EM, verifier-boosted EM, Pass@N)
- Save comparison table for the report

---

## 4. Nanochat-Style One-Button Pipeline

### Main entry point: `speedrun.sh`

```bash
#!/bin/bash
# Modern LLM: One-button frontier training pipeline
# Usage:
#   bash speedrun.sh local      # full pipeline on RTX 3060 (small config)
#   bash speedrun.sh tacc       # full pipeline on TACC (large config)
#   bash speedrun.sh tacc-smoke # short smoke test on TACC to catch errors

# 1. Setup environment (venv, deps)
# 2. Pretrain (or load existing checkpoint based on config)
# 3. SFT (instruction tuning)
# 4. DPO (preference optimization)
# 5. Train Verifier (math/QA correctness)
# 6. Evaluate all stages (Base, SFT, DPO, +Verifier)
# 7. Generate report.md with metrics + samples
```

This single script is what you launch once on TACC (via a small SLURM wrapper) to run the **entire pipeline end-to-end**. It also supports a `tacc-smoke` mode that runs tiny configs (few steps, tiny model) to verify everything works before you commit a long H100 run.

### Supporting scripts in `scripts/`

- `scripts/pretrain.py` - Pretraining with hardware + data config selection
- `scripts/sft.py` - SFT stage
- `scripts/dpo.py` - DPO stage
- `scripts/train_verifier.py` - Verifier training
- `scripts/evaluate_pipeline.py` - Full evaluation suite (Base, SFT, DPO, +Verifier)
- `scripts/generate_report.py` - Markdown report generation

Each of these can be run **individually** for debugging, or orchestrated by `speedrun.sh`.

### Report generation: `src/modern_llm/report.py`

Like nanochat, generate a `report.md` with:

- Model architecture summary (your ModernDecoderLM + verifier)
- Training curves (loss, perplexity, EM curves per stage)
- Evaluation table (Base → SFT → DPO → +Verifier)
- Sample generations and GSM8K examples with verifier scores

---

## 5. Evaluation Suite

### Benchmarks for YOUR model

- **Perplexity:** WikiText-2 validation
- **Generation quality:** Qualitative samples
- **Math (GSM8K subset):** Raw EM, Verifier-reranked EM, Pass@N
- **Attention sinks:** Long-context stability test (using your sinks implementation)

### Comparison baselines

- GPT-2 (same size, zero-shot) via `compare_scratch_vs_gpt2.sh`
- Your model at each alignment stage (Base, SFT, DPO, +Verifier)

### Output: `experiments/results/`

- `base_eval.json`
- `sft_eval.json`
- `dpo_eval.json`
- `verifier_eval.json`
- `comparison_table.csv`

These are the artifacts your ACL-style report and portfolio README will highlight.

---

## 6. Directory Structure After Cleanup

```text
modern_llm/
├── configs/
│   ├── local_rtx3060.json
│   ├── tacc_a100.json
│   └── tacc_h100.json
├── src/modern_llm/
│   ├── models/          # Your architecture (keep all)
│   ├── training/        # train_lm, train_sft, train_dpo, train_verifier
│   ├── alignment/       # dpo_loss, alignment_pipeline
│   ├── data/            # lm_datasets, instruction_datasets, preference_datasets
│   ├── evaluation/      # metrics, analysis, attention_viz
│   └── config/          # model_config, train_config, hardware_config
├── scripts/
│   ├── pretrain.py
│   ├── sft.py
│   ├── dpo.py
│   ├── evaluate_pipeline.py
│   ├── generate_report.py
│   └── tacc/            # SLURM job scripts (e.g., submit_speedrun.sh)
├── speedrun.sh          # One-button local/TACC pipeline (with smoke mode)
├── experiments/
│   ├── runs/scratch-*   # Your model checkpoints only
│   └── results/         # JSON/CSV evaluation tables
└── report/              # Generated reports
```

---

## 7. Data Scale & Local Inference Guarantees

To stay true to your proposal and hardware:

- **Configurable data scale**: Add `data_config` sections to your JSON configs that specify:
  - Which corpora to use: `wikitext-2`, `tinystories`, optional extra HF corpora (e.g., `openwebtext`, `bookcorpus` if feasible)
  - Target tokens (e.g., `tokens_target: 50M` for local, `tokens_target: 5B` for TACC)
  - Shuffling/epoch policy (avoid too many epochs over tiny data)
- **Local vs TACC presets**:
  - `local_rtx3060.json`: small-ish model (<= ~200M params), smaller token budget, guaranteed to fit inference + small finetunes on your 12GB GPU
  - `tacc_a100.json` / `tacc_h100.json`: larger token budget, possibly somewhat larger model depth, but **kept within a size that still fits inference on 12GB** (e.g., ~300–400M params max)
- **Inference tests**:
  - Add a small script/notebook that:
    - Loads the TACC-trained checkpoint on your 3060
    - Runs a short generation and GSM8K inference batch
    - Saves a simple `inference_ok.json` flag in `experiments/results/`

This ensures you can push pretraining hard on TACC while always being able to **run inference and alignment locally**.

---

## 8. Implementation Order

1. **Cleanup** - Delete Phase 2 clutter
2. **Hardware + data config** - Add multi-GPU/TACC + data scale support
3. **SFT implementation** - Instruction tuning for your model
4. **DPO implementation** - Preference optimization
5. **Verifier implementation** - Math/QA scoring
6. **Pipeline orchestration** - End-to-end flow (Base→SFT→DPO→Verifier)
7. **Speedrun + smoke mode** - One-button entry point for local and TACC
8. **Evaluation suite** - Standardized metrics across stages
9. **Report generation** - Automated markdown report for portfolio + paper

### To-dos

- [ ] Delete Phase 2 HF training files and clutter from scripts/, src/hf/, experiments/
- [ ] Add HardwareConfig and preset configs for RTX 3060, A100, H100 with data scale options
- [ ] Create SLURM job scripts for TACC (submit_speedrun.sh + any helpers)
- [ ] Implement train_sft.py with instruction dataset loading and Trainer wiring
- [ ] Implement train_dpo.py using existing dpo_loss.py and preference_datasets.py
- [ ] Complete verifier.py forward pass and train_verifier.py with GSM8K-based correctness labels
- [ ] Implement alignment_pipeline.py for Base->SFT->DPO->Verifier flow and metrics logging
- [ ] Create speedrun.sh with local/tacc/tacc-smoke modes that orchestrate the full pipeline
- [ ] Implement evaluation suite (evaluate_pipeline.py) for Base/SFT/DPO/Verifier and comparison_table.csv
- [ ] Add report generation (report.py + generate_report.py) with metrics tables and qualitative samples