<!-- 799ff3a7-ec85-4f4b-85e3-a0794f3edb4f 82dd2ddf-acc4-4dec-85a8-6007a3a0ff3e -->
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
- All `src/modern_llm/training/` (trainer, train_lm.py)
- All `src/modern_llm/alignment/` (DPO loss, pipeline stubs)
- All `src/modern_llm/data/` (dataset loaders)
- `src/modern_llm/hf/lora_utils.py` (useful for PEFT on your model)
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

### TACC job script template: `scripts/tacc/submit_pretrain.sh`

SLURM script that:

- Loads TACC modules (cuda, python)
- Activates venv
- Runs distributed training with torchrun
- Saves checkpoints to $SCRATCH

---

## 3. Phase 3: Alignment Pipeline Implementation

### 3.1 Supervised Fine-Tuning (SFT)

**File:** `src/modern_llm/training/train_sft.py`

Purpose: Instruction-tune YOUR scratch model on QA/dialog data.

Key components:

- Load your pretrained scratch checkpoint
- Use instruction dataset (Alpaca-style or OpenAssistant subset)
- Same Trainer abstraction, just different data format
- Save SFT checkpoint

**Dataset:** `src/modern_llm/data/instruction_datasets.py`

- Load from HF: `tatsu-lab/alpaca` or `OpenAssistant/oasst1`
- Format: `{"instruction": ..., "input": ..., "output": ...}`
- Tokenize with chat template

### 3.2 Direct Preference Optimization (DPO)

**File:** `src/modern_llm/training/train_dpo.py` (currently stub)

Implement full DPO training:

- Load SFT checkpoint as policy model
- Load preference pairs from `preference_datasets.py`
- Use existing `dpo_loss.py` (already implemented correctly)
- Compute log-probs for chosen/rejected
- Optimize with DPO objective

**Dataset options:**

- `Anthropic/hh-rlhf` (human preference data)
- Synthetic: generate pairs from your model + GPT-4 judge

### 3.3 Verifier Model

**File:** `src/modern_llm/models/verifier.py` (currently stub)

Complete implementation:

- Small encoder (reuse your architecture or use nn.TransformerEncoder)
- Binary classifier: correct/incorrect
- Scoring API: `score(problem: str, answer: str) -> float`

**Training:** `src/modern_llm/training/train_verifier.py`

- Dataset: GSM8K with correct/incorrect labels
- Train to predict answer correctness
- Use for reranking at inference

### 3.4 Alignment Pipeline Orchestration

**File:** `src/modern_llm/alignment/alignment_pipeline.py`

Implement end-to-end:

```
Base LM → SFT → DPO → +Verifier Reranking
```

For each stage:

- Load checkpoint
- Evaluate on GSM8K/math subset
- Log metrics (EM, verifier-boosted EM)
- Save comparison table

---

## 4. Nanochat-Style One-Button Pipeline

### Main entry point: `speedrun.sh`

```bash
#!/bin/bash
# Modern LLM: One-button frontier training pipeline
# Usage: bash speedrun.sh [local|tacc]

# 1. Setup environment
# 2. Pretrain (or load checkpoint)
# 3. SFT
# 4. DPO  
# 5. Train Verifier
# 6. Evaluate all stages
# 7. Generate report
```

### Supporting scripts in `scripts/`

- `scripts/pretrain.py` - Pretraining with config selection
- `scripts/sft.py` - SFT stage
- `scripts/dpo.py` - DPO stage
- `scripts/train_verifier.py` - Verifier training
- `scripts/evaluate_pipeline.py` - Full evaluation suite
- `scripts/generate_report.py` - Markdown report generation

### Report generation: `src/modern_llm/report.py`

Like nanochat, generate a `report.md` with:

- Model architecture summary
- Training curves
- Evaluation table (Base → SFT → DPO → +Verifier)
- Sample generations

---

## 5. Evaluation Suite

### Benchmarks for YOUR model

- **Perplexity:** WikiText-2 validation
- **Generation quality:** Qualitative samples
- **Math (GSM8K subset):** Raw EM, Verifier-reranked EM, Pass@N
- **Attention sinks:** Long-context stability test

### Comparison baselines

- GPT-2 (same size, zero-shot)
- Your model at each alignment stage

### Output: `experiments/results/`

- `base_eval.json`
- `sft_eval.json`
- `dpo_eval.json`
- `verifier_eval.json`
- `comparison_table.csv`

---

## 6. Directory Structure After Cleanup

```
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
│   ├── evaluate.py
│   └── tacc/            # SLURM job scripts
├── speedrun.sh          # One-button pipeline
├── experiments/
│   └── runs/scratch-*   # Your model checkpoints only
└── report/              # Generated reports
```

---

## Implementation Order

1. **Cleanup** - Delete Phase 2 clutter
2. **Hardware config** - Add multi-GPU/TACC support
3. **SFT implementation** - Instruction tuning for your model
4. **DPO implementation** - Preference optimization
5. **Verifier implementation** - Math/QA scoring
6. **Pipeline orchestration** - End-to-end flow
7. **Speedrun script** - One-button entry point
8. **Report generation** - Automated markdown report

### To-dos

- [ ] Add scripts/notebooks to evaluate existing LM checkpoints and collect loss/perplexity metrics for WikiText-2 runs
- [ ] Run a full GPT-2+LoRA SST-2 finetune and implement evaluation to report validation accuracy and sample errors
- [ ] Choose and implement one additional HF finetune (SAMSum or GSM8K) plus matching evaluation
- [ ] Implement minimal prompting baselines for SST-2 and the chosen second task using HF models
- [ ] Implement train_dpo.py using dpo_loss and a small preference dataset to run at least one DPO experiment
- [ ] Implement a small verifier model and training loop on a tiny math/QA correctness dataset, with a simple scoring API
- [ ] Wire up alignment_pipeline.py to run a reduced Base → DPO → +Verifier pipeline and log metrics
- [ ] Implement and run at least one advanced feature experiment (GQA or MoE) and summarize its impact
- [ ] Implement evaluation scripts, attention viz, error analysis, and draft the final ACL-style report and README updates