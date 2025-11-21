# Modern LLM

Modern LLM is a from-scratch implementation of a modern decoder-only language model, plus a full training and evaluation stack that mirrors how small-scale frontier-style systems are built: custom Transformer architecture, shared training loop, HF LoRA baselines, and an alignment pipeline scaffold (SFT → DPO → verifier).

The code is written for inspection. Each major component points back to specific papers and equations, and the repository is structured so a reviewer can drop directly into the part they care about: architecture, training, evaluation, or alignment.

---

## What this repo contains

- **Custom decoder-only LM** with:
  - RoPE positional embeddings
  - RMSNorm instead of LayerNorm
  - SwiGLU feedforward blocks
  - Attention sinks for long-context stability
  - Optional Grouped Query Attention (GQA) and Mixture-of-Experts (MoE) hooks
- **Shared training loop** with gradient accumulation, mixed precision, checkpointing, and evaluation.
- **HF finetuning stack**:
  - GPT‑2 + LoRA on SST‑2 (classification)
  - T5‑small + LoRA on SAMSum (summarization)
  - GPT‑2 + LoRA on GSM8K (math reasoning)
  - Zero-/few-shot prompting baselines for all three tasks.
- **Alignment scaffolding**:
  - DPO loss implementation
  - Verifier model skeleton for math/QA correctness
  - Alignment pipeline entrypoint (Base → SFT → DPO → Verifier).
- **Experiment orchestration**:
  - One-command scripts to run Phase 1 (scratch LM) and Phase 2 (HF + prompting), plus evaluation-only runners.

This is a single-GPU, RTX‑3060‑calibrated codebase; configs and scripts are tuned to that constraint.

---

## Code map

### Core package: `src/modern_llm/`

- `models/`
  - `transformer.py`
    - `ModernDecoderLM`: the main decoder-only LM.
    - Uses:
      - `DecoderBlock` (RMSNorm → MultiHeadAttention → RMSNorm → SwiGLU)
      - RoPE for positional information
      - attention sinks / GQA / MoE toggles via `ModernLLMConfig`.
  - `attention.py`
    - `AttentionConfig`, `MultiHeadAttention`:
      - Scaled dot-product attention per Vaswani et al. (2017).
      - RoPE (Su et al. 2021) applied to Q/K.
      - Attention sinks (Press et al. 2021) via learnable “sink” tokens.
      - Optional GQA (shared K/V heads across Q heads) to reduce KV cache.
  - `layers.py`
    - `RMSNorm`: root-mean-square normalization (Zhang & Sennrich 2019).
    - `SwiGLU`: SwiGLU feedforward module (Shazeer 2020; PaLM 2022).
  - `moe.py`
    - MoE FFN scaffold: top‑k expert routing; wired but optional in configs.
  - `verifier.py`
    - `VerifierConfig`, `VerifierModel`: encoder-only classifier for correctness scoring on (problem, answer) pairs.
    - Architecture: small Transformer encoder + linear head; forward is intentionally left for alignment work (Phase 3).

- `config/`
  - `model_config.py`
    - `ModernLLMConfig`, `MoEConfig`: strict-validated hyperparameters for the decoder LM (dimensions, RoPE, sinks, GQA, MoE, etc.).
  - `train_config.py`
    - `TrainingConfig`: batch sizes, gradient accumulation, LR schedule, logging/ckpt cadence, mixed precision.

- `data/`
  - `lm_datasets.py`
    - `LanguageModelingDatasetConfig`, `load_causal_lm_dataset`:
      - WikiText‑2 / TinyStories style language modeling.
      - Returns tokenized `datasets.Dataset` with `input_ids`, `attention_mask`, `labels`.
  - `task_datasets.py`
    - `TaskDatasetConfig`, `load_supervised_text_dataset`:
      - SST‑2, SAMSum, GSM8K, etc. via Hugging Face `datasets`.
      - Handles both classification (`task_type="classification"`) and seq2seq (`task_type="seq2seq"`).
  - `preference_datasets.py`
    - `PreferenceDatasetConfig`, `load_preference_dataset`:
      - Loader for (prompt, chosen, rejected) pairs for DPO.

- `training/`
  - `trainer_base.py`
    - `Trainer`: single place where optimization happens.
      - Gradient accumulation
      - Mixed precision (fp16/bf16)
      - Grad clipping
      - LR warmup
      - TQDM progress bar with remaining steps
      - Periodic eval + checkpointing.
  - `train_lm.py`
    - CLI entrypoint for language modeling on WikiText‑2 / TinyStories.
    - Wires `ModernDecoderLM` + LM datasets + `Trainer`.
    - Includes `generate_text(...)` helper used across the repo.
  - `train_sft.py`, `train_dpo.py`, `train_verifier.py`
    - Stubs for Phase 3 alignment work (SFT, DPO training loop, verifier training).

- `hf/`
  - `lora_utils.py`
    - `LoraConfig`, `prepare_lora_model`: thin wrapper around `peft` to inject LoRA adapters into HF models (GPT‑2, T5, etc.).
  - `finetune_gpt2_sst2.py`
    - GPT‑2 / DistilGPT2 + LoRA on SST‑2 (GLUE).
    - Reuses `TaskDatasetConfig` + `Trainer` for consistency with scratch LM.
  - `finetune_t5_samsum.py`
    - T5‑small / FLAN‑T5 + optional LoRA on SAMSum summarization.
    - Supports a dummy in‑memory dataset mode for fast smoke tests; full SAMSum via `datasets` for real runs.
  - `finetune_math_gsm8k.py`
    - GPT‑2 / TinyLlama (via `AutoModelForCausalLM`) + LoRA on GSM8K subset.
    - Generates EM-style math accuracy metrics.
  - `prompting_baselines.py`
    - Zero-/few-shot prompting for SST‑2, SAMSum, GSM8K.
    - Uses the same `evaluation.metrics` utilities so prompting vs finetuning vs scratch are directly comparable.

- `alignment/`
  - `dpo_loss.py`
    - Implementation of Direct Preference Optimization objective (Rafailov et al. 2023).
  - `alignment_pipeline.py`
    - Orchestrator stub for Base → SFT → DPO → Verifier. To be filled in Phase 3 with real runs.

- `evaluation/`
  - `metrics.py`
    - `EvaluationResult`, `compute_metrics`, `compute_f1`:
      - Accuracy and macro‑F1 for classification (SST‑2).
      - Exact match for GSM8K/math.
      - ROUGE‑1/2/L for summarization (via `evaluate` + `rouge-score`).
  - `analysis.py`, `attention_viz.py`, `verifier_eval.py`
    - Stubs/utility scaffolds for error analysis, attention visualization, and verifier ablations.

- `utils/`
  - `checkpointing.py`
    - `save_checkpoint`, `load_checkpoint`: central checkpoint IO.
    - New checkpoints store `model_state`, optimizer state, and model config (for rebuilding `ModernDecoderLM`).
  - `logging_utils.py`
    - Creates per‑run loggers used by `Trainer`.

---

## Scripts and orchestration

All of the “glue” lives under `scripts/`. They are intentionally thin and call into `modern_llm` rather than re‑implementing logic.

- **Phase 1 – scratch LM:**
  - `train_lm_from_config.py`
    - Reads a JSON config (e.g. `configs/lm_max_rtx3060.json`) and runs full LM training.
  - `evaluate_lm_checkpoints.py`
    - Iterates over `experiments/runs/**.pt`, runs validation, writes CSV of loss/perplexity.
  - `generate_from_checkpoints.py`
    - Samples from checkpoints on fixed prompts, writes JSON with generations.
  - `experiment_attention_sinks.py`
    - Trains small models with/without attention sinks and samples at extended context to compare stability.

- **Phase 2 – HF finetuning + prompting:**
  - `evaluate_hf_sst2.py`
    - Loads a GPT‑2/DistilGPT2 SST‑2 checkpoint and computes accuracy + macro‑F1; logs misclassified examples.
  - `evaluate_hf_samsum.py`
    - Loads a T5 SAMSum checkpoint and computes ROUGE‑1/2/L.
  - `evaluate_hf_gsm8k.py`
    - Evaluates a GSM8K checkpoint, extracting numeric answers and computing EM; logs error cases.
  - `setup_check.py`
    - Quick import check for `torch`, `transformers`, `datasets`, `peft`, `evaluate`, `rouge_score`, etc.

- **One-button runners:**
  - `run_phase1_phase2.sh`
    - Runs:
      - LM checkpoint eval + generation
      - Max‑size LM training on WikiText‑2
      - Attention sinks experiment
      - GPT‑2 LoRA SST‑2
      - T5 LoRA SAMSum
      - GPT‑2 LoRA GSM8K
      - Prompting baselines
  - `run_phase1_only.sh` / `run_phase2_only.sh`
    - Isolated Phase 1 or Phase 2 baselines.
  - `run_all_evaluations.sh`
    - Re‑runs all evaluation scripts assuming checkpoints already exist.
  - `smoke_test_phase1_phase2.sh`
    - Short, low‑step sanity check that all paths (LM, GPT‑2 SST‑2, T5 SAMSum) are wired correctly.
  - `run_experiments.py`
    - Python entrypoint mirroring the bash scripts:
      - `--phase 1` / `2` / `all` / `eval` / `smoke`.

---

## Model choices (why this architecture)

- **RMSNorm + SwiGLU**
  - Matches the “modern” design of PaLM/LLaMA‑style models and has better training stability than vanilla LayerNorm + GELU at comparable parameter counts.
  - Implemented explicitly with equations in `layers.py` so reviewers can map code ↔ paper.

- **RoPE positional embeddings**
  - Used by GPT‑NeoX/LLaMA; improves extrapolation vs fixed absolute embeddings.
  - Implemented inside `MultiHeadAttention` in `attention.py`, following Su et al. (2021).

- **Attention sinks**
  - Inspired by Press et al. (2021) to improve generation stability beyond the training context.
  - Configurable via `ModernLLMConfig.use_attention_sinks` and demonstrated by `experiment_attention_sinks.py`.

- **GQA and MoE toggles**
  - GQA (grouped K/V heads) reduces KV cache memory; MoE allows sparsely activated FFNs.
  - Both are wired into the configs and attention/FFN layers so ablations can be done without touching core training code.

- **Shared Trainer**
  - The same `Trainer` drives:
    - Scratch LM training
    - HF GPT‑2 SST‑2 finetuning
    - T5 SAMSum
    - GSM8K math runs.
  - This keeps comparisons about *models* and *objectives* rather than “different training code.”

---

## Getting started

Create and activate a virtualenv, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
pytest  # optional: run quick tests
```

---

## Running experiments

### End-to-end Phase 1 & 2 (baseline stack)

```bash
cd /home/ayman/modern_llm  # adjust if needed
bash scripts/run_phase1_phase2.sh
```

This:
- Evaluates existing LM checkpoints and generates samples.
- Trains a max‑size scratch LM on WikiText‑2 (`configs/lm_max_rtx3060.json`).
- Runs the attention sinks experiment.
- Runs GPT‑2 + LoRA on SST‑2, T5 + LoRA on SAMSum, GPT‑2 + LoRA on GSM8K.
- Runs prompting baselines for all three tasks.

### Per-phase

```bash
# Phase 1 only (scratch LM + max-size model + attention sinks)
bash scripts/run_phase1_only.sh

# Phase 2 only (HF finetuning + prompting baselines)
bash scripts/run_phase2_only.sh

# Evaluations only (if checkpoints already exist)
bash scripts/run_all_evaluations.sh
```

### Python-only orchestration

```bash
python scripts/run_experiments.py --phase smoke   # quick sanity run
python scripts/run_experiments.py --phase all     # full Phase 1 + 2
```

### Max-size scratch LM (direct)

```bash
python scripts/train_lm_from_config.py --config configs/lm_max_rtx3060.json
```

This uses a ~PaLM/LLaMA‑style stack (RMSNorm + SwiGLU + RoPE + sinks) at a size calibrated for a single RTX 3060 (12 GB) and runs for many hours to reach a strong WikiText‑2 baseline.

---

## Status and next work

- **Implemented and exercised:**
  - Modern decoder-only LM with RoPE, RMSNorm, SwiGLU, attention sinks, optional GQA/MoE.
  - Shared Trainer with AMP, grad accumulation, and checkpointing.
  - GPT‑2 + LoRA on SST‑2, T5 + LoRA on SAMSum (with synthetic smoke mode), GPT‑2 + LoRA on GSM8K.
  - Prompting baselines for SST‑2/SAMSum/GSM8K.
  - Evaluation scripts for all of the above.

- **Scaffolded for Phase 3+ (alignment & advanced features):**
  - DPO loss and preference dataset loader.
  - Verifier architecture and training entrypoint.
  - Alignment pipeline orchestrator.
  - GQA/MoE ablation hooks and attention visualization utilities.

The current training run focuses on the max‑size scratch LM; as it completes, the same evaluation scripts will populate experiment tables under `experiments/` that can be dropped directly into the final report or portfolio. The alignment and verifier pieces are wired but intentionally left for the next phase of work so they can be developed against a stable base LM + HF baseline stack.
