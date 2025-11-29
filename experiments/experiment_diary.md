## Experiment Diary

### 2025-11-18 — Phase 1 LM training + first generations

- **Context**: Finished Phase 0–1 scaffolding for `ModernDecoderLM` (configs, data loader, trainer, and model) and verified end-to-end training on WikiText-2.
- **Hardware**: AMD Ryzen 9 7900X (12C/24T), 61 GiB RAM, NVIDIA RTX 3060 12 GiB.
- **Model**: Decoder-only Transformer with RMSNorm, SwiGLU FFN, RoPE, causal masking, optional attention sinks/GQA/MoE wired but disabled.
- **Run**: `scratch-wikitext2-medium` style configuration (e.g., `d_model≈256`, `n_layers≈4`, `n_heads≈4`, `max_seq_len≈256`, batch size in the teens, a few thousand steps on WikiText-2).
- **Training loop**: Custom `Trainer` with gradient accumulation, optional AMP, learning-rate warmup, AdamW, gradient clipping, tqdm progress bar, and periodic logging.
- **Generation setup**: After training, the script calls `generate_text(...)` with a short prompt (e.g., `\"The meaning of life is\"` or soccer-related text) and samples ~80 tokens using temperature + top-k from the trained scratch model.
- **Observed behavior**: Samples show coherent sentence structure and domain-specific phrases (e.g., football commentary and historical references), with locally consistent grammar but global drift and repetition—typical of a small LM at this training depth.
- **Interpretation**: Even a relatively small decoder-only model with modern components (RMSNorm, SwiGLU, RoPE) learns strong local patterns on WikiText-2 with a few thousand steps, confirming that the architecture + data + trainer are all wired correctly.
- **Next steps**:
  - Run more systematic long-context evaluations and log train/validation perplexity over steps.
  - Add small utilities/notebooks to visualize loss curves and show qualitative generations side-by-side for different checkpoints.
  - Move into Phase 2: implement LoRA-based HF finetuning (starting with SST-2) and compare scratch vs. HF baselines on shared metrics.

### 2025-11-18 — Phase 2 start: GPT-2 LoRA finetuning on SST-2

- **Goal**: Establish a strong yet lightweight HF baseline by finetuning GPT-2 on SST-2 with LoRA, using the same training loop abstractions as the scratch LM.
- **Model**: `AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)` with pad token set to EOS; LoRA applied to attention projections (e.g., `c_attn`, `c_proj`) via `LoraConfig(task_type="SEQ_CLS")`.
- **Data**: GLUE SST-2 via `TaskDatasetConfig` + `load_supervised_text_dataset` with `text_fields=("sentence",)` and `task_type="classification"`, producing `input_ids`, `attention_mask`, `labels` tensors.
- **Training loop**: Reused `Trainer` with `TrainingConfig` (AdamW, warmup schedule, gradient accumulation, tqdm progress bar). A smoke run (`max_steps≈50`, `batch_size≈8`) showed rapid loss drop from ~1.1 toward the 0.2–0.6 range.
- **Outcome**: Confirms the HF + LoRA pipeline is wired correctly, the dataset helpers generalize beyond pure LM, and GPT-2 quickly adapts to binary sentiment classification under the shared training abstraction.
- **Next steps**:
  - Run a longer GPT-2 LoRA finetune to reach competitive SST-2 accuracy (to be quantified later via `evaluation.metrics`).
  - Design and implement a scratch `ModernDecoderLM` adaptation to sentiment (e.g., generative \"Review ... Sentiment:\" prompts) for direct comparison against GPT-2 LoRA.

### 2025-11-21 — Phase 1 & 2 implementation complete

- **Phase 1 additions**:
  - Created `scripts/evaluate_lm_checkpoints.py` to compute validation loss/perplexity for all saved checkpoints and write CSV tables.
  - Created `scripts/generate_from_checkpoints.py` to sample text from checkpoints with fixed prompts for qualitative comparison.
  - Designed `configs/lm_max_rtx3060.json` for a max-size model (768-dim, 12-layer, 12-head, 1024 context) pushing RTX 3060 limits with gradient accumulation and BF16.
  - Implemented `scripts/experiment_attention_sinks.py` to train two small models (with/without sinks) and compare generation stability at 3x trained context.
  - Added `notebooks/visualize_lm_results.py` to plot loss/perplexity curves from checkpoint metrics.

- **Phase 2 additions**:
  - Completed `hf/finetune_t5_samsum.py` for T5/FLAN-T5 LoRA finetuning on SAMSum summarization.
  - Completed `hf/finetune_math_gsm8k.py` for GPT-2/TinyLlama LoRA finetuning on GSM8K math reasoning.
  - Completed `hf/prompting_baselines.py` to run zero-/few-shot prompting on SST-2, SAMSum, and GSM8K.
  - Extended `evaluation/metrics.py` to support ROUGE metrics for summarization and F1 for classification.
  - Created evaluation scripts: `evaluate_hf_sst2.py`, `evaluate_hf_samsum.py`, `evaluate_hf_gsm8k.py` with accuracy/ROUGE/EM computation and error logging.

- **Status**: All Phase 1 and Phase 2 baseline experiments are now runnable end-to-end. Ready to execute training runs and collect metrics for the report.

- **Next steps**: Execute the training runs, then move to Phase 3 (SFT → DPO → Verifier alignment pipeline).

### 2025-11-28 — Full pipeline complete

- **Hardware**: H100 GPU (80GB VRAM)
- **Training**: Full alignment pipeline executed:
  - Pretrain: 30,000 steps on WikiText-103 + TinyStories (~600M tokens)
  - SFT: 5,000 steps on Alpaca 52K
  - DPO: 2,000 steps on HH-RLHF
  - Verifier: 3,000 steps on GSM8K

- **Results** (WikiText-2 perplexity):
  | Model | Params | PPL |
  |-------|--------|-----|
  | GPT-2 (baseline) | 124M | 40.64 |
  | **Ours (pretrain)** | 253M | **27.03** |
  | Ours (SFT) | 253M | 34.14 |
  | Ours (DPO) | 253M | 34.32 |

- **Key findings**:
  - Pretrain model outperforms GPT-2 by 33% on perplexity despite being trained from scratch
  - SFT/DPO increase perplexity (expected: alignment trades raw LM ability for instruction-following)
  - Attention sinks + RoPE enable stable generation beyond training context

- **Architecture validated**:
  - RMSNorm + SwiGLU + RoPE + attention sinks work well together
  - Flash Attention (SDPA) provides 2-4x speedup
  - GQA/MoE hooks are wired but not used in final model

### 2025-11-29 — Final cleanup and portfolio preparation

- **Cleanup tasks completed**:
  - Deleted orphaned files: `run_alignment_pipeline.py`, `run_ablations.py`
  - Deleted Phase 2 HF artifacts from experiments/ (not relevant to final project)
  - Created unified `scripts/run_pipeline.py` entry point
  - Added overwrite protection to report generation
  - Wrote portfolio-grade README.md with architecture deep-dive
  - Removed cluster-specific code and configs

- **Final project structure**:
  - Single Python entry point: `scripts/run_pipeline.py`
  - Pre-trained checkpoints in `checkpoints/`
  - Protected results in `experiments/results/`, `report/`, `comparison_log_v2.txt`

- **Status**: Project complete. All code paths verified, documentation updated, ready for submission.
