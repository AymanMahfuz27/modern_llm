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


