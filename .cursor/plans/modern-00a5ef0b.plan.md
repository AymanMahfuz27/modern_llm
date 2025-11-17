<!-- 00a5ef0b-8f02-44ec-918c-ba5e030a7e59 b834a6cf-b493-44e8-95eb-3924f9543eb2 -->
# Modern LLM Project Plan

## High-level phases

- **Phase 0 — Environment & scaffolding**: Set up dependencies, repo structure, and configuration so you can focus only on model and experiments afterwards.
- **Phase 1 — From-scratch Transformer LM**: Implement a clean decoder-only Transformer with RoPE, RMSNorm, SwiGLU, attention sinks, and training on WikiText-2/TinyStories.
- **Phase 2 — HF finetuning & prompting**: LoRA/QLoRA-based finetuning of GPT-2/TinyLlama/T5-small on SST-2, SAMSum, GSM8K, plus prompting baselines.
- **Phase 3 — Alignment pipeline**: SFT → DPO → verifier for math/QA tasks, with a minimal but clear PyTorch/DPO implementation.
- **Phase 4 — Advanced features & ablations**: GQA/MoE, test-time compute scaling, structured outputs, and ablations (with/without RoPE, sinks, DPO, verifier).
- **Phase 5 — Evaluation, visualization, and report**: Metric tables, attention viz, error analysis, final ACL-style paper, and polished README/demo.

## Repository and directory structure

### Top-level layout

- `README.md`: High-level overview, quickstart, experiment recipes (copy-pastable commands).
- `requirements.txt` or `pyproject.toml`: Pin core dependencies (`torch`, `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `evaluate`, `rouge-score`, etc.).
- `setup.cfg`/`pyproject.toml` (optional): Tooling config (ruff/black/mypy) if you decide to enforce style.
- `src/modern_llm/`: Main Python package with all code.
- `scripts/`: Thin CLI entrypoints that import from `modern_llm` (no heavy logic here).
- `configs/`: Simple JSON/YAML or Python dataclass configs for models and experiments.
- `experiments/`: Saved configs, run logs, and result summaries (CSV/JSON).
- `data/`: Downloaded/processed datasets (or symbolic links) — kept out of git or gitignored.
- `tests/`: Unit tests for core model blocks, training utilities, and evaluation.
- `report/`: Drafts and final ACL-style paper, plus figure-generating notebooks.
- `notebooks/`: Optional EDA, sanity-check notebooks, and demo interface.

### `src/modern_llm/` package structure

- `__init__.py`: Lightweight; expose key components (`ModernLLMConfig`, `ModernDecoderLM`, etc.).
- `config/`:
- `model_config.py`: Dataclasses for model hyperparameters (d_model, n_layers, RoPE, GQA/MoE toggles).
- `train_config.py`: Dataclasses for training hyperparams (batch sizes, lr, schedulers, paths).
- `data/`:
- `lm_datasets.py`: WikiText-2/TinyStories loading, tokenization, and causal LM batching.
- `task_datasets.py`: SST-2, SAMSum, GSM8K, SVAMP/MAWPS loaders using `datasets` with clear task-specific collators.
- `preference_datasets.py`: Loader for DPO preference pairs (synthetic or open-source).
- `models/`:
- `layers.py`: Core building blocks — `RMSNorm`, `SwiGLU`, attention (including RoPE), optional GQA/MoE FFN.
- `attention.py`: Multi-head attention with RoPE, attention sinks, and optional GQA.
- `moe.py`: (Stretch) MoE feedforward with top-k routing.
- `transformer.py`: `ModernDecoderLM` (from-scratch decoder-only Transformer) using the above components.
- `verifier.py`: Lightweight verifier model (e.g., small encoder or classifier) and scoring interface.
- `training/`:
- `trainer_base.py`: Thin training loop abstraction for PyTorch models (step, eval, checkpointing) without over-architecting.
- `train_lm.py`: LM training for from-scratch Transformer (WikiText-2/TinyStories).
- `train_verifier.py`: Training loop for the verifier on math/QA correctness labels.
- `train_sft.py`: SFT stage for instruction-tuned data.
- `train_dpo.py`: DPO implementation (custom DPO loss) for pairwise preference data.
- `hf/`:
- `lora_utils.py`: Utilities to wrap HF models with LoRA/QLoRA (using PEFT) and load adapters.
- `finetune_gpt2_sst2.py`: Task-specific finetuning pipeline for GPT-2/DistilGPT2 on SST-2.
- `finetune_t5_samsum.py`: Finetuning FLAN-T5/T5-small on SAMSum summarization.
- `finetune_math_gsm8k.py`: Finetuning GPT-2/TinyLlama on GSM8K subset.
- `prompting_baselines.py`: Zero-/few-shot prompting runners for each task.
- `alignment/`:
- `dpo_loss.py`: Implementation of DPO objective that can be plugged into HF or custom trainers.
- `alignment_pipeline.py`: Orchestrates base → SFT → DPO → verifier evaluations on chosen tasks.
- `evaluation/`:
- `metrics.py`: PPL, accuracy, F1, ROUGE, EM, Pass@N, verifier-improvement metrics.
- `analysis.py`: Error categorization utilities (misclassification types, math error types).
- `attention_viz.py`: Attention visualization tools for scratch model and HF models.
- `verifier_eval.py`: Scripts to measure verifier impact (before/after reranking, false accept/reject).
- `utils/`:
- `logging_utils.py`: Minimal logging (no overkill); tensorboard or simple CSV logging.
- `checkpointing.py`: Save/load checkpoints (model, optimizer, config) with versioned filenames.
- `distributed_utils.py`: If needed, wrappers for `accelerate` or simple DDP setup.

### `scripts/` entrypoints (thin)

- `train_lm.py`: Calls `modern_llm.training.train_lm.main()` with CLI args or config path.
- `finetune_sst2.py`, `finetune_samsum.py`, `finetune_gsm8k.py`: Call into `modern_llm.hf.*` utilities.
- `run_alignment_pipeline.py`: End-to-end evaluation from base → SFT → DPO → verifier.
- `run_ablations.py`: Run toggles like `--no_rope`, `--no_sinks`, `--no_verifier` and log results.

## Phase 0 — Environment, scaffolding, and datasets

- **0.1 Create environment and dependencies**
- Decide on `pip`/`conda` and set up environment with pinned versions of PyTorch and HF libraries.
- Add `requirements.txt` listing core libraries, keeping it minimal and task-focused.
- **0.2 Initialize repo structure**
- Create `src/modern_llm/` with the subpackages described above, plus stub modules with minimal code and docstrings so imports work.
- Add `scripts/`, `configs/`, `tests/`, `report/`, and `notebooks/` folders with placeholder files.
- **0.3 Dataset download and preprocessing**
- Implement dataset loaders in `data/` using `datasets.load_dataset` with clear wrappers that return tokenized PyTorch `Dataset`/`DataLoader` objects.
- Ensure reproducible train/val/test splits and tokenizer consistency across scratch/HF models.

## Phase 1 — From-scratch Transformer LM (RoPE, RMSNorm, SwiGLU, sinks)

- **1.1 Core layers and blocks**
- Implement `RMSNorm`, `SwiGLU`, and a standard MLP block in `models/layers.py` with clear input validation.
- Implement RoPE in `attention.py` with a focused function to apply rotational embeddings to Q/K.
- Implement multi-head self-attention, attention masking, and attention sinks (e.g., first N tokens reserved as sink positions).
- **1.2 Decoder-only Transformer**
- Implement `ModernDecoderLM` in `transformer.py` using the core layers: token embeddings, positional RoPE, stacked RMSNorm+attention+SwiGLU blocks, final LM head.
- Add options in `ModernLLMConfig` for enabling/disabling RoPE, sinks, GQA, and MoE (GQA/MoE wired but can be `None` initially).
- **1.3 Training loop for LM**
- Implement `trainer_base.py` with a simple train/eval loop supporting gradient accumulation, mixed precision, and gradient clipping.
- Implement `train_lm.py` to train on WikiText-2 or TinyStories, logging PPL and saving checkpoints.
- **1.4 Long-context and sinks experiment**
- Add generation script (or notebook) to sample from the trained LM and demonstrate stable generation beyond training context length via attention sinks.
- Log and save curves for PPL vs. context length with and without sinks.

## Phase 2 — HF finetuning and prompting baselines

- **2.1 LoRA/QLoRA utilities**
- Implement `lora_utils.py` to add LoRA adapters to GPT-2/DistilGPT2/TinyLlama using PEFT, with config options for rank, alpha, and target modules.
- Support 4-bit quantization (QLoRA) for larger models like TinyLlama if memory requires it.
- **2.2 Task-specific finetuning scripts**
- `finetune_gpt2_sst2.py`: LoRA finetune GPT-2/DistilGPT2 on SST-2 (classification), logging accuracy/F1.
- `finetune_t5_samsum.py`: LoRA finetune T5-small/FLAN-T5 on SAMSum summarization, logging ROUGE.
- `finetune_math_gsm8k.py`: LoRA/QLoRA finetune GPT-2/TinyLlama on GSM8K subset.
- **2.3 Prompting baselines**
- Implement `prompting_baselines.py` to run zero-/few-shot prompting for the same tasks, using standard HF generation and evaluation.
- Standardize evaluation via `evaluation.metrics` so scratch vs. HF vs. prompting outputs are directly comparable.

## Phase 3 — Alignment pipeline: SFT → DPO → Verifier

- **3.1 Supervised finetuning (SFT)**
- Implement `train_sft.py` to perform supervised instruction tuning for one decoder model (e.g., GPT-2 or your scratch LM) on a curated instruction dataset.
- Keep the SFT implementation close to the LM training loop to avoid duplicated logic.
- **3.2 DPO implementation**
- Implement a clear DPO loss in `alignment/dpo_loss.py` based on log-probs of chosen vs. rejected responses.
- Implement `train_dpo.py` that:
- Loads a base/SFT model (HF or scratch if feasible).
- Loads preference pairs from `preference_datasets.py`.
- Optimizes DPO loss with logging of preference-aligned metrics.
- **3.3 Verifier model and training**
- Implement `verifier.py` as a small encoder/classifier model (can be a small Transformer encoder or shallow MLP on pooled token representations).
- Implement `train_verifier.py` to train on labeled correctness data for math/QA (GSM8K, SVAMP/MAWPS, synthetic arithmetic).
- **3.4 Alignment pipeline orchestration**
- Implement `alignment_pipeline.py` to:
- Evaluate base model on tasks.
- Apply SFT, re-evaluate.
- Apply DPO, re-evaluate.
- Integrate verifier for reranking or triggering retries.
- Output a summary table showing incremental gains from each alignment stage.

## Phase 4 — Advanced architectural features and test-time compute

- **4.1 Grouped Query Attention (GQA)**
- Extend `attention.py` to support GQA (fewer K/V heads than Q heads) with a clean configuration flag.
- Add experiments comparing memory usage and speed vs. standard MHA.
- **4.2 Mixture-of-Experts (MoE)**
- Implement MoE feedforward in `moe.py` with top-1 or top-2 gating and optional capacity factor.
- Add a variant of `ModernDecoderLM` that swaps standard FFN for MoE in some layers.
- **4.3 Test-time compute scaling**
- Add self-consistency sampling (sample N outputs, majority vote) for GSM8K-style math tasks.
- Implement verifier-gated "revise and retry" loop where low-scoring generations are revised once or twice.
- **4.4 Structured outputs and tools**
- Implement a simple calculator/tool stub for symbolic arithmetic problems: parse model outputs for expressions and check against ground-truth evaluation.

## Phase 5 — Evaluation, ablations, visualization, and report

- **5.1 Metric computation and evaluation scripts**
- Complete `evaluation.metrics` and `evaluation.verifier_eval` to compute all required metrics (PPL, accuracy, F1, ROUGE, EM, Pass@N, verifier gains).
- Add `scripts/run_ablations.py` that toggles features (RoPE, sinks, DPO, verifier, GQA, MoE) and logs results to `experiments/`.
- **5.2 Attention and behavior analysis**
- Implement `attention_viz.py` to visualize attention maps over input tokens for both scratch and HF models.
- Use notebooks in `notebooks/` to generate figures for the report.
- **5.3 Error analysis and qualitative examples**
- Implement utilities in `analysis.py` to categorize errors (e.g., sentiment flips, hallucinated facts, math arithmetic errors, reasoning chain breaks).
- Save representative examples for inclusion in the final report.
- **5.4 Final documentation and report**
- Polish `README.md` with a clear hierarchy: quickstart, experiments, alignment pipeline, and stretch features.
- Finalize the ACL-style report in `report/` with tables, figures, and ablation findings.
- Optionally add a lightweight demo notebook that loads a trained model and verifier for interactive querying.

### To-dos

- [ ] Set up Python environment, core dependencies, and base repo structure (src/modern_llm, scripts, configs, tests, report, notebooks).
- [ ] Implement core from-scratch decoder-only Transformer with RoPE, RMSNorm, SwiGLU, attention sinks, and configuration dataclasses.
- [ ] Build and run the language modeling training loop on WikiText-2/TinyStories, including long-context and sinks experiments.
- [ ] Implement LoRA/QLoRA utilities and finetune HF models (GPT-2/DistilGPT2, T5-small, TinyLlama) on SST-2, SAMSum, and GSM8K with prompting baselines.
- [ ] Implement SFT and DPO training loops plus preference datasets and run the full alignment pipeline on at least one model.
- [ ] Implement and train a verifier model for math/QA tasks and integrate it for reranking and revise-and-retry loops.
- [ ] Add GQA, MoE, and test-time compute scaling (self-consistency, verifier-gated retries, structured outputs) to the scratch model.
- [ ] Implement metrics, ablations, attention visualization, and error analysis across scratch and HF models.
- [ ] Polish README, write the ACL-style final report, and add an optional demo notebook for interactive usage.