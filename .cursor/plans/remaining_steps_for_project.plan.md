<!-- 799ff3a7-ec85-4f4b-85e3-a0794f3edb4f 82dd2ddf-acc4-4dec-85a8-6007a3a0ff3e -->
# Modern LLM – Full Remaining Work Plan

## 1. Phase 1 – From-scratch Transformer LM (complete + max-size model)

- **Baseline LM experiments (small/medium)**
- Use `training/train_lm.py` and existing configs to clearly define at least one "small" and one "medium" scratch LM on WikiText-2 (and/or TinyStories).
- For each configuration, run training to stable convergence (within hardware limits), saving checkpoints and logging train/val loss and perplexity.
- Add a small evaluation script or notebook that loads checkpoints, runs `Trainer.evaluate()` on the validation set, and writes a metrics table (CSV/JSON) under `experiments/`.

- **Max-size LM tuned to hardware limits**
- Design a "max" configuration that pushes the RTX 3060 (12 GB) close to capacity using mixed precision, gradient accumulation, and reasonable context length (based on your hardware plan in the proposal).
- Implement a corresponding config or CLI recipe and run a long training job (on the order of many hours / ~day) to obtain a final large scratch LM checkpoint.
- Record exact hyperparameters (model size, batch sizes, learning rate schedule, steps, wall-clock time) for inclusion in the report.

- **Long-context & attention sinks experiment**
- Finalize the attention-sinks implementation (in `models/attention.py` / `models/transformer.py` as needed) and design a protocol to test context extrapolation: train at a given max context, then generate at 2–4× that length.
- Implement a script or notebook that:
- Generates continuations from checkpoints with sinks enabled vs disabled on the same prompts.
- Measures qualitative stability (lack of collapse, topic retention) and logs basic quantitative signals (e.g., repetition scores, simple heuristic metrics) if feasible.
- Produce a small figure or table summarizing the effect of sinks for the report.

## 2. Phase 2 – HF Finetuning & Prompting (SST-2, SAMSum, GSM8K)

- **SST-2 – GPT-2/DistilGPT2 + LoRA/QLoRA**
- Run `hf/finetune_gpt2_sst2.py` (with optional LoRA via `lora_utils.py`) to full convergence on SST-2, using a non-smoke `run_name` and appropriately tuned `max_steps` and learning rate.
- Implement evaluation logic (script or notebook) that:
- Loads the final checkpoint.
- Runs inference on SST-2 validation.
- Uses `evaluation.metrics.compute_metrics("sst-2", preds, labels)` to compute accuracy (and F1 if you extend metrics).
- Saves metrics and misclassified examples to `experiments/`.

- **SAMSum – T5/FLAN-T5 summarization finetune**
- Flesh out `hf/finetune_t5_samsum.py` to:
- Load SAMSum via a `TaskDatasetConfig`-style helper.
- Tokenize, collate, and finetune T5-small or FLAN-T5 with LoRA where appropriate.
- Add evaluation that computes ROUGE-1/ROUGE-L (via `evaluation.metrics` wrappers or external libraries) on the validation set and logs results to `experiments/`.

- **GSM8K – math reasoning finetune**
- Implement `hf/finetune_math_gsm8k.py` to finetune GPT-2 or TinyLlama (using QLoRA if needed) on a GSM8K subset.
- Add evaluation that computes Exact Match (EM) for GSM8K, integrated with `evaluation.metrics` where appropriate.

- **Prompting baselines for all three tasks**
- Implement `hf/prompting_baselines.py` to run zero-/few-shot prompting for:
- SST-2 (classification accuracy).
- SAMSum (ROUGE for summaries).
- GSM8K (EM for answers).
- Ensure outputs are evaluated with the same `evaluation.metrics` (or consistent wrappers) used for the finetuned models, so you can directly compare scratch LM, HF finetune, and prompting baselines.

## 3. Phase 3 – Alignment Pipeline: SFT → DPO → Verifier

- **Supervised finetuning (SFT)**
- Implement `training/train_sft.py` to perform supervised instruction tuning on a small instruction-following dataset (e.g., QA/dialog style), using either an HF decoder or your scratch LM.
- Reuse the `Trainer` abstraction and data loaders so SFT shares as much structure as LM and HF finetuning.
- Evaluate SFT models on an appropriate held-out set or small battery of instruction-following prompts, logging basic metrics and qualitative examples.

- **DPO training loop using `alignment/dpo_loss.py`**
- Implement `training/train_dpo.py` to:
- Load a base or SFT checkpoint (HF or scratch decoder-style model).
- Load preference pairs from a `preference_datasets.py`-style loader (synthetic or open) returning (prompt, chosen, rejected) triples.
- Compute chosen/rejected log-probabilities and apply `dpo_loss` to optimize the model.
- Log DPO loss and evaluate the aligned model on a downstream task (e.g., a small QA or preference-sensitive benchmark), comparing to the base/SFT model.

- **Verifier model and training**
- Implement `models/verifier.py` as a compact classifier/encoder that scores (problem, answer) pairs for correctness, targeted at math/QA tasks (e.g., GSM8K subset, synthetic arithmetic).
- Implement `training/train_verifier.py` to:
- Prepare a dataset of labeled (problem, answer, correct/incorrect) examples.
- Train the verifier and measure validation accuracy/EM on correctness labels.
- Expose a simple scoring interface such as `score(problem: str, answer: str) -> float` for use during reranking and test-time compute.

- **Alignment pipeline orchestration**
- Flesh out `alignment/alignment_pipeline.py` to run a full, if small-scale, pipeline:
- Base (or SFT) model → DPO-aligned model → +Verifier reranking.
- For at least one math/QA task, evaluate each stage and log metrics: EM, verifier-assisted EM, possibly Pass@N.
- Save a compact summary table under `experiments/` showing the incremental gains from SFT, DPO, and verifier reranking.

## 4. Phase 4 – Advanced Features & Test-time Compute (all required)

- **Grouped Query Attention (GQA)**
- Complete and verify GQA support in `models/attention.py` and configuration toggles in `ModernLLMConfig` / `ModernDecoderLM`.
- Train at least one LM configuration with GQA enabled and compare memory usage, speed, and perplexity against a standard multi-head attention baseline of comparable size.

- **Mixture-of-Experts (MoE)**
- Implement or finalize `models/moe.py` with top-1 or top-2 gating and capacity controls.
- Integrate MoE into a variant of `ModernDecoderLM` (e.g., MoE in some FFN layers) controlled by config flags.
- Run at least one LM experiment with MoE enabled and compare training stability and perplexity vs. a dense baseline.

- **Test-time compute scaling**
- Implement self-consistency and/or majority-vote routines for GSM8K-like tasks: generate N candidate answers per problem and select the most common or best-scoring.
- Combine this with the verifier: if the verifier score is below a threshold, trigger a re-sample and compare EM/Pass@N before and after this gating.

- **Structured outputs and tools**
- Implement a simple calculator/tool stub for symbolic arithmetic problems:
- Parse arithmetic expressions from model outputs.
- Evaluate them programmatically and compare to ground-truth answers.
- Use this to analyze when the model is arithmetically correct vs linguistically plausible but wrong.

## 5. Phase 5 – Evaluation, Visualization, and Report

- **Metrics and evaluation scripts**
- Extend `evaluation/metrics.py` and/or wrap external libraries to support all required metrics: perplexity, accuracy, F1, ROUGE, EM, Pass@N, verifier-improvement metrics.
- Implement small CLI scripts or notebooks that run standardized evaluations for:
- LM tasks (WikiText-2/TinyStories),
- SST-2, SAMSum, GSM8K,
- Alignment stages (Base/SFT/DPO/Verifier),
- Advanced features (GQA, MoE, test-time compute).
- Write results to consistent CSV/JSON tables under `experiments/` for easy plotting and report inclusion.

- **Attention visualization and behavior analysis**
- Implement `evaluation/attention_viz.py` to visualize attention maps for selected layers/heads in the scratch LM (and optionally HF models) over key examples.
- Use a notebook in `notebooks/` to generate a small set of high-quality attention visualizations for the final report.

- **Error analysis utilities**
- Implement `evaluation/analysis.py` to categorize and log errors for:
- SST-2 (e.g., sentiment flips, ambiguous labels),
- GSM8K/math (e.g., arithmetic errors, reasoning-chain breaks),
- Alignment-related failures (e.g., preference violations).
- Save example-driven artifacts (inputs, outputs, labels, error category) under `experiments/` for use in the paper.

- **Final report and documentation**
- Populate `report/` with an ACL-style paper draft summarizing methods, experiments, results, ablations, and alignment findings, using the metrics and visualizations produced above.
- Update `README.md` with:
- Clear one-command recipes for key experiments (LM training, HF finetunes, DPO, verifier, advanced features).
- Pointers to configs and result tables.

## 6. Orchestration & One-button Pipelines (nanochat-inspired)

- **End-to-end orchestration script(s)**
- Inspired by the "nanochat" style in your other repos, design one or more top-level orchestrator scripts (e.g., under `scripts/`):
- A script to launch the full **max-size LM pretraining** run with appropriate config (pushing the 3060 to its limits), including checkpointing and logging.
- A script to run the **end-to-end evaluation suite** for a chosen configuration: load best checkpoints, run all relevant evaluations, and write metrics tables and plots.
- Optionally, a script to **chain training + posttraining** (e.g., LM pretraining → SFT → DPO → Verifier) for a small configuration, so you can reproduce a full pipeline with minimal manual steps.
- Ensure these scripts encapsulate the core workflows you want to showcase and are documented in `README.md` as "one-button" entry points for the project.

## 7. Sequencing (all tasks required, but ordered for sanity)

- **First focus:** finalize Phase 1 (including the max-size LM) and Phase 2 HF finetunes and prompting baselines so the default project requirements are locked in.
- **Next:** implement the full alignment pipeline (SFT → DPO → Verifier) and the orchestration around it.
- **Then:** complete all Phase 4 advanced features (GQA, MoE, test-time compute, structured outputs) and run at least one focused experiment for each.
- **In parallel as results arrive:** build out Phase 5 metrics, visualization, analysis, and the ACL-style report, so you are continuously converting experiments into polished artifacts.

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