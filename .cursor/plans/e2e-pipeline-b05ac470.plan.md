<!-- b05ac470-3e5a-465c-b16f-dd3229445944 6e2eccd0-ef6d-4d37-8a8d-d98ba0743f68 -->
# End-to-End Pipeline Verification and Evaluation Suite

## CRITICAL GAPS TO FILL (from review)

The reviewer identified these **must-fix** items for full marks:

1. **Task-level comparisons** - SST-2 accuracy, GSM8K EM (scratch vs HF baselines)
2. **Stage-wise alignment gains** - Show where SFT/DPO improve task metrics despite PPL increase
3. **Verifier impact proof** - EM before/after, error taxonomy, false pos/neg examples
4. **Interpretability** - Attention heatmaps on 2-3 examples
5. **Reproducibility** - Exact commands for each result/table

---

## 0. Dataset Loading Fix

Before training, verify all datasets in `DATASET_REGISTRY` load correctly:

- `wikitext-2-raw-v1`
- `wikitext-103-raw-v1`
- `roneneldan/TinyStories`
- `openwebtext` (may need trust_remote_code)
- `bookcorpus` (may need auth token)

Create `scripts/verify_datasets.py` to test each and report failures.

---

## 1. Pipeline Verification (TACC Run)

Single SLURM script for full pipeline with checkpointing:

- **Pretrain**: WikiText-103 + TinyStories (**40K steps max** - avoid slowdown after 31K)
- **SFT**: Alpaca 52K (5K steps, ~3h)
- **DPO**: HH-RLHF (3K steps, ~2h)
- **Verifier**: GSM8K (3K steps, ~2h)
- **Total**: ~35h target (well under 48h limit)

**Speed optimizations**:

- Cap pretrain at 40K steps (slowdown observed after 31K on H100)
- Use Flash Attention (disable attention_sinks for SDPA)
- Batch size tuned for H100 memory

Files:

- `scripts/tacc_pipeline.slurm` - SLURM job with auto-resume
- Update `gpu_full_config()` in [`src/modern_llm/config/pipeline_config.py`](src/modern_llm/config/pipeline_config.py)

---

## 2. Task Evaluation Scripts

**Organized structure:**

```
scripts/evaluation/
├── evaluate_tasks.py      # Unified task runner
├── eval_sst2.py           # SST-2 few-shot
├── eval_generation.py     # TinyStories perplexity
└── eval_gsm8k.py          # GSM8K with verifier
```

| Task | Metric | Why Achievable |

|------|--------|----------------|

| SST-2 | Accuracy | Binary sentiment, simple prompting |

| TinyStories | Perplexity | Model trained on this |

| GSM8K (easy) | EM | With verifier reranking |

---

## 3. HF Baseline Comparisons

Add to evaluation scripts:

- GPT-2 (124M) - prompted/few-shot
- DistilGPT-2 (82M) - prompted/few-shot

Output: `experiments/results/baseline_comparison.json`

---

## 4. Stage-wise Gains Table

Extend [`src/modern_llm/evaluation/pipeline_eval.py`](src/modern_llm/evaluation/pipeline_eval.py):

- Add task metrics per stage: `pretrain_acc`, `sft_acc`, `dpo_acc`
- Show alignment payoff: task improvement despite PPL increase

Output: `experiments/results/stage_gains.csv`

---

## 5. Verifier Impact Documentation

Enhance [`scripts/benchmark_gsm8k.py`](scripts/benchmark_gsm8k.py):

- Error taxonomy: extraction / arithmetic / reasoning
- 3 false-positive examples, 3 false-negative examples
- Markdown output section

---

## 6. Lightweight Interpretability

Create `scripts/visualize_attention.py`:

- Extract attention from 2-3 short examples
- Heatmap plots (matplotlib)
- Save to `report/figures/attention_*.png`

---

## 7. MoE (Stretch Goal)

If time permits:

- `TopKRouter.forward()` - softmax gating
- `MixtureOfExperts.forward()` - expert dispatch
- Load balancing loss
- Smoke test only

---

## 8. Final Report (Dense Bullet Format)

Generate `report/final_report.md` as **dense bullet list** of facts:

**Architecture:**

- Model size, d_model, n_layers, n_heads, vocab_size
- Features: RoPE, RMSNorm, SwiGLU, attention sinks, GQA

**Training:**

- Datasets: names, sizes, token counts
- Steps per stage, batch sizes, learning rates
- Hardware: GPU type, VRAM, training time

**Results:**

- PPL table (WikiText-2)
- Task metrics table (SST-2, GSM8K)
- Stage-wise gains table
- Verifier impact (+X% EM)

**Interpretability:**

- Attention pattern observations

**Reproducibility:**

- Exact commands for each table/figure
- Seeds, config paths

---

## Organized Output Structure

```
experiments/
├── results/
│   ├── baseline_comparison.json
│   ├── stage_gains.csv
│   ├── task_metrics.json
│   └── gsm8k_analysis.json
└── runs/
    └── gpu-full/
        ├── *-pretrain_final.pt
        ├── *-sft_final.pt
        ├── *-dpo_final.pt
        └── *-verifier_final.pt

report/
├── figures/
│   ├── attention_sentiment.png
│   └── attention_entity.png
└── final_report.md
```

---

## File Changes Summary

| File | Action |

|------|--------|

| `scripts/verify_datasets.py` | Create - test all dataset loading |

| `scripts/tacc_pipeline.slurm` | Create - SLURM job with auto-resume |

| `scripts/evaluation/evaluate_tasks.py` | Create - unified task evaluation |

| `scripts/evaluation/eval_sst2.py` | Create - SST-2 few-shot |

| `scripts/evaluation/eval_gsm8k.py` | Create - GSM8K + verifier analysis |

| `scripts/visualize_attention.py` | Create - attention heatmaps |

| `src/modern_llm/config/pipeline_config.py` | Update - 40K steps, speed opts |

| `src/modern_llm/evaluation/pipeline_eval.py` | Extend - task metrics per stage |

| `scripts/benchmark_gsm8k.py` | Extend - error taxonomy |

| `src/modern_llm/models/moe.py` | (Stretch) - implement routing |

| `report/final_report.md` | Create - dense bullet list |

### To-dos

- [ ] Create TACC SLURM script with auto-resume checkpointing
- [ ] Build unified task evaluation script (SST-2, TinyStories, GSM8K)
- [ ] Add HF baseline evaluation (GPT-2, DistilGPT-2) to task script
- [ ] Extend pipeline_eval.py with per-stage task metrics
- [ ] Add error taxonomy and example analysis to GSM8K benchmark
- [ ] Create attention visualization script with heatmaps
- [ ] (Stretch) Implement MoE routing and load balancing
- [ ] Generate comprehensive final report with all tables/figures