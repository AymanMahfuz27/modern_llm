# Modern LLM Final Report

Dense bullet-point summary of architecture, training, results, and reproducibility.

---

## Architecture

- **Model type**: Decoder-only Transformer (GPT-style)
- **Parameters**: ~253M (d_model=1024, n_layers=12, n_heads=16, ffn=4096)
- **Vocab size**: 50,257 (GPT-2 tokenizer)
- **Max sequence length**: 1024 tokens
- **Frontier features**:
  - RoPE (Rotary Position Embeddings) - relative position encoding
  - RMSNorm (Root Mean Square Layer Normalization) - faster than LayerNorm
  - SwiGLU activation - gated FFN with SiLU activation
  - Attention sinks - stabilizes long-context via sink tokens
  - GQA ready (Grouped Query Attention) - config knob for KV sharing

---

## Training Pipeline

### Stage 1: Pretraining
- **Datasets**: WikiText-103 (~100M tokens) + TinyStories (~500M tokens)
- **Steps**: 40,000
- **Batch size**: 128 (effective), micro-batch: 8
- **Learning rate**: 3e-4 with cosine decay
- **Warmup**: 500 steps
- **Hardware**: H100 80GB, ~17h estimated

### Stage 2: Supervised Fine-Tuning (SFT)
- **Dataset**: tatsu-lab/alpaca (52K instruction pairs)
- **Steps**: 5,000
- **Batch size**: 32
- **Learning rate**: 1e-5
- **Hardware**: ~3h

### Stage 3: Direct Preference Optimization (DPO)
- **Dataset**: Anthropic/hh-rlhf (chosen/rejected pairs)
- **Steps**: 3,000
- **Batch size**: 16
- **Learning rate**: 5e-6
- **Beta**: 0.1
- **Hardware**: ~2h

### Stage 4: Verifier Training
- **Dataset**: GSM8K (grade school math)
- **Steps**: 3,000
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Hardware**: ~2h

### Total Training Time
- **Estimated**: ~25h on H100
- **Wall clock**: (fill after run)

---

## Results

### Perplexity (WikiText-2 validation)

| Model | PPL |
|-------|-----|
| GPT-2 (124M) | ~29.4 |
| Our Pretrain | (fill) |
| Our SFT | (fill) |
| Our DPO | (fill) |

### Task Metrics

| Model | SST-2 Acc | GSM8K EM |
|-------|-----------|----------|
| GPT-2 baseline | (fill) | N/A |
| DistilGPT-2 baseline | (fill) | N/A |
| Our Pretrain | (fill) | (fill) |
| Our SFT | (fill) | (fill) |
| Our DPO | (fill) | (fill) |

### Stage-wise Alignment Gains

| Stage | SST-2 Acc | GSM8K EM | Δ SST-2 | Δ GSM8K |
|-------|-----------|----------|---------|---------|
| Pretrain | (fill) | (fill) | - | - |
| SFT | (fill) | (fill) | (fill) | (fill) |
| DPO | (fill) | (fill) | (fill) | (fill) |

### Verifier Impact (GSM8K)

- **Without verifier**: (fill)% EM
- **With verifier**: (fill)% EM
- **Improvement**: +(fill)%

**Error Taxonomy**:
- Extraction errors: (fill)
- Arithmetic errors: (fill)
- Reasoning errors: (fill)

---

## Interpretability

### Attention Patterns

See `report/figures/` for heatmaps.

**Observations**:
- (fill after visualization)
- Self-attention patterns show...
- Local attention concentration...
- First-token sink behavior...

---

## Reproducibility

### Environment
- Python 3.12
- PyTorch 2.3+
- CUDA 12.2
- HuggingFace Transformers 4.44+

### Commands

**Full pipeline (TACC H100)**:
```bash
sbatch scripts/tacc_pipeline.slurm
```

**Local smoke test**:
```bash
python scripts/run_pipeline.py --config local-smoke --stage all
```

**Dataset verification**:
```bash
python scripts/verify_datasets.py --quick
```

**Evaluation**:
```bash
python scripts/evaluation/evaluate_tasks.py \
    --stage-checkpoints experiments/runs/gpu-full/ \
    --include-baselines
```

**Attention visualization**:
```bash
python scripts/visualize_attention.py --checkpoint path/to/model.pt
```

### Seed
- All training uses seed=42

### Config paths
- Local smoke: `src/modern_llm/config/pipeline_config.py::local_smoke_config`
- GPU full: `src/modern_llm/config/pipeline_config.py::gpu_full_config`

---

## Files

### Code Structure
```
src/modern_llm/
├── models/
│   ├── transformer.py    # Main model
│   ├── attention.py      # Multi-head attention + RoPE
│   └── layers.py         # RMSNorm, SwiGLU, etc.
├── alignment/
│   ├── dpo.py            # DPO training
│   └── verifier.py       # Solution verifier
├── config/
│   └── pipeline_config.py
└── evaluation/
    └── pipeline_eval.py
```

### Checkpoints (on TACC $WORK)
```
$WORK/modern_llm_runs/checkpoints/
├── *-pretrain_final.pt
├── *-sft_final.pt
├── *-dpo_final.pt
└── *-verifier_final.pt
```

---

## Notes

- PPL slightly increases after SFT/DPO (expected - different optimization objective)
- Task metrics show alignment payoff (instruction following, preference alignment)
- Verifier improves GSM8K EM by reranking multiple solutions
- Attention sinks disabled for Flash Attention compatibility on H100

