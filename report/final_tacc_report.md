# Modern LLM Final Report (TACC H100 Run)

Generated: 2025-12-03 13:11:20

---

## Architecture

- **Model type**: Decoder-only Transformer (GPT-style)
- **Parameters**: ~253M (d_model=1024, n_layers=12, n_heads=16, ffn=4096)
- **Vocab size**: 50,257 (GPT-2 tokenizer)
- **Max sequence length**: 1024 tokens
- **Features**: RoPE, RMSNorm, SwiGLU, Flash Attention (SDPA)

---

## Training Pipeline

### Stage 1: Pretraining
- **Datasets**: WikiText-2 + WikiText-103 + TinyStories
- **Steps**: 40,000
- **Batch size**: 128 (micro=32)
- **Learning rate**: 3e-4
- **Checkpoint**: `/work/09999/aymanmahfuz/ls6/modern_llm_runs/checkpoints/gpu-full-pretrain/gpu-full-pretrain_final.pt`

### Stage 2: SFT
- **Dataset**: tatsu-lab/alpaca (52K)
- **Steps**: 5,000
- **Checkpoint**: `/work/09999/aymanmahfuz/ls6/modern_llm_runs/checkpoints/gpu-full-sft/gpu-full-sft_final.pt`

### Stage 3: DPO
- **Dataset**: Anthropic/hh-rlhf
- **Steps**: 3,000
- **Beta**: 0.1
- **Checkpoint**: `/work/09999/aymanmahfuz/ls6/modern_llm_runs/checkpoints/gpu-full-dpo/gpu-full-dpo_final.pt`

### Stage 4: Verifier
- **Dataset**: GSM8K
- **Steps**: 3,000
- **Checkpoint**: `/work/09999/aymanmahfuz/ls6/modern_llm_runs/checkpoints/gpu-full-verifier/gpu-full-verifier_final.pt`

---

## Results

### Task Metrics

| Model | SST-2 Acc | GSM8K EM | Notes |
|-------|-----------|----------|-------|
| gpt2 | 56.0% | N/A | HF Baseline |
| distilgpt2 | 56.0% | N/A | HF Baseline |
| Our PRETRAIN | 49.5% | N/A | |
| Our SFT | 53.5% | N/A | |
| Our DPO | 49.5% | N/A | |

### Stage-wise Gains

| Stage | SST-2 Acc | GSM8K EM | Δ SST-2 | Δ GSM8K |
|-------|-----------|----------|---------|---------|
| PRETRAIN | 49.5% | 0.0% | +0.0% | +0.0% |
| SFT | 53.5% | 0.0% | +4.0% | +0.0% |
| DPO | 49.5% | 0.0% | -4.0% | +0.0% |

---

## Interpretability

### Attention Patterns


**Observations** (see `report/figures/attention_observations.md`):
  # Attention Pattern Observations
  
  Model: /work/09999/aymanmahfuz/ls6/modern_llm_runs/checkpoints/gpu-full-dpo/gpu-full-dpo_final.pt
  Layer visualized: -1 (negative = from end)
  
  ## Summary Statistics
  
  | Example | Self-Attn | Local-Attn | First-Token | Entropy |
  |---------|-----------|------------|-------------|--------|
  | sentiment_positive | 0.276 | 0.118 | 0.286 | 1.589 |

---

## Figures

- `report/figures/attention_sentiment_positive.png`
- `report/figures/attention_sentiment_negative.png`
- `report/figures/attention_entity.png`
- `report/figures/attention_math.png`

---

## Reproducibility

```bash
# Full pipeline
sbatch scripts/tacc_pipeline.slurm

# Local smoke test
python scripts/run_pipeline.py --config local-smoke --stage all
```

- **Seed**: 42
- **Hardware**: NVIDIA H100 PCIe 80GB
- **CUDA**: 12.2
