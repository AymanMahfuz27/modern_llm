# Scripts Directory

Thin CLI entrypoints for training, evaluation, and orchestration.

## Quick Start

### Run All Phase 1 & 2 Experiments

```bash
chmod +x scripts/*.sh
bash scripts/run_phase1_phase2.sh
```

### Run Only Evaluations (After Training)

```bash
bash scripts/run_all_evaluations.sh
```

## Individual Scripts

### Training

- `train_lm_from_config.py` – Train scratch LM from JSON config
- `experiment_attention_sinks.py` – Long-context stability experiment

### Evaluation

- `evaluate_lm_checkpoints.py` – Compute loss/perplexity for all LM checkpoints
- `generate_from_checkpoints.py` – Generate text samples from LM checkpoints
- `evaluate_hf_sst2.py` – Evaluate SST-2 checkpoint (accuracy + errors)
- `evaluate_hf_samsum.py` – Evaluate SAMSum checkpoint (ROUGE)
- `evaluate_hf_gsm8k.py` – Evaluate GSM8K checkpoint (exact match)

### Orchestration

- `run_phase1_phase2.sh` – Run all Phase 1 & 2 experiments
- `run_phase1_only.sh` – Run only Phase 1 (scratch LM)
- `run_phase2_only.sh` – Run only Phase 2 (HF finetuning)
- `run_all_evaluations.sh` – Run all evaluations (assumes checkpoints exist)

## Module Entry Points

Some scripts use `python -m modern_llm.hf.{module}` format:

- `python -m modern_llm.hf.finetune_gpt2_sst2` – GPT-2 on SST-2
- `python -m modern_llm.hf.finetune_t5_samsum` – T5 on SAMSum
- `python -m modern_llm.hf.finetune_math_gsm8k` – GPT-2 on GSM8K
- `python -m modern_llm.hf.prompting_baselines` – Zero/few-shot prompting

These can also be run via the orchestration scripts above.



