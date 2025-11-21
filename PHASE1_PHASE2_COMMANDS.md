# Phase 1 & 2 Execution Commands

## Quick Start – Run Everything

To execute all Phase 1 and Phase 2 experiments in sequence:

```bash
chmod +x scripts/run_phase1_phase2.sh
bash scripts/run_phase1_phase2.sh
```

**Note:** The max-size LM training (Phase 1.3) will run for many hours. Consider running the script in `tmux` or `screen`, or run steps individually.

---

## Individual Commands (for selective execution)

### Phase 1 – Scratch LM Experiments

#### 1.1 Evaluate Existing LM Checkpoints

```bash
python scripts/evaluate_lm_checkpoints.py \
    --runs_dir experiments/runs \
    --output_csv experiments/lm_checkpoint_metrics.csv \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tokenizer_name gpt2 \
    --max_seq_len 512 \
    --batch_size 8
```

#### 1.2 Generate Text Samples from Checkpoints

```bash
python scripts/generate_from_checkpoints.py \
    --runs_dir experiments/runs \
    --output_json experiments/lm_generation_samples.json \
    --tokenizer_name gpt2 \
    --max_new_tokens 80 \
    --temperature 1.0
```

#### 1.3 Train MAX-SIZE Scratch LM (Long Run)

```bash
python scripts/train_lm_from_config.py \
    --config configs/lm_max_rtx3060.json \
    --output_dir experiments/runs \
    --num_proc 4 \
    --gen_prompt "The meaning of life is" \
    --gen_max_new_tokens 100
```

#### 1.4 Attention Sinks Long-Context Experiment

```bash
python scripts/experiment_attention_sinks.py \
    --output_dir experiments/runs \
    --output_json experiments/attention_sinks_results.json \
    --train_context 512 \
    --gen_context_multiplier 3 \
    --max_steps 500
```

#### 1.5 Visualize LM Results

```bash
python notebooks/visualize_lm_results.py \
    --metrics_csv experiments/lm_checkpoint_metrics.csv \
    --output_dir experiments/plots
```

---

### Phase 2 – HF Finetuning & Prompting

#### 2.1 GPT-2 + LoRA on SST-2 (Full Run)

```bash
python -m modern_llm.hf.finetune_gpt2_sst2 \
    --run_name gpt2-sst2-lora-main \
    --model_name gpt2 \
    --max_seq_len 128 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 2e-4 \
    --max_steps 2000 \
    --warmup_steps 100 \
    --eval_every 200 \
    --save_every 500 \
    --log_every 50 \
    --mixed_precision bf16 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32
```

#### 2.2 Evaluate GPT-2 SST-2 Checkpoint

```bash
python scripts/evaluate_hf_sst2.py \
    --checkpoint_path experiments/runs/gpt2-sst2-lora-main/gpt2-sst2-lora-main_final.pt \
    --model_name gpt2 \
    --max_seq_len 128 \
    --output_metrics_csv experiments/sst2_eval_metrics.csv \
    --output_errors_json experiments/sst2_misclassified.json
```

#### 2.3 T5 + LoRA on SAMSum

```bash
python -m modern_llm.hf.finetune_t5_samsum \
    --run_name t5-samsum-lora-main \
    --model_name t5-small \
    --max_source_length 512 \
    --max_target_length 64 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 5e-4 \
    --max_steps 2000 \
    --warmup_steps 100 \
    --eval_every 200 \
    --save_every 500 \
    --log_every 50 \
    --mixed_precision bf16 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32
```

#### 2.4 Evaluate T5 SAMSum Checkpoint

```bash
python scripts/evaluate_hf_samsum.py \
    --checkpoint_path experiments/runs/t5-samsum-lora-main/t5-samsum-lora-main_final.pt \
    --model_name t5-small \
    --max_source_length 512 \
    --max_target_length 64 \
    --output_csv experiments/samsum_eval_metrics.csv
```

#### 2.5 GPT-2 + LoRA on GSM8K

```bash
python -m modern_llm.hf.finetune_math_gsm8k \
    --run_name gpt2-gsm8k-lora-main \
    --model_name gpt2 \
    --max_seq_len 512 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --learning_rate 3e-4 \
    --max_steps 3000 \
    --warmup_steps 150 \
    --eval_every 300 \
    --save_every 1000 \
    --log_every 50 \
    --mixed_precision bf16 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32
```

#### 2.6 Evaluate GPT-2 GSM8K Checkpoint

```bash
python scripts/evaluate_hf_gsm8k.py \
    --checkpoint_path experiments/runs/gpt2-gsm8k-lora-main/gpt2-gsm8k-lora-main_final.pt \
    --model_name gpt2 \
    --max_seq_len 512 \
    --max_new_tokens 128 \
    --output_csv experiments/gsm8k_eval_metrics.csv \
    --output_errors_json experiments/gsm8k_errors.json
```

#### 2.7 Prompting Baselines (Zero-Shot)

```bash
python -m modern_llm.hf.prompting_baselines \
    --model_name gpt2 \
    --tasks sst2 samsum gsm8k \
    --num_shots 0 \
    --max_samples 500 \
    --output_csv experiments/prompting_baseline_results.csv
```

---

## Expected Outputs

After running Phase 1 & 2:

- `experiments/lm_checkpoint_metrics.csv` – validation loss/perplexity for all scratch LM checkpoints
- `experiments/lm_generation_samples.json` – text samples from checkpoints
- `experiments/attention_sinks_results.json` – long-context stability comparison
- `experiments/plots/lm_metrics.png` – loss/perplexity curves
- `experiments/sst2_eval_metrics.csv` – SST-2 accuracy and F1
- `experiments/sst2_misclassified.json` – misclassified examples
- `experiments/samsum_eval_metrics.csv` – ROUGE scores
- `experiments/gsm8k_eval_metrics.csv` – exact match scores
- `experiments/gsm8k_errors.json` – error examples
- `experiments/prompting_baseline_results.csv` – zero-shot prompting results

Checkpoints saved under `experiments/runs/{run_name}/`.



