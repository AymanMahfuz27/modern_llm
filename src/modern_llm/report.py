"""Report generation for Modern LLM pipeline.

Generates a markdown report with:
- Model architecture summary
- Training curves (if available)
- Evaluation metrics table
- Sample generations
- GSM8K examples with verifier scores
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from modern_llm.alignment.alignment_pipeline import PipelineState
from modern_llm.config import ModernLLMConfig, PipelineConfig
from modern_llm.models.transformer import ModernDecoderLM
from modern_llm.utils.checkpointing import load_checkpoint


def generate_text(
    model: ModernDecoderLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text from a prompt.

    Pre: model is on device and in eval mode.
    Post: Returns generated text string.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def generate_report(
    state: PipelineState,
    config: PipelineConfig,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate a markdown report for the pipeline run.

    Pre: Pipeline state has valid checkpoint paths.
    Post: Returns path to generated report.md file.
    """
    output_dir = output_dir or Path("report")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{config.run_name}_report.md"

    # Load evaluation results if available
    eval_results = None
    eval_path = Path("experiments/results") / f"{config.run_name}_eval.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

    # Generate report
    lines = []

    # Header
    lines.append("# Modern LLM Pipeline Report")
    lines.append("")
    lines.append(f"**Run Name:** {config.run_name}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Model Architecture
    lines.append("## Model Architecture")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Hidden Dimension | {config.d_model} |")
    lines.append(f"| Layers | {config.n_layers} |")
    lines.append(f"| Attention Heads | {config.n_heads} |")
    lines.append(f"| FFN Hidden Size | {config.ffn_hidden_size} |")
    lines.append(f"| Max Sequence Length | {config.max_seq_len} |")
    lines.append(f"| Vocabulary Size | {config.vocab_size} |")
    lines.append(f"| RoPE | {'Yes' if config.use_rope else 'No'} |")
    lines.append(f"| Attention Sinks | {'Yes' if config.use_attention_sinks else 'No'} |")
    lines.append(f"| SwiGLU | {'Yes' if config.use_swiglu else 'No'} |")
    lines.append(f"| GQA | {'Yes' if config.use_gqa else 'No'} |")
    lines.append("")

    # Pipeline Stages
    lines.append("## Pipeline Stages")
    lines.append("")

    lines.append("### 1. Pretraining")
    if state.pretrain_checkpoint:
        lines.append(f"- **Checkpoint:** `{state.pretrain_checkpoint}`")
        lines.append(f"- **Max Steps:** {config.pretrain_max_steps}")
        lines.append(f"- **Learning Rate:** {config.pretrain_lr}")
        lines.append(f"- **Batch Size:** {config.pretrain_batch_size}")
    else:
        lines.append("*Not completed*")
    lines.append("")

    lines.append("### 2. Supervised Fine-Tuning (SFT)")
    if state.sft_checkpoint:
        lines.append(f"- **Checkpoint:** `{state.sft_checkpoint}`")
        lines.append(f"- **Dataset:** {config.sft_dataset}")
        lines.append(f"- **Max Steps:** {config.sft_max_steps}")
        lines.append(f"- **Learning Rate:** {config.sft_lr}")
    else:
        lines.append("*Not completed*")
    lines.append("")

    lines.append("### 3. Direct Preference Optimization (DPO)")
    if state.dpo_checkpoint:
        lines.append(f"- **Checkpoint:** `{state.dpo_checkpoint}`")
        lines.append(f"- **Dataset:** {config.dpo_dataset}")
        lines.append(f"- **Beta:** {config.dpo_beta}")
        lines.append(f"- **Max Steps:** {config.dpo_max_steps}")
    else:
        lines.append("*Not completed*")
    lines.append("")

    lines.append("### 4. Verifier")
    if state.verifier_checkpoint:
        lines.append(f"- **Checkpoint:** `{state.verifier_checkpoint}`")
        lines.append(f"- **Max Steps:** {config.verifier_max_steps}")
    else:
        lines.append("*Not completed*")
    lines.append("")

    # Evaluation Results
    lines.append("## Evaluation Results")
    lines.append("")

    if eval_results:
        lines.append("### Perplexity Comparison")
        lines.append("")
        lines.append("| Stage | Perplexity | Loss | Parameters |")
        lines.append("|-------|------------|------|------------|")

        for stage_key in ["base", "sft", "dpo"]:
            stage = eval_results.get(stage_key)
            if stage:
                params_m = stage.get("num_params", 0) / 1e6
                lines.append(
                    f"| {stage_key.upper()} | {stage['perplexity']:.2f} | "
                    f"{stage['loss']:.4f} | {params_m:.1f}M |"
                )
        lines.append("")
    else:
        lines.append("*Evaluation not yet completed. Run `python scripts/evaluate_pipeline.py`*")
        lines.append("")

    # Sample Generations
    lines.append("## Sample Generations")
    lines.append("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample_prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In machine learning,",
    ]

    # Generate from final model if available
    final_checkpoint = state.dpo_checkpoint or state.sft_checkpoint or state.pretrain_checkpoint
    if final_checkpoint and final_checkpoint.exists():
        try:
            ckpt = load_checkpoint(final_checkpoint)
            model_config = ModernLLMConfig(**ckpt["config"])
            model = ModernDecoderLM(model_config)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
            model.eval()

            for prompt in sample_prompts:
                lines.append(f"**Prompt:** {prompt}")
                lines.append("")
                try:
                    generation = generate_text(
                        model, tokenizer, prompt,
                        max_new_tokens=50, temperature=0.7, device=device
                    )
                    lines.append(f"**Output:** {generation}")
                except Exception as e:
                    lines.append(f"*Generation failed: {e}*")
                lines.append("")

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            lines.append(f"*Could not load model for generation: {e}*")
            lines.append("")
    else:
        lines.append("*No model checkpoint available for generation*")
        lines.append("")

    # Conclusion
    lines.append("## Summary")
    lines.append("")
    lines.append("This report summarizes the Modern LLM pipeline execution.")
    lines.append("The pipeline implements a frontier-style training workflow:")
    lines.append("")
    lines.append("1. **Pretraining** - Language model training on text corpora")
    lines.append("2. **SFT** - Instruction tuning (Ouyang et al., 2022)")
    lines.append("3. **DPO** - Preference alignment (Rafailov et al., 2023)")
    lines.append("4. **Verifier** - Answer correctness scoring (Lightman et al., 2023)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Modern LLM Pipeline*")

    # Write report
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {report_path}")
    return report_path



