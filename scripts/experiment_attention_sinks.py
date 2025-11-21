"""Long-context and attention sinks experiment.

Compares generation stability when trained context is exceeded, with and
without attention sinks enabled (Press et al., 2021).

The experiment trains two small models (one with sinks, one without) then
generates at 2-4x the trained context length to measure stability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig, TrainingConfig
from modern_llm.data import LanguageModelingDatasetConfig, load_causal_lm_dataset
from modern_llm.models import ModernDecoderLM
from modern_llm.training.trainer_base import Trainer
from modern_llm.training.train_lm import generate_text


def _count_repetitions(text: str, window: int = 10) -> float:
    """Measure repetition by counting repeated n-gram windows.

    Pre:
        - text is non-empty, window > 0.
    Post:
        - returns fraction of windows that repeat at least once later in the text.
    """
    if not text or window <= 0:
        raise ValueError("text must be non-empty and window must be positive")
    words = text.split()
    if len(words) < window:
        return 0.0
    ngrams = [" ".join(words[i : i + window]) for i in range(len(words) - window + 1)]
    repeated = sum(1 for i, ng in enumerate(ngrams) if ng in ngrams[i + 1 :])
    return repeated / len(ngrams) if ngrams else 0.0


def _train_model_variant(
    run_name: str,
    use_sinks: bool,
    tokenizer,
    output_dir: Path,
    max_seq_len: int,
    max_steps: int,
    num_proc: int,
) -> ModernDecoderLM:
    """Train a small LM with or without attention sinks.

    Pre:
        - max_seq_len > 0, max_steps > 0.
    Post:
        - returns trained model in eval mode.
    """
    dataset_config = LanguageModelingDatasetConfig(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        split="train",
        max_length=max_seq_len,
        num_proc=num_proc,
    )
    train_dataset = load_causal_lm_dataset(dataset_config, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)

    model_config = ModernLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        ffn_hidden_size=1024,
        max_seq_len=max_seq_len,
        rope_theta=10000.0,
        dropout=0.0,
        use_rope=True,
        use_attention_sinks=use_sinks,
        num_attention_sinks=4,
        use_swiglu=True,
        tie_embeddings=True,
    )
    model = ModernDecoderLM(model_config)

    training_config = TrainingConfig(
        run_name=run_name,
        dataset_name="wikitext",
        tokenizer_name="gpt2",
        output_dir=output_dir / run_name,
        batch_size=16,
        micro_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        max_steps=max_steps,
        warmup_steps=50,
        weight_decay=0.01,
        eval_every=0,
        save_every=0,
        log_every=50,
        mixed_precision="bf16",
        max_grad_norm=1.0,
        compile_model=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < training_config.warmup_steps:
            return float(step + 1) / float(training_config.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        eval_dataloader=None,
        config=training_config,
        lr_scheduler=scheduler,
    )
    trainer.train()

    model.eval()
    return model


def main() -> None:
    """Run attention sinks experiment comparing stability at extended context."""

    parser = argparse.ArgumentParser(description="Attention sinks long-context experiment.")
    parser.add_argument("--output_dir", type=str, default="experiments/runs")
    parser.add_argument("--output_json", type=str, default="experiments/attention_sinks_results.json")
    parser.add_argument("--train_context", type=int, default=512)
    parser.add_argument("--gen_context_multiplier", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["The meaning of life is", "In a distant galaxy,"],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.train_context

    print(f"Training model WITH attention sinks (context={args.train_context})...")
    model_with_sinks = _train_model_variant(
        run_name="sinks-enabled",
        use_sinks=True,
        tokenizer=tokenizer,
        output_dir=output_dir,
        max_seq_len=args.train_context,
        max_steps=args.max_steps,
        num_proc=args.num_proc,
    )

    print(f"\nTraining model WITHOUT attention sinks (context={args.train_context})...")
    model_no_sinks = _train_model_variant(
        run_name="sinks-disabled",
        use_sinks=False,
        tokenizer=tokenizer,
        output_dir=output_dir,
        max_seq_len=args.train_context,
        max_steps=args.max_steps,
        num_proc=args.num_proc,
    )

    gen_tokens = args.train_context * args.gen_context_multiplier
    results = {"train_context": args.train_context, "gen_tokens": gen_tokens, "comparisons": []}

    for prompt in args.prompts:
        print(f"\n--- Generating for prompt: '{prompt}' ---")
        with_sinks_text = generate_text(
            model_with_sinks,
            tokenizer,
            prompt,
            max_new_tokens=gen_tokens,
            temperature=1.0,
            top_k=50,
        )
        no_sinks_text = generate_text(
            model_no_sinks,
            tokenizer,
            prompt,
            max_new_tokens=gen_tokens,
            temperature=1.0,
            top_k=50,
        )

        rep_with = _count_repetitions(with_sinks_text, window=10)
        rep_no = _count_repetitions(no_sinks_text, window=10)

        results["comparisons"].append({
            "prompt": prompt,
            "with_sinks_text": with_sinks_text,
            "without_sinks_text": no_sinks_text,
            "with_sinks_repetition": rep_with,
            "without_sinks_repetition": rep_no,
        })

        print(f"WITH sinks: repetition={rep_with:.3f}")
        print(f"WITHOUT sinks: repetition={rep_no:.3f}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote attention sinks experiment results to {output_path}")


if __name__ == "__main__":
    main()



