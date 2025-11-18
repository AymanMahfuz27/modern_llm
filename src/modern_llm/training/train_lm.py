"""Language modeling training entrypoint (causal LM on WikiText-2/TinyStories)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig, TrainingConfig
from modern_llm.data import LanguageModelingDatasetConfig, load_causal_lm_dataset
from modern_llm.models import ModernDecoderLM
from modern_llm.training.trainer_base import Trainer


def _sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tensor:
    """Sample the next token id from logits with optional temperature and top-k truncation.

    Pre:
        - logits: shape (vocab_size,) on a single device.
        - temperature > 0.
    Post:
        - returns a scalar tensor containing the sampled token id.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, received {temperature}")

    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        values, indices = torch.topk(logits, top_k)
        min_topk = values[..., -1]
        logits = torch.where(logits < min_topk, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(
    model: ModernDecoderLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
) -> str:
    """Generate text from a trained `ModernDecoderLM` given a prompt.

    Pre:
        - `max_new_tokens` > 0.
        - `prompt` is a non-empty string whose tokenized length is < model.config.max_seq_len.
    Post:
        - returns the decoded prompt + continuation string.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, received {max_new_tokens}")
    if not prompt:
        raise ValueError("prompt must be a non-empty string")

    model.eval()
    device = next(model.parameters()).device

    encoded = tokenizer.encode(prompt, return_tensors="pt")
    if encoded.dim() != 2 or encoded.size(0) != 1:
        raise ValueError("tokenizer.encode must return tensors of shape (1, seq_len)")
    input_ids = encoded.to(device)

    max_seq_len = model.config.max_seq_len
    if input_ids.size(1) >= max_seq_len:
        raise ValueError(
            f"Prompt length {input_ids.size(1)} exceeds or equals model max_seq_len {max_seq_len}"
        )

    available_tokens = max_seq_len - input_ids.size(1)
    steps = min(max_new_tokens, available_tokens)

    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        for _ in range(steps):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            if logits is None:
                raise ValueError("Model did not return logits during generation.")
            next_token_logits = logits[:, -1, :].squeeze(0)
            next_token = _sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            attention_mask = torch.ones_like(input_ids, device=device)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main() -> None:
    """Train the from-scratch decoder LM on WikiText-2 or TinyStories."""

    parser = argparse.ArgumentParser(description="Train Modern Decoder LM on a causal LM corpus.")
    parser.add_argument("--run_name", type=str, default="scratch-lm")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--ffn_hidden_size", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="experiments/runs")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument(
        "--gen_prompt",
        type=str,
        default="The meaning of life is",
        help="Prompt used for post-training text generation.",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=64,
        help="Number of new tokens to sample after training (set to 0 to disable).",
    )
    parser.add_argument(
        "--gen_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for post-training generation.",
    )
    parser.add_argument(
        "--gen_top_k",
        type=int,
        default=50,
        help="Top-k truncation for sampling (<=0 disables top-k).",
    )
    args = parser.parse_args()

    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size for gradient accumulation.")

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    train_dataset = load_causal_lm_dataset(
        LanguageModelingDatasetConfig(
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            split="train",
            max_length=args.max_seq_len,
            num_proc=args.num_proc,
        ),
        tokenizer,
    )
    eval_dataset = load_causal_lm_dataset(
        LanguageModelingDatasetConfig(
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            split="validation",
            max_length=args.max_seq_len,
            num_proc=args.num_proc,
        ),
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
    )

    model_config = ModernLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        dropout=args.dropout,
    )
    model = ModernDecoderLM(model_config)

    training_config = TrainingConfig(
        run_name=args.run_name,
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        output_dir=output_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        mixed_precision=args.mixed_precision,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)

    scheduler = None
    if training_config.warmup_steps > 0:
        def lr_lambda(step: int) -> float:
            if step < training_config.warmup_steps:
                return float(step + 1) / float(training_config.warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=training_config,
        lr_scheduler=scheduler,
    )
    trainer.train()

    if args.gen_max_new_tokens > 0:
        top_k = args.gen_top_k if args.gen_top_k > 0 else None
        sample = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.gen_prompt,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            top_k=top_k,
        )
        separator = "=" * 80
        print(separator)
        print("Post-training sample generation:")
        print(sample)
        print(separator)
