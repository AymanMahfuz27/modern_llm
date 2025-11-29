#!/usr/bin/env python3
"""Verify all checkpoints load correctly and produce sensible outputs.

Usage:
    python scripts/verify_checkpoints.py
    python scripts/verify_checkpoints.py --checkpoint-dir /path/to/checkpoints

This script:
1. Loads each checkpoint (pretrain, sft, dpo, verifier)
2. Verifies model architecture matches config
3. Runs a forward pass to ensure weights are valid
4. Generates sample text from language models
5. Scores a sample problem with the verifier
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
from modern_llm.models.verifier import VerifierConfig, VerifierModel


def load_lm_checkpoint(path: Path, device: torch.device):
    """Load a language model checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    if "config" in ckpt:
        config = ModernLLMConfig(**ckpt["config"])
    else:
        # Default config for 253M model
        config = ModernLLMConfig(
            vocab_size=50257,
            d_model=768,
            n_layers=12,
            n_heads=12,
            ffn_hidden_size=3072,
            max_seq_len=1024,
        )
    
    model = ModernDecoderLM(config)
    
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    return model, config


def load_verifier_checkpoint(path: Path, device: torch.device):
    """Load a verifier checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    if "config" in ckpt:
        config = VerifierConfig(**ckpt["config"])
    else:
        config = VerifierConfig(
            vocab_size=50257,
            d_model=512,
            num_layers=4,
            n_heads=8,
            max_position_embeddings=1024,
        )
    
    model = VerifierModel(config)
    
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    return model, config


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, device: torch.device = None):
    """Generate text from a model."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_token_logits = logits[:, -1, :] / 0.8  # temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Verify checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing checkpoints",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Expected checkpoints
    checkpoints = {
        "pretrain": args.checkpoint_dir / "pretrain_best.pt",
        "sft": args.checkpoint_dir / "sft_final.pt",
        "dpo": args.checkpoint_dir / "dpo_final.pt",
        "verifier": args.checkpoint_dir / "verifier_final.pt",
    }
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    print("=" * 60)
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)
    
    # Test language models
    for name in ["pretrain", "sft", "dpo"]:
        path = checkpoints[name]
        print(f"\n[{name.upper()}] {path}")
        print("-" * 40)
        
        if not path.exists():
            print(f"  ❌ Checkpoint not found")
            results[name] = {"status": "missing"}
            continue
        
        try:
            model, config = load_lm_checkpoint(path, device)
            model.to(device)
            model.eval()
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Loaded successfully")
            print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
            print(f"  Config: d={config.d_model}, L={config.n_layers}, H={config.n_heads}")
            
            # Test forward pass
            test_input = torch.randint(0, config.vocab_size, (1, 32), device=device)
            with torch.no_grad():
                output = model(test_input)
            print(f"  ✓ Forward pass OK (output shape: {output['logits'].shape})")
            
            # Generate sample
            prompt = "The meaning of life is"
            generation = generate_text(model, tokenizer, prompt, max_new_tokens=30, device=device)
            print(f"  ✓ Generation OK")
            print(f"  Sample: \"{generation[:100]}...\"")
            
            results[name] = {
                "status": "ok",
                "params": params,
                "sample": generation,
            }
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[name] = {"status": "error", "error": str(e)}
    
    # Test verifier
    name = "verifier"
    path = checkpoints[name]
    print(f"\n[{name.upper()}] {path}")
    print("-" * 40)
    
    if not path.exists():
        print(f"  ❌ Checkpoint not found")
        results[name] = {"status": "missing"}
    else:
        try:
            model, config = load_verifier_checkpoint(path, device)
            model.to(device)
            model.eval()
            
            params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Loaded successfully")
            print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
            print(f"  Config: d={config.d_model}, L={config.num_layers}, H={config.n_heads}")
            
            # Test forward pass with sample problem
            problem = "What is 2 + 2?"
            answer = "4"
            
            # Verifier expects concatenated input: "Question: ... Answer: ..."
            full_input = f"Question: {problem} Answer: {answer}"
            input_ids = tokenizer.encode(full_input, return_tensors="pt", truncation=True, max_length=256).to(device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                score = probs[0, 1].item()  # Probability of "correct"
            
            print(f"  ✓ Forward pass OK (logits shape: {logits.shape})")
            print(f"  Sample: \"{problem}\" → \"{answer}\" = {score:.4f} (correct prob)")
            
            results[name] = {
                "status": "ok",
                "params": params,
                "sample_score": score,
            }
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[name] = {"status": "error", "error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for name, result in results.items():
        status = result["status"]
        if status == "ok":
            params = result.get("params", 0)
            print(f"  ✓ {name}: OK ({params/1e6:.1f}M params)")
        else:
            print(f"  ❌ {name}: {status}")
            all_ok = False
    
    if all_ok:
        print("\n✅ All checkpoints verified successfully!")
        return 0
    else:
        print("\n⚠️  Some checkpoints failed verification")
        return 1


if __name__ == "__main__":
    sys.exit(main())

