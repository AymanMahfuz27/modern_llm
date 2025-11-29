#!/usr/bin/env python3
"""Evaluate trained models and compare against GPT-2 baseline."""

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modern_llm.config import get_pipeline_preset
from modern_llm.models import ModernDecoderLM


def load_model(checkpoint_path: Path, config):
    """Load a model from checkpoint."""
    model_config = config.get_model_config()
    model = ModernDecoderLM(model_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    
    return model


def compute_perplexity(model, tokenizer, texts, device, max_length=512):
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts[:100]:  # Limit to 100 samples for speed
            if not text.strip():
                continue
            encoded = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            )
            input_ids = encoded["input_ids"].to(device)
            
            if input_ids.size(1) < 2:
                continue
            
            # Get model output
            if hasattr(model, "forward"):
                outputs = model(input_ids=input_ids, attention_mask=encoded["attention_mask"].to(device))
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs.logits
            else:
                outputs = model(input_ids)
                logits = outputs.logits
            
            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    if total_tokens == 0:
        return float("inf")
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def generate_text(model, tokenizer, prompt, device, max_new_tokens=50):
    """Generate text from a prompt."""
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if hasattr(model, "forward"):
                outputs = model(input_ids=input_ids)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs.logits
            else:
                outputs = model(input_ids)
                logits = outputs.logits
            
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    project_dir = Path(__file__).parent.parent
    checkpoints = {
        "pretrain": project_dir / "checkpoints" / "pretrain_best.pt",
        "sft": project_dir / "checkpoints" / "sft_final.pt",
        "dpo": project_dir / "checkpoints" / "dpo_final.pt",
    }
    
    # Load config and tokenizer
    config = get_pipeline_preset("local")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load validation data
    print("\nLoading WikiText-2 validation set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    val_texts = [t for t in dataset["text"] if t.strip()]
    print(f"Loaded {len(val_texts)} validation samples")
    
    # Load GPT-2 baseline
    print("\n" + "="*60)
    print("Loading GPT-2 baseline...")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_params = sum(p.numel() for p in gpt2.parameters())
    print(f"  Parameters: {gpt2_params:,} ({gpt2_params/1e6:.1f}M)")
    
    gpt2_ppl = compute_perplexity(gpt2, tokenizer, val_texts, device)
    print(f"  WikiText-2 Perplexity: {gpt2_ppl:.2f}")
    
    # Results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "gpt2": {
            "parameters": gpt2_params,
            "perplexity": gpt2_ppl,
        },
        "stages": {}
    }
    
    # Evaluate each stage
    for stage_name, ckpt_path in checkpoints.items():
        print("\n" + "="*60)
        print(f"Evaluating {stage_name.upper()} model...")
        print("="*60)
        
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue
        
        model = load_model(ckpt_path, config).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
        
        # Perplexity
        ppl = compute_perplexity(model, tokenizer, val_texts, device)
        print(f"  WikiText-2 Perplexity: {ppl:.2f}")
        
        results["stages"][stage_name] = {
            "parameters": params,
            "perplexity": ppl,
            "checkpoint": str(ckpt_path),
        }
        
        del model
        torch.cuda.empty_cache()
    
    # Generation comparison
    print("\n" + "="*60)
    print("GENERATION COMPARISON")
    print("="*60)
    
    prompts = [
        "The history of artificial intelligence began",
        "In a groundbreaking scientific discovery,",
        "The economic implications of",
    ]
    
    # Load DPO model for generation (best aligned model)
    dpo_path = checkpoints["dpo"]
    if dpo_path.exists():
        dpo_model = load_model(dpo_path, config).to(device)
        
        generations = {"prompts": [], "your_model": [], "gpt2": []}
        
        for prompt in prompts:
            print(f"\nPROMPT: {prompt}")
            print("-" * 40)
            
            your_gen = generate_text(dpo_model, tokenizer, prompt, device)
            gpt2_gen = generate_text(gpt2, tokenizer, prompt, device)
            
            print(f"YOUR MODEL (DPO): {your_gen[:200]}...")
            print(f"GPT-2:            {gpt2_gen[:200]}...")
            
            generations["prompts"].append(prompt)
            generations["your_model"].append(your_gen)
            generations["gpt2"].append(gpt2_gen)
        
        results["generations"] = generations
    
    # Summary table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\n┌─────────────────────┬──────────────────┬──────────────────┐")
    print("│ Model               │ Parameters       │ WikiText-2 PPL   │")
    print("├─────────────────────┼──────────────────┼──────────────────┤")
    print(f"│ GPT-2 (baseline)    │ {gpt2_params/1e6:>14.1f}M │ {gpt2_ppl:>16.2f} │")
    
    for stage_name, stage_data in results["stages"].items():
        params = stage_data["parameters"]
        ppl = stage_data["perplexity"]
        print(f"│ Your {stage_name:<13} │ {params/1e6:>14.1f}M │ {ppl:>16.2f} │")
    
    print("└─────────────────────┴──────────────────┴──────────────────┘")
    
    # Save results
    results_dir = project_dir / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "full_eval.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save comparison log
    log_file = project_dir / "experiments" / "comparison_log_v2.txt"
    with open(log_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("YOUR MODEL vs GPT-2 COMPARISON (v2 - Better Training)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("CHECKPOINTS:\n")
        for stage, path in checkpoints.items():
            f.write(f"  {stage}: {path}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("PERPLEXITY RESULTS (lower is better)\n")
        f.write("="*60 + "\n\n")
        f.write(f"  GPT-2 (baseline):     {gpt2_ppl:.2f}\n")
        for stage_name, stage_data in results["stages"].items():
            f.write(f"  Your {stage_name:<15}: {stage_data['perplexity']:.2f}\n")
        
        if "generations" in results:
            f.write("\n" + "="*60 + "\n")
            f.write("GENERATION SAMPLES\n")
            f.write("="*60 + "\n")
            for i, prompt in enumerate(results["generations"]["prompts"]):
                f.write(f"\nPROMPT: {prompt}\n")
                f.write("-"*40 + "\n")
                f.write(f"YOUR MODEL: {results['generations']['your_model'][i][:300]}...\n")
                f.write(f"GPT-2:      {results['generations']['gpt2'][i][:300]}...\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("COMPARISON COMPLETE!\n")
        f.write("="*60 + "\n")
    
    print(f"Comparison log saved to {log_file}")
    
    # Generate markdown report
    report_file = project_dir / "report" / "full_report.md"
    with open(report_file, "w") as f:
        f.write("# Modern LLM Training Report (v2)\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")
        
        f.write("## Model Architecture\n\n")
        f.write(f"- **d_model:** {config.d_model}\n")
        f.write(f"- **n_layers:** {config.n_layers}\n")
        f.write(f"- **n_heads:** {config.n_heads}\n")
        f.write(f"- **max_seq_len:** {config.max_seq_len}\n")
        f.write(f"- **Parameters:** ~253M\n\n")
        
        f.write("## Training Data\n\n")
        f.write("- **Pretrain:** WikiText-103 + TinyStories (~600M tokens, 30K steps)\n")
        f.write("- **SFT:** tatsu-lab/alpaca (52K examples, 5K steps)\n")
        f.write("- **DPO:** Anthropic/hh-rlhf (161K pairs, 3K steps)\n")
        f.write("- **Verifier:** GSM8K (15K examples, 3K steps)\n\n")
        
        f.write("## Evaluation Results\n\n")
        f.write("### Perplexity (WikiText-2 Validation)\n\n")
        f.write("| Model | Parameters | Perplexity |\n")
        f.write("|-------|------------|------------|\n")
        f.write(f"| GPT-2 (baseline) | 124M | {gpt2_ppl:.2f} |\n")
        for stage_name, stage_data in results["stages"].items():
            f.write(f"| Your {stage_name} | {stage_data['parameters']/1e6:.0f}M | {stage_data['perplexity']:.2f} |\n")
        
        if "generations" in results:
            f.write("\n### Sample Generations\n\n")
            for i, prompt in enumerate(results["generations"]["prompts"]):
                f.write(f"**Prompt:** {prompt}\n\n")
                f.write(f"- **Your Model:** {results['generations']['your_model'][i][:200]}...\n")
                f.write(f"- **GPT-2:** {results['generations']['gpt2'][i][:200]}...\n\n")
        
        f.write("## Checkpoints\n\n")
        for stage, path in checkpoints.items():
            f.write(f"- **{stage}:** `{path}`\n")
    
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()

