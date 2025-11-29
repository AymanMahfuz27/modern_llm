#!/usr/bin/env python3
"""GSM8K Math Benchmark: Compare your model vs GPT-2 with verifier reranking.

Usage:
    python scripts/benchmark_gsm8k.py
    python scripts/benchmark_gsm8k.py --num-samples 100 --num-candidates 5

This script:
1. Loads GSM8K test problems
2. Generates answers from your DPO model and GPT-2
3. Uses the verifier to rerank candidate answers
4. Computes exact match accuracy with and without verifier
5. Generates a report with examples
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
from modern_llm.models.verifier import VerifierConfig, VerifierModel


def extract_answer(text: str) -> str:
    """Extract numeric answer from generated text.
    
    Looks for patterns like "#### 42" or "the answer is 42" or just the last number.
    """
    # Try GSM8K format first: #### answer
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Try "answer is X" pattern
    match = re.search(r"answer\s+is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    # Try "= X" at end
    match = re.search(r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Fall back to last number in text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return ""


def load_model(checkpoint_path: Path, device: torch.device):
    """Load language model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "config" in ckpt:
        config = ModernLLMConfig(**ckpt["config"])
    else:
        config = ModernLLMConfig(
            vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
            ffn_hidden_size=3072, max_seq_len=1024,
        )
    
    model = ModernDecoderLM(config)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    
    return model.to(device).eval()


def load_verifier(checkpoint_path: Path, device: torch.device):
    """Load verifier from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "config" in ckpt:
        config = VerifierConfig(**ckpt["config"])
    else:
        config = VerifierConfig(
            vocab_size=50257, d_model=512, num_layers=4,
            n_heads=8, max_position_embeddings=1024,
        )
    
    model = VerifierModel(config)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    
    return model.to(device).eval()


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 150,
                   temperature: float = 0.7, device: torch.device = None) -> str:
    """Generate an answer to a math question."""
    prompt = f"Question: {question}\nAnswer: Let me solve this step by step.\n"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if hasattr(model, "forward"):
                outputs = model(input_ids)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs.logits
            else:
                outputs = model(input_ids)
                logits = outputs.logits
            
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop at newline or EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            if input_ids.size(1) > 512:  # Max length safety
                break
    
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # Return just the answer part
    answer_start = full_text.find("Answer:")
    if answer_start >= 0:
        return full_text[answer_start + 7:].strip()
    return full_text[len(prompt):].strip()


def generate_candidates(model, tokenizer, question: str, num_candidates: int,
                       device: torch.device) -> List[str]:
    """Generate multiple candidate answers."""
    candidates = []
    for _ in range(num_candidates):
        answer = generate_answer(
            model, tokenizer, question,
            temperature=0.8 + 0.2 * torch.rand(1).item(),  # Vary temperature
            device=device,
        )
        candidates.append(answer)
    return candidates


def score_with_verifier(verifier, tokenizer, question: str, answer: str,
                       device: torch.device) -> float:
    """Score an answer using the verifier."""
    # Verifier expects concatenated input
    full_input = f"Question: {question} Answer: {answer}"
    input_ids = tokenizer.encode(full_input, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = verifier(input_ids)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        score = probs[0, 1].item()  # Probability of "correct"
    
    return score


def main():
    parser = argparse.ArgumentParser(description="GSM8K Math Benchmark")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--num-samples", type=int, default=50, help="Number of problems to evaluate")
    parser.add_argument("--num-candidates", type=int, default=3, help="Candidates for verifier reranking")
    parser.add_argument("--output", type=Path, default=Path("experiments/gsm8k_benchmark.json"))
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    print("\nLoading models...")
    
    dpo_path = args.checkpoint_dir / "dpo_final.pt"
    verifier_path = args.checkpoint_dir / "verifier_final.pt"
    
    if not dpo_path.exists():
        print(f"DPO checkpoint not found: {dpo_path}")
        sys.exit(1)
    
    your_model = load_model(dpo_path, device)
    your_params = sum(p.numel() for p in your_model.parameters())
    print(f"  Your model: {your_params/1e6:.1f}M params")
    
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    gpt2_params = sum(p.numel() for p in gpt2.parameters())
    print(f"  GPT-2: {gpt2_params/1e6:.1f}M params")
    
    verifier = None
    if verifier_path.exists():
        verifier = load_verifier(verifier_path, device)
        v_params = sum(p.numel() for p in verifier.parameters())
        print(f"  Verifier: {v_params/1e6:.1f}M params")
    else:
        print("  Verifier: Not found (skipping reranking)")
    
    # Load GSM8K
    print("\nLoading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = list(dataset.select(range(min(args.num_samples, len(dataset)))))
    print(f"  Evaluating {len(samples)} problems")
    
    # Evaluate
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(samples),
        "num_candidates": args.num_candidates,
        "your_model": {"correct": 0, "correct_reranked": 0, "examples": []},
        "gpt2": {"correct": 0, "correct_reranked": 0, "examples": []},
    }
    
    print("\nEvaluating...")
    for i, sample in enumerate(tqdm(samples)):
        question = sample["question"]
        gold_answer = extract_answer(sample["answer"])
        
        # Your model
        your_candidates = generate_candidates(your_model, tokenizer, question, args.num_candidates, device)
        your_answers = [extract_answer(c) for c in your_candidates]
        your_best = your_answers[0]  # Default: first candidate
        your_correct = (your_best == gold_answer)
        
        # Verifier reranking for your model
        your_best_reranked = your_best
        if verifier and your_candidates:
            scores = [score_with_verifier(verifier, tokenizer, question, c, device) for c in your_candidates]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            your_best_reranked = your_answers[best_idx]
        your_correct_reranked = (your_best_reranked == gold_answer)
        
        results["your_model"]["correct"] += int(your_correct)
        results["your_model"]["correct_reranked"] += int(your_correct_reranked)
        
        # GPT-2
        gpt2_candidates = generate_candidates(gpt2, tokenizer, question, args.num_candidates, device)
        gpt2_answers = [extract_answer(c) for c in gpt2_candidates]
        gpt2_best = gpt2_answers[0] if gpt2_answers else ""
        gpt2_correct = (gpt2_best == gold_answer)
        
        # Verifier reranking for GPT-2
        gpt2_best_reranked = gpt2_best
        if verifier and gpt2_candidates:
            scores = [score_with_verifier(verifier, tokenizer, question, c, device) for c in gpt2_candidates]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            gpt2_best_reranked = gpt2_answers[best_idx]
        gpt2_correct_reranked = (gpt2_best_reranked == gold_answer)
        
        results["gpt2"]["correct"] += int(gpt2_correct)
        results["gpt2"]["correct_reranked"] += int(gpt2_correct_reranked)
        
        # Save example
        if i < 10:  # Save first 10 examples
            results["your_model"]["examples"].append({
                "question": question,
                "gold": gold_answer,
                "predicted": your_best,
                "predicted_reranked": your_best_reranked,
                "correct": your_correct,
                "correct_reranked": your_correct_reranked,
            })
            results["gpt2"]["examples"].append({
                "question": question,
                "gold": gold_answer,
                "predicted": gpt2_best,
                "predicted_reranked": gpt2_best_reranked,
                "correct": gpt2_correct,
                "correct_reranked": gpt2_correct_reranked,
            })
    
    # Compute accuracy
    n = len(samples)
    results["your_model"]["accuracy"] = results["your_model"]["correct"] / n
    results["your_model"]["accuracy_reranked"] = results["your_model"]["correct_reranked"] / n
    results["gpt2"]["accuracy"] = results["gpt2"]["correct"] / n
    results["gpt2"]["accuracy_reranked"] = results["gpt2"]["correct_reranked"] / n
    
    # Print results
    print("\n" + "=" * 60)
    print("GSM8K MATH BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n{'Model':<25} {'Accuracy':<15} {'+Verifier':<15}")
    print("-" * 55)
    print(f"{'Your Model (DPO)':<25} {results['your_model']['accuracy']*100:>6.1f}%        {results['your_model']['accuracy_reranked']*100:>6.1f}%")
    print(f"{'GPT-2':<25} {results['gpt2']['accuracy']*100:>6.1f}%        {results['gpt2']['accuracy_reranked']*100:>6.1f}%")
    
    if verifier:
        your_gain = results["your_model"]["accuracy_reranked"] - results["your_model"]["accuracy"]
        gpt2_gain = results["gpt2"]["accuracy_reranked"] - results["gpt2"]["accuracy"]
        print(f"\nVerifier improvement:")
        print(f"  Your Model: +{your_gain*100:.1f}%")
        print(f"  GPT-2:      +{gpt2_gain*100:.1f}%")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    # Generate report
    report_path = Path("report") / "gsm8k_benchmark_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# GSM8K Math Benchmark Report\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")
        f.write(f"**Samples:** {n}\n\n")
        f.write(f"**Candidates per problem:** {args.num_candidates}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Model | Accuracy | +Verifier Reranking |\n")
        f.write("|-------|----------|---------------------|\n")
        f.write(f"| Your Model (DPO, 253M) | {results['your_model']['accuracy']*100:.1f}% | {results['your_model']['accuracy_reranked']*100:.1f}% |\n")
        f.write(f"| GPT-2 (124M) | {results['gpt2']['accuracy']*100:.1f}% | {results['gpt2']['accuracy_reranked']*100:.1f}% |\n\n")
        
        if verifier:
            f.write("## Verifier Impact\n\n")
            f.write(f"- Your Model: +{(results['your_model']['accuracy_reranked'] - results['your_model']['accuracy'])*100:.1f}% accuracy gain\n")
            f.write(f"- GPT-2: +{(results['gpt2']['accuracy_reranked'] - results['gpt2']['accuracy'])*100:.1f}% accuracy gain\n\n")
        
        f.write("## Sample Problems\n\n")
        for i, (your_ex, gpt2_ex) in enumerate(zip(
            results["your_model"]["examples"][:5],
            results["gpt2"]["examples"][:5]
        )):
            f.write(f"### Problem {i+1}\n\n")
            f.write(f"**Question:** {your_ex['question'][:200]}...\n\n")
            f.write(f"**Gold Answer:** {your_ex['gold']}\n\n")
            f.write(f"| Model | Prediction | Correct? |\n")
            f.write(f"|-------|------------|----------|\n")
            f.write(f"| Your Model | {your_ex['predicted_reranked']} | {'✓' if your_ex['correct_reranked'] else '✗'} |\n")
            f.write(f"| GPT-2 | {gpt2_ex['predicted_reranked']} | {'✓' if gpt2_ex['correct_reranked'] else '✗'} |\n\n")
    
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

