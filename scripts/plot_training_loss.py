#!/usr/bin/env python3
"""
Parse TACC training log and generate loss curves.

Usage:
    python scripts/plot_training_loss.py [--log-file PATH] [--output PATH]
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def parse_log_file(log_path: str) -> dict:
    """Parse the SLURM error log file to extract loss values by stage."""
    
    stages = defaultdict(lambda: {"steps": [], "losses": [], "lrs": []})
    current_stage = None
    
    # Pattern to match log lines with loss info
    # Format 1: step=100 loss=1.9034 lr=6.060e-05
    # Format 2: step=0 loss=0.1835 accuracy=100.00% lr=5.000e-06 (DPO)
    loss_pattern = re.compile(r'step=(\d+)\s+loss=([\d.]+).*?lr=([\d.e+-]+)')
    
    # Pattern to detect stage changes
    stage_patterns = {
        'pretrain': re.compile(r'(pretrain|Pretraining|trainer\.gpu-full-pretrain)', re.IGNORECASE),
        'sft': re.compile(r'(sft|fine-tuning|Stage 2|trainer\.gpu-full-sft)', re.IGNORECASE),
        'dpo': re.compile(r'(dpo|preference|Stage 3)', re.IGNORECASE),
        'verifier': re.compile(r'(verifier|Stage 4)', re.IGNORECASE),
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Split on carriage returns to handle tqdm output
    lines = content.replace('\r', '\n').split('\n')
    
    for line in lines:
        # Check for stage changes
        for stage_name, pattern in stage_patterns.items():
            if pattern.search(line):
                current_stage = stage_name
                break
        
        # Extract loss info
        match = loss_pattern.search(line)
        if match and current_stage:
            step = int(match.group(1))
            loss = float(match.group(2))
            lr = float(match.group(3))
            
            # Skip duplicate step entries and zero-loss artifacts
            if step not in stages[current_stage]["steps"] and loss > 0.001:
                stages[current_stage]["steps"].append(step)
                stages[current_stage]["losses"].append(loss)
                stages[current_stage]["lrs"].append(lr)
    
    # Sort by step for each stage
    for stage_name in stages:
        data = stages[stage_name]
        if data["steps"]:
            sorted_indices = np.argsort(data["steps"])
            data["steps"] = [data["steps"][i] for i in sorted_indices]
            data["losses"] = [data["losses"][i] for i in sorted_indices]
            data["lrs"] = [data["lrs"][i] for i in sorted_indices]
    
    return dict(stages)


def smooth_curve(values: list, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')
    # Pad to match original length
    pad_size = len(values) - len(smoothed)
    return np.concatenate([values[:pad_size], smoothed])


def plot_losses(stages: dict, output_path: str, smoothing: int = 10):
    """Generate a multi-panel loss plot."""
    
    # Set up the figure with a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Color scheme
    colors = {
        'pretrain': '#2E86AB',   # Blue
        'sft': '#A23B72',        # Magenta
        'dpo': '#F18F01',        # Orange
        'verifier': '#C73E1D',   # Red
    }
    
    stage_labels = {
        'pretrain': 'Pretraining (80K steps)',
        'sft': 'SFT (10K steps)',
        'dpo': 'DPO (6K steps)',
        'verifier': 'Verifier (3K steps)',
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss Curves - Modern LLM Pipeline', fontsize=16, fontweight='bold', y=0.98)
    
    stage_order = ['pretrain', 'sft', 'dpo', 'verifier']
    
    for idx, stage_name in enumerate(stage_order):
        ax = axes[idx // 2, idx % 2]
        
        if stage_name not in stages or len(stages[stage_name]["steps"]) == 0:
            ax.text(0.5, 0.5, f'No data for {stage_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(stage_labels.get(stage_name, stage_name))
            continue
        
        data = stages[stage_name]
        steps = np.array(data["steps"])
        losses = np.array(data["losses"])
        
        # Plot raw data (faded)
        ax.plot(steps, losses, alpha=0.3, color=colors[stage_name], linewidth=0.5, label='Raw')
        
        # Plot smoothed curve
        smoothed = smooth_curve(losses, window=smoothing)
        ax.plot(steps, smoothed, color=colors[stage_name], linewidth=2, label=f'Smoothed (w={smoothing})')
        
        # Add final loss annotation
        final_loss = losses[-1]
        final_step = steps[-1]
        ax.annotate(f'Final: {final_loss:.4f}', 
                   xy=(final_step, final_loss),
                   xytext=(-60, 20), textcoords='offset points',
                   fontsize=10, color=colors[stage_name],
                   arrowprops=dict(arrowstyle='->', color=colors[stage_name], alpha=0.7))
        
        # Add min loss annotation
        min_idx = np.argmin(losses)
        min_loss = losses[min_idx]
        min_step = steps[min_idx]
        if min_step != final_step:  # Only if different from final
            ax.scatter([min_step], [min_loss], color=colors[stage_name], s=50, zorder=5, marker='*')
            ax.annotate(f'Min: {min_loss:.4f}', 
                       xy=(min_step, min_loss),
                       xytext=(10, -25), textcoords='offset points',
                       fontsize=9, color=colors[stage_name], alpha=0.8)
        
        ax.set_title(stage_labels.get(stage_name, stage_name), fontsize=12, fontweight='bold')
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        y_min = max(0, min(losses) * 0.9)
        y_max = max(losses) * 1.05
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved loss plot to: {output_path}")
    
    # Also save individual stage plots
    output_dir = Path(output_path).parent
    for stage_name in stage_order:
        if stage_name in stages and len(stages[stage_name]["steps"]) > 0:
            fig_single, ax_single = plt.subplots(figsize=(10, 6))
            
            data = stages[stage_name]
            steps = np.array(data["steps"])
            losses = np.array(data["losses"])
            smoothed = smooth_curve(losses, window=smoothing)
            
            ax_single.fill_between(steps, losses, alpha=0.2, color=colors[stage_name])
            ax_single.plot(steps, losses, alpha=0.4, color=colors[stage_name], linewidth=0.5)
            ax_single.plot(steps, smoothed, color=colors[stage_name], linewidth=2.5)
            
            ax_single.set_title(f'{stage_labels.get(stage_name, stage_name)}', fontsize=14, fontweight='bold')
            ax_single.set_xlabel('Training Step', fontsize=12)
            ax_single.set_ylabel('Loss', fontsize=12)
            ax_single.grid(True, alpha=0.3)
            
            # Stats box
            stats_text = f'Steps: {len(steps)}\nFinal: {losses[-1]:.4f}\nMin: {min(losses):.4f}'
            ax_single.text(0.98, 0.98, stats_text, transform=ax_single.transAxes,
                          fontsize=10, verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            single_path = output_dir / f'loss_{stage_name}.png'
            fig_single.savefig(single_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_single)
            print(f"Saved {stage_name} plot to: {single_path}")
    
    plt.close(fig)
    
    return stages


def print_summary(stages: dict):
    """Print a summary of training statistics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for stage_name in ['pretrain', 'sft', 'dpo', 'verifier']:
        if stage_name not in stages or len(stages[stage_name]["steps"]) == 0:
            print(f"\n{stage_name.upper()}: No data")
            continue
        
        data = stages[stage_name]
        steps = data["steps"]
        losses = data["losses"]
        
        print(f"\n{stage_name.upper()}:")
        print(f"  Total logged steps: {len(steps)}")
        print(f"  Step range: {min(steps)} - {max(steps)}")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Min loss: {min(losses):.4f} (step {steps[losses.index(min(losses))]})")
        print(f"  Max loss: {max(losses):.4f}")
        if losses[0] > 0:
            print(f"  Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        else:
            print(f"  Improvement: N/A (initial loss was 0)")


def main():
    parser = argparse.ArgumentParser(description='Parse training logs and generate loss curves')
    parser.add_argument('--log-file', type=str, 
                       default='logs/modern-llm-h100_2782103.err',
                       help='Path to the SLURM error log file')
    parser.add_argument('--output', type=str,
                       default='report/figures/training_loss.png',
                       help='Output path for the loss plot')
    parser.add_argument('--smoothing', type=int, default=10,
                       help='Smoothing window size')
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    log_path = project_root / args.log_file
    output_path = project_root / args.output
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing log file: {log_path}")
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    # Parse and plot
    stages = parse_log_file(str(log_path))
    
    if not stages:
        print("No training data found in log file!")
        return
    
    print_summary(stages)
    plot_losses(stages, str(output_path), smoothing=args.smoothing)
    
    print(f"\nDone! Main plot saved to: {output_path}")


if __name__ == '__main__':
    main()

