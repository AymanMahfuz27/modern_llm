#!/bin/bash
# Modern LLM: One-button frontier training pipeline
#
# Usage:
#   bash speedrun.sh local        # Full pipeline on RTX 3060 (small config)
#   bash speedrun.sh local-smoke  # Quick smoke test locally
#   bash speedrun.sh gpu          # Full pipeline on high-end GPU (large config)
#   bash speedrun.sh gpu-smoke    # Short smoke test on GPU
#   bash speedrun.sh help         # Show this help
#
# Pipeline Stages:
#   1. Pretrain  - Language model pretraining on text corpora
#   2. SFT       - Supervised fine-tuning on instruction data
#   3. DPO       - Direct preference optimization for alignment
#   4. Verifier  - Train answer correctness model
#   5. Evaluate  - Compare all stages on benchmarks
#   6. Report    - Generate markdown report with metrics

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

print_header() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Modern LLM Speedrun Pipeline${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
}

print_help() {
    cat << EOF
Modern LLM Speedrun Pipeline
=============================

One-button training for the complete frontier LLM pipeline:
Pretrain -> SFT -> DPO -> Verifier -> Evaluate -> Report

USAGE:
    bash speedrun.sh <mode> [options]

MODES:
    local         Full pipeline on local GPU (RTX 3060, ~12GB VRAM)
    local-smoke   Quick smoke test locally (~5 minutes)
    gpu           Full pipeline on high-end GPU (A100/H100)
    gpu-smoke     Quick smoke test on GPU (~10 minutes)
    help          Show this help message

OPTIONS:
    --skip-pretrain    Use existing pretrain checkpoint
    --skip-sft         Skip SFT stage
    --skip-dpo         Skip DPO stage
    --skip-verifier    Skip verifier training
    --checkpoint PATH  Path to existing pretrain checkpoint

EXAMPLES:
    # Quick local test
    bash speedrun.sh local-smoke

    # Full local training (will take hours)
    bash speedrun.sh local

    # Use existing pretrain checkpoint
    bash speedrun.sh local --skip-pretrain --checkpoint checkpoints/pretrain_best.pt

OUTPUT:
    experiments/runs/<run_name>/
    ├── *-pretrain_final.pt    # Base model checkpoint
    ├── *-sft_final.pt         # Instruction-tuned model
    ├── *-dpo_final.pt         # Preference-aligned model
    ├── *-verifier_final.pt    # Answer correctness model
    ├── pipeline_state.json    # Pipeline progress tracking
    └── report.md              # Evaluation report

EOF
}

setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"
    
    cd "${PROJECT_DIR}"
    
    # Create venv if needed
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate venv
    source .venv/bin/activate
    
    # Install dependencies
    echo "Checking dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    
    # Set PYTHONPATH
    export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
    export TOKENIZERS_PARALLELISM=false
    
    echo -e "${GREEN}Environment ready.${NC}"
}

run_pipeline() {
    local mode=$1
    shift
    local extra_args="$@"
    
    echo -e "${YELLOW}Running pipeline with mode: ${mode}${NC}"
    echo "Extra args: ${extra_args}"
    echo ""
    
    # Map mode to config
    local config=""
    case $mode in
        local)
            config="local"
            ;;
        local-smoke)
            config="local-smoke"
            ;;
        gpu)
            config="gpu"
            ;;
        gpu-smoke)
            config="gpu-smoke"
            ;;
        *)
            echo -e "${RED}Unknown mode: ${mode}${NC}"
            print_help
            exit 1
            ;;
    esac
    
    # Run the Python pipeline
    python scripts/speedrun_pipeline.py --config "${config}" ${extra_args}
}

# Main
main() {
    if [ $# -lt 1 ]; then
        print_header
        print_help
        exit 1
    fi
    
    local mode=$1
    shift
    
    case $mode in
        help|--help|-h)
            print_header
            print_help
            exit 0
            ;;
        local|local-smoke|gpu|gpu-smoke)
            print_header
            setup_environment
            run_pipeline "$mode" "$@"
            ;;
        *)
            echo -e "${RED}Unknown mode: ${mode}${NC}"
            echo "Run 'bash speedrun.sh help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
