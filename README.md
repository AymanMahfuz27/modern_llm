# Modern LLM

A portfolio-grade NLP final project that implements a modern decoder-only language model from scratch, compares it against Hugging Face baselines, and reproduces a frontier alignment pipeline (SFT → DPO → verifier).

## Repository Layout

- `src/modern_llm/`: Core Python package (configs, data loaders, models, training, evaluation).
- `scripts/`: Thin CLI entrypoints that import functions from `modern_llm`.
- `configs/`: JSON/YAML experiment configs (add new files per run).
- `experiments/`: Summaries and logs for each run (`runs/` is gitignored).
- `data/`: Local copies of datasets (ignored in Git, see `data/README.md`).
- `notebooks/`: Optional exploratory notebooks and demos.
- `report/`: Course deliverables (proposal, progress report, ACL-format paper).
- `tests/`: Pytest-based regression/unit tests.

## Getting Started

```bash
python -m venv .venv
.venv\\Scripts\\activate  # PowerShell
pip install -r requirements.txt
pytest  # run fast validation checks
```

## Next Steps

- Implement the decoder block forward pass plus attention (Phase 1).
- Build huggingface finetuning pipelines (Phase 2).
- Layer on SFT/DPO/verifier alignment and advanced features (Phases 3–4).

