# Data Directory

This directory stores downloaded or preprocessed datasets that are too large for source control.

- Place raw Hugging Face `datasets` cache references or exported parquet/JSON files here if you need offline access.
- Do **not** commit dataset artifacts. The `.gitignore` entry keeps everything in `data/` out of Git except this README.
- Keep preprocessing scripts inside `src/modern_llm/data` so experiments remain reproducible from the original public sources.

