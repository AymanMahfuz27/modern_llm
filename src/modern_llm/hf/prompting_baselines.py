"""Prompting baselines (zero/few-shot per Brown et al., 2020)."""

from __future__ import annotations


def run_prompting_baselines() -> None:
    """Execute zero-/few-shot prompting evaluations for configured tasks.

    Pre:
        - Tasks defined in config files.
    Post:
        - Logs metrics comparable to finetuning runs.
    Complexity:
        - Dominated by model inference cost across evaluation sets.
    """

    raise NotImplementedError("Prompting baselines will be added in the evaluation phase.")

