"""Lightweight metrics utilities with explicit math definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence


@dataclass(slots=True)
class EvaluationResult:
    """Container for metric scores keyed by metric name."""
    task_name: str
    metrics: Dict[str, float] = field(default_factory=dict)


def compute_metrics(
    task_name: str,
    predictions: Sequence,
    references: Sequence,
) -> EvaluationResult:
    """Compute accuracy- or exact-match-style metrics.

    Math:
        accuracy = (1/N) Σ 1[p_i = y_i]
        exact_match = (1/N) Σ 1[normalize(p_i) = normalize(y_i)]

    These correspond to the standard definitions used in GLUE and GSM8K papers.

    Pre:
        - len(predictions) == len(references).
    Post:
        - returns EvaluationResult with at least one metric entry.
    Complexity:
        - O(N) comparisons for N samples.
    """

    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must be the same length ("
            f"{len(predictions)} vs {len(references)})"
        )
    n = len(predictions)
    metrics: Dict[str, float] = {}

    if task_name.lower() in {"sst-2", "classification"}:
        correct = sum(int(p == r) for p, r in zip(predictions, references))
        metrics["accuracy"] = correct / n if n else 0.0
    elif task_name.lower() in {"gsm8k", "math"}:
        exact = sum(int(str(p).strip() == str(r).strip()) for p, r in zip(predictions, references))
        metrics["exact_match"] = exact / n if n else 0.0
    else:
        raise NotImplementedError(f"Metrics for task '{task_name}' are not defined yet.")

    return EvaluationResult(task_name=task_name, metrics=metrics)

