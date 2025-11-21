"""Lightweight metrics utilities with explicit math definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


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
    elif task_name.lower() in {"samsum", "summarization"}:
        try:
            from evaluate import load as load_metric
        except ImportError as exc:
            raise ImportError("Install `evaluate` for ROUGE: pip install evaluate rouge-score") from exc
        rouge = load_metric("rouge")
        result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        metrics["rouge1"] = result["rouge1"]
        metrics["rouge2"] = result["rouge2"]
        metrics["rougeL"] = result["rougeL"]
    else:
        raise NotImplementedError(f"Metrics for task '{task_name}' are not defined yet.")

    return EvaluationResult(task_name=task_name, metrics=metrics)


def compute_f1(
    predictions: Sequence[int],
    references: Sequence[int],
    num_classes: int = 2,
) -> Dict[str, float]:
    """Compute macro-averaged F1 score for classification.

    Math (per-class):
        precision_c = TP_c / (TP_c + FP_c)
        recall_c = TP_c / (TP_c + FN_c)
        F1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c)
        macro_F1 = (1/C) Σ F1_c

    Pre:
        - predictions and references are integer labels in [0, num_classes).
    Post:
        - returns dict with macro_f1 and per-class scores.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for pred, ref in zip(predictions, references):
        if pred == ref:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[ref] += 1

    f1_scores: List[float] = []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / num_classes if num_classes else 0.0
    return {"macro_f1": macro_f1, "per_class_f1": f1_scores}

