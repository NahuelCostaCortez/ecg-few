"""Metrics for QRS finding and derived Brugada evaluation."""

from __future__ import annotations

from collections.abc import Iterable

from ecg_few.findings import LABEL_NAMES


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def binary_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    return {
        "accuracy": _safe_divide(tp + tn, tp + tn + fp + fn),
        "sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "precision": _safe_divide(tp, tp + fp),
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
    }


def brugada_metrics(
    y_true: Iterable[int | bool],
    y_pred: Iterable[int | bool],
) -> dict[str, object]:
    """Return patient-level metrics for the derived Brugada decision."""
    counts = _empty_counts()
    true_values = [int(bool(value)) for value in y_true]
    pred_values = [int(bool(value)) for value in y_pred]
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length.")

    for expected, predicted in zip(true_values, pred_values, strict=True):
        if expected == 1 and predicted == 1:
            counts["tp"] += 1
        elif expected == 0 and predicted == 0:
            counts["tn"] += 1
        elif expected == 0 and predicted == 1:
            counts["fp"] += 1
        else:
            counts["fn"] += 1

    metrics = binary_metrics(counts)
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    total = len(true_values)
    return {
        **metrics,
        "balanced_accuracy": balanced_accuracy,
        "counts": counts,
        "prevalence": _safe_divide(sum(true_values), total),
        "predicted_prevalence": _safe_divide(sum(pred_values), total),
    }


def multilabel_metrics(
    rows: Iterable[dict[str, object]],
    label_names: Iterable[str] = LABEL_NAMES,
) -> dict[str, object]:
    labels = list(label_names)
    counts_by_label = {label: _empty_counts() for label in labels}
    exact_matches = 0
    total = 0

    for row in rows:
        expected = row["expected"]
        predicted = row["predicted"]
        if not isinstance(expected, dict) or not isinstance(predicted, dict):
            raise TypeError("Each row must provide dict values for expected and predicted.")

        total += 1
        row_match = True
        for label in labels:
            exp = bool(expected[label])
            pred = bool(predicted[label])
            if exp and pred:
                counts_by_label[label]["tp"] += 1
            elif not exp and not pred:
                counts_by_label[label]["tn"] += 1
            elif not exp and pred:
                counts_by_label[label]["fp"] += 1
            else:
                counts_by_label[label]["fn"] += 1
            row_match = row_match and exp == pred

        if row_match:
            exact_matches += 1

    per_label = {
        label: {"counts": counts_by_label[label], **binary_metrics(counts_by_label[label])}
        for label in labels
    }

    macro: dict[str, float | None] = {}
    for metric_name in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
        values = [
            per_label[label][metric_name]
            for label in labels
            if per_label[label][metric_name] is not None
        ]
        macro[metric_name] = sum(values) / len(values) if values else None

    return {
        "n": total,
        "exact_match_accuracy": exact_matches / total if total else None,
        "per_label": per_label,
        "macro": macro,
    }
