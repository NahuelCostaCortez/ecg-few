#!/usr/bin/env python3
"""Evaluate binary finding JSON predictions against ECG visual QA labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_vlm.evaluation.metrics import binary_metrics
from ecg_vlm.simulator.constants import LABEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ECG binary finding predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file.")
    parser.add_argument("--output", default="", help="Optional metrics JSON output path.")
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Skip records with API/parsing errors instead of failing.",
    )
    return parser.parse_args()


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def _load_counts(path: Path, allow_errors: bool) -> tuple[int, dict[str, dict[str, int]]]:
    counts_by_label = {label_name: _empty_counts() for label_name in LABEL_NAMES}
    total = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        if record.get("error"):
            if allow_errors:
                continue
            raise ValueError(f"{path}:{line_no} has error: {record['error']}")
        prediction = record.get("prediction")
        expected = record.get("expected_answer")
        if not isinstance(prediction, dict) or not isinstance(expected, dict):
            if allow_errors:
                continue
            raise ValueError(f"{path}:{line_no} has no parsed prediction/expected answer.")

        finding = str(expected["finding"])
        if finding not in counts_by_label:
            raise ValueError(f"{path}:{line_no} has invalid expected finding: {finding}")
        if prediction.get("finding") != finding:
            if allow_errors:
                continue
            raise ValueError(
                f"{path}:{line_no} predicted finding {prediction.get('finding')} "
                f"but expected {finding}."
            )

        exp = bool(expected["present"])
        pred = bool(prediction["present"])
        counts = counts_by_label[finding]
        if exp and pred:
            counts["tp"] += 1
        elif not exp and not pred:
            counts["tn"] += 1
        elif not exp and pred:
            counts["fp"] += 1
        else:
            counts["fn"] += 1
        total += 1
    return total, counts_by_label


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.predictions)
    total, counts_by_label = _load_counts(prediction_path, allow_errors=args.allow_errors)
    per_label = {
        label_name: {"counts": counts_by_label[label_name], **binary_metrics(counts_by_label[label_name])}
        for label_name in LABEL_NAMES
    }
    macro: dict[str, float | None] = {}
    for metric_name in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
        values = [
            per_label[label_name][metric_name]
            for label_name in LABEL_NAMES
            if per_label[label_name][metric_name] is not None
        ]
        macro[metric_name] = sum(values) / len(values) if values else None

    payload = {
        "predictions": prediction_path.as_posix(),
        "n_evaluated": total,
        "metrics": {
            "per_label": per_label,
            "macro": macro,
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
