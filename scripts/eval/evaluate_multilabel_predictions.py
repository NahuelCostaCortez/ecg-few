#!/usr/bin/env python3
"""Evaluate multi-label JSON predictions against ECG visual QA labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_few.evaluation.metrics import LABEL_NAMES, multilabel_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ECG multi-label predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file.")
    parser.add_argument("--output", default="", help="Optional metrics JSON output path.")
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Skip records with API/parsing errors instead of failing.",
    )
    return parser.parse_args()


def _load_rows(path: Path, allow_errors: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        if record.get("error"):
            if allow_errors:
                continue
            raise ValueError(f"{path}:{line_no} has error: {record['error']}")
        prediction = record.get("prediction")
        if not isinstance(prediction, dict):
            if allow_errors:
                continue
            raise ValueError(f"{path}:{line_no} has no parsed prediction.")
        rows.append(
            {
                "id": record["id"],
                "expected": record["expected_answer"],
                "predicted": prediction,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.predictions)
    rows = _load_rows(prediction_path, allow_errors=args.allow_errors)
    metrics = multilabel_metrics(rows, label_names=LABEL_NAMES)

    payload = {
        "predictions": prediction_path.as_posix(),
        "n_evaluated": len(rows),
        "metrics": metrics,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
