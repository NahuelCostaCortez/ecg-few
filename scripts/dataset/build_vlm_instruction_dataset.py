#!/usr/bin/env python3
"""
Build visual QA datasets from synthetic ECG image labels.

The outputs are intentionally strict JSON tasks:
- multi-label QA: one question per image for all findings
- binary QA: one question per image per finding
- chat-message JSONL mirrors for VLM instruction tuning
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

from ecg_vlm.prompts import (
    SYNTHETIC_RATIONALE_TEXT,
    TASK_BINARY,
    TASK_MULTILABEL,
    load_markdown_prompt,
)
from ecg_vlm.simulator.constants import LABEL_NAMES

LABEL_COLUMNS = {
    "RBBB": "label_rbbb",
    "ST_ELEVATION": "label_st_elevation",
    "T_WAVE_INVERSION": "label_t_wave_inversion",
}

SPLITS = ("train", "val", "test", "transfer")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BINARY_PROMPT_FILES = {
    "RBBB": "rbbb.md",
    "ST_ELEVATION": "st_elevation.md",
    "T_WAVE_INVERSION": "t_wave_inversion.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ECG visual QA JSONL files.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data",
        help="Root dataset directory produced by build_ecg_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for visual QA datasets (default: <dataset_root>/vlm).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Reserved for reproducible dataset variants.",
    )
    parser.add_argument(
        "--multilabel-prompt-file",
        type=str,
        default=(PROJECT_ROOT / "prompts" / "multilabel" / "label_only.md").as_posix(),
        help="Markdown prompt file for multi-label records.",
    )
    parser.add_argument(
        "--binary-prompt-dir",
        type=str,
        default=(PROJECT_ROOT / "prompts" / "binary").as_posix(),
        help="Directory containing rbbb.md, st_elevation.md, and t_wave_inversion.md.",
    )
    parser.add_argument(
        "--absolute-image-paths",
        action="store_true",
        help="Write absolute image paths instead of paths relative to dataset root.",
    )
    parser.add_argument(
        "--include-synthetic-rationales",
        action="store_true",
        help=(
            "Add short label-derived rationales to target JSON. Keep disabled for primary "
            "evaluation because these are not clinician-reviewed visual explanations."
        ),
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def _image_path_value(dataset_root: Path, row_image_path: str, absolute: bool) -> str:
    rel_path = row_image_path.replace("\\", "/")
    if not absolute:
        return rel_path
    return (dataset_root / rel_path).resolve().as_posix()


def _labels_from_row(row: dict[str, str]) -> dict[str, bool]:
    return {label_name: bool(int(row[LABEL_COLUMNS[label_name]])) for label_name in LABEL_NAMES}


def _base_metadata(row: dict[str, str]) -> dict[str, object]:
    return {
        "split": row["split"],
        "source_family": row["source_family"],
        "subtype": row["subtype"] or None,
        "seed": int(row["seed"]),
        "attempt": int(row["attempt"]),
        "sample_index": int(row["sample_index"]),
        "labels": _labels_from_row(row),
    }


def _synthetic_rationale(labels: dict[str, bool]) -> str:
    present = [label_name for label_name in LABEL_NAMES if labels[label_name]]
    if not present:
        return "The simulator labels do not mark any of the target findings as present."
    return " ".join(SYNTHETIC_RATIONALE_TEXT[label_name] for label_name in present)


def _multilabel_answer(
    row: dict[str, str],
    include_synthetic_rationales: bool,
) -> dict[str, object]:
    labels = _labels_from_row(row)
    answer: dict[str, object] = dict(labels)
    if include_synthetic_rationales:
        answer["rationale"] = _synthetic_rationale(labels)
    return answer


def _binary_answer(
    row: dict[str, str],
    finding: str,
    include_synthetic_rationales: bool,
) -> dict[str, object]:
    labels = _labels_from_row(row)
    answer: dict[str, object] = {"finding": finding, "present": labels[finding]}
    if include_synthetic_rationales:
        if labels[finding]:
            answer["rationale"] = SYNTHETIC_RATIONALE_TEXT[finding]
        else:
            answer["rationale"] = f"The simulator label does not mark {finding} as present."
    return answer


def _message_record(eval_record: dict[str, object]) -> dict[str, object]:
    return {
        "id": eval_record["id"],
        "image_path": eval_record["image_path"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": eval_record["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            eval_record["expected_answer"],
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                    }
                ],
            },
        ],
        "metadata": eval_record["metadata"],
    }


def build_multilabel_records(
    rows: Iterable[dict[str, str]],
    dataset_root: Path,
    split_name: str,
    prompt: str,
    absolute_paths: bool,
    include_synthetic_rationales: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        records.append(
            {
                "id": f"{split_name}_multilabel_{idx:06d}",
                "task_type": TASK_MULTILABEL,
                "image_path": _image_path_value(dataset_root, row["image_path"], absolute_paths),
                "prompt": prompt,
                "expected_answer": _multilabel_answer(row, include_synthetic_rationales),
                "metadata": _base_metadata(row),
            }
        )
    return records


def build_binary_records(
    rows: Iterable[dict[str, str]],
    dataset_root: Path,
    split_name: str,
    prompts_by_finding: dict[str, str],
    absolute_paths: bool,
    include_synthetic_rationales: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        for finding in LABEL_NAMES:
            records.append(
                {
                    "id": f"{split_name}_binary_{finding.lower()}_{idx:06d}",
                    "task_type": TASK_BINARY,
                    "finding": finding,
                    "image_path": _image_path_value(
                        dataset_root,
                        row["image_path"],
                        absolute_paths,
                    ),
                    "prompt": prompts_by_finding[finding],
                    "expected_answer": _binary_answer(
                        row,
                        finding=finding,
                        include_synthetic_rationales=include_synthetic_rationales,
                    ),
                    "metadata": _base_metadata(row),
                }
            )
    return records


def _validate_multilabel_record(record: dict[str, object]) -> None:
    answer = record["expected_answer"]
    if not isinstance(answer, dict):
        raise TypeError("expected_answer must be a dict.")
    for label_name in LABEL_NAMES:
        if type(answer.get(label_name)) is not bool:
            raise ValueError(f"{record['id']} has invalid {label_name} value.")


def _validate_binary_record(record: dict[str, object]) -> None:
    answer = record["expected_answer"]
    if not isinstance(answer, dict):
        raise TypeError("expected_answer must be a dict.")
    if answer.get("finding") not in LABEL_NAMES:
        raise ValueError(f"{record['id']} has invalid finding.")
    if type(answer.get("present")) is not bool:
        raise ValueError(f"{record['id']} has invalid present value.")


def _schema_payload(include_synthetic_rationales: bool) -> dict[str, object]:
    return {
        "label_names": LABEL_NAMES,
        "tasks": {
            TASK_MULTILABEL: {
                "prompt": "Detect all target findings in one image.",
                "answer_schema": {
                    "RBBB": "boolean",
                    "ST_ELEVATION": "boolean",
                    "T_WAVE_INVERSION": "boolean",
                    "rationale": "optional string",
                },
            },
            TASK_BINARY: {
                "prompt": "Detect one target finding in one image.",
                "answer_schema": {
                    "finding": LABEL_NAMES,
                    "present": "boolean",
                    "rationale": "optional string",
                },
            },
        },
        "include_synthetic_rationales": include_synthetic_rationales,
        "note": (
            "Synthetic rationales are label-derived and are not clinician-reviewed visual "
            "explanations. Use strict boolean fields as the source of truth."
        ),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    labels_dir = dataset_root / "labels"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (dataset_root / "vlm")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, str]]] = {}
    for split_name in SPLITS:
        csv_path = labels_dir / f"{split_name}_labels.csv"
        if not csv_path.exists():
            if split_name == "transfer":
                split_rows[split_name] = []
                continue
            raise FileNotFoundError(f"Missing expected label file: {csv_path}")
        split_rows[split_name] = read_rows(csv_path)

    counts: dict[str, dict[str, int]] = {}
    multilabel_prompt = load_markdown_prompt(args.multilabel_prompt_file)
    binary_prompt_dir = Path(args.binary_prompt_dir)
    binary_prompts = {
        finding: load_markdown_prompt(binary_prompt_dir / BINARY_PROMPT_FILES[finding])
        for finding in LABEL_NAMES
    }

    for split_name in SPLITS:
        rows = split_rows[split_name]
        multilabel_records = build_multilabel_records(
            rows,
            dataset_root=dataset_root,
            split_name=split_name,
            prompt=multilabel_prompt,
            absolute_paths=args.absolute_image_paths,
            include_synthetic_rationales=args.include_synthetic_rationales,
        )
        binary_records = build_binary_records(
            rows,
            dataset_root=dataset_root,
            split_name=split_name,
            prompts_by_finding=binary_prompts,
            absolute_paths=args.absolute_image_paths,
            include_synthetic_rationales=args.include_synthetic_rationales,
        )

        for record in multilabel_records:
            _validate_multilabel_record(record)
        for record in binary_records:
            _validate_binary_record(record)

        multilabel_messages = [_message_record(record) for record in multilabel_records]
        binary_messages = [_message_record(record) for record in binary_records]

        counts[split_name] = {
            "input_rows": len(rows),
            "multilabel_records": write_jsonl(
                output_dir / "eval" / "multilabel" / f"{split_name}.jsonl",
                multilabel_records,
            ),
            "binary_records": write_jsonl(
                output_dir / "eval" / "binary" / f"{split_name}.jsonl",
                binary_records,
            ),
            "multilabel_messages": write_jsonl(
                output_dir / "messages" / "multilabel" / f"{split_name}.jsonl",
                multilabel_messages,
            ),
            "binary_messages": write_jsonl(
                output_dir / "messages" / "binary" / f"{split_name}.jsonl",
                binary_messages,
            ),
        }

    # Backward-compatible convenience files for the primary fine-tuning format.
    for split_name in SPLITS:
        src = output_dir / "messages" / "multilabel" / f"{split_name}.jsonl"
        dst = output_dir / f"{split_name}_messages.jsonl"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    schema_path = output_dir / "qa_schema.json"
    schema_path.write_text(
        json.dumps(_schema_payload(args.include_synthetic_rationales), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = output_dir / "qa_dataset_summary.json"
    summary_path.write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")

    for split_name, split_counts in counts.items():
        print(
            f"[OK] {split_name}: {split_counts['multilabel_records']} multilabel, "
            f"{split_counts['binary_records']} binary records"
        )
    print(f"[OK] Wrote schema:  {schema_path}")
    print(f"[OK] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
