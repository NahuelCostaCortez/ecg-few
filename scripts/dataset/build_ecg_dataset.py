#!/usr/bin/env python3
"""
Build a synthetic ECG image dataset for multi-label finding detection.

Workflow:
1. Generate verification examples (images only).
2. Generate many beats per source family.
3. Create a stratified train/val/test split by source family.
4. Save beats as images and write binary-label manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecg_few.simulator import LABEL_NAMES, SOURCE_FAMILIES, generate_beat
from ecg_few.simulator.plotting import plot_beat


LABEL_COLUMNS = {
    "RBBB": "label_rbbb",
    "ST_ELEVATION": "label_st_elevation",
    "T_WAVE_INVERSION": "label_t_wave_inversion",
}


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _reset_managed_output_dirs(outdir: Path) -> None:
    for name in ("train", "val", "test", "transfer", "labels", "examples", "vlm"):
        path = outdir / name
        if path.exists():
            shutil.rmtree(path)


def _subtype_for_source_family(source_family: str) -> str | None:
    if source_family == "BRUGADA":
        return "coved"
    return None


def _feature_text_from_row(row: Dict[str, object]) -> str:
    present = [
        label_name
        for label_name in LABEL_NAMES
        if int(row[LABEL_COLUMNS[label_name]]) == 1
    ]
    return "NORMAL" if not present else "+".join(present)


def generate_verification_examples(
    fs: int,
    seed: int,
    outdir: Path,
    dpi: int,
) -> Dict[str, Dict[str, object]]:
    """Generate one reproducible example beat per source family and save previews."""
    outdir.mkdir(parents=True, exist_ok=True)
    examples_dir = outdir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    example_info: Dict[str, Dict[str, object]] = {}
    for i, source_family in enumerate(SOURCE_FAMILIES):
        example_seed = seed + i
        subtype = _subtype_for_source_family(source_family)
        beat, metadata = generate_beat(
            class_name=source_family,
            seed=example_seed,
            subtype=subtype,
            fs=fs,
        )
        family_example_dir = examples_dir / source_family
        family_example_dir.mkdir(parents=True, exist_ok=True)
        plot_beat(
            beat,
            fs=fs,
            save_path=str(family_example_dir / "example.png"),
            dpi=dpi,
        )

        ax = axes[i]
        plot_beat(beat, fs=fs, ax=ax)
        label_text = _feature_text_from_row(
            {
                LABEL_COLUMNS[label_name]: metadata["feature_labels"][label_name]
                for label_name in LABEL_NAMES
            }
        )
        ax.set_title(f"{source_family}: {label_text}")
        ax.axhline(0.0, color="gray", linewidth=0.9, linestyle="--", alpha=0.6)

        example_info[source_family] = {
            "seed": example_seed,
            "subtype": metadata["subtype"],
            "attempt": metadata["attempt"],
            "feature_labels": metadata["feature_labels"],
            "validation_metrics": metadata["validation_metrics"],
        }

    axes[-1].axis("off")
    preview_path = examples_dir / "verification_grid_by_family.png"
    fig.savefig(preview_path, dpi=dpi)
    fig.savefig(examples_dir / "verification_grid.png", dpi=dpi)
    plt.close(fig)

    print(f"[OK] Verification family grid: {preview_path}")
    print(f"[OK] Verification examples under: {examples_dir}")
    for source_family in SOURCE_FAMILIES:
        print(
            f"  - {source_family}: accepted on attempt {example_info[source_family]['attempt']} "
            f"(seed={example_info[source_family]['seed']})"
        )
    return example_info


def generate_dataset(
    samples_per_family: int,
    fs: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, List[Dict[str, object]]]:
    """Generate beats, source-family ids, and sample-level metadata."""
    rng = np.random.default_rng(seed)
    family_to_id = {name: idx for idx, name in enumerate(SOURCE_FAMILIES)}

    beats: List[np.ndarray] = []
    source_ids: List[int] = []
    metadata_rows: List[Dict[str, object]] = []

    for source_family in SOURCE_FAMILIES:
        family_id = family_to_id[source_family]
        subtype = _subtype_for_source_family(source_family)
        for _ in range(samples_per_family):
            sample_seed = int(rng.integers(0, 2_147_483_647))
            beat, meta = generate_beat(
                class_name=source_family,
                seed=sample_seed,
                subtype=subtype,
                fs=fs,
            )
            feature_labels = meta["feature_labels"]
            beats.append(beat.astype(np.float32))
            source_ids.append(family_id)
            metadata_rows.append(
                {
                    "sample_index": len(beats) - 1,
                    "source_family": source_family,
                    "seed": sample_seed,
                    "attempt": meta["attempt"],
                    "subtype": meta["subtype"] or "",
                    LABEL_COLUMNS["RBBB"]: int(feature_labels["RBBB"]),
                    LABEL_COLUMNS["ST_ELEVATION"]: int(feature_labels["ST_ELEVATION"]),
                    LABEL_COLUMNS["T_WAVE_INVERSION"]: int(feature_labels["T_WAVE_INVERSION"]),
                }
            )
        print(f"[OK] Generated {samples_per_family} beats for {source_family}")

    return beats, np.asarray(source_ids, dtype=np.int64), metadata_rows


def stratified_split_indices(
    source_ids: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split by source-family ids."""
    rng = np.random.default_rng(seed)
    classes = np.unique(source_ids)
    train_idx_parts: List[np.ndarray] = []
    val_idx_parts: List[np.ndarray] = []
    test_idx_parts: List[np.ndarray] = []

    for source_id in classes:
        idx = np.where(source_ids == source_id)[0]
        idx = idx[rng.permutation(idx.size)]

        if idx.size < 3:
            raise ValueError("Need at least 3 samples per source family for train/val/test split.")

        n_test = int(round(idx.size * test_ratio))
        n_test = max(1, min(idx.size - 2, n_test))
        remaining = idx.size - n_test

        n_val = int(round(idx.size * val_ratio))
        n_val = max(1, min(remaining - 1, n_val))

        test_idx_parts.append(idx[:n_test])
        val_idx_parts.append(idx[n_test : n_test + n_val])
        train_idx_parts.append(idx[n_test + n_val :])

    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    test_idx = np.concatenate(test_idx_parts)

    train_idx = train_idx[rng.permutation(train_idx.size)]
    val_idx = val_idx[rng.permutation(val_idx.size)]
    test_idx = test_idx[rng.permutation(test_idx.size)]
    return train_idx, val_idx, test_idx


def tiny_poc_split_indices(
    source_ids: np.ndarray,
    component_dev_count: int,
    component_test_count: int,
    transfer_count: int,
    transfer_source_family: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the first proof-of-concept split with a held-out transfer family."""
    rng = np.random.default_rng(seed)
    family_to_id = {name: idx for idx, name in enumerate(SOURCE_FAMILIES)}
    transfer_source_family = transfer_source_family.upper()
    if transfer_source_family not in family_to_id:
        raise ValueError(
            f"Unknown transfer source family '{transfer_source_family}'. "
            f"Use one of {SOURCE_FAMILIES}."
        )

    train_idx_parts: List[np.ndarray] = []
    test_idx_parts: List[np.ndarray] = []
    transfer_idx_parts: List[np.ndarray] = []

    for source_family, source_id in family_to_id.items():
        idx = np.where(source_ids == source_id)[0]
        idx = idx[rng.permutation(idx.size)]

        if source_family == transfer_source_family:
            if idx.size < transfer_count:
                raise ValueError(
                    f"Need at least {transfer_count} samples for {transfer_source_family}; "
                    f"found {idx.size}."
                )
            transfer_idx_parts.append(idx[:transfer_count])
            continue

        needed = component_dev_count + component_test_count
        if idx.size < needed:
            raise ValueError(
                f"Need at least {needed} samples for {source_family}; found {idx.size}."
            )
        train_idx_parts.append(idx[:component_dev_count])
        test_idx_parts.append(idx[component_dev_count:needed])

    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    transfer_idx = np.concatenate(transfer_idx_parts)
    val_idx = np.asarray([], dtype=np.int64)

    train_idx = train_idx[rng.permutation(train_idx.size)]
    test_idx = test_idx[rng.permutation(test_idx.size)]
    transfer_idx = transfer_idx[rng.permutation(transfer_idx.size)]
    return train_idx, val_idx, test_idx, transfer_idx


def save_image_dataset(
    outdir: Path,
    beats: List[np.ndarray],
    source_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fs: int,
    metadata_rows: List[Dict[str, object]],
    dpi: int,
    seed: int,
    transfer_idx: np.ndarray | None = None,
) -> None:
    """Save image dataset layout and CSV/JSON manifests."""
    outdir.mkdir(parents=True, exist_ok=True)
    train_root = outdir / "train"
    val_root = outdir / "val"
    test_root = outdir / "test"
    transfer_root = outdir / "transfer"
    labels_root = outdir / "labels"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)
    if transfer_idx is not None:
        transfer_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    for source_family in SOURCE_FAMILIES:
        (train_root / source_family).mkdir(parents=True, exist_ok=True)
        (val_root / source_family).mkdir(parents=True, exist_ok=True)
        (test_root / source_family).mkdir(parents=True, exist_ok=True)
        if transfer_idx is not None:
            (transfer_root / source_family).mkdir(parents=True, exist_ok=True)

    label_schema = {
        "labels": LABEL_NAMES,
        "label_columns": [LABEL_COLUMNS[label_name] for label_name in LABEL_NAMES],
        "normal_token": "NORMAL",
        "positive_text_separator": ", ",
    }
    _save_json(labels_root / "label_schema.json", label_schema)

    headers = [
        "image_path",
        "split",
        "source_family",
        "seed",
        "attempt",
        "subtype",
        "sample_index",
        LABEL_COLUMNS["RBBB"],
        LABEL_COLUMNS["ST_ELEVATION"],
        LABEL_COLUMNS["T_WAVE_INVERSION"],
    ]
    all_rows: List[Dict[str, object]] = []
    split_counters: Dict[Tuple[str, str], int] = {}

    def _write_split(split_name: str, indices: np.ndarray) -> None:
        split_root_map = {
            "train": train_root,
            "val": val_root,
            "test": test_root,
            "transfer": transfer_root,
        }
        split_root = split_root_map[split_name]
        for idx in indices:
            row_meta = metadata_rows[int(idx)]
            source_family = str(row_meta["source_family"])
            key = (split_name, source_family)
            split_counters[key] = split_counters.get(key, 0) + 1
            file_name = f"beat_{split_counters[key]:06d}.png"
            img_path = split_root / source_family / file_name

            beat = beats[int(idx)]
            plot_beat(beat, fs=fs, save_path=str(img_path), dpi=dpi)

            all_rows.append(
                {
                    "image_path": img_path.relative_to(outdir).as_posix(),
                    "split": split_name,
                    "source_family": source_family,
                    "seed": int(row_meta["seed"]),
                    "attempt": int(row_meta["attempt"]),
                    "subtype": str(row_meta["subtype"]),
                    "sample_index": int(row_meta["sample_index"]),
                    LABEL_COLUMNS["RBBB"]: int(row_meta[LABEL_COLUMNS["RBBB"]]),
                    LABEL_COLUMNS["ST_ELEVATION"]: int(row_meta[LABEL_COLUMNS["ST_ELEVATION"]]),
                    LABEL_COLUMNS["T_WAVE_INVERSION"]: int(row_meta[LABEL_COLUMNS["T_WAVE_INVERSION"]]),
                }
            )

    _write_split("train", train_idx)
    _write_split("val", val_idx)
    _write_split("test", test_idx)
    if transfer_idx is not None:
        _write_split("transfer", transfer_idx)

    def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    train_rows = [row for row in all_rows if row["split"] == "train"]
    val_rows = [row for row in all_rows if row["split"] == "val"]
    test_rows = [row for row in all_rows if row["split"] == "test"]
    transfer_rows = [row for row in all_rows if row["split"] == "transfer"]
    _save_csv(labels_root / "train_labels.csv", train_rows)
    _save_csv(labels_root / "val_labels.csv", val_rows)
    _save_csv(labels_root / "test_labels.csv", test_rows)
    if transfer_idx is not None:
        _save_csv(labels_root / "transfer_labels.csv", transfer_rows)
    _save_csv(labels_root / "all_labels.csv", all_rows)

    def _source_family_counts(rows: List[Dict[str, object]]) -> Dict[str, int]:
        counts = Counter(str(row["source_family"]) for row in rows)
        return {source_family: int(counts.get(source_family, 0)) for source_family in SOURCE_FAMILIES}

    def _positive_counts(rows: List[Dict[str, object]]) -> Dict[str, int]:
        return {
            label_name: int(sum(int(row[LABEL_COLUMNS[label_name]]) for row in rows))
            for label_name in LABEL_NAMES
        }

    def _combination_counts(rows: List[Dict[str, object]]) -> Dict[str, int]:
        counts = Counter(_feature_text_from_row(row) for row in rows)
        return {key: int(value) for key, value in sorted(counts.items())}

    brugada_rows = [row for row in all_rows if row["source_family"] == "BRUGADA"]
    summary = {
        "seed": seed,
        "fs_hz": fs,
        "n_samples_total": int(len(beats)),
        "n_samples_train": int(train_idx.size),
        "n_samples_val": int(val_idx.size),
        "n_samples_test": int(test_idx.size),
        "n_samples_transfer": int(transfer_idx.size) if transfer_idx is not None else 0,
        "signal_length_samples": int(beats[0].shape[0]) if beats else 0,
        "source_family_distribution_total": _source_family_counts(all_rows),
        "source_family_distribution_train": _source_family_counts(train_rows),
        "source_family_distribution_val": _source_family_counts(val_rows),
        "source_family_distribution_test": _source_family_counts(test_rows),
        "source_family_distribution_transfer": _source_family_counts(transfer_rows),
        "label_positive_counts_total": _positive_counts(all_rows),
        "label_positive_counts_train": _positive_counts(train_rows),
        "label_positive_counts_val": _positive_counts(val_rows),
        "label_positive_counts_test": _positive_counts(test_rows),
        "label_positive_counts_transfer": _positive_counts(transfer_rows),
        "feature_combination_counts_total": _combination_counts(all_rows),
        "feature_combination_counts_train": _combination_counts(train_rows),
        "feature_combination_counts_val": _combination_counts(val_rows),
        "feature_combination_counts_test": _combination_counts(test_rows),
        "feature_combination_counts_transfer": _combination_counts(transfer_rows),
        "brugada_only_feature_combination_counts": _combination_counts(brugada_rows),
    }
    _save_json(labels_root / "dataset_summary.json", summary)
    _save_json(labels_root / "sanity_counts.json", summary)

    print(f"[OK] Saved train images under: {train_root}")
    print(f"[OK] Saved val images under:   {val_root}")
    print(f"[OK] Saved test images under:  {test_root}")
    if transfer_idx is not None:
        print(f"[OK] Saved transfer images under: {transfer_root}")
    print(f"[OK] Saved train labels: {labels_root / 'train_labels.csv'}")
    print(f"[OK] Saved val labels:   {labels_root / 'val_labels.csv'}")
    print(f"[OK] Saved test labels:  {labels_root / 'test_labels.csv'}")
    if transfer_idx is not None:
        print(f"[OK] Saved transfer labels: {labels_root / 'transfer_labels.csv'}")
    print(f"[OK] Saved all labels:   {labels_root / 'all_labels.csv'}")
    print(f"[OK] Saved label schema: {labels_root / 'label_schema.json'}")
    print(f"[OK] Saved summary:      {labels_root / 'dataset_summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic ECG multi-label dataset.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Output directory for dataset files.",
    )
    parser.add_argument(
        "--samples-per-family",
        "--samples-per-class",
        dest="samples_per_family",
        type=int,
        default=1000,
        help="Number of beats to generate per source family.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of each source family assigned to test split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of each source family assigned to validation split.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=500,
        help="Sampling frequency (Hz).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=112,
        help="Image DPI used when saving beat images.",
    )
    parser.add_argument(
        "--tiny-poc",
        action="store_true",
        help="Create the minimal component-development/component-test/held-out-transfer split.",
    )
    parser.add_argument(
        "--component-dev-count",
        type=int,
        default=20,
        help="Examples per non-transfer source family for the tiny-poc development split.",
    )
    parser.add_argument(
        "--component-test-count",
        type=int,
        default=20,
        help="Examples per non-transfer source family for the tiny-poc component test split.",
    )
    parser.add_argument(
        "--transfer-count",
        type=int,
        default=40,
        help="Examples from the transfer source family for the tiny-poc held-out split.",
    )
    parser.add_argument(
        "--transfer-source-family",
        type=str,
        default="BRUGADA",
        help="Source family to hold out as transfer data in tiny-poc mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_family < 3:
        raise ValueError("--samples-per-family must be >= 3 for train/val/test splitting.")
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be between 0 and 1.")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1.")
    if (args.val_ratio + args.test_ratio) >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1.")
    if args.tiny_poc:
        if args.component_dev_count < 1 or args.component_test_count < 1:
            raise ValueError("--component-dev-count and --component-test-count must be >= 1.")
        if args.transfer_count < 1:
            raise ValueError("--transfer-count must be >= 1.")

    outdir = Path(args.outdir).resolve()
    _reset_managed_output_dirs(outdir)

    print("[1/3] Generating verification examples...")
    family_example_info = generate_verification_examples(
        fs=args.fs,
        seed=args.seed,
        outdir=outdir,
        dpi=args.dpi,
    )
    _save_json(
        outdir / "examples" / "sanity_metadata.json",
        {
            "seed": args.seed,
            "source_family_examples": family_example_info,
            "label_names": LABEL_NAMES,
        },
    )

    print("[2/3] Generating dataset...")
    beats, source_ids, metadata_rows = generate_dataset(
        samples_per_family=args.samples_per_family,
        fs=args.fs,
        seed=args.seed + 1000,
    )

    print("[3/3] Creating split and saving...")
    transfer_idx = None
    if args.tiny_poc:
        train_idx, val_idx, test_idx, transfer_idx = tiny_poc_split_indices(
            source_ids=source_ids,
            component_dev_count=args.component_dev_count,
            component_test_count=args.component_test_count,
            transfer_count=args.transfer_count,
            transfer_source_family=args.transfer_source_family,
            seed=args.seed + 2000,
        )
    else:
        train_idx, val_idx, test_idx = stratified_split_indices(
            source_ids=source_ids,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed + 2000,
        )
    save_image_dataset(
        outdir=outdir,
        beats=beats,
        source_ids=source_ids,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        fs=args.fs,
        metadata_rows=metadata_rows,
        dpi=args.dpi,
        seed=args.seed,
        transfer_idx=transfer_idx,
    )
    print("[DONE] Dataset creation completed.")


if __name__ == "__main__":
    main()
