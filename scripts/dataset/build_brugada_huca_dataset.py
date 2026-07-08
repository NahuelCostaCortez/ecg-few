#!/usr/bin/env python3
"""Build the Brugada-HUCA clinical-reference image dataset for patient-level LOOCV."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_few.loocv import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    LABEL_COLUMNS,
    LABEL_NAMES,
    BrugadaImageRow,
    build_fold_plan,
    write_jsonl,
    write_manifest,
)

DATASET_URL = "https://physionet.org/content/brugada-huca/1.0.0/"
DATASET_VERSION = "1.0.0"
RIGHT_PRECORDIAL_LEADS = ("V1",)
R_DETECTION_LEADS = ("V1", "II")
DEFAULT_PRE_R_MS = 300
DEFAULT_POST_R_MS = 600
LEAD_GRID = (
    ("V1",),
)


@dataclass(frozen=True)
class BrugadaRecord:
    patient_id: str
    basal_pattern: int
    sudden_death: int
    brugada: int


@dataclass(frozen=True)
class ExcludedRecord:
    patient_id: str
    basal_pattern: int
    sudden_death: int
    brugada: int
    reason: str


@dataclass(frozen=True)
class BeatExtraction:
    r_peak_sample: int
    r_peak_lead: str
    r_peak_detector: str
    pre_r_ms: int
    post_r_ms: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Brugada-HUCA LOOCV images with clinical labels for evaluation."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/brugada-huca/1.0.0"))
    parser.add_argument("--outdir", type=Path, default=Path("data/brugada_huca"))
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--pre-r-ms", type=int, default=DEFAULT_PRE_R_MS)
    parser.add_argument("--post-r-ms", type=int, default=DEFAULT_POST_R_MS)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--val-per-class", type=int, default=4)
    parser.add_argument(
        "--include-borderline-positive",
        action="store_true",
        help="Include basal_pattern=1 and brugada=1 patients after clinical review.",
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reset_output_dir(outdir: Path, *, overwrite: bool) -> None:
    if outdir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing dataset: {outdir}")
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)


def read_metadata(
    raw_root: Path,
    *,
    include_borderline_positive: bool = False,
) -> list[BrugadaRecord]:
    records, _excluded = read_metadata_with_exclusions(
        raw_root,
        include_borderline_positive=include_borderline_positive,
    )
    return records


def read_metadata_with_exclusions(
    raw_root: Path,
    *,
    include_borderline_positive: bool = False,
) -> tuple[list[BrugadaRecord], list[ExcludedRecord]]:
    metadata = read_metadata_frame(raw_root)
    records: list[BrugadaRecord] = []
    excluded: list[ExcludedRecord] = []
    for row in metadata.to_dict(orient="records"):
        record = BrugadaRecord(
            patient_id=str(int(row["patient_id"])),
            basal_pattern=int(row["basal_pattern"]),
            sudden_death=int(row["sudden_death"]),
            brugada=int(row["brugada"]),
        )
        if record.brugada == 2:
            excluded.append(_excluded(record, "brugada_2_atypical"))
            continue
        if record.brugada not in {0, 1}:
            raise ValueError(
                f"Unsupported brugada label {record.brugada!r} for {record.patient_id}"
            )
        if (
            not include_borderline_positive
            and record.basal_pattern == 1
            and record.brugada == 1
        ):
            excluded.append(_excluded(record, "borderline_basal_pattern_1_brugada_1"))
            continue
        records.append(record)
    records.sort(key=lambda item: int(item.patient_id))
    excluded.sort(key=lambda item: int(item.patient_id))
    return records, excluded


def read_metadata_frame(raw_root: Path) -> pd.DataFrame:
    metadata_path = raw_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Download Brugada-HUCA v{DATASET_VERSION} first."
        )
    metadata = pd.read_csv(metadata_path)
    if len(metadata.columns) == 1:
        metadata = pd.read_csv(metadata_path, sep=r"\s+")
    required = {"patient_id", "basal_pattern", "sudden_death", "brugada"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"{metadata_path} is missing required columns: {missing}")
    return metadata


def _excluded(record: BrugadaRecord, reason: str) -> ExcludedRecord:
    return ExcludedRecord(
        patient_id=record.patient_id,
        basal_pattern=record.basal_pattern,
        sudden_death=record.sudden_death,
        brugada=record.brugada,
        reason=reason,
    )


def wfdb_record_path(raw_root: Path, patient_id: str) -> Path:
    return raw_root / "files" / patient_id / patient_id


def read_wfdb_record(raw_root: Path, patient_id: str) -> Any:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "wfdb is required to build Brugada-HUCA. Install with "
            "`uv sync --extra real-data`."
        ) from exc
    return wfdb.rdrecord(wfdb_record_path(raw_root, patient_id).as_posix())


def lead_map(record: Any) -> dict[str, np.ndarray]:
    signals = np.asarray(record.p_signal, dtype=np.float32)
    names = [str(name) for name in record.sig_name]
    if signals.ndim != 2:
        raise ValueError(f"Expected a 2D WFDB signal array, found shape {signals.shape}.")
    return {lead_name: signals[:, index] for index, lead_name in enumerate(names)}


def detect_central_r_peak(record: Any, leads: dict[str, np.ndarray]) -> BeatExtraction:
    fs = float(record.fs)
    center_sample = int(np.asarray(next(iter(leads.values()))).shape[0] // 2)
    for lead_name in R_DETECTION_LEADS:
        signal = leads.get(lead_name)
        if signal is None:
            continue
        for detector in ("xqrs_detect", "gqrs_detect"):
            peaks = _detect_r_peaks(signal, fs=fs, detector=detector)
            if peaks.size:
                peak = int(peaks[int(np.argmin(np.abs(peaks - center_sample)))])
                return BeatExtraction(
                    r_peak_sample=peak,
                    r_peak_lead=lead_name,
                    r_peak_detector=detector,
                    pre_r_ms=DEFAULT_PRE_R_MS,
                    post_r_ms=DEFAULT_POST_R_MS,
                )
    return BeatExtraction(
        r_peak_sample=center_sample,
        r_peak_lead="center_fallback",
        r_peak_detector="center_sample",
        pre_r_ms=DEFAULT_PRE_R_MS,
        post_r_ms=DEFAULT_POST_R_MS,
    )


def _detect_r_peaks(signal: np.ndarray, *, fs: float, detector: str) -> np.ndarray:
    try:
        from wfdb import processing
    except ImportError as exc:
        raise RuntimeError(
            "wfdb is required to build Brugada-HUCA. Install with "
            "`uv sync --extra real-data`."
        ) from exc
    finite_signal = np.nan_to_num(np.asarray(signal, dtype=np.float64), copy=False)
    try:
        if detector == "xqrs_detect":
            peaks = processing.xqrs_detect(sig=finite_signal, fs=fs, verbose=False)
        elif detector == "gqrs_detect":
            peaks = processing.gqrs_detect(sig=finite_signal, fs=fs)
        else:
            raise ValueError(f"Unsupported detector: {detector}")
    except Exception:
        return np.asarray([], dtype=np.int64)
    return np.asarray(peaks, dtype=np.int64)


def extract_beat_window(
    signal: np.ndarray,
    *,
    fs: float,
    r_peak_sample: int,
    pre_r_ms: int,
    post_r_ms: int,
) -> np.ndarray:
    pre_samples = int(round(fs * pre_r_ms / 1000.0))
    post_samples = int(round(fs * post_r_ms / 1000.0))
    start = int(r_peak_sample) - pre_samples
    end = int(r_peak_sample) + post_samples
    clipped = np.asarray(signal[max(0, start) : min(int(signal.shape[0]), end)], dtype=np.float32)
    pad_left = max(0, -start)
    pad_right = max(0, end - int(signal.shape[0]))
    if pad_left or pad_right:
        clipped = np.pad(clipped, (pad_left, pad_right), mode="edge")
    expected = pre_samples + post_samples
    if clipped.shape[0] != expected:
        clipped = clipped[:expected] if clipped.shape[0] > expected else np.pad(
            clipped,
            (0, expected - clipped.shape[0]),
            mode="edge",
        )
    return clipped.astype(np.float32)


def render_single_beat(beat: np.ndarray, output_path: Path, *, fs: float, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seconds = np.arange(beat.shape[0], dtype=np.float32) / float(fs)
    finite = beat[np.isfinite(beat)]
    max_abs = max(0.25, float(np.nanpercentile(np.abs(finite), 99.0)) * 1.25)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(seconds, beat, color="#111111", linewidth=2)
    ax.set_xlim(float(seconds[0]), float(seconds[-1]))
    ax.set_ylim(-max_abs, max_abs)
    ax.grid(which="major", color="#e8b8b8", linewidth=0.65)
    ax.grid(which="minor", color="#f4dcdc", linewidth=0.35)
    ax.minorticks_on()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_12lead_ecg(record: Any, output_path: Path, *, patient_id: str, dpi: int) -> None:
    leads = lead_map(record)
    missing = [lead for row in LEAD_GRID for lead in row if lead not in leads]
    if missing:
        raise ValueError(f"Patient {patient_id} is missing expected leads: {missing}")
    fs = float(record.fs)
    seconds = np.arange(leads[LEAD_GRID[0][0]].shape[0], dtype=np.float32) / fs
    stacked = np.concatenate([leads[lead] for row in LEAD_GRID for lead in row])
    finite = stacked[np.isfinite(stacked)]
    max_abs = max(0.5, float(np.nanpercentile(np.abs(finite), 99.0)) * 1.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 1, figsize=(3.2, 3.2), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(LEAD_GRID), len(LEAD_GRID[0]))
    for row_index, lead_row in enumerate(LEAD_GRID):
        for col_index, lead_name in enumerate(lead_row):
            axis = axes[row_index, col_index]
            axis.plot(seconds, leads[lead_name], color="#111111", linewidth=0.85)
            axis.set_title(lead_name, loc="left", fontsize=10, fontweight="bold")
            axis.set_ylim(-max_abs, max_abs)
            axis.set_xlim(float(seconds[0]), float(seconds[-1]))
            axis.grid(which="major", color="#e6b4b4", linewidth=0.55)
            axis.grid(which="minor", color="#f3d8d8", linewidth=0.35)
            axis.minorticks_on()
            axis.tick_params(labelsize=7, length=2)
            if col_index == 0:
                axis.set_ylabel("mV", fontsize=8)
            if row_index == 2:
                axis.set_xlabel("s", fontsize=8)
    fig.suptitle(f"Brugada-HUCA patient {patient_id}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_dataset(
    *,
    records: Sequence[BrugadaRecord],
    excluded: Sequence[ExcludedRecord],
    raw_root: Path,
    outdir: Path,
    dpi: int,
    pre_r_ms: int,
    post_r_ms: int,
    k_values: Sequence[int],
    seeds: Sequence[int],
    val_per_class: int,
    include_borderline_positive: bool,
) -> list[BrugadaImageRow]:
    rows: list[BrugadaImageRow] = []
    for record in records:
        source_family = "CLINICAL_BRUGADA" if record.brugada else "CLINICAL_NORMAL"
        wfdb_record = read_wfdb_record(raw_root, record.patient_id)
        leads = lead_map(wfdb_record)
        missing = [lead for lead in RIGHT_PRECORDIAL_LEADS if lead not in leads]
        if missing:
            raise ValueError(f"Patient {record.patient_id} is missing expected leads: {missing}")
        extraction = detect_central_r_peak(wfdb_record, leads)
        extraction = BeatExtraction(
            r_peak_sample=extraction.r_peak_sample,
            r_peak_lead=extraction.r_peak_lead,
            r_peak_detector=extraction.r_peak_detector,
            pre_r_ms=pre_r_ms,
            post_r_ms=post_r_ms,
        )
        for lead_name in RIGHT_PRECORDIAL_LEADS:
            image_path = Path("images") / source_family / f"{record.patient_id}_{lead_name}.png"
            beat = extract_beat_window(
                leads[lead_name],
                fs=float(wfdb_record.fs),
                r_peak_sample=extraction.r_peak_sample,
                pre_r_ms=pre_r_ms,
                post_r_ms=post_r_ms,
            )
            render_single_beat(beat, outdir / image_path, fs=float(wfdb_record.fs), dpi=dpi)
            rows.append(
                BrugadaImageRow(
                    image_path=image_path.as_posix(),
                    patient_id=record.patient_id,
                    lead=lead_name,
                    source_family=source_family,
                    label_rbbb=None,
                    label_st_elevation=None,
                    label_t_wave_inversion=None,
                    clinical_brugada=record.brugada,
                    basal_pattern=record.basal_pattern,
                    sudden_death=record.sudden_death,
                    sample_index=int(record.patient_id),
                    aggregation_group_id=record.patient_id,
                    r_peak_sample=extraction.r_peak_sample,
                    r_peak_lead=extraction.r_peak_lead,
                    r_peak_detector=extraction.r_peak_detector,
                    pre_r_ms=pre_r_ms,
                    post_r_ms=post_r_ms,
                )
            )

    labels_dir = outdir / "labels"
    write_manifest(labels_dir / "all_labels.csv", rows)
    write_excluded(labels_dir / "excluded_patients.csv", excluded)
    save_json(
        labels_dir / "label_schema.json",
        {
            "labels": LABEL_NAMES,
            "label_columns": LABEL_COLUMNS,
            "clinical_reference_column": "clinical_brugada",
            "normal_token": "NORMAL",
            "task": "qrs_finding_detection_with_derived_brugada_evaluation",
            "brugada_rule": "all QRS finding labels must be true",
            "qrs_labels_available": False,
        },
    )
    fold_plan = build_fold_plan(
        rows,
        k_values=k_values,
        seeds=seeds,
        val_per_class=val_per_class,
    )
    write_jsonl(labels_dir / "loocv_folds.jsonl", fold_plan)
    write_jsonl(outdir / "vlm" / "all_records.jsonl", vlm_records(rows))
    save_json(
        labels_dir / "dataset_summary.json",
        dataset_summary(
            rows=rows,
            records=records,
            excluded=excluded,
            raw_root=raw_root,
            outdir=outdir,
            dpi=dpi,
            pre_r_ms=pre_r_ms,
            post_r_ms=post_r_ms,
            k_values=k_values,
            seeds=seeds,
            val_per_class=val_per_class,
            include_borderline_positive=include_borderline_positive,
        ),
    )
    build_examples_preview(outdir)
    return rows


def write_excluded(path: Path, excluded: Sequence[ExcludedRecord]) -> None:
    fieldnames = ["patient_id", "basal_pattern", "sudden_death", "brugada", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in excluded:
            writer.writerow(record.__dict__)


def vlm_records(rows: Sequence[BrugadaImageRow]) -> Iterable[dict[str, object]]:
    for row in rows:
        yield {
            "id": f"{row.patient_id}_{row.lead.lower()}",
            "image_path": row.image_path,
            "prompt": (
                f"Analyze this right precordial ECG beat image from lead {row.lead}. "
                "Return JSON booleans for RBBB, ST_ELEVATION, and T_WAVE_INVERSION."
            ),
            "expected_answer": None,
            "metadata": {
                "patient_id": row.patient_id,
                "lead": row.lead,
                "aggregation_group_id": row.aggregation_group_id,
                "source_family": row.source_family,
                "basal_pattern": row.basal_pattern,
                "sudden_death": row.sudden_death,
                "clinical_brugada": row.clinical_brugada,
            },
        }


def dataset_summary(
    *,
    rows: Sequence[BrugadaImageRow],
    records: Sequence[BrugadaRecord],
    excluded: Sequence[ExcludedRecord],
    raw_root: Path,
    outdir: Path,
    dpi: int,
    pre_r_ms: int,
    post_r_ms: int,
    k_values: Sequence[int],
    seeds: Sequence[int],
    val_per_class: int,
    include_borderline_positive: bool,
) -> dict[str, object]:
    patient_counts = Counter(record.brugada for record in records)
    excluded_counts = Counter(record.reason for record in excluded)
    return {
        "source": DATASET_URL,
        "version": DATASET_VERSION,
        "raw_root": raw_root.as_posix(),
        "outdir": outdir.as_posix(),
        "labels": LABEL_NAMES,
        "qrs_labels_available": False,
        "n_patients": len(records),
        "n_images": len(rows),
        "clinical_label_counts": {
            "normal": int(patient_counts.get(0, 0)),
            "brugada": int(patient_counts.get(1, 0)),
        },
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "excluded_patients": len(excluded),
        "include_borderline_positive": include_borderline_positive,
        "right_precordial_leads": list(RIGHT_PRECORDIAL_LEADS),
        "image_layout": "single central beat for V1 only",
        "aggregation": "single_v1_condition_probability_then_all_conditions_true",
        "dpi": dpi,
        "pre_r_ms": pre_r_ms,
        "post_r_ms": post_r_ms,
        "k_values": list(k_values),
        "seeds": list(seeds),
        "val_per_class": val_per_class,
        "fold_plan": "labels/loocv_folds.jsonl",
    }


def build_examples_preview(outdir: Path) -> None:
    examples_dir = outdir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for source_family in ("CLINICAL_NORMAL", "CLINICAL_BRUGADA"):
        candidates = sorted((outdir / "images" / source_family).glob("*.png"))
        if not candidates:
            continue
        destination = examples_dir / f"{source_family.lower()}_example.png"
        shutil.copyfile(candidates[0], destination)
        manifest.append({"source_family": source_family, "image_path": destination.as_posix()})
    save_json(examples_dir / "manifest.json", manifest)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    outdir = args.outdir.resolve()
    reset_output_dir(outdir, overwrite=bool(args.overwrite))
    records, excluded = read_metadata_with_exclusions(
        raw_root,
        include_borderline_positive=bool(args.include_borderline_positive),
    )
    rows = build_dataset(
        records=records,
        excluded=excluded,
        raw_root=raw_root,
        outdir=outdir,
        dpi=int(args.dpi),
        pre_r_ms=int(args.pre_r_ms),
        post_r_ms=int(args.post_r_ms),
        k_values=parse_int_list(args.k_values),
        seeds=parse_int_list(args.seeds),
        val_per_class=int(args.val_per_class),
        include_borderline_positive=bool(args.include_borderline_positive),
    )
    print(f"[OK] Wrote Brugada-HUCA LOOCV dataset under: {outdir}")
    print(f"[OK] Patients: {len(records)} | images: {len(rows)} | excluded: {len(excluded)}")


if __name__ == "__main__":
    main()
