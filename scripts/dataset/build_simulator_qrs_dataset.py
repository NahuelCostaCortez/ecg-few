#!/usr/bin/env python3
"""Build a synthetic QRS/ST multi-label dataset for derived Brugada evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from ecg_few.findings import LABEL_COLUMNS, LABEL_NAMES, findings_to_brugada, findings_to_text
from ecg_few.loocv import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    BrugadaImageRow,
    build_fold_plan,
    parse_int_list,
    write_jsonl,
    write_manifest,
)
from ecg_few.simulator import SOURCE_FAMILIES, generate_beat
from ecg_few.simulator.plotting import plot_beat

LEADS = ("V1", "V2", "V3")
DISPLAY_SOURCE_FAMILY = {"BRUGADA": "COMBINED_QRS_ST"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic QRS/ST LOOCV dataset.")
    parser.add_argument("--outdir", type=Path, default=Path("data/simulator_qrs"))
    parser.add_argument("--patients-per-source-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fs", type=int, default=500)
    parser.add_argument("--duration-ms", type=int, default=900)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--val-per-class", type=int, default=4)
    parser.add_argument("--max-patient-attempts", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_dataset(
        outdir=args.outdir.resolve(),
        patients_per_source_family=int(args.patients_per_source_family),
        base_seed=int(args.seed),
        fs=int(args.fs),
        duration_ms=int(args.duration_ms),
        dpi=int(args.dpi),
        k_values=parse_int_list(str(args.k_values)),
        seeds=parse_int_list(str(args.seeds)),
        val_per_class=int(args.val_per_class),
        max_patient_attempts=int(args.max_patient_attempts),
        overwrite=bool(args.overwrite),
    )
    patients = {row.patient_id for row in rows}
    print(f"[OK] Wrote synthetic QRS/ST LOOCV dataset under: {args.outdir.resolve()}")
    print(f"[OK] Patients: {len(patients)} | images: {len(rows)}")


def build_dataset(
    *,
    outdir: Path,
    patients_per_source_family: int,
    base_seed: int,
    fs: int,
    duration_ms: int,
    dpi: int,
    k_values: list[int],
    seeds: list[int],
    val_per_class: int,
    max_patient_attempts: int,
    overwrite: bool,
) -> list[BrugadaImageRow]:
    if patients_per_source_family < 1:
        raise ValueError("--patients-per-source-family must be positive.")
    reset_output_dir(outdir, overwrite=overwrite)
    rows: list[BrugadaImageRow] = []
    metadata_rows: list[dict[str, object]] = []
    sample_index = 0
    for family_index, generator_family in enumerate(SOURCE_FAMILIES):
        display_family = DISPLAY_SOURCE_FAMILY.get(generator_family, generator_family)
        for class_index in range(patients_per_source_family):
            patient_number = family_index * patients_per_source_family + class_index + 1
            patient_id = f"SIM{patient_number:05d}"
            patient_seed = base_seed + family_index * 1_000_003 + class_index * 10_007
            waveform, metadata = generate_patient_waveform(
                generator_family=generator_family,
                seed=patient_seed,
                fs=fs,
                duration_ms=duration_ms,
                max_patient_attempts=max_patient_attempts,
            )
            findings = {
                label_name: int(metadata["feature_labels"][label_name])
                for label_name in LABEL_NAMES
            }
            feature_text = findings_to_text(findings)
            for lead in LEADS:
                image_path = Path("images") / feature_text / f"{patient_id}_{lead}.png"
                (outdir / image_path).parent.mkdir(parents=True, exist_ok=True)
                plot_beat(waveform, fs=fs, save_path=str(outdir / image_path), dpi=dpi)
                row = BrugadaImageRow(
                    image_path=image_path.as_posix(),
                    patient_id=patient_id,
                    lead=lead,
                    source_family=display_family,
                    label_rbbb=findings["RBBB"],
                    label_st_elevation=findings["ST_ELEVATION"],
                    label_t_wave_inversion=findings["T_WAVE_INVERSION"],
                    clinical_brugada=None,
                    basal_pattern=0,
                    sudden_death=0,
                    sample_index=sample_index,
                    aggregation_group_id=patient_id,
                    r_peak_sample=None,
                    r_peak_lead=lead,
                    r_peak_detector="synthetic_known_template",
                    pre_r_ms=300,
                    post_r_ms=600,
                )
                rows.append(row)
                metadata_rows.append(
                    {
                        "patient_id": patient_id,
                        "lead": lead,
                        "generator_source_family": generator_family,
                        "source_family": display_family,
                        "seed": patient_seed,
                        "fs": fs,
                        "duration_ms": duration_ms,
                        "attempt": int(metadata["attempt"]),
                        "feature_labels": findings,
                        "derived_brugada": findings_to_brugada(findings),
                        "atoms": metadata["atoms"],
                    }
                )
                sample_index += 1

    labels_dir = outdir / "labels"
    write_manifest(labels_dir / "all_labels.csv", rows)
    write_excluded(labels_dir / "excluded_patients.csv")
    write_label_schema(labels_dir / "label_schema.json")
    fold_plan = build_fold_plan(
        rows,
        k_values=k_values,
        seeds=seeds,
        val_per_class=val_per_class,
    )
    write_jsonl(labels_dir / "loocv_folds.jsonl", fold_plan)
    write_jsonl(outdir / "vlm" / "all_records.jsonl", vlm_records(rows))
    save_json(outdir / "metadata" / "simulator_parameters.json", metadata_rows)
    save_json(
        labels_dir / "dataset_summary.json",
        dataset_summary(
            rows=rows,
            patients_per_source_family=patients_per_source_family,
            base_seed=base_seed,
            fs=fs,
            duration_ms=duration_ms,
            dpi=dpi,
            k_values=k_values,
            seeds=seeds,
            val_per_class=val_per_class,
        ),
    )
    build_examples_preview(outdir)
    return rows


def generate_patient_waveform(
    *,
    generator_family: str,
    seed: int,
    fs: int,
    duration_ms: int,
    max_patient_attempts: int,
) -> tuple[object, dict[str, object]]:
    subtype = "coved" if generator_family == "BRUGADA" else None
    for offset in range(max_patient_attempts):
        waveform, metadata = generate_beat(
            class_name=generator_family,
            seed=seed + offset * 97_531,
            subtype=subtype,
            fs=fs,
            duration_ms=duration_ms,
        )
        findings = metadata["feature_labels"]
        if accepts_source_family(generator_family, findings):
            return waveform, metadata
    raise RuntimeError(
        f"Could not generate a {generator_family} patient with compatible QRS labels "
        f"after {max_patient_attempts} attempts."
    )


def accepts_source_family(generator_family: str, findings: dict[str, int]) -> bool:
    if generator_family == "NORMAL":
        return not any(int(findings[label_name]) for label_name in LABEL_NAMES)
    if generator_family == "BRUGADA":
        return bool(findings_to_brugada(findings))
    return int(findings[generator_family]) == 1


def reset_output_dir(outdir: Path, *, overwrite: bool) -> None:
    if outdir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing dataset: {outdir}")
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)


def write_excluded(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patient_id", "source_family", "reason"],
        )
        writer.writeheader()


def write_label_schema(path: Path) -> None:
    save_json(
        path,
        {
            "labels": LABEL_NAMES,
            "label_columns": LABEL_COLUMNS,
            "normal_token": "NORMAL",
            "task": "qrs_finding_detection",
            "derived_brugada_rule": "RBBB and ST_ELEVATION and T_WAVE_INVERSION",
        },
    )


def vlm_records(rows: list[BrugadaImageRow]) -> list[dict[str, object]]:
    return [
        {
            "id": f"{row.patient_id}_{row.lead.lower()}",
            "image_path": row.image_path,
            "expected_answer": row.expected_answer(),
            "metadata": {
                "patient_id": row.patient_id,
                "lead": row.lead,
                "source_family": row.source_family,
                "derived_brugada": row.derived_brugada,
            },
        }
        for row in rows
    ]


def dataset_summary(
    *,
    rows: list[BrugadaImageRow],
    patients_per_source_family: int,
    base_seed: int,
    fs: int,
    duration_ms: int,
    dpi: int,
    k_values: list[int],
    seeds: list[int],
    val_per_class: int,
) -> dict[str, object]:
    patient_rows = {row.patient_id: row for row in rows}
    derived_counts = Counter(row.derived_brugada for row in patient_rows.values())
    source_counts = Counter(row.source_family for row in patient_rows.values())
    combo_counts = Counter(findings_to_text(row.findings) for row in patient_rows.values())
    return {
        "source": "ecg_few.simulator",
        "task": "qrs_finding_detection_with_derived_brugada",
        "n_patients": len(patient_rows),
        "n_images": len(rows),
        "patients_per_source_family": patients_per_source_family,
        "source_family_counts": dict(sorted(source_counts.items())),
        "derived_brugada_counts": {
            "normal_or_incomplete": int(derived_counts.get(0, 0)),
            "all_conditions_present": int(derived_counts.get(1, 0)),
        },
        "finding_combination_counts": dict(sorted(combo_counts.items())),
        "right_precordial_leads": list(LEADS),
        "image_layout": "single simulated beat copied across V1/V2/V3 lead slots",
        "aggregation": "mean_condition_probability_by_patient_then_all_conditions_true",
        "base_seed": base_seed,
        "fs": fs,
        "duration_ms": duration_ms,
        "dpi": dpi,
        "k_values": k_values,
        "seeds": seeds,
        "val_per_class": val_per_class,
        "fold_plan": "labels/loocv_folds.jsonl",
    }


def build_examples_preview(outdir: Path) -> None:
    examples_dir = outdir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for family_dir in sorted((outdir / "images").glob("*")):
        if not family_dir.is_dir():
            continue
        candidates = sorted(family_dir.glob("*.png"))
        if not candidates:
            continue
        destination = examples_dir / f"{family_dir.name.lower()}_example.png"
        shutil.copyfile(candidates[0], destination)
        manifest.append(
            {
                "finding_combination": family_dir.name,
                "image_path": destination.as_posix(),
            }
        )
    save_json(examples_dir / "manifest.json", manifest)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
