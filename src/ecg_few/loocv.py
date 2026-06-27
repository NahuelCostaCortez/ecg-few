"""Shared QRS-finding LOOCV fold planning utilities."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecg_few.findings import LABEL_COLUMNS, LABEL_NAMES, findings_to_brugada, normalize_findings

DEFAULT_K_VALUES = (2, 4, 8, 16, 32)
DEFAULT_SEEDS = (42, 123, 2026)


@dataclass(frozen=True)
class BrugadaImageRow:
    """One ECG image row with QRS labels and optional clinical Brugada reference."""

    image_path: str
    patient_id: str
    lead: str
    source_family: str
    label_rbbb: int | None
    label_st_elevation: int | None
    label_t_wave_inversion: int | None
    clinical_brugada: int | None
    basal_pattern: int
    sudden_death: int
    sample_index: int
    aggregation_group_id: str
    r_peak_sample: int | None = None
    r_peak_lead: str = ""
    r_peak_detector: str = ""
    pre_r_ms: int | None = None
    post_r_ms: int | None = None

    @property
    def findings(self) -> dict[str, int]:
        if not self.has_qrs_labels:
            raise ValueError(
                f"Row {self.patient_id}/{self.lead} has no QRS finding labels. "
                "Use it only for clinical Brugada evaluation, not detector training."
            )
        return {
            "RBBB": int(self.label_rbbb),
            "ST_ELEVATION": int(self.label_st_elevation),
            "T_WAVE_INVERSION": int(self.label_t_wave_inversion),
        }

    @property
    def has_qrs_labels(self) -> bool:
        return (
            self.label_rbbb is not None
            and self.label_st_elevation is not None
            and self.label_t_wave_inversion is not None
        )

    @property
    def derived_brugada(self) -> int | None:
        if not self.has_qrs_labels:
            return None
        return findings_to_brugada(self.findings)

    @property
    def reference_brugada(self) -> int:
        if self.clinical_brugada is not None:
            return int(self.clinical_brugada)
        derived = self.derived_brugada
        if derived is None:
            raise ValueError(f"Row {self.patient_id}/{self.lead} has no reference label.")
        return int(derived)

    @property
    def label(self) -> int:
        return self.reference_brugada

    def expected_answer(self) -> dict[str, bool]:
        return {label_name: bool(self.findings[label_name]) for label_name in LABEL_NAMES}


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    reference_brugada: int
    source_family: str
    basal_pattern: int
    sudden_death: int
    findings: dict[str, int]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def resolve_folds_path(dataset_root: Path, folds: str | Path) -> Path:
    text = str(folds)
    if text in {"", "."}:
        return dataset_root / "labels" / "loocv_folds.jsonl"
    return Path(folds)


def read_manifest(path: Path) -> list[BrugadaImageRow]:
    rows: list[BrugadaImageRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            findings = _optional_findings(record)
            clinical_brugada = _optional_int(record.get("clinical_brugada"))
            if findings is None and clinical_brugada is None:
                raise ValueError(
                    f"{path} has neither QRS finding labels nor clinical_brugada. "
                    "Rebuild obsolete direct-Brugada manifests with the current builders."
                )
            rows.append(
                BrugadaImageRow(
                    image_path=str(record["image_path"]),
                    patient_id=str(record["patient_id"]),
                    lead=str(record["lead"]),
                    source_family=str(record["source_family"]),
                    label_rbbb=None if findings is None else findings["RBBB"],
                    label_st_elevation=None if findings is None else findings["ST_ELEVATION"],
                    label_t_wave_inversion=None
                    if findings is None
                    else findings["T_WAVE_INVERSION"],
                    clinical_brugada=clinical_brugada,
                    basal_pattern=int(record.get("basal_pattern", 0) or 0),
                    sudden_death=int(record.get("sudden_death", 0) or 0),
                    sample_index=int(record["sample_index"]),
                    aggregation_group_id=str(record["aggregation_group_id"]),
                    r_peak_sample=_optional_int(record.get("r_peak_sample")),
                    r_peak_lead=str(record.get("r_peak_lead", "") or ""),
                    r_peak_detector=str(record.get("r_peak_detector", "") or ""),
                    pre_r_ms=_optional_int(record.get("pre_r_ms")),
                    post_r_ms=_optional_int(record.get("post_r_ms")),
                )
            )
    return sorted(rows, key=lambda row: (_patient_sort_key(row.patient_id), row.lead))


def write_manifest(path: Path, rows: Iterable[BrugadaImageRow]) -> None:
    fieldnames = [
        "image_path",
        "patient_id",
        "lead",
        "source_family",
        LABEL_COLUMNS["RBBB"],
        LABEL_COLUMNS["ST_ELEVATION"],
        LABEL_COLUMNS["T_WAVE_INVERSION"],
        "derived_brugada",
        "clinical_brugada",
        "basal_pattern",
        "sudden_death",
        "sample_index",
        "aggregation_group_id",
        "r_peak_sample",
        "r_peak_lead",
        "r_peak_detector",
        "pre_r_ms",
        "post_r_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row.image_path,
                    "patient_id": row.patient_id,
                    "lead": row.lead,
                    "source_family": row.source_family,
                    LABEL_COLUMNS["RBBB"]: row.label_rbbb,
                    LABEL_COLUMNS["ST_ELEVATION"]: row.label_st_elevation,
                    LABEL_COLUMNS["T_WAVE_INVERSION"]: row.label_t_wave_inversion,
                    "derived_brugada": "" if row.derived_brugada is None else row.derived_brugada,
                    "clinical_brugada": ""
                    if row.clinical_brugada is None
                    else row.clinical_brugada,
                    "basal_pattern": row.basal_pattern,
                    "sudden_death": row.sudden_death,
                    "sample_index": row.sample_index,
                    "aggregation_group_id": row.aggregation_group_id,
                    "r_peak_sample": "" if row.r_peak_sample is None else row.r_peak_sample,
                    "r_peak_lead": row.r_peak_lead,
                    "r_peak_detector": row.r_peak_detector,
                    "pre_r_ms": "" if row.pre_r_ms is None else row.pre_r_ms,
                    "post_r_ms": "" if row.post_r_ms is None else row.post_r_ms,
                }
            )


def patients_from_rows(rows: Sequence[BrugadaImageRow]) -> list[PatientRecord]:
    by_patient: dict[str, BrugadaImageRow] = {}
    for row in rows:
        current = by_patient.get(row.patient_id)
        if current is None:
            by_patient[row.patient_id] = row
            continue
        if current.reference_brugada != row.reference_brugada:
            raise ValueError(f"Patient {row.patient_id} has inconsistent reference labels.")
    return [
        PatientRecord(
            patient_id=row.patient_id,
            reference_brugada=row.reference_brugada,
            source_family=row.source_family,
            basal_pattern=row.basal_pattern,
            sudden_death=row.sudden_death,
            findings=row.findings if row.has_qrs_labels else {},
        )
        for row in sorted(by_patient.values(), key=lambda item: _patient_sort_key(item.patient_id))
    ]


def rows_for_patient_ids(
    rows: Sequence[BrugadaImageRow],
    patient_ids: Iterable[str],
) -> list[BrugadaImageRow]:
    wanted = {str(patient_id) for patient_id in patient_ids}
    return [row for row in rows if row.patient_id in wanted]


def patient_label_map(rows: Sequence[BrugadaImageRow]) -> dict[str, int]:
    return {patient.patient_id: patient.reference_brugada for patient in patients_from_rows(rows)}


def build_fold_plan(
    rows: Sequence[BrugadaImageRow],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    val_per_class: int = 4,
) -> list[dict[str, Any]]:
    patients = patients_from_rows(rows)
    if not patients:
        raise ValueError("Cannot build LOOCV folds from an empty manifest.")
    folds: list[dict[str, Any]] = []
    for fold_id, test_patient in enumerate(patients):
        selections: dict[str, dict[str, dict[str, list[str]]]] = {}
        for seed in seeds:
            seed_payload: dict[str, dict[str, list[str]]] = {}
            for k in k_values:
                if k < 1:
                    raise ValueError("CNN/VLM few-shot LOOCV requires k >= 1.")
                selection_seed = int(seed) + fold_id * 100_003 + int(k) * 1_009
                context_ids = select_context_patient_ids(
                    patients,
                    test_patient_id=test_patient.patient_id,
                    k=int(k),
                    seed=selection_seed,
                )
                validation_ids = select_validation_patient_ids(
                    patients,
                    excluded_patient_ids={test_patient.patient_id, *context_ids},
                    seed=selection_seed + 17,
                    val_per_class=val_per_class,
                )
                seed_payload[str(k)] = {
                    "context_patient_ids": context_ids,
                    "validation_patient_ids": validation_ids,
                }
            selections[str(seed)] = seed_payload
        folds.append(
            {
                "fold_id": fold_id,
                "test_patient_id": test_patient.patient_id,
                "reference_brugada": test_patient.reference_brugada,
                "source_family": test_patient.source_family,
                "findings": test_patient.findings,
                "selections": selections,
            }
        )
    return folds


def select_context_patient_ids(
    patients: Sequence[PatientRecord],
    *,
    test_patient_id: str,
    k: int,
    seed: int,
) -> list[str]:
    available = [patient for patient in patients if patient.patient_id != str(test_patient_id)]
    if k >= len(available):
        return [patient.patient_id for patient in sorted(available, key=_patient_record_sort_key)]

    rng = random.Random(seed)
    by_label: dict[int, list[PatientRecord]] = {0: [], 1: []}
    for patient in available:
        by_label.setdefault(patient.reference_brugada, []).append(patient)

    selected: list[PatientRecord] = []
    selected_ids: set[str] = set()
    if k >= 2:
        for label in (0, 1):
            candidates = sorted(by_label.get(label, []), key=_patient_record_sort_key)
            if candidates:
                choice = rng.choice(candidates)
                selected.append(choice)
                selected_ids.add(choice.patient_id)
                if len(selected) >= k:
                    return [patient.patient_id for patient in selected]

    remaining = [
        patient
        for patient in sorted(available, key=_patient_record_sort_key)
        if patient.patient_id not in selected_ids
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, k - len(selected))])
    return [patient.patient_id for patient in selected[:k]]


def select_validation_patient_ids(
    patients: Sequence[PatientRecord],
    *,
    excluded_patient_ids: set[str],
    seed: int,
    val_per_class: int,
) -> list[str]:
    if val_per_class <= 0:
        return []
    rng = random.Random(seed)
    selected: list[PatientRecord] = []
    for label in (0, 1):
        candidates = [
            patient
            for patient in patients
            if patient.reference_brugada == label and patient.patient_id not in excluded_patient_ids
        ]
        candidates = sorted(candidates, key=_patient_record_sort_key)
        rng.shuffle(candidates)
        selected.extend(candidates[:val_per_class])
    return [
        patient.patient_id
        for patient in sorted(
            selected,
            key=lambda item: (item.reference_brugada, _patient_sort_key(item.patient_id)),
        )
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_fold_plan(
    folds: Sequence[dict[str, Any]],
    rows: Sequence[BrugadaImageRow],
    *,
    k_values: Sequence[int],
    seeds: Sequence[int],
) -> None:
    patient_ids = {patient.patient_id for patient in patients_from_rows(rows)}
    tested = [str(fold["test_patient_id"]) for fold in folds]
    if set(tested) != patient_ids or len(tested) != len(patient_ids):
        raise ValueError("LOOCV fold plan must test every patient exactly once.")
    for fold in folds:
        test_id = str(fold["test_patient_id"])
        if test_id not in patient_ids:
            raise ValueError(f"Fold tests unknown patient {test_id}.")
        selections = fold.get("selections", {})
        for seed in seeds:
            seed_payload = selections.get(str(seed))
            if not isinstance(seed_payload, dict):
                raise ValueError(f"Fold {fold['fold_id']} is missing seed {seed}.")
            for k in k_values:
                selection = seed_payload.get(str(k))
                if not isinstance(selection, dict):
                    raise ValueError(f"Fold {fold['fold_id']} is missing k={k}, seed={seed}.")
                context_ids = [str(item) for item in selection.get("context_patient_ids", [])]
                validation_ids = [
                    str(item) for item in selection.get("validation_patient_ids", [])
                ]
                if test_id in context_ids or test_id in validation_ids:
                    raise ValueError(f"Fold {fold['fold_id']} leaks the test patient.")
                if set(context_ids).intersection(validation_ids):
                    raise ValueError(f"Fold {fold['fold_id']} overlaps context and validation.")
                if len(context_ids) != len(set(context_ids)):
                    raise ValueError(f"Fold {fold['fold_id']} has duplicate context patients.")
                unknown = (set(context_ids) | set(validation_ids)) - patient_ids
                if unknown:
                    raise ValueError(
                        f"Fold {fold['fold_id']} references unknown patients: {unknown}."
                    )


def selection_for(
    fold: dict[str, Any],
    *,
    k: int,
    seed: int,
) -> dict[str, list[str]]:
    try:
        selection = fold["selections"][str(seed)][str(k)]
    except KeyError as exc:
        raise KeyError(
            f"Fold {fold.get('fold_id')} has no selection for k={k}, seed={seed}."
        ) from exc
    return {
        "context_patient_ids": [str(item) for item in selection["context_patient_ids"]],
        "validation_patient_ids": [
            str(item) for item in selection.get("validation_patient_ids", [])
        ],
    }


def _optional_findings(record: Mapping[str, object]) -> dict[str, int] | None:
    columns_present = [
        column in record and record[column] != "" for column in LABEL_COLUMNS.values()
    ]
    labels_present = [
        label_name in record and record[label_name] != "" for label_name in LABEL_NAMES
    ]
    if all(columns_present) or all(labels_present):
        return normalize_findings(record)
    if any(columns_present) or any(labels_present):
        raise ValueError("Manifest row has partial QRS finding labels.")
    return None


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _patient_sort_key(patient_id: str) -> tuple[int, str]:
    text = str(patient_id)
    return (int(text), text) if text.isdigit() else (10**12, text)


def _patient_record_sort_key(patient: PatientRecord) -> tuple[int, str]:
    return _patient_sort_key(patient.patient_id)


def finding_dict_from_arrays(values: Sequence[int | bool]) -> dict[str, int]:
    if len(values) != len(LABEL_NAMES):
        raise ValueError(f"Expected {len(LABEL_NAMES)} finding values, got {len(values)}.")
    return {
        label_name: int(bool(value))
        for label_name, value in zip(LABEL_NAMES, values, strict=True)
    }
