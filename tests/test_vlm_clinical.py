import argparse
import importlib.util
from pathlib import Path

from ecg_few.loocv import BrugadaImageRow

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "eval" / "run_vlm_loocv.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_vlm_loocv", RUNNER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(
    patient_id: str,
    lead: str,
    clinical_brugada: int,
    *,
    qrs_labels: bool = False,
) -> BrugadaImageRow:
    return BrugadaImageRow(
        image_path=f"{patient_id}_{lead}.png",
        patient_id=patient_id,
        lead=lead,
        source_family="huca",
        label_rbbb=1 if qrs_labels else None,
        label_st_elevation=1 if qrs_labels else None,
        label_t_wave_inversion=1 if qrs_labels else None,
        clinical_brugada=clinical_brugada,
        basal_pattern=0,
        sudden_death=0,
        sample_index=0,
        aggregation_group_id=patient_id,
    )


def test_clinical_fold_aggregates_requested_leads() -> None:
    runner = load_runner()
    args = argparse.Namespace(dry_run_predictions="expected")
    rows = [row("1", lead, 1, qrs_labels=True) for lead in ("V1", "V2", "V3")]

    record = runner.run_clinical_fold(
        rows=rows,
        context_rows=[],
        dataset_root=ROOT,
        context_dataset_root=ROOT,
        test_patient_id="1",
        fold_id=0,
        k=0,
        seed=42,
        key="zero_shot:0:1:0:42",
        condition=runner.ZERO_SHOT_CONDITION,
        runtime=runner.REMOTE_RUNTIME,
        model="google/gemma-4-E4B-it",
        api_base=None,
        api_key="",
        system_prompt="",
        prompt_template="lead {lead}",
        include_support_images=True,
        clinical_leads=["V1", "V2", "V3"],
        clinical_aggregation="majority",
        started=0.0,
        selection={"context_patient_ids": [], "validation_patient_ids": []},
        context_ids=[],
        args=args,
        generator=None,
    )

    assert record["pred_label"] == 1
    assert record["valid_leads"] == 3
    assert record["invalid_leads"] == 0
    assert record["clinical_leads"] == "V1|V2|V3"


def test_clinical_normal_uses_fold_selection_and_balanced_uses_stratified() -> None:
    runner = load_runner()
    rows = [
        row("1", lead, 1)
        for lead in ("V1", "V2", "V3")
    ] + [
        row("2", lead, 1)
        for lead in ("V1", "V2", "V3")
    ] + [
        row("3", lead, 0)
        for lead in ("V1", "V2", "V3")
    ] + [
        row("4", lead, 1)
        for lead in ("V1", "V2", "V3")
    ]
    fold = {
        "fold_id": 0,
        "test_patient_id": "1",
        "selections": {
            "42": {
                "2": {
                    "context_patient_ids": ["2", "4"],
                    "validation_patient_ids": [],
                }
            }
        },
    }

    normal = runner.selection_for_condition(
        fold=fold,
        rows=rows,
        context_pool_rows=rows,
        test_patient_id="1",
        k=2,
        seed=42,
        fold_id=0,
        task=runner.CLINICAL_TASK,
        condition=runner.NORMAL_CONDITION,
    )
    balanced = runner.selection_for_condition(
        fold=fold,
        rows=rows,
        context_pool_rows=rows,
        test_patient_id="1",
        k=2,
        seed=42,
        fold_id=0,
        task=runner.CLINICAL_TASK,
        condition=runner.BALANCED_CONDITION,
    )

    assert normal["context_patient_ids"] == ["2", "4"]
    assert {
        next(item for item in rows if item.patient_id == patient_id).reference_brugada
        for patient_id in balanced["context_patient_ids"]
    } == {0, 1}
