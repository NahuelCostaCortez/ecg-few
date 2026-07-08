import argparse
import importlib.util
from pathlib import Path

from ecg_few.loocv import BrugadaImageRow, build_fold_plan

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


def morphology_row(patient_id: str, label: int) -> BrugadaImageRow:
    value = int(label)
    return BrugadaImageRow(
        image_path=f"{patient_id}_V1.png",
        patient_id=patient_id,
        lead="V1",
        source_family="sim" if patient_id.startswith("SIM") else "huca",
        label_rbbb=value,
        label_st_elevation=value,
        label_t_wave_inversion=value,
        clinical_brugada=None,
        basal_pattern=0,
        sudden_death=0,
        sample_index=0,
        aggregation_group_id=patient_id,
    )


def test_clinical_fold_uses_v1_only() -> None:
    runner = load_runner()
    args = argparse.Namespace(dry_run_predictions="expected")
    rows = [row("1", "V1", 1, qrs_labels=True)]

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
        clinical_leads=["V1"],
        clinical_aggregation="majority",
        started=0.0,
        selection={"context_patient_ids": [], "validation_patient_ids": []},
        context_ids=[],
        args=args,
        generator=None,
    )

    assert record["pred_label"] == 1
    assert record["valid_leads"] == 1
    assert record["invalid_leads"] == 0
    assert record["clinical_leads"] == "V1"


def test_clinical_estandar_uses_fold_selection_and_balanced_uses_stratified() -> None:
    runner = load_runner()
    rows = [
        row("1", lead, 1)
        for lead in ("V1",)
    ] + [
        row("2", lead, 1)
        for lead in ("V1",)
    ] + [
        row("3", lead, 0)
        for lead in ("V1",)
    ] + [
        row("4", lead, 1)
        for lead in ("V1",)
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

    estandar = runner.selection_for_condition(
        fold=fold,
        rows=rows,
        context_pool_rows=rows,
        test_patient_id="1",
        k=2,
        seed=42,
        fold_id=0,
        task=runner.CLINICAL_TASK,
        condition=runner.ESTANDAR_CONDITION,
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

    assert estandar["context_patient_ids"] == ["2", "4"]
    assert {
        next(item for item in rows if item.patient_id == patient_id).reference_brugada
        for patient_id in balanced["context_patient_ids"]
    } == {0, 1}


def test_same_context_root_is_treated_as_internal(tmp_path: Path) -> None:
    runner = load_runner()
    dataset_root = tmp_path.resolve()

    context_root, has_context_dataset = runner.resolve_context_dataset(
        dataset_root,
        dataset_root,
    )

    assert context_root == dataset_root
    assert has_context_dataset is False


def test_internal_morphology_context_never_uses_test_patient(tmp_path: Path) -> None:
    runner = load_runner()
    rows = [
        morphology_row(str(patient_id), 1 if patient_id <= 3 else 0)
        for patient_id in range(1, 9)
    ]
    folds = build_fold_plan(rows, k_values=[2, 4], seeds=[42], val_per_class=0)
    args = argparse.Namespace(resume=False, dry_run_predictions="expected")

    for condition in (runner.ESTANDAR_CONDITION, runner.BALANCED_CONDITION):
        for k in (2, 4):
            predictions = runner.run_k_seed(
                rows=rows,
                context_pool_rows=rows,
                folds=folds,
                dataset_root=ROOT,
                context_dataset_root=ROOT,
                run_dir=tmp_path / condition / f"k{k}",
                k=k,
                seed=42,
                task=runner.MORPHOLOGY_TASK,
                condition=condition,
                runtime=runner.REMOTE_RUNTIME,
                model="google/gemma-4-E4B-it",
                api_base=None,
                api_key="",
                system_prompt="",
                prompt_template="lead {lead}",
                args=args,
                generator=None,
            )

            for prediction in predictions:
                context_ids = [
                    item
                    for item in str(prediction["context_patient_ids"]).split("|")
                    if item
                ]
                assert str(prediction["test_patient_id"]) not in context_ids


def test_external_morphology_controls_use_context_dataset_ids(tmp_path: Path) -> None:
    runner = load_runner()
    rows = [morphology_row("1", 1), morphology_row("2", 0), morphology_row("3", 1)]
    context_rows = [
        morphology_row("SIM001", 0),
        morphology_row("SIM002", 1),
        morphology_row("SIM003", 0),
        morphology_row("SIM004", 1),
    ]
    fold = {
        "fold_id": 0,
        "test_patient_id": "1",
        "selections": {
            "42": {
                "2": {
                    "context_patient_ids": ["2", "3"],
                    "validation_patient_ids": [],
                }
            }
        },
    }
    args = argparse.Namespace(resume=False, dry_run_predictions="expected")

    for condition in (runner.PERMUTED_CONDITION, runner.NO_SUPPORT_IMAGES_CONDITION):
        predictions = runner.run_k_seed(
            rows=rows,
            context_pool_rows=context_rows,
            folds=[fold],
            dataset_root=ROOT,
            context_dataset_root=ROOT,
            run_dir=tmp_path / condition,
            k=2,
            seed=42,
            task=runner.MORPHOLOGY_TASK,
            condition=condition,
            runtime=runner.REMOTE_RUNTIME,
            model="google/gemma-4-E4B-it",
            api_base=None,
            api_key="",
            system_prompt="",
            prompt_template="lead {lead}",
            args=args,
            generator=None,
        )

        context_ids = predictions[0]["context_patient_ids"].split("|")
        assert context_ids
        assert all(patient_id.startswith("SIM") for patient_id in context_ids)
