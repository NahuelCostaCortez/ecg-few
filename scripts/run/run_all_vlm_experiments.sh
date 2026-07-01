#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
RUN_DIR="$ROOT_DIR/scripts/run"

MPLBACKEND="${MPLBACKEND:-Agg}"
VLM_RUNTIME="${VLM_RUNTIME:-remote_api}"
VLM_API_BASE="${VLM_API_BASE:-}"
VLM_MODELS="${VLM_MODELS:-google/gemma-4-E4B-it,google/medgemma-4b-it}"
VLM_K_VALUES="${VLM_K_VALUES:-0,2,4,8,16,32}"
CONTROL_K_VALUES="${CONTROL_K_VALUES:-8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
CONDITIONS="${CONDITIONS:-zero_shot,normal,balanced,permuted,no_support_images}"
CLINICAL_LEAD="${CLINICAL_LEAD:-V2}"

RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_COMPARISONS="${RUN_COMPARISONS:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
COMPARE_VLM_CONDITION="${COMPARE_VLM_CONDITION:-normal}"
COMPARE_VLM_MODEL="${COMPARE_VLM_MODEL:-}"
AUDIT_K_VALUES="${AUDIT_K_VALUES:-2,4,8,16,32}"

SIMULATOR_DATASET_ROOT="${SIMULATOR_DATASET_ROOT:-$ROOT_DIR/data/simulator_qrs}"
HUCA_DATASET_ROOT="${HUCA_DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"

export MPLBACKEND VLM_RUNTIME VLM_API_BASE VLM_MODELS CONTROL_K_VALUES
export SEEDS CONDITIONS CLINICAL_LEAD

if [ "$VLM_RUNTIME" = "remote_api" ] && [ "$VLM_API_BASE" = "" ]; then
  echo "Set VLM_API_BASE before running remote_api experiments."
  echo "Example: VLM_API_BASE=http://your-host:8000/v1 sh scripts/run/run_all_vlm_experiments.sh"
  exit 2
fi

run_step() {
  label="$1"
  shift
  printf "\n==> %s\n" "$label"
  "$@"
}

run_vlm_campaign() {
  campaign_name="$1"
  campaign_task="$2"
  campaign_dataset_root="$3"
  campaign_context_root="$4"
  campaign_output_root="$5"
  campaign_report_dir="$6"

  if [ "$RUN_VALIDATE" = "1" ]; then
    run_step "Validate $campaign_name" env \
      TASK="$campaign_task" \
      DATASET_ROOT="$campaign_dataset_root" \
      CONTEXT_DATASET_ROOT="$campaign_context_root" \
      REPORT_DIR="$campaign_report_dir" \
      OUTPUT="$campaign_report_dir/vlm_setup_validation.json" \
      K_VALUES="$VLM_K_VALUES" \
      sh "$RUN_DIR/validate_vlm_loocv.sh"
  fi

  run_step "Run $campaign_name" env \
    TASK="$campaign_task" \
    DATASET_ROOT="$campaign_dataset_root" \
    CONTEXT_DATASET_ROOT="$campaign_context_root" \
    OUTPUT_ROOT="$campaign_output_root" \
    REPORT_DIR="$campaign_report_dir" \
    K_VALUES="$VLM_K_VALUES" \
    sh "$RUN_DIR/run_vlm_loocv.sh"
}

printf "VLM runtime: %s\n" "$VLM_RUNTIME"
printf "VLM models: %s\n" "$VLM_MODELS"
printf "VLM k values: %s\n" "$VLM_K_VALUES"
printf "Seeds: %s\n" "$SEEDS"
printf "Conditions: %s\n" "$CONDITIONS"

run_vlm_campaign \
  "VLM simulator QRS morphology LOOCV" \
  "morphology" \
  "$SIMULATOR_DATASET_ROOT" \
  "$SIMULATOR_DATASET_ROOT" \
  "$ROOT_DIR/outputs/vlm_simulator_qrs_loocv" \
  "$ROOT_DIR/reports/loocv/vlm_simulator_qrs"

run_vlm_campaign \
  "VLM HUCA morphology LOOCV with simulator context" \
  "morphology" \
  "$HUCA_DATASET_ROOT" \
  "$SIMULATOR_DATASET_ROOT" \
  "$ROOT_DIR/outputs/vlm_loocv" \
  "$ROOT_DIR/reports/loocv/vlm"

run_vlm_campaign \
  "VLM HUCA clinical LOOCV with real context" \
  "clinical" \
  "$HUCA_DATASET_ROOT" \
  "" \
  "$ROOT_DIR/outputs/vlm_real_context_loocv" \
  "$ROOT_DIR/reports/loocv/vlm_real_context"

if [ "$RUN_COMPARISONS" = "1" ]; then
  run_step "Compare CNN vs VLM on simulator QRS" env \
    CNN_SUMMARY="$ROOT_DIR/reports/loocv/cnn_simulator_qrs/cnn_summary_by_seed.csv" \
    VLM_SUMMARY="$ROOT_DIR/reports/loocv/vlm_simulator_qrs/vlm_summary_by_seed.csv" \
    VLM_CONDITION="$COMPARE_VLM_CONDITION" \
    VLM_MODEL="$COMPARE_VLM_MODEL" \
    OUTPUT_DIR="$ROOT_DIR/reports/loocv/comparison_vlm_simulator_qrs" \
    sh "$RUN_DIR/build_loocv_comparison.sh"

  run_step "Compare CNN vs VLM on HUCA morphology" env \
    CNN_SUMMARY="$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
    VLM_SUMMARY="$ROOT_DIR/reports/loocv/vlm/vlm_summary_by_seed.csv" \
    VLM_CONDITION="$COMPARE_VLM_CONDITION" \
    VLM_MODEL="$COMPARE_VLM_MODEL" \
    OUTPUT_DIR="$ROOT_DIR/reports/loocv/comparison" \
    sh "$RUN_DIR/build_loocv_comparison.sh"

  run_step "Compare CNN vs VLM on HUCA clinical" env \
    CNN_SUMMARY="$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
    VLM_SUMMARY="$ROOT_DIR/reports/loocv/vlm_real_context/vlm_summary_by_seed.csv" \
    VLM_CONDITION="$COMPARE_VLM_CONDITION" \
    VLM_MODEL="$COMPARE_VLM_MODEL" \
    OUTPUT_DIR="$ROOT_DIR/reports/loocv/comparison_vlm_real_context" \
    sh "$RUN_DIR/build_loocv_comparison.sh"
fi

if [ "$RUN_AUDIT" = "1" ]; then
  run_step "Audit LOOCV results" env \
    K_VALUES="$AUDIT_K_VALUES" \
    SEEDS="$SEEDS" \
    sh "$RUN_DIR/audit_loocv_results.sh"
fi

printf "\n[OK] Finished VLM experiment campaign.\n"
