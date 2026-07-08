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
CLINICAL_LEADS="${CLINICAL_LEADS:-V1,V2,V3}"
CLINICAL_AGGREGATION="${CLINICAL_AGGREGATION:-majority}"

RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_COMPARISONS="${RUN_COMPARISONS:-auto}"
RUN_AUDIT="${RUN_AUDIT:-auto}"
COMPARE_VLM_CONDITION="${COMPARE_VLM_CONDITION:-normal}"
COMPARE_VLM_MODEL="${COMPARE_VLM_MODEL:-}"
AUDIT_K_VALUES="${AUDIT_K_VALUES:-2,4,8,16,32}"

SIMULATOR_DATASET_ROOT="${SIMULATOR_DATASET_ROOT:-$ROOT_DIR/data/simulator_qrs}"
HUCA_DATASET_ROOT="${HUCA_DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"

export MPLBACKEND VLM_RUNTIME VLM_API_BASE VLM_MODELS CONTROL_K_VALUES
export SEEDS CONDITIONS CLINICAL_LEAD CLINICAL_LEADS CLINICAL_AGGREGATION

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

missing_file() {
  for path in "$@"; do
    if [ ! -f "$path" ]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

run_comparison() {
  label="$1"
  cnn_summary="$2"
  vlm_summary="$3"
  output_dir="$4"

  if [ "$RUN_COMPARISONS" = "0" ]; then
    return
  fi
  missing="$(missing_file "$cnn_summary" "$vlm_summary" || true)"
  if [ "$missing" != "" ]; then
    if [ "$RUN_COMPARISONS" = "auto" ]; then
      printf "\n==> Skip %s\nMissing required summary: %s\n" "$label" "$missing"
      return
    fi
    echo "Missing required summary for $label: $missing"
    exit 1
  fi

  run_step "$label" env \
    CNN_SUMMARY="$cnn_summary" \
    VLM_SUMMARY="$vlm_summary" \
    VLM_CONDITION="$COMPARE_VLM_CONDITION" \
    VLM_MODEL="$COMPARE_VLM_MODEL" \
    OUTPUT_DIR="$output_dir" \
    sh "$RUN_DIR/build_loocv_comparison.sh"
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
printf "Clinical leads: %s\n" "$CLINICAL_LEADS"
printf "Clinical aggregation: %s\n" "$CLINICAL_AGGREGATION"
printf "Comparisons: %s\n" "$RUN_COMPARISONS"
printf "Audit: %s\n" "$RUN_AUDIT"

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

run_comparison \
  "Compare CNN vs VLM on simulator QRS" \
  "$ROOT_DIR/reports/loocv/cnn_simulator_qrs/cnn_summary_by_seed.csv" \
  "$ROOT_DIR/reports/loocv/vlm_simulator_qrs/vlm_summary_by_seed.csv" \
  "$ROOT_DIR/reports/loocv/comparison_vlm_simulator_qrs"

run_comparison \
  "Compare CNN vs VLM on HUCA morphology" \
  "$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
  "$ROOT_DIR/reports/loocv/vlm/vlm_summary_by_seed.csv" \
  "$ROOT_DIR/reports/loocv/comparison"

run_comparison \
  "Compare CNN vs VLM on HUCA clinical" \
  "$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
  "$ROOT_DIR/reports/loocv/vlm_real_context/vlm_summary_by_seed.csv" \
  "$ROOT_DIR/reports/loocv/comparison_vlm_real_context"

if [ "$RUN_AUDIT" != "0" ]; then
  missing="$(missing_file \
    "$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
    "$ROOT_DIR/reports/loocv/cnn_simulator_qrs/cnn_summary_by_seed.csv" \
    || true)"
  if [ "$missing" != "" ] && [ "$RUN_AUDIT" = "auto" ]; then
    printf "\n==> Skip Audit LOOCV results\nMissing required CNN summary: %s\n" "$missing"
  else
    if [ "$missing" != "" ]; then
      echo "Missing required CNN summary for audit: $missing"
      exit 1
    fi
  run_step "Audit LOOCV results" env \
    K_VALUES="$AUDIT_K_VALUES" \
    SEEDS="$SEEDS" \
    sh "$RUN_DIR/audit_loocv_results.sh"
  fi
fi

printf "\n[OK] Finished VLM experiment campaign.\n"
