#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
UV="${UV:-uv}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"
CONTEXT_DATASET_ROOT="${CONTEXT_DATASET_ROOT:-$ROOT_DIR/data/simulator_qrs}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports/loocv/vlm}"
K_VALUES="${K_VALUES:-0,2,4,8,16,32}"
CONTROL_K_VALUES="${CONTROL_K_VALUES:-8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
VLM_RUNTIME="${VLM_RUNTIME:-remote_api}"
MODELS="${MODELS:-${VLM_MODELS:-${MODEL:-${VLM_MODEL:-google/gemma-4-E4B-it,google/medgemma-4b-it}}}}"
CONDITIONS="${CONDITIONS:-zero_shot,normal,balanced,permuted,no_support_images}"
API_BASE="${API_BASE:-${VLM_API_BASE:-}}"
OUTPUT="${OUTPUT:-$REPORT_DIR/vlm_setup_validation.json}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

ARGS=""
if [ "$API_BASE" != "" ]; then
  ARGS="$ARGS --api-base $API_BASE"
fi

"$UV" run --no-sync python \
  "$ROOT_DIR/scripts/eval/validate_vlm_loocv.py" \
  --dataset-root "$DATASET_ROOT" \
  --context-dataset-root "$CONTEXT_DATASET_ROOT" \
  --k-values "$K_VALUES" \
  --control-k-values "$CONTROL_K_VALUES" \
  --seeds "$SEEDS" \
  --models "$MODELS" \
  --conditions "$CONDITIONS" \
  --vlm-runtime "$VLM_RUNTIME" \
  --output "$OUTPUT" \
  $ARGS
