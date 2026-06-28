#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
UV="${UV:-uv}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"
CONTEXT_DATASET_ROOT="${CONTEXT_DATASET_ROOT:-$ROOT_DIR/data/simulator_qrs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/vlm_loocv}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports/loocv/vlm}"
K_VALUES="${K_VALUES:-0,2,4,8,16,32}"
CONTROL_K_VALUES="${CONTROL_K_VALUES:-8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
VLM_RUNTIME="${VLM_RUNTIME:-remote_api}"
MODELS="${MODELS:-${VLM_MODELS:-${MODEL:-${VLM_MODEL:-google/gemma-4-E4B-it,google/medgemma-4b-it}}}}"
CONDITIONS="${CONDITIONS:-zero_shot,normal,balanced,permuted,no_support_images}"
API_BASE="${API_BASE:-${VLM_API_BASE:-}}"
LIMIT_FOLDS="${LIMIT_FOLDS:-0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-96}"
DRY_RUN_PREDICTIONS="${DRY_RUN_PREDICTIONS:-none}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH MPLBACKEND

ARGS=""
if [ "$LIMIT_FOLDS" != "0" ]; then
  ARGS="$ARGS --limit-folds $LIMIT_FOLDS"
fi
if [ "$API_BASE" != "" ]; then
  ARGS="$ARGS --api-base $API_BASE"
fi
if [ "$DRY_RUN_PREDICTIONS" != "none" ]; then
  ARGS="$ARGS --dry-run-predictions $DRY_RUN_PREDICTIONS"
fi

"$UV" run --no-sync python \
  "$ROOT_DIR/scripts/eval/run_vlm_loocv.py" \
  --dataset-root "$DATASET_ROOT" \
  --context-dataset-root "$CONTEXT_DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --report-dir "$REPORT_DIR" \
  --k-values "$K_VALUES" \
  --control-k-values "$CONTROL_K_VALUES" \
  --seeds "$SEEDS" \
  --models "$MODELS" \
  --conditions "$CONDITIONS" \
  --vlm-runtime "$VLM_RUNTIME" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  $ARGS
