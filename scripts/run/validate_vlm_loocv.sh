#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"
CONTEXT_DATASET_ROOT="${CONTEXT_DATASET_ROOT:-$ROOT_DIR/data/simulator_qrs}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports/loocv/vlm}"
K_VALUES="${K_VALUES:-2,4,8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
VLM_RUNTIME="${VLM_RUNTIME:-remote_api}"
MODEL="${MODEL:-${VLM_MODEL:-google/medgemma-4b-it}}"
API_BASE="${API_BASE:-${VLM_API_BASE:-}}"
OUTPUT="${OUTPUT:-$REPORT_DIR/vlm_setup_validation.json}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

ARGS=""
if [ "$API_BASE" != "" ]; then
  ARGS="$ARGS --api-base $API_BASE"
fi

uv run --no-sync python \
  "$ROOT_DIR/scripts/eval/validate_vlm_loocv.py" \
  --dataset-root "$DATASET_ROOT" \
  --context-dataset-root "$CONTEXT_DATASET_ROOT" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --vlm-runtime "$VLM_RUNTIME" \
  --model "$MODEL" \
  --output "$OUTPUT" \
  $ARGS
