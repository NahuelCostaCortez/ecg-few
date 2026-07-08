#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
UV="${UV:-uv}"

CNN_SUMMARY="${CNN_SUMMARY:-$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv}"
VLM_SUMMARY="${VLM_SUMMARY:-$ROOT_DIR/reports/loocv/vlm/vlm_summary_by_seed.csv}"
VLM_CONDITION="${VLM_CONDITION:-estandar}"
VLM_MODEL="${VLM_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/reports/loocv/comparison}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH MPLBACKEND

run_python() {
  if command -v "$UV" >/dev/null 2>&1; then
    "$UV" run --no-sync python "$@"
  else
    python "$@"
  fi
}

run_python \
  "$ROOT_DIR/scripts/eval/compare_loocv_reports.py" \
  --cnn-summary "$CNN_SUMMARY" \
  --vlm-summary "$VLM_SUMMARY" \
  --vlm-condition "$VLM_CONDITION" \
  --vlm-model "$VLM_MODEL" \
  --output-dir "$OUTPUT_DIR"
