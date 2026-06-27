#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

CNN_SUMMARY="${CNN_SUMMARY:-$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv}"
VLM_SUMMARY="${VLM_SUMMARY:-$ROOT_DIR/reports/loocv/vlm/vlm_summary_by_seed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/reports/loocv/comparison}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH MPLBACKEND

uv run --no-sync python \
  "$ROOT_DIR/scripts/eval/compare_loocv_reports.py" \
  --cnn-summary "$CNN_SUMMARY" \
  --vlm-summary "$VLM_SUMMARY" \
  --output-dir "$OUTPUT_DIR"
