#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

RAW_ROOT="${RAW_ROOT:-$ROOT_DIR/data/raw/brugada-huca/1.0.0}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"
CNN_REPORT_DIR="${CNN_REPORT_DIR:-$ROOT_DIR/reports/loocv/cnn}"
VLM_REPORT_DIR="${VLM_REPORT_DIR:-$ROOT_DIR/reports/loocv/vlm}"
COMPARISON_DIR="${COMPARISON_DIR:-$ROOT_DIR/reports/loocv/comparison}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/reports/loocv/audit}"
K_VALUES="${K_VALUES:-2,4,8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

uv run --no-sync python \
  "$ROOT_DIR/scripts/eval/audit_loocv_results.py" \
  --raw-root "$RAW_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --cnn-report-dir "$CNN_REPORT_DIR" \
  --vlm-report-dir "$VLM_REPORT_DIR" \
  --comparison-dir "$COMPARISON_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS"
