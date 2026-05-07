#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
MODEL="${MODEL:-gpt-4.5-preview}"
K_VALUES="${K_VALUES:-0,1,2,4,8,12}"
SEEDS="${SEEDS:-42,123,2026}"
EXPERIMENTS="${EXPERIMENTS:-multilabel_label_only,multilabel_morphology_described,binary_morphology_described}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/data/vlm_outputs/openai/pattern_detection}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports/pattern_detection/openai}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=""
if [ "$DRY_RUN" = "1" ]; then
  ARGS="$ARGS --dry-run"
fi
if [ "$LIMIT" != "0" ]; then
  ARGS="$ARGS --limit $LIMIT"
fi

UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync python \
  "$ROOT_DIR/scripts/eval/run_pattern_detection_experiments.py" \
  --provider openai \
  --model "$MODEL" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --experiments "$EXPERIMENTS" \
  --output-root "$OUTPUT_ROOT" \
  --report-dir "$REPORT_DIR" \
  --reasoning-effort "$REASONING_EFFORT" \
  $ARGS
