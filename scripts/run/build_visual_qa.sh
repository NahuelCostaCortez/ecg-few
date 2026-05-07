#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_ROOT/vlm}"
SEED="${SEED:-42}"

UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync python \
  "$ROOT_DIR/scripts/dataset/build_vlm_instruction_dataset.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED"
