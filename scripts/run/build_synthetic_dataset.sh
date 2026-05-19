#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export MPLBACKEND="Agg"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
mkdir -p "$MPLCONFIGDIR"

run_python() {
  if UV_CACHE_DIR="$ROOT_DIR/.uv-cache" uv run --no-sync python "$@"; then
    return 0
  fi

  PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
}

OUTDIR="$ROOT_DIR/data/synthetic"
SAMPLES_PER_FAMILY=40
TEST_RATIO=0.2
VAL_RATIO=0.1
FS=500
SEED=42
DPI=112

run_python "$ROOT_DIR/scripts/dataset/build_ecg_dataset.py" \
  --outdir "$OUTDIR" \
  --samples-per-family "$SAMPLES_PER_FAMILY" \
  --test-ratio "$TEST_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --fs "$FS" \
  --seed "$SEED" \
  --dpi "$DPI" \
  --tiny-poc \
  --component-dev-count 20 \
  --component-test-count 20 \
  --transfer-count 40 \
  --transfer-source-family BRUGADA