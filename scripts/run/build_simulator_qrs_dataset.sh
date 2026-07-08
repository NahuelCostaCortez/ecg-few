#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
UV="${UV:-uv}"

OUTDIR="${OUTDIR:-$ROOT_DIR/data/simulator_qrs}"
PATIENTS_PER_SOURCE_FAMILY="${PATIENTS_PER_SOURCE_FAMILY:-20}"
SEED="${SEED:-2026}"
K_VALUES="${K_VALUES:-2,4,8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
VAL_PER_CLASS="${VAL_PER_CLASS:-4}"

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
  "$ROOT_DIR/scripts/dataset/build_simulator_qrs_dataset.py" \
  --outdir "$OUTDIR" \
  --patients-per-source-family "$PATIENTS_PER_SOURCE_FAMILY" \
  --seed "$SEED" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --val-per-class "$VAL_PER_CLASS"
