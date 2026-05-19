#!/usr/bin/env bash
set -eu

SLRCORES=4

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT_DIR" ] || [ ! -f "$ROOT_DIR/scripts/run/run_vllm_pattern_detection.sh" ]; then
  ROOT_DIR="$(git -C "${PWD:-.}" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$ROOT_DIR" ] || [ ! -f "$ROOT_DIR/scripts/run/run_vllm_pattern_detection.sh" ]; then
  ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
fi
if [ ! -f "$ROOT_DIR/scripts/run/run_vllm_pattern_detection.sh" ]; then
  echo "Could not locate project root containing scripts/run/run_vllm_pattern_detection.sh." >&2
  echo "Submit this job from the ecg-vlm repository root or set SLURM_SUBMIT_DIR accordingly." >&2
  exit 1
fi

MODEL="${MODEL:-google/gemma-4-E4B-it}" \
K_VALUES="${K_VALUES:-12,24,32}" \
EXPERIMENTS="${EXPERIMENTS:-multilabel_label_only,binary_morphology_described}" \
FEW_SHOT_CONTROLS="${FEW_SHOT_CONTROLS:-shuffled_answers,text_only_examples}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/vlm_outputs/vllm/pattern_detection}" \
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports/pattern_detection/vllm_icl_controls}" \
"$ROOT_DIR/scripts/run/run_vllm_pattern_detection.sh"
