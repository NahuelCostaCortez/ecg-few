#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

VLLM_HOST="${VLLM_HOST:-slave2.pir.uo}"
API_BASE="${API_BASE:-http://$VLLM_HOST:8000/v1}"
MODEL="${MODEL:-open-source-vlm}"
K_VALUES="${K_VALUES:-0,1,2,4,8,12,16,24,32}"
SEEDS="${SEEDS:-42,123,2026}"
EXPERIMENTS="${EXPERIMENTS:-multilabel_label_only,multilabel_morphology_described,binary_morphology_described}"
FEW_SHOT_CONTROLS="${FEW_SHOT_CONTROLS:-normal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/vlm_outputs/vllm/pattern_detection}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports/pattern_detection/vllm}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-json_schema}"
LIMIT="${LIMIT:-0}"
RESUME="${RESUME:-1}"

ARGS=""
if [ "$LIMIT" != "0" ]; then
  ARGS="$ARGS --limit $LIMIT"
fi
if [ "$RESUME" = "0" ]; then
  ARGS="$ARGS --no-resume"
fi

MODELS_URL="${API_BASE%/}/models"
if ! curl -fsS -m 5 "$MODELS_URL" >/dev/null; then
  echo "Could not reach the vLLM server at $API_BASE." >&2
  echo "Set API_BASE=http://<host>:8000/v1 if the server is running on another machine." >&2
  exit 1
fi

UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync python \
  "$ROOT_DIR/scripts/eval/run_pattern_detection_experiments.py" \
  --provider vllm \
  --api-base "$API_BASE" \
  --model "$MODEL" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --few-shot-controls "$FEW_SHOT_CONTROLS" \
  --experiments "$EXPERIMENTS" \
  --output-root "$OUTPUT_ROOT" \
  --report-dir "$REPORT_DIR" \
  --response-format "$RESPONSE_FORMAT" \
  $ARGS
