#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
RUN_DIR="$ROOT_DIR/scripts/run"

VLM_MODEL="${VLM_MODEL:-google/gemma-4-E4B-it}"
PORT="${PORT:-8000}"
VLM_API_BASE="${VLM_API_BASE:-http://127.0.0.1:${PORT}/v1}"
VLLM_DIR="${VLLM_DIR:-/home/nahuel/vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
START_VLLM="${START_VLLM:-1}"
CNN_CUDA_VISIBLE_DEVICES="${CNN_CUDA_VISIBLE_DEVICES:-}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export VLM_MODEL VLM_API_BASE

VLLM_PID=""

cleanup_vllm() {
  if [ "$VLLM_PID" != "" ]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
  fi
}

run_step() {
  label="$1"
  shift
  printf "\n==> %s\n" "$label"
  "$@"
}

wait_for_vllm() {
  printf "Waiting for vLLM at %s ...\n" "$VLM_API_BASE"
  i=0
  until curl -fsS "$VLM_API_BASE/models" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 120 ]; then
      printf "vLLM did not become ready. Check: %s\n" "$ROOT_DIR/reports/logs/vllm_server.log"
      exit 1
    fi
    sleep 10
  done
}

run_cnn_step() {
  if [ "$CNN_CUDA_VISIBLE_DEVICES" != "" ]; then
    CUDA_VISIBLE_DEVICES="$CNN_CUDA_VISIBLE_DEVICES" DEVICE=cuda "$@"
  else
    DEVICE=auto "$@"
  fi
}

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/reports/logs"

if command -v uv >/dev/null 2>&1; then
  run_step "Sync Python environment" uv sync --extra dev --extra cnn --extra real-data
fi

run_step "Build simulator QRS V1 dataset" sh "$RUN_DIR/build_simulator_qrs_dataset.sh"
run_step "Build HUCA V1 dataset" sh "$RUN_DIR/build_brugada_huca_dataset.sh"

if [ "$START_VLLM" = "1" ]; then
  printf "\n==> Start vLLM server\n"
  (
    cd "$VLLM_DIR"
    if [ -f ".venv/bin/activate" ]; then
      . ".venv/bin/activate"
    fi
    exec vllm serve \
      --model "$VLM_MODEL" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  ) > "$ROOT_DIR/reports/logs/vllm_server.log" 2>&1 &
  VLLM_PID="$!"
  trap cleanup_vllm EXIT INT TERM
fi

wait_for_vllm

run_step "Run VLM/ICL V1 experiments" env \
  VLM_RUNTIME=remote_api \
  VLM_API_BASE="$VLM_API_BASE" \
  VLM_MODELS="$VLM_MODEL" \
  COMPARE_VLM_MODEL="$VLM_MODEL" \
  COMPARE_VLM_CONDITION=estandar \
  CONDITIONS=zero_shot,estandar,balanced \
  CLINICAL_LEADS=V1 \
  CLINICAL_LEAD=V1 \
  RUN_COMPARISONS=0 \
  RUN_AUDIT=0 \
  sh "$RUN_DIR/run_all_vlm_experiments.sh"

if [ "$START_VLLM" = "1" ]; then
  run_step "Stop vLLM server" cleanup_vllm
  trap - EXIT INT TERM
  sleep 15
fi

run_step "Run CNN simulator QRS V1 LOOCV" \
  run_cnn_step env RESUME=0 sh "$RUN_DIR/run_cnn_simulator_qrs_loocv.sh"
run_step "Run CNN HUCA V1 LOOCV" \
  run_cnn_step env RESUME=0 sh "$RUN_DIR/run_cnn_loocv.sh"
run_step "Run CNN SSL domain adaptation" \
  run_cnn_step env METHOD=ssl RESUME=0 sh "$RUN_DIR/run_cnn_domain_adaptation_loocv.sh"
run_step "Run CNN CORAL domain adaptation" \
  run_cnn_step env METHOD=coral RESUME=0 sh "$RUN_DIR/run_cnn_domain_adaptation_loocv.sh"
run_step "Run CNN MMD domain adaptation" \
  run_cnn_step env METHOD=mmd RESUME=0 sh "$RUN_DIR/run_cnn_domain_adaptation_loocv.sh"
run_step "Run CNN DANN domain adaptation" \
  run_cnn_step env METHOD=dann RESUME=0 sh "$RUN_DIR/run_cnn_domain_adaptation_loocv.sh"

run_step "Build TFG reports and figures" env \
  STRICT=1 \
  VLM_MODEL="$VLM_MODEL" \
  RUN_COMPARISONS=1 \
  RUN_RENDERERS=1 \
  sh "$RUN_DIR/build_tfg_report_figures.sh"

printf "\n[OK] TFG end-to-end pipeline finished.\n"
printf "Reports: %s\n" "$ROOT_DIR/reports/loocv"
printf "TFG figures: %s\n" "$ROOT_DIR/reports/tfg_figures"
printf "vLLM log: %s\n" "$ROOT_DIR/reports/logs/vllm_server.log"
