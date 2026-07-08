#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
UV="${UV:-uv}"

STRICT="${STRICT:-1}"
RUN_COMPARISONS="${RUN_COMPARISONS:-1}"
RUN_RENDERERS="${RUN_RENDERERS:-1}"
REPORT_FIGURES_DIR="${REPORT_FIGURES_DIR:-$ROOT_DIR/reports/tfg_figures}"
VLM_MODEL="${VLM_MODEL:-google/gemma-4-E4B-it}"

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

require_files() {
  label="$1"
  shift
  missing=""
  for path in "$@"; do
    if [ ! -f "$path" ]; then
      missing="$missing
$path"
    fi
  done
  if [ "$missing" != "" ]; then
    printf "\n[ERROR] Missing inputs for %s:%s\n" "$label" "$missing"
    if [ "$STRICT" = "1" ]; then
      exit 1
    fi
    return 1
  fi
  return 0
}

run_step() {
  label="$1"
  shift
  printf "\n==> %s\n" "$label"
  "$@"
}

if [ "$RUN_COMPARISONS" = "1" ]; then
  if require_files "CNN sim-vs-real comparison" \
    "$ROOT_DIR/reports/loocv/cnn_simulator_qrs/cnn_summary_by_k.csv" \
    "$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_k.csv"; then
    run_step "Build CNN sim-vs-real report plots" run_python \
      "$ROOT_DIR/scripts/eval/compare_cnn_sim_real_reports.py" \
      --output-dir "$ROOT_DIR/reports/loocv/cnn_comparison"
  fi

  if require_files "CNN domain-adaptation comparison" \
    "$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_k.csv" \
    "$ROOT_DIR/reports/loocv/cnn_domain_adaptation/ssl/cnn_summary_by_k.csv" \
    "$ROOT_DIR/reports/loocv/cnn_domain_adaptation/coral/cnn_summary_by_k.csv" \
    "$ROOT_DIR/reports/loocv/cnn_domain_adaptation/mmd/cnn_summary_by_k.csv" \
    "$ROOT_DIR/reports/loocv/cnn_domain_adaptation/dann/cnn_summary_by_k.csv"; then
    run_step "Build CNN domain-adaptation report plots" run_python \
      "$ROOT_DIR/scripts/eval/compare_cnn_domain_adaptation_reports.py" \
      --output-dir "$ROOT_DIR/reports/loocv/cnn_domain_adaptation/comparison"
  fi

  if require_files "CNN vs VLM simulator comparison" \
    "$ROOT_DIR/reports/loocv/cnn_simulator_qrs/cnn_summary_by_seed.csv" \
    "$ROOT_DIR/reports/loocv/vlm_simulator_qrs/vlm_summary_by_seed.csv"; then
    run_step "Build CNN vs VLM simulator report plots" env \
      CNN_SUMMARY="$ROOT_DIR/reports/loocv/cnn_simulator_qrs/cnn_summary_by_seed.csv" \
      VLM_SUMMARY="$ROOT_DIR/reports/loocv/vlm_simulator_qrs/vlm_summary_by_seed.csv" \
      VLM_CONDITION=estandar \
      VLM_MODEL="$VLM_MODEL" \
      OUTPUT_DIR="$ROOT_DIR/reports/loocv/comparison_vlm_simulator_qrs" \
      sh "$ROOT_DIR/scripts/run/build_loocv_comparison.sh"
  fi

  if require_files "CNN vs VLM HUCA morphology comparison" \
    "$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
    "$ROOT_DIR/reports/loocv/vlm/vlm_summary_by_seed.csv"; then
    run_step "Build CNN vs VLM HUCA morphology report plots" env \
      CNN_SUMMARY="$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
      VLM_SUMMARY="$ROOT_DIR/reports/loocv/vlm/vlm_summary_by_seed.csv" \
      VLM_CONDITION=estandar \
      VLM_MODEL="$VLM_MODEL" \
      OUTPUT_DIR="$ROOT_DIR/reports/loocv/comparison" \
      sh "$ROOT_DIR/scripts/run/build_loocv_comparison.sh"
  fi

  if require_files "CNN vs VLM HUCA clinical comparison" \
    "$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
    "$ROOT_DIR/reports/loocv/vlm_real_context/vlm_summary_by_seed.csv"; then
    run_step "Build CNN vs VLM HUCA clinical report plots" env \
      CNN_SUMMARY="$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_seed.csv" \
      VLM_SUMMARY="$ROOT_DIR/reports/loocv/vlm_real_context/vlm_summary_by_seed.csv" \
      VLM_CONDITION=estandar \
      VLM_MODEL="$VLM_MODEL" \
      OUTPUT_DIR="$ROOT_DIR/reports/loocv/comparison_vlm_real_context" \
      sh "$ROOT_DIR/scripts/run/build_loocv_comparison.sh"
  fi
fi

if [ "$RUN_RENDERERS" = "1" ]; then
  if require_files "CNN paper-style thesis renderer" \
    "$ROOT_DIR/reports/loocv/cnn_simulator_qrs/cnn_summary_by_k.csv" \
    "$ROOT_DIR/reports/loocv/cnn/cnn_summary_by_k.csv" \
    "$ROOT_DIR/reports/loocv/cnn_comparison/cnn_simulated_vs_real_by_k.csv" \
    "$ROOT_DIR/reports/loocv/cnn_domain_adaptation/comparison/cnn_domain_adaptation_by_k.csv"; then
    run_step "Render CNN paper-style thesis figures" run_python \
      "$ROOT_DIR/scripts/thesis/render_paper_style_result_figures.py"
  fi

  if require_files "VLM/ICL condition thesis renderer" \
    "$ROOT_DIR/reports/loocv/vlm_simulator_qrs/vlm_summary_by_k.csv" \
    "$ROOT_DIR/reports/loocv/vlm_real_context/vlm_summary_by_k.csv"; then
    run_step "Render VLM/ICL condition thesis figures" run_python \
      "$ROOT_DIR/scripts/thesis/render_vlm_icl_condition_figures.py" \
      --results-dir "$ROOT_DIR"
  fi

  if require_files "comparative thesis renderer" \
    "$ROOT_DIR/reports/loocv/vlm_simulator_qrs/vlm_summary_by_model_condition_k.csv" \
    "$ROOT_DIR/reports/loocv/vlm_real_context/vlm_summary_by_model_condition_k.csv" \
    "$ROOT_DIR/reports/loocv/vlm/vlm_summary_by_model_condition_k.csv"; then
    run_step "Render comparative thesis figures" run_python \
      "$ROOT_DIR/scripts/thesis/render_comparative_result_figures.py" \
      --results-dir "$ROOT_DIR" \
      --model "$VLM_MODEL"
  fi
fi

run_step "Collect TFG-used figures into reports" env \
  ROOT_DIR="$ROOT_DIR" \
  THESIS_ROOT="$ROOT_DIR/thesis/thesis" \
  REPORT_FIGURES_DIR="$REPORT_FIGURES_DIR" \
  STRICT="$STRICT" \
  sh "$ROOT_DIR/scripts/run/collect_tfg_figures.sh"

printf "\n[OK] TFG report figures are under: %s\n" "$REPORT_FIGURES_DIR"
