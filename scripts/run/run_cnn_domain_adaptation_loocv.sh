#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

METHOD="${METHOD:-coral}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-$ROOT_DIR/data/simulator_qrs}"
TARGET_DATASET_ROOT="${TARGET_DATASET_ROOT:-$DATASET_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/cnn_domain_adaptation/$METHOD}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports/loocv/cnn_domain_adaptation/$METHOD}"
K_VALUES="${K_VALUES:-16,32}"
SEEDS="${SEEDS:-42,123,2026}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
RESNET_WEIGHTS="${RESNET_WEIGHTS:-default}"
DOMAIN_ADAPTATION_WEIGHT="${DOMAIN_ADAPTATION_WEIGHT:-0.1}"
MMD_KERNEL_SCALES="${MMD_KERNEL_SCALES:-0.5,1.0,2.0,4.0}"
SSL_PRETRAIN_EPOCHS="${SSL_PRETRAIN_EPOCHS:-0}"
SSL_PRETRAIN_LR="${SSL_PRETRAIN_LR:-0.0001}"
SSL_PROJECTION_DIM="${SSL_PROJECTION_DIM:-128}"
SSL_TEMPERATURE="${SSL_TEMPERATURE:-0.2}"
GRADCAM_COUNT="${GRADCAM_COUNT:-6}"
LIMIT_FOLDS="${LIMIT_FOLDS:-0}"
RESUME="${RESUME:-0}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH MPLBACKEND

ARGS=""
if [ "$LIMIT_FOLDS" != "0" ]; then
  ARGS="$ARGS --limit-folds $LIMIT_FOLDS"
fi
if [ "$RESUME" = "0" ]; then
  ARGS="$ARGS --no-resume"
fi

uv run --no-sync python \
  "$ROOT_DIR/scripts/eval/run_cnn_loocv.py" \
  --dataset-root "$DATASET_ROOT" \
  --train-dataset-root "$TRAIN_DATASET_ROOT" \
  --target-dataset-root "$TARGET_DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --report-dir "$REPORT_DIR" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --image-size "$IMAGE_SIZE" \
  --resnet-weights "$RESNET_WEIGHTS" \
  --domain-adaptation "$METHOD" \
  --domain-adaptation-weight "$DOMAIN_ADAPTATION_WEIGHT" \
  --mmd-kernel-scales "$MMD_KERNEL_SCALES" \
  --ssl-pretrain-epochs "$SSL_PRETRAIN_EPOCHS" \
  --ssl-pretrain-lr "$SSL_PRETRAIN_LR" \
  --ssl-projection-dim "$SSL_PROJECTION_DIM" \
  --ssl-temperature "$SSL_TEMPERATURE" \
  --gradcam-count "$GRADCAM_COUNT" \
  $ARGS
