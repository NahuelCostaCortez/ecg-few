#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

"$ROOT_DIR/scripts/run/run_proprietary_pattern_detection.sh"
"$ROOT_DIR/scripts/run/run_vllm_pattern_detection.sh"
