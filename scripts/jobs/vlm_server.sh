#!/usr/bin/env bash
SLRCORES=1

# GPU
# 1 H100 (slave 6), 2 A100 (slave1), 2 A6000 (slave2), 1 P100 (slave3), 5 2080 (slave 4-5)
#SLRNGPU=H100
SLRNGPU=A6000
SLRGPUS=1
MODEL="google/gemma-4-E4B-it"

set -euo pipefail

cd /home/nahuel/vllm

source .venv/bin/activate

vllm serve \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port 8000

# check if the server is up
# curl http://slave2.pir.uo:8000/v1/models

# inference example
# curl http://slave2.pir.uo:8000/v1/responses \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "google/gemma-4-E4B-it",
#     "input": [
#       {
#         "type": "message",
#         "role": "user",
#         "content": [
#           {"type": "input_text", "text": "Explain quantum entanglement in simple terms."}
#         ]
#       }
#     ],
#     "max_output_tokens": 512,
#     "temperature": 0.7
#   }'



