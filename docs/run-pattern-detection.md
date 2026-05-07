# Running Pattern Detection Experiments

## Build Data

Generate the tiny synthetic dataset and visual QA files:

```bash
scripts/run/build_synthetic_dataset.sh
```

Regenerate only the visual QA files:

```bash
scripts/run/build_visual_qa.sh
```

## Proprietary VLM: GPT-4.5

Dry run:

```bash
DRY_RUN=1 LIMIT=2 scripts/run/run_proprietary_pattern_detection.sh
```

Full run:

```bash
scripts/run/run_proprietary_pattern_detection.sh
```

The default proprietary model is:

```text
gpt-4.5-preview
```

Override it if needed:

```bash
MODEL=gpt-4.1 scripts/run/run_proprietary_pattern_detection.sh
```

## Open-Source VLM Served With vLLM

Start your vLLM server separately, then run:

```bash
MODEL=your-vllm-model-name scripts/run/run_vllm_pattern_detection.sh
```

If your server is not on `http://localhost:8000/v1`:

```bash
API_BASE=http://localhost:9000/v1 \
MODEL=your-vllm-model-name \
scripts/run/run_vllm_pattern_detection.sh
```

Some vLLM/model combinations may not support JSON schema response format. If needed:

```bash
RESPONSE_FORMAT=json_object \
MODEL=your-vllm-model-name \
scripts/run/run_vllm_pattern_detection.sh
```

or:

```bash
RESPONSE_FORMAT=none \
MODEL=your-vllm-model-name \
scripts/run/run_vllm_pattern_detection.sh
```

## Run Both

```bash
MODEL=your-vllm-model-name scripts/run/run_all_pattern_detection.sh
```

If you use `MODEL=...` with `run_all_pattern_detection.sh`, it applies to both runners. To avoid that, run the proprietary and vLLM scripts separately.

## Useful Smaller Runs

Only multi-label experiments:

```bash
EXPERIMENTS=multilabel_label_only,multilabel_morphology_described \
scripts/run/run_proprietary_pattern_detection.sh
```

Short k curve:

```bash
K_VALUES=0,1,2,4 \
scripts/run/run_proprietary_pattern_detection.sh
```

One-seed debug run:

```bash
SEEDS=42 LIMIT=5 scripts/run/run_proprietary_pattern_detection.sh
```

## Outputs

Proprietary model outputs:

```text
data/vlm_outputs/openai/pattern_detection/
reports/pattern_detection/openai/
```

vLLM model outputs:

```text
data/vlm_outputs/vllm/pattern_detection/
reports/pattern_detection/vllm/
```

The report CSV is the easiest file to compare across:

```text
reports/pattern_detection/<provider>/icl_sweep_summary.csv
```
