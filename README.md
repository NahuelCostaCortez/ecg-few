# ECG VLM

Synthetic ECG beat image experiments for testing whether vision-language models can identify specific morphology patterns in single-beat ECG images.

## Current Focus

The current hypothesis is:

> Do VLMs work for identifying specific patterns in beat images?

This first phase compares:

- a proprietary VLM, defaulting to `gpt-5.5`
- an open-source VLM served through vLLM

If this phase works next steps should include:
1) CNN baselines
2) Teacher-student distillation/fine-tuning for open source models -> only if SOTA perform well and open-source fails
3) Evaluation on real ECG data

## Setup

Install dependencies:

```bash
uv sync --extra dev
```

Run commands through the synced environment:

```bash
UV_CACHE_DIR=.uv-cache uv run --no-sync pytest
```

## Data and Prompts

Build a tiny synthetic dataset and visual QA files:

```bash
scripts/run/build_synthetic_dataset.sh
```

Regenerate only the visual QA files:

```bash
scripts/run/build_visual_qa.sh
```

Editable prompts for the LLMs live in:

```text
prompts/
```

The Python prompt loader lives in:

```text
src/ecg_vlm/prompts/
```

## Run Experiments

Dry-run the proprietary model sweep:

```bash
DRY_RUN=1 LIMIT=2 scripts/run/run_proprietary_pattern_detection.sh
```

Run the proprietary model sweep:

```bash
scripts/run/run_proprietary_pattern_detection.sh
```

Run the vLLM-served open-source model sweep:

```bash
MODEL=your-vllm-model-name scripts/run/run_vllm_pattern_detection.sh
```

Run both:

```bash
MODEL=your-vllm-model-name scripts/run/run_all_pattern_detection.sh
```

## Outputs

```text
data/vlm_outputs/
reports/pattern_detection/
```

* Generated predictions and reports are ignored by git.

The report CSV is the easiest summary to inspect:

```text
reports/pattern_detection/<provider>/icl_sweep_summary.csv
```

## Repository Layout

```text
src/ecg_vlm/
  simulator/     Synthetic beat simulator
  evaluation/    Metric utilities
  prompts/       Markdown prompt loading and response schemas
scripts/
  dataset/       Dataset and visual QA builders
  eval/          Shared VLM evaluator, experiment sweeps, and metrics
  run/           Shell entrypoints for common runs
prompts/         Editable Markdown prompt files
docs/            Study notes and runbooks
data/            Generated datasets and model outputs, ignored by git
reports/         Generated summaries, ignored by git
tests/           Smoke tests
```
