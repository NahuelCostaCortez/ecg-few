# ECG beat analysis with vision models

## Main hypothesis

> In situations where there is so little data that machine learning isn't even considered, can ICL be a useful tool until enough data is collected to perform effective fine-tuning? If so, when should ICL be used, and when should it be set aside in favor of fine-tuning?

The focus of our study is on ECG beat analysis. We'll test first our hypothesis on synthetic data and then we'll use a real case scenario, focused on Brugada syndrome, which is a rare hereditary disease, to test this idea.

## Setup

Install dependencies:

```bash
uv sync --extra dev
```

## Data

As the final tests will be done on Brugada data, we have chosen to focus on characteristic features of this condition, specifically:
- RBBB (Right Bundle Branch Block)
- ST elevated
- T-wave inverted

In Brugada cases all these three features are true. We'll use them to generate a synthetic dataset to first test the initial hypthesis.
- Synthetic data generator is in:
  ```bash
  src/ecg_few/simulator
  ```
- Synthetic dataset can be generated with:
  ```bash
  scripts/run/build_synthetic_dataset.sh
  ```
- [ TO-DO ]: add real data


## ICL
We want to know whether ICL can be used to classify ECG images. Since ICL can depend on various factors (model, n of in-context examples, prompts, etc) we will organize experiments in the following way:

- Experiments:
  - `multilabel_label_only`: just ask the model for presence/absence of specific labels. Prompt in: 
    ```text
    prompts/multilabel/label_only.md
    ```
  - `multilabel_morphology_described`: same as `multilabel_label_only` but describing what we are interested in for each label. Prompt in:
    ```text
    prompts/multilabel/morphology_described.md
     ``
  - `binary_morphology_described`: same as `multilabel_morphology_described` but asks for each label independently (so 3 requests for each image instead of 1). Prompt in:
    ```text
    prompts/multilabel/morphology_described.md
     ``
- The number of in-context examples are referred to as `k`. The values of `k` are: `0, 1, 2, 4, 8, 12, 16, 24, 32`
- Seeds: `42, 123, 2026`: to check variability
- Special case: `k=0` is zero-shot and runs only once per experiment, using the first seed.

That means *each* experiment runs:

- `k=0`
- `k=1` with seeds `42, 123, 2026`
- `k=2` with seeds `42, 123, 2026`
- ...

Default run counts:

- `25` runs per experiment (`8` k x `3` seeds + 1 `k=0`)
- `75` runs per provider (`3` experiments x `25`)

The main experiment script can be found in:
  ```bash
  scripts/eval/run_pattern_detection_experiments.py
  ```

There are two scripts to run this experiments, one that runs proprietary models and another one that runs local models served with vllm.

The proprietary model sweep is in:

```bash
scripts/run/run_proprietary_pattern_detection.sh
```

, and the vLLM-served open-source model sweep is in:

```bash
MODEL=your-vllm-model-name scripts/run/run_vllm_pattern_detection.sh
```

The vLLM runner defaults to:

```text
http://slave2.pir.uo:8000/v1
```

But you can override the server location if needed with:

```bash
API_BASE=http://your-host:8000/v1 \
MODEL=your-vllm-model-name \
scripts/run/run_vllm_pattern_detection.sh
```

By default the full experiment matrix is calculated. You can narrow it manually with `K_VALUES`, `SEEDS`, `EXPERIMENTS`, and `LIMIT` when debugging.
Reasoning effort is only sent for the OpenAI Responses API path. The vLLM
runner currently passes `--api-base` and `--response-format`, but not a
`reasoning` field in the Responses API payload.

## Outputs

```text
outputs/vlm_outputs/
reports/pattern_detection/
```

* Generated predictions and reports are ignored by git.

The report CSV is the easiest summary to inspect:

```text
reports/pattern_detection/<provider>/icl_sweep_summary.csv
```

## CNN
- [ TO-DO ]: the idea is not to test fine-tuning with VLM as these models may be prompt to hallucinations but with CNNs, which are more reliable and explicables.

## Repository Layout

```text
src/ecg_few/
  simulator/     Synthetic beat simulator
  evaluation/    Metric utilities
  prompts/       Markdown prompt loading and response schemas
scripts/
  dataset/       Dataset builders
  eval/          Shared VLM evaluator, experiment sweeps, and metrics
  run/           Shell entrypoints for common runs
prompts/         Editable Markdown prompt files
data/            ECG datasets
outputs/         Model outputs, ignored by git
reports/         Results reports, ignored by git
```
