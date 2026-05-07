# Pattern Detection Hypothesis

## Current Focus

The first project goal is to test this hypothesis:

> Do vision-language models identify specific ECG morphology patterns in single-beat images?

This phase does not test knowledge distillation, fine-tuning, teacher-student learning, CNN baselines, or real ECG data.

## Models to Compare

1. Proprietary VLM
   - Default script model: `gpt-4.5-preview`
   - Note: OpenAI currently marks GPT-4.5 Preview as deprecated. The script keeps it as the default because this is the requested comparison, but the model can be changed with `MODEL=...`.

2. Open-source VLM served with vLLM
   - The model is expected to be exposed through a vLLM OpenAI-compatible server.
   - Default API base: `http://localhost:8000/v1`
   - The model name is provided with `MODEL=...`.

## Data

The first dataset is synthetic and image-only.

It contains:

- `NORMAL`
- `RBBB`
- `ST_ELEVATION`
- `T_WAVE_INVERSION`
- `BRUGADA`

The prediction targets are:

- `RBBB`
- `ST_ELEVATION`
- `T_WAVE_INVERSION`

## Primary Evaluation

Use multi-label classification:

```json
{
  "RBBB": true,
  "ST_ELEVATION": false,
  "T_WAVE_INVERSION": true
}
```

This asks whether the model can identify all target patterns in the same beat image.

## Diagnostic Evaluation

Use binary finding prompts:

```json
{
  "finding": "RBBB",
  "present": true
}
```

This asks about one finding at a time, then aggregates results across findings. It helps diagnose whether failures come from visual recognition or from the multi-label output format.

## Prompt Families

Prompt text lives in Markdown under `prompts/`.

Current prompt families:

- `multilabel_label_only`
- `multilabel_morphology_described`
- `binary_morphology_described`

Doctors can edit the Markdown files directly without touching Python code.

## ICL Sweep

Default ICL values:

```text
k = 0, 1, 2, 4, 8, 12
seeds = 42, 123, 2026
```

`k=0` is zero-shot. For `k>0`, examples are selected from the training split using the given seed.

Success in this phase means:

- the proprietary VLM performs above chance on the target patterns
- ICL improves or stabilizes performance as k increases
- the open-source vLLM-served model can be evaluated with the same protocol
- we can compare proprietary versus open-source behavior before deciding whether distillation/fine-tuning is worth pursuing
