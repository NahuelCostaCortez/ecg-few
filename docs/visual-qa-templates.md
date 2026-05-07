# Visual QA Templates

Generated for `SD-03` on 2026-05-05.

## Purpose

The visual QA dataset turns each synthetic ECG beat image into compact image-question-answer records for two uses:

- VLM evaluation with strict expected answers
- future VLM instruction tuning with chat-message JSONL

The target findings are:

- `RBBB`
- `ST_ELEVATION`
- `T_WAVE_INVERSION`

The clinician-editable prompt templates live in:

```text
prompts/
```

Python keeps only prompt loading and JSON schema helpers in:

```text
src/ecg_vlm/prompts/
```

## Primary Task: Multi-Label Finding Detection

Zero-shot question shape:

```text
Analyze this single-lead ECG beat image. Decide whether each finding is present:
RBBB, ST_ELEVATION, T_WAVE_INVERSION. Return only valid JSON with boolean values
for exactly these keys: RBBB, ST_ELEVATION, T_WAVE_INVERSION.
```

Answer shape:

```json
{
  "RBBB": true,
  "ST_ELEVATION": false,
  "T_WAVE_INVERSION": true
}
```

This is the main format for model testing.

The zero-shot evaluator can use the `prompt` field already stored in each JSONL record, but the experiment runners normally override it with the selected Markdown prompt file. That keeps prompt editing in `prompts/` instead of Python.

Few-shot evaluation uses the same final prompt, but prepends `k` demonstrations from the train split. Each demonstration includes:

- the example image
- the example question
- the correct JSON answer

The current default is `--few-shot-k 4`.

In the VLM runners, these are represented as true ICL turns: user image/question, then assistant JSON answer. The final test image/question comes after those demonstrations.

## Secondary Task: Binary Finding Detection

Question shape:

```text
Analyze this single-lead ECG beat image. Is RBBB present?
Return only valid JSON with this schema:
{"finding": "RBBB", "present": boolean}.
```

Answer shape:

```json
{
  "finding": "RBBB",
  "present": true
}
```

This format is useful for per-finding diagnostics and sanity checks.

## Rationale Policy

Rationales are optional and disabled by default.

The builder supports `--include-synthetic-rationales`, but these rationales are label-derived rather than clinician-reviewed visual explanations. The boolean labels remain the source of truth.

For the first VLM evaluation, use the no-rationale files.

## Output Layout

The builder writes:

```text
data/vlm/
  qa_schema.json
  qa_dataset_summary.json
  eval/
    multilabel/{train,val,test,transfer}.jsonl
    binary/{train,val,test,transfer}.jsonl
  messages/
    multilabel/{train,val,test,transfer}.jsonl
    binary/{train,val,test,transfer}.jsonl
```

Backward-compatible convenience files are also written:

```text
data/vlm/train_messages.jsonl
data/vlm/val_messages.jsonl
data/vlm/test_messages.jsonl
data/vlm/transfer_messages.jsonl
```

These point to the multi-label chat-message format.

## Current Counts

| Split | Multi-label records | Binary records |
| --- | ---: | ---: |
| train | 80 | 240 |
| val | 0 | 0 |
| test | 80 | 240 |
| transfer | 40 | 120 |

The `transfer` split contains only held-out `BRUGADA` images.
