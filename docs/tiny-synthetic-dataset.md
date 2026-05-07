# Tiny Synthetic Dataset

Generated for `SD-02` on 2026-05-01.

## Purpose

This dataset is the first proof-of-concept split for component-level visual transfer.

It keeps `BRUGADA` coved/type-1 beats held out from component development and component testing, then uses them only as transfer examples.

## Generation command

```bash
MPLCONFIGDIR=.cache/matplotlib \
MPLBACKEND=Agg \
XDG_CACHE_HOME=.cache \
UV_CACHE_DIR=.uv-cache \
uv run --no-sync python scripts/dataset/build_ecg_dataset.py \
  --outdir data \
  --samples-per-family 40 \
  --fs 500 \
  --seed 42 \
  --dpi 112 \
  --tiny-poc \
  --component-dev-count 20 \
  --component-test-count 20 \
  --transfer-count 40 \
  --transfer-source-family BRUGADA
```

The convenience runner `scripts/run/build_synthetic_dataset.sh` uses the same tiny proof-of-concept settings.

## Split counts

| Split | Count | Source families |
| --- | ---: | --- |
| train | 80 | 20 each from `NORMAL`, `RBBB`, `ST_ELEVATION`, `T_WAVE_INVERSION` |
| val | 0 | intentionally empty for the first proof of concept |
| test | 80 | 20 each from `NORMAL`, `RBBB`, `ST_ELEVATION`, `T_WAVE_INVERSION` |
| transfer | 40 | 40 from held-out `BRUGADA` |

Total: 200 images.

## Output locations

- Images: `data/train`, `data/test`, and `data/transfer`
- Verification examples: `data/examples`
- Labels and summaries: `data/labels`
- Main summary: `data/labels/dataset_summary.json`
- Label schema: `data/labels/label_schema.json`

## Validation

Checked after generation:

- `data/labels/all_labels.csv` has 200 rows.
- `data/labels/train_labels.csv` has 80 rows.
- `data/labels/test_labels.csv` has 80 rows.
- `data/labels/transfer_labels.csv` has 40 rows.
- `BRUGADA` appears only in `transfer`.
- Rendered beat images are 896 x 896 RGBA PNG files.
