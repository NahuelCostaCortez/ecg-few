# How To Tune The Simulator

If you want a hands-on walkthrough first, start with [`notebooks/tuning_morphologies.ipynb`](notebooks/tuning_morphologies.ipynb).

## Where To Edit

- [`src/ecg_vlm/simulator/tuning.py`](src/ecg_vlm/simulator/tuning.py): change how beats look
- [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py): change what counts as `RBBB`, `ST_ELEVATION`, or `T_WAVE_INVERSION`
- [`src/ecg_vlm/simulator/plotting.py`](src/ecg_vlm/simulator/plotting.py): change image style
- [`src/ecg_vlm/simulator/simulator.py`](src/ecg_vlm/simulator/simulator.py): change the sampling loop itself

## Most Common Changes

### Make A Morphology Stronger Or Weaker

Edit [`src/ecg_vlm/simulator/tuning.py`](src/ecg_vlm/simulator/tuning.py).

The main knobs are:

- `BASE_SCAFFOLD_RANGES`: shared defaults for all beats
- `CLASS_PARAM_RANGES`: family-specific ranges for `NORMAL`, `RBBB`, `ST_ELEVATION`, `T_WAVE_INVERSION`, and `BRUGADA`
- `_apply_morphology_rules(...)`: hand-written rules that keep sampled atoms anatomically coherent

Typical edits:

- change `a` to make a component taller, deeper, stronger, or weaker
- change `mu` to move a component earlier or later
- change `sigma` to make a component narrower or broader
- change `_apply_morphology_rules(...)` when independently sampled atoms do not combine into realistic shapes

## Change Label Meaning

Edit [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py).

This file defines the numeric acceptance thresholds and time windows for:

- `_validate_rbbb(...)`
- `_validate_st_elevation(...)`
- `_validate_t_wave_inversion(...)`
- `_validate_normal(...)`
- `_validate_brugada(...)`

Important: changing these functions changes the dataset semantics, not just the visuals.

## Change The Rendered Images

Edit [`src/ecg_vlm/simulator/plotting.py`](src/ecg_vlm/simulator/plotting.py).

The main entry point is `plot_beat(...)`.

## Safe Working Pattern

1. Make a small change in [`src/ecg_vlm/simulator/tuning.py`](src/ecg_vlm/simulator/tuning.py) or [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py).
2. Generate a few preview beats with a fixed seed.
3. Check whether the result still looks correct and still passes validation.
4. If acceptance rate collapses, the morphology ranges and validator thresholds are probably no longer aligned.

## Quick Preview Commands

Generate one example grid:

```bash
uv run python -c "from ecg_vlm.simulator import demo_plot_all; demo_plot_all()"
```

Generate one beat interactively:

```bash
uv run python -c "from ecg_vlm.simulator import generate_beat; beat, meta = generate_beat('RBBB', seed=123); print(meta['feature_labels']); print(meta['validation_metrics'])"
```