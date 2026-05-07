# Beat Simulator

This repository generates synthetic single-beat ECG waveforms, renders them as ECG-style images, and optionally converts the labels into instruction-tuning data for vision-language models (VLM).

The simulator is organized into small modules:

- [`src/ecg_vlm/simulator/tuning.py`](src/ecg_vlm/simulator/tuning.py): morphology ranges and hand-written shape rules
- [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py): label and acceptance criteria
- [`src/ecg_vlm/simulator/simulator.py`](src/ecg_vlm/simulator/simulator.py): beat generation loop
- [`src/ecg_vlm/simulator/plotting.py`](src/ecg_vlm/simulator/plotting.py): ECG rendering helpers

The dataset builders live in `scripts/dataset/`.

If you want to change the simulator behaviour, start with [`HOW_TO_TUNE.md`](HOW_TO_TUNE.md).

## What This Is

The simulator does not try to model a full ECG recording or a specific lead system. It generates one beat at a time and focuses on right-precordial-like morphologies that are useful for these source families:

- `NORMAL`
- `RBBB`
- `ST_ELEVATION`
- `T_WAVE_INVERSION`
- `BRUGADA` (`coved` / type-1 only yet)

The prediction targets are not the same as the source families. The dataset labels are only these three findings:

- `RBBB`
- `ST_ELEVATION`
- `T_WAVE_INVERSION`

That distinction matters: a generated beat from one source family can satisfy more than one label validator.

## Simple Explanation

If you want the short version, think of the simulator like this:

1. It builds one ECG beat by stacking several bell-shaped curves.
2. Each curve represents a familiar ECG component such as `P`, `Q`, `R`, `S`, `R'`, J-point/ST pieces, and the T wave.
3. For each beat type, the code samples those curve parameters from ranges that make that morphology more likely.
4. It then applies a few hand-written rules so the beat still looks anatomically coherent.
5. If the result does not pass a validator for the requested morphology, it throws that beat away and tries again.
6. Once it gets a valid beat, it can save the waveform as an ECG image, attach rule-based labels, and export the data for downstream training.

So the whole project is basically:

`sample Gaussian pieces -> sum them into a waveform -> reject invalid beats -> save images and labels`

## Thorough Explanation

### 1. Waveform Model

Each beat is modeled as a sum of Gaussian atoms:

`a * exp(-((t - mu)^2) / (2 * sigma^2))`

Each atom has:

- `a`: amplitude
- `mu`: center in milliseconds
- `sigma`: width in milliseconds

The simulator uses this fixed atom order:

- `P`
- `Q`
- `R`
- `S`
- `R_prime`
- `J_elev`
- `ST1`
- `ST2`
- `T1`
- `T2`

You can think of them as a scaffold:

- `P`, `Q`, `R`, `S`, `R_prime` shape the pre-QRS and QRS region.
- `J_elev`, `ST1`, `ST2` shape the J-point and ST segment.
- `T1`, `T2` shape the T wave.

The waveform is sampled on a fixed time grid. By default:

- `duration_ms = 800`
- `fs = 500`

That gives `400` samples per beat.

### 2. How A Beat Is Generated

`generate_beat(...)` follows this process:

1. Start from a base scaffold of parameter ranges shared by all beats.
2. Merge in source-family-specific parameter ranges.
3. Sample atom parameters uniformly from those ranges.
4. Apply hand-written morphology rules so the timing and shape relationships make sense.
5. Sum all atoms into one waveform.
6. Validate the waveform against the requested class.
7. If validation fails, repeat until success or `max_attempts` is reached.

This is a rejection-sampling simulator. It does not guarantee the first sample will be acceptable; it keeps trying until the waveform satisfies the class validator.

### 3. Source Families vs Final Labels

This project uses two different concepts:

- `source_family`: how the beat was generated
- `feature_labels`: what rule-based findings are present in the final waveform

That means the generator is not doing a one-to-one mapping like:

- `RBBB source_family -> RBBB label only`
- `ST_ELEVATION source_family -> ST_ELEVATION label only`

Instead, labels are extracted from the final waveform after generation. Because the validators are independent, one beat can carry multiple labels. This is especially important for:

- `ST_ELEVATION`, which can also satisfy the `RBBB` validator
- `T_WAVE_INVERSION`, which can appear with or without `RBBB` or `ST_ELEVATION`
- `BRUGADA`, which is a source family but not a target label; its beats can map to combinations of the three target findings

This is why the dataset builder stores both:

- the source family
- the final binary labels

### 4. What The Validators Check

Each label is defined by explicit numeric rules over windows of the generated waveform.

High level:

- `RBBB`: looks for two positive QRS peaks with enough separation, a deep enough dip between them, a late enough second peak, and enough terminal width.
- `ST_ELEVATION`: checks strong QRS activity plus positive J-point/ST mean values.
- `T_WAVE_INVERSION`: checks a negative T wave that is late enough, deep enough, and lasts long enough, while also requiring compatible QRS/ST features.
- `NORMAL`: requires a healthy-looking QRS and upright T wave while explicitly rejecting the other target findings.
- `BRUGADA`: checks elevated J/ST morphology with a coved-looking descending ST segment and a negative T wave.

If you need the exact thresholds, read the validator functions in [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py).

### 5. What Metadata You Get Back

`generate_beat(...)` returns:

- the waveform as a NumPy array
- a metadata dictionary

The metadata includes:

- `source_family`
- `class_key`
- `subtype`
- `fs`
- `duration_ms`
- `attempt`
- `validation_metrics`
- `feature_labels`
- `atoms`

The `atoms` field is especially useful if you want to inspect or debug why a waveform looks the way it does.

## How To Use It

### Setup

This repository is already configured for `uv`.

```bash
uv sync
```

### Generate One Beat In Python

```python
from ecg_vlm.simulator import generate_beat, plot_beat

beat, meta = generate_beat(
    class_name="RBBB",
    seed=123,
    fs=500,
)

print(meta["feature_labels"])
print(meta["validation_metrics"])
print(meta["attempt"])

plot_beat(beat, fs=500, save_path="rbbb_example.png")
```

### Generate One Example Per Source Family

```bash
uv run python -c "from ecg_vlm.simulator import demo_plot_all; demo_plot_all()"
```

This writes a grid image of example beats using the built-in defaults.

### Build The Image Dataset

```bash
uv run python scripts/dataset/build_ecg_dataset.py \
  --outdir data \
  --samples-per-family 1000 \
  --test-ratio 0.2 \
  --val-ratio 0.1 \
  --fs 500 \
  --seed 42 \
  --dpi 112
```

What this script does:

1. Generates one verification example per source family.
2. Generates `samples_per_family` beats for each source family.
3. Splits them into train/val/test, stratified by source family.
4. Saves ECG-style images plus CSV and JSON metadata.

Important detail: the split is stratified by `source_family`, not by final label combination.

### Build The VLM Instruction Dataset

After the image dataset exists:

```bash
uv run python scripts/dataset/build_vlm_instruction_dataset.py \
  --dataset-root data \
  --seed 42
```

This reads the label CSV files and writes JSONL records with:

- an image reference
- a user prompt
- the assistant target text
- metadata

Use `--absolute-image-paths` if your downstream training pipeline expects absolute image paths instead of dataset-relative ones.

The target text is either:

- `NORMAL`
- or a comma-separated label list in canonical order:
  `RBBB, ST_ELEVATION, T_WAVE_INVERSION`

### Convenience Shell Script

If you want the default full build from the repository's job script:

```bash
cd jobs
sh build_datasets.sh
```

## Output Structure

After `build_ecg_dataset.py`, the dataset root looks like this:

```text
data/
  examples/
    NORMAL/example.png
    RBBB/example.png
    ST_ELEVATION/example.png
    T_WAVE_INVERSION/example.png
    BRUGADA/example.png
    verification_grid.png
    verification_grid_by_family.png
    sanity_metadata.json
  train/
    NORMAL/*.png
    RBBB/*.png
    ST_ELEVATION/*.png
    T_WAVE_INVERSION/*.png
    BRUGADA/*.png
  val/
    ...
  test/
    ...
  labels/
    train_labels.csv
    val_labels.csv
    test_labels.csv
    all_labels.csv
    label_schema.json
    dataset_summary.json
    sanity_counts.json
```

After `build_vlm_instruction_dataset.py`, you also get:

```text
data/
  vlm/
    train_messages.jsonl
    val_messages.jsonl
    test_messages.jsonl
```

### Label And Metadata Files

The main CSV manifest columns are:

```text
image_path, split, source_family, seed, attempt, subtype, sample_index, label_rbbb, label_st_elevation, label_t_wave_inversion
```

The most useful metadata artifacts are:

- `labels/label_schema.json`: canonical label names and text-formatting conventions
- `labels/dataset_summary.json`: sample counts by split, source family, label, and label combination
- `examples/sanity_metadata.json`: seeds, attempts, and validation metadata for the verification beats

## How To Manipulate It

### If You Want Different Random Beats

Change:

- `seed`
- `samples-per-family`
- `fs`
- `duration_ms` in `generate_beat(...)`
- `max_attempts` in `generate_beat(...)` if stricter settings make valid beats harder to sample

Use a fixed `seed` when you want reproducibility.

If you generate `BRUGADA`, pass `subtype="coved"` or leave `subtype=None`; no other subtype is currently supported.

### If You Want To Change How A Morphology Looks

The main knobs are in [`src/ecg_vlm/simulator/tuning.py`](src/ecg_vlm/simulator/tuning.py):

- `BASE_SCAFFOLD_RANGES` -> shared default sampling ranges for every atom across all families
- `CLASS_PARAM_RANGES` -> family-specific overrides layered on top of that base for NORMAL, RBBB, ST_ELEVATION, T_WAVE_INVERSION, and BRUGADA
- `_apply_morphology_rules(...)` -> hand-written morphology rules

Use them like this:

- widen or narrow amplitude ranges to make a feature stronger or weaker
- move `mu` ranges to shift components earlier or later
- change `sigma` ranges to sharpen or broaden components
- add or tighten rule logic when the independently sampled atoms do not combine into realistic shapes

If the beats stop being accepted often, you probably made the parameter ranges inconsistent with the validator thresholds.

Validators live in [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py). They define acceptance thresholds and time windows on the finished waveform, like “look in 240-470 ms,” “peak > 0.12,” or “ST mean > 0.09”.

### If You Want Different Labels

The target labels are defined by:

- `LABEL_NAMES`
- `extract_feature_labels(...)`
- the validator functions in [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py)

To change label behavior, edit the validator thresholds or logic.

Be careful: changing validators changes the dataset semantics, not just the visuals.

### If You Want Different Images

Adjust:

- [`src/ecg_vlm/simulator/plotting.py`](src/ecg_vlm/simulator/plotting.py) and `plot_beat(...)` for styling
- `dpi` in the dataset builder

Right now the renderer:

- uses a square canvas
- draws an ECG-like pink grid
- plots a single black waveform
- saves PNG files

### If You Want To Add A New Source Family

The usual path is:

1. Add the family name to `SOURCE_FAMILIES` in [`src/ecg_vlm/simulator/constants.py`](src/ecg_vlm/simulator/constants.py).
2. Add parameter ranges to `CLASS_PARAM_RANGES` in [`src/ecg_vlm/simulator/tuning.py`](src/ecg_vlm/simulator/tuning.py).
3. Add or extend morphology rules in [`src/ecg_vlm/simulator/tuning.py`](src/ecg_vlm/simulator/tuning.py).
4. Add a validator in [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py).
5. Decide whether the family should expose a subtype.
6. Update dataset-building helpers if the subtype needs special handling.

If the new family should also become a training label, update:

- `LABEL_NAMES`
- `LABEL_COLUMNS`
- `extract_feature_labels(...)`
- the CSV/JSON export code
- the VLM target formatting/parsing logic

## How To Improve It

### 1. Improve Physiologic Realism

Right now the simulator is intentionally simple and controllable, but it is still synthetic. It does not model:

- multiple leads
- beat-to-beat rhythm changes
- baseline wander
- muscle noise
- electrode noise
- explicit PR/QRS/QT interval physiology beyond the hand-written constraints

Useful next steps:

- add noise and baseline-drift models
- generate short sequences instead of isolated beats
- support multiple leads with shared latent structure
- introduce heart-rate-dependent timing changes

### 2. Improve Label Quality

Labels are rule-based, not expert annotations. That is fine for a synthetic pipeline, but it has consequences.

Useful next steps:

- add unit tests for each validator
- create golden examples for each morphology
- measure how often each source family maps to each label combination
- decide whether splits should be stratified by final labels instead of only source family

### 3. Improve Extensibility

The code is clean to navigate now, but most of the domain knowledge is still hand-coded rather than config-driven.

Useful next steps:

- add a config-driven format for morphology definitions
- store versioned simulation settings with every dataset export
- expand the notebook or add scripts for visualizing sampled atom distributions and acceptance rates

### 4. Improve Documentation And Evaluation

A new contributor will move faster if they can quickly see whether a change improved or broke the simulator.

Useful next steps:

- save acceptance-rate statistics per source family
- save more sample grids during dataset generation
- add comparison reports between old and new parameter sets
- document known failure modes for each morphology

## Important Limitations

Before improving or training on this data, keep these constraints in mind:

- This is a single-beat simulator, not a full ECG simulator.
- It does not model explicit lead identity.
- `BRUGADA` only supports the `coved` / type-1 subtype.
- Final labels are derived from validators, so source families and labels are not one-to-one.
- Train/val/test splits are source-family-stratified, not label-stratified.
- The project is designed for controllable morphology synthesis, not for clinical use.

## Key Files

- [`src/ecg_vlm/simulator/tuning.py`](src/ecg_vlm/simulator/tuning.py): morphology ranges and hand-written rules
- [`src/ecg_vlm/simulator/validators.py`](src/ecg_vlm/simulator/validators.py): waveform acceptance and label semantics
- [`src/ecg_vlm/simulator/simulator.py`](src/ecg_vlm/simulator/simulator.py): generation loop and metadata assembly
- [`src/ecg_vlm/simulator/plotting.py`](src/ecg_vlm/simulator/plotting.py): ECG-style rendering helpers
- [`notebooks/tuning_morphologies.ipynb`](notebooks/tuning_morphologies.ipynb): interactive tuning examples and atom-distribution visualizations
- [`HOW_TO_TUNE.md`](HOW_TO_TUNE.md): collaborator-oriented guide for safe edits
- `scripts/dataset/build_ecg_dataset.py`: image dataset generation and label export
- `scripts/dataset/build_vlm_instruction_dataset.py`: VLM instruction JSONL export
- `scripts/run/build_synthetic_dataset.sh`: convenience shell workflow for both dataset stages

## Mental Model

If you are new to the project, this is the one mental model worth remembering:

- geometry comes from Gaussian atoms
- realism comes from family-specific ranges plus morphology rules
- correctness comes from validators
- training data comes from rendering validated beats and exporting their labels
