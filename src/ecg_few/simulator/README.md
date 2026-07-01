# ECG Beat Simulator

This package generates synthetic single-beat ECG waveforms for QRS/ST morphology
experiments. It is not a direct `Brugada vs Normal` simulator and it is not a
clinical diagnosis engine.

The public training targets are always the three QRS/ST findings:

- `RBBB`
- `ST_ELEVATION`
- `T_WAVE_INVERSION`

The Brugada decision is derived downstream only when all three findings are
present at the same time.

## How It Fits The Project

The simulator is used by:

```bash
scripts/run/build_simulator_qrs_dataset.sh
```

That wrapper writes the current synthetic dataset under:

```text
data/simulator_qrs/
```

The exported manifest is:

```text
data/simulator_qrs/labels/all_labels.csv
```

Each row contains the three canonical label columns:

```text
label_rbbb
label_st_elevation
label_t_wave_inversion
```

and a convenience column:

```text
derived_brugada
```

`derived_brugada` is not a training target. It is just:

```text
RBBB and ST_ELEVATION and T_WAVE_INVERSION
```

## Source Families

Internally the generator samples from morphology source families:

- `NORMAL`
- `RBBB`
- `ST_ELEVATION`
- `T_WAVE_INVERSION`
- `BRUGADA`

The last family is retained as an internal shape prior for combined coved
QRS/ST morphology. The dataset builder exposes it as `COMBINED_QRS_ST` so the
downstream task does not become a direct Brugada classifier.

The important distinction is:

```text
source_family      = how the synthetic waveform was sampled
label_* columns    = what the model is trained to predict
derived_brugada    = final rule-based interpretation
```

## Main Modules

- `constants.py`: shared atom, family, and label names.
- `tuning.py`: morphology parameter ranges.
- `validators.py`: acceptance rules for generated beats.
- `simulator.py`: waveform generation loop.
- `plotting.py`: ECG-style rendering helpers.

## Minimal Usage

```python
from ecg_few.simulator import generate_beat

beat, metadata = generate_beat(class_name="RBBB", seed=42, fs=500)
print(metadata["feature_labels"])
```

The returned `feature_labels` dictionary uses the canonical QRS/ST labels, not
a direct clinical Brugada label.
