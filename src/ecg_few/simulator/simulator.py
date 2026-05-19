"""Beat generation logic for the ECG simulator."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import ATOM_ORDER, SOURCE_FAMILIES
from .model import GaussianAtom
from .tuning import (
    BASE_SCAFFOLD_RANGES,
    CLASS_PARAM_RANGES,
    _adapt_ranges_for_duration,
    _apply_morphology_rules,
    _merge_ranges,
    _resolve_class_key,
    _sample_atoms,
)
from .validators import _validate_beat, extract_feature_labels


def _sum_gaussians(atoms: List[GaussianAtom], t_ms: np.ndarray) -> np.ndarray:
    x = np.zeros_like(t_ms, dtype=float)
    for atom in atoms:
        x += atom.evaluate(t_ms)
    return x


def generate_beat(
    class_name: str,
    seed: Optional[int] = None,
    subtype: Optional[str] = None,
    fs: int = 500,
    duration_ms: float = 800.0,
    max_attempts: int = 400,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Generate one synthetic ECG beat and metadata."""
    class_key = _resolve_class_key(class_name, subtype)
    if class_key not in CLASS_PARAM_RANGES:
        raise ValueError(f"Unknown class_name='{class_name}'. Use one of {SOURCE_FAMILIES}.")

    ranges = _merge_ranges(BASE_SCAFFOLD_RANGES, CLASS_PARAM_RANGES[class_key])
    ranges = _adapt_ranges_for_duration(ranges, duration_ms)
    rng = np.random.default_rng(seed)

    n_samples = int(np.round((duration_ms / 1000.0) * fs))
    t_ms = np.arange(n_samples, dtype=float) / fs * 1000.0

    last_metrics: Dict[str, float] = {}
    for attempt in range(1, max_attempts + 1):
        sampled_atoms = _sample_atoms(ranges, rng)
        _apply_morphology_rules(class_key, sampled_atoms, ranges)

        atom_list = [sampled_atoms[name] for name in ATOM_ORDER]
        x = _sum_gaussians(atom_list, t_ms)
        is_valid, metrics = _validate_beat(class_key, x, t_ms, fs)
        last_metrics = metrics
        if is_valid:
            metadata = {
                "source_family": class_name.upper(),
                "class_key": class_key,
                "subtype": "coved" if class_key == "BRUGADA" else None,
                "fs": fs,
                "duration_ms": float(duration_ms),
                "attempt": attempt,
                "validation_metrics": metrics,
                "feature_labels": extract_feature_labels(x, fs),
                "atoms": [atom.as_dict() for atom in atom_list],
            }
            return x, metadata

    raise RuntimeError(
        f"Failed to generate valid beat for '{class_name}' within {max_attempts} attempts. "
        f"Last metrics: {last_metrics}"
    )


__all__ = ["generate_beat", "_sum_gaussians"]