"""Morphology tuning surfaces for the ECG beat simulator.

This is the main file to edit when you want a morphology to look different
without changing the dataset label semantics.
"""

from __future__ import annotations

import numpy as np

from .constants import ATOM_ORDER, REFERENCE_DURATION_MS, REPOLARIZATION_PIVOT_MS
from .model import GaussianAtom

# Shared default ranges for every Gaussian atom before family-specific overrides.
BASE_SCAFFOLD_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "P": {"a": (0.04, 0.12), "mu": (90, 150), "sigma": (18, 34)},
    "Q": {"a": (0, 0.0), "mu": (245, 280), "sigma": (5, 11)},
    "R": {"a": (0.05, 0.70), "mu": (285, 330), "sigma": (8, 15)},
    "S": {"a": (-0.85, -0.28), "mu": (320, 360), "sigma": (14, 30)},
    "R_prime": {"a": (0.0, 0.0), "mu": (355, 430), "sigma": (12, 25)},
    "J_elev": {"a": (-0.01, 0.01), "mu": (385, 430), "sigma": (10, 20)},
    "ST1": {"a": (-0.01, 0.01), "mu": (460, 520), "sigma": (30, 65)},
    "ST2": {"a": (-0.01, 0.01), "mu": (530, 620), "sigma": (35, 85)},
    "T1": {"a": (0.14, 0.34), "mu": (520, 620), "sigma": (55, 105)},
    "T2": {"a": (0.02, 0.14), "mu": (600, 720), "sigma": (45, 110)},
}


# Family-specific morphology overrides layered on top of the shared scaffold.
CLASS_PARAM_RANGES: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "NORMAL": {
        #"R": {"a": (0.26, 0.62), "mu": (286, 326), "sigma": (8, 15)},
        "S": {"a": (-0.72, -0.24), "mu": (320, 356), "sigma": (13, 26)},
        "R_prime": {"a": (0.0, 0.01), "mu": (354, 414), "sigma": (10, 22)},
        "J_elev": {"a": (-0.02, 0.02), "mu": (388, 420), "sigma": (10, 22)},
        "ST1": {"a": (-0.02, 0.02), "mu": (455, 525), "sigma": (32, 72)},
        "ST2": {"a": (-0.02, 0.02), "mu": (540, 630), "sigma": (35, 86)},
        #"T1": {"a": (0.16, 0.32), "mu": (635, 735), "sigma": (58, 110)},
        #"T2": {"a": (0.04, 0.16), "mu": (710, 835), "sigma": (50, 118)},
    },
    "RBBB": {
        #"R": {"a": (0.16, 0.48), "mu": (278, 314), "sigma": (7, 13)},
        "S": {"a": (-0.95, -0.38), "mu": (320, 355), "sigma": (14, 30)},
        "R_prime": {"a": (0.45, 1.00), "mu": (360, 430), "sigma": (16, 34)},
        "T1": {"a": (0.05, 0.22), "mu": (535, 650), "sigma": (60, 115)},
        "T2": {"a": (0.00, 0.12), "mu": (625, 740), "sigma": (50, 120)},
    },
    "ST_ELEVATION": {
        #"R": {"a": (0.24, 0.66), "mu": (286, 328), "sigma": (8, 15)},
        "S": {"a": (-0.88, -0.28), "mu": (322, 360), "sigma": (13, 28)},
        "J_elev": {"a": (0.12, 0.35), "mu": (386, 420), "sigma": (10, 24)},
        "ST1": {"a": (0.12, 0.28), "mu": (450, 520), "sigma": (40, 85)},
        "ST2": {"a": (0.06, 0.20), "mu": (530, 620), "sigma": (45, 95)},
        "T1": {"a": (0.10, 0.30), "mu": (520, 635), "sigma": (55, 110)},
        "T2": {"a": (0.00, 0.12), "mu": (600, 730), "sigma": (48, 120)},
    },
    "T_WAVE_INVERSION": {
        #"R": {"a": (0.18, 0.42), "mu": (282, 322), "sigma": (7, 13)},
        "S": {"a": (-1.05, -0.48), "mu": (320, 360), "sigma": (15, 34)},
        "R_prime": {"a": (0.12, 0.55), "mu": (348, 430), "sigma": (12, 30)},
        "J_elev": {"a": (0.02, 0.16), "mu": (388, 425), "sigma": (10, 24)},
        "ST1": {"a": (-0.02, 0.05), "mu": (455, 535), "sigma": (32, 76)},
        "ST2": {"a": (-0.02, 0.03), "mu": (540, 630), "sigma": (34, 82)},
        "T1": {"a": (-0.1, -0.02), "mu": (550, 640), "sigma": (38, 78)},
        "T2": {"a": (-0.05, 0), "mu": (635, 750), "sigma": (35, 88)},
    },
    "BRUGADA": {
        #"R": {"a": (0.16, 0.52), "mu": (280, 320), "sigma": (7, 14)},
        "S": {"a": (-1.05, -0.38), "mu": (320, 360), "sigma": (13, 30)},
        "R_prime": {"a": (0.08, 0.36), "mu": (340, 392), "sigma": (10, 24)},
        "J_elev": {"a": (0.18, 0.46), "mu": (392, 430), "sigma": (10, 24)},
        "ST1": {"a": (0.14, 0.35), "mu": (455, 530), "sigma": (40, 95)},
        "ST2": {"a": (-0.12, 0.03), "mu": (550, 640), "sigma": (45, 95)},
        "T1": {"a": (-0.15, -0.02), "mu": (550, 640), "sigma": (65, 125)},
        "T2": {"a": (-0.07, 0), "mu": (640, 750), "sigma": (55, 130)},
    },
}


def _deepcopy_ranges(
    ranges: dict[str, dict[str, tuple[float, float]]]
) -> dict[str, dict[str, tuple[float, float]]]:
    return {k: dict(v) for k, v in ranges.items()}


def _merge_ranges(
    base: dict[str, dict[str, tuple[float, float]]],
    override: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    out = _deepcopy_ranges(base)
    for atom_name, atom_params in override.items():
        if atom_name not in out:
            out[atom_name] = {}
        out[atom_name].update(atom_params)
    return out


def _resolve_class_key(class_name: str, subtype: str | None) -> str:
    class_name = class_name.upper()
    if class_name != "BRUGADA":
        return class_name
    if subtype is None or subtype == "" or subtype.lower() == "coved":
        return "BRUGADA"
    if subtype.lower() == "saddleback":
        raise ValueError("BRUGADA only supports the coved/type-1 subtype in this project.")
    raise ValueError("BRUGADA subtype must be None or 'coved'.")


def _sample_uniform(rng: np.random.Generator, low_high: tuple[float, float]) -> float:
    low, high = low_high
    return float(rng.uniform(low, high))


def _clip_to_range(value: float, low_high: tuple[float, float]) -> float:
    low, high = low_high
    return float(np.clip(value, low, high))


def _scale_repolarization_time(time_ms: float, duration_ms: float) -> float:
    if time_ms <= REPOLARIZATION_PIVOT_MS:
        return float(time_ms)

    reference_span = REFERENCE_DURATION_MS - REPOLARIZATION_PIVOT_MS
    duration_span = max(duration_ms - REPOLARIZATION_PIVOT_MS, 1.0)
    scale = duration_span / reference_span
    return float(REPOLARIZATION_PIVOT_MS + (time_ms - REPOLARIZATION_PIVOT_MS) * scale)


def _adapt_ranges_for_duration(
    ranges: dict[str, dict[str, tuple[float, float]]], duration_ms: float
) -> dict[str, dict[str, tuple[float, float]]]:
    out = _deepcopy_ranges(ranges)
    reference_span = REFERENCE_DURATION_MS - REPOLARIZATION_PIVOT_MS
    duration_span = max(duration_ms - REPOLARIZATION_PIVOT_MS, 1.0)
    late_scale = max(duration_span / reference_span, 0.2)

    for atom_name in ("J_elev", "ST1", "ST2", "T1", "T2"):
        if atom_name not in out:
            continue

        mu_low, mu_high = out[atom_name]["mu"]
        out[atom_name]["mu"] = (
            _scale_repolarization_time(mu_low, duration_ms),
            _scale_repolarization_time(mu_high, duration_ms),
        )

        sigma_low, sigma_high = out[atom_name]["sigma"]
        scaled_sigma_low = max(3.0, sigma_low * late_scale)
        scaled_sigma_high = max(scaled_sigma_low, sigma_high * late_scale)
        out[atom_name]["sigma"] = (scaled_sigma_low, scaled_sigma_high)

    return out


def _sample_atoms(
    ranges: dict[str, dict[str, tuple[float, float]]], rng: np.random.Generator
) -> dict[str, GaussianAtom]:
    atoms: dict[str, GaussianAtom] = {}
    for name in ATOM_ORDER:
        atom_ranges = ranges[name]
        atoms[name] = GaussianAtom(
            name=name,
            a=_sample_uniform(rng, atom_ranges["a"]),
            mu=_sample_uniform(rng, atom_ranges["mu"]),
            sigma=_sample_uniform(rng, atom_ranges["sigma"]),
        )
    return atoms


def _apply_morphology_rules(
    class_key: str,
    atoms: dict[str, GaussianAtom],
    ranges: dict[str, dict[str, tuple[float, float]]],
) -> None:
    atoms["Q"].a = _clip_to_range(max(atoms["Q"].a, -0.018), ranges["Q"]["a"])
    atoms["Q"].mu = _clip_to_range(min(atoms["Q"].mu, atoms["R"].mu - 12), ranges["Q"]["mu"])
    atoms["S"].mu = _clip_to_range(max(atoms["S"].mu, atoms["R"].mu + 10), ranges["S"]["mu"])
    atoms["S"].sigma = _clip_to_range(
        max(atoms["S"].sigma, atoms["R"].sigma + 5), ranges["S"]["sigma"]
    )
    atoms["J_elev"].mu = _clip_to_range(
        max(atoms["J_elev"].mu, atoms["S"].mu + 8), ranges["J_elev"]["mu"]
    )
    atoms["ST1"].mu = _clip_to_range(
        max(atoms["ST1"].mu, atoms["J_elev"].mu + 20), ranges["ST1"]["mu"]
    )
    atoms["ST2"].mu = _clip_to_range(
        max(atoms["ST2"].mu, atoms["ST1"].mu + 28), ranges["ST2"]["mu"]
    )
    atoms["T1"].mu = _clip_to_range(max(atoms["T1"].mu, atoms["ST2"].mu + 50), ranges["T1"]["mu"])
    atoms["T2"].mu = _clip_to_range(max(atoms["T2"].mu, atoms["T1"].mu + 20), ranges["T2"]["mu"])

    if class_key == "NORMAL":
        atoms["R_prime"].a = _clip_to_range(
            min(abs(atoms["R_prime"].a), 0.05), ranges["R_prime"]["a"]
        )
        atoms["R_prime"].sigma = _clip_to_range(
            min(atoms["R_prime"].sigma, atoms["R"].sigma + 4), ranges["R_prime"]["sigma"]
        )
        for name in ("J_elev", "ST1", "ST2"):
            atoms[name].a = _clip_to_range(atoms[name].a * 0.4, ranges[name]["a"])
        atoms["T1"].a = _clip_to_range(abs(atoms["T1"].a), ranges["T1"]["a"])
        atoms["T2"].a = _clip_to_range(abs(atoms["T2"].a), ranges["T2"]["a"])

    if class_key == "RBBB":
        atoms["R_prime"].mu = _clip_to_range(
            max(atoms["R_prime"].mu, atoms["S"].mu + 24), ranges["R_prime"]["mu"]
        )
        atoms["R_prime"].a = _clip_to_range(
            max(atoms["R_prime"].a, 0.8 * atoms["R"].a), ranges["R_prime"]["a"]
        )
        atoms["R_prime"].sigma = _clip_to_range(
            max(atoms["R_prime"].sigma, atoms["R"].sigma + 5), ranges["R_prime"]["sigma"]
        )

    if class_key == "ST_ELEVATION":
        for name in ("J_elev", "ST1", "ST2"):
            atoms[name].a = _clip_to_range(abs(atoms[name].a), ranges[name]["a"])
        atoms["ST1"].sigma = _clip_to_range(max(atoms["ST1"].sigma, 46.0), ranges["ST1"]["sigma"])

    if class_key == "T_WAVE_INVERSION":
        atoms["Q"].a = _clip_to_range(max(atoms["Q"].a, -0.012), ranges["Q"]["a"])
        atoms["S"].a = _clip_to_range(min(atoms["S"].a, -0.58), ranges["S"]["a"])
        atoms["S"].sigma = _clip_to_range(
            max(atoms["S"].sigma, atoms["R"].sigma + 7), ranges["S"]["sigma"]
        )
        atoms["R_prime"].mu = _clip_to_range(
            max(atoms["R_prime"].mu, atoms["S"].mu + 14), ranges["R_prime"]["mu"]
        )
        atoms["R_prime"].a = _clip_to_range(
            max(atoms["R_prime"].a, 0.38 * atoms["R"].a), ranges["R_prime"]["a"]
        )
        atoms["R_prime"].sigma = _clip_to_range(
            max(atoms["R_prime"].sigma, atoms["R"].sigma + 1), ranges["R_prime"]["sigma"]
        )
        atoms["J_elev"].a = _clip_to_range(abs(atoms["J_elev"].a), ranges["J_elev"]["a"])
        atoms["ST1"].a = _clip_to_range(max(atoms["ST1"].a, -0.01), ranges["ST1"]["a"])
        atoms["ST2"].a = _clip_to_range(max(atoms["ST2"].a, -0.015), ranges["ST2"]["a"])
        atoms["T1"].mu = _clip_to_range(
            max(atoms["T1"].mu, atoms["ST2"].mu + 85), ranges["T1"]["mu"]
        )
        atoms["T2"].mu = _clip_to_range(
            max(atoms["T2"].mu, atoms["T1"].mu + 24), ranges["T2"]["mu"]
        )
        atoms["T1"].a = _clip_to_range(-abs(atoms["T1"].a), ranges["T1"]["a"])
        atoms["T2"].a = _clip_to_range(-abs(atoms["T2"].a), ranges["T2"]["a"])
        atoms["T1"].sigma = _clip_to_range(max(atoms["T1"].sigma, 44.0), ranges["T1"]["sigma"])
        atoms["T2"].sigma = _clip_to_range(max(atoms["T2"].sigma, 40.0), ranges["T2"]["sigma"])

    if class_key == "BRUGADA":
        atoms["J_elev"].a = _clip_to_range(abs(atoms["J_elev"].a), ranges["J_elev"]["a"])
        atoms["ST1"].a = _clip_to_range(abs(atoms["ST1"].a), ranges["ST1"]["a"])
        atoms["T1"].a = _clip_to_range(-abs(atoms["T1"].a), ranges["T1"]["a"])
        atoms["T2"].a = _clip_to_range(-abs(atoms["T2"].a), ranges["T2"]["a"])
        atoms["R_prime"].mu = _clip_to_range(
            max(atoms["R_prime"].mu, atoms["S"].mu + 8), ranges["R_prime"]["mu"]
        )
        atoms["ST2"].a = _clip_to_range(min(atoms["ST2"].a, 0.02), ranges["ST2"]["a"])


__all__ = [
    "BASE_SCAFFOLD_RANGES",
    "CLASS_PARAM_RANGES",
    "_deepcopy_ranges",
    "_merge_ranges",
    "_resolve_class_key",
    "_sample_uniform",
    "_clip_to_range",
    "_scale_repolarization_time",
    "_adapt_ranges_for_duration",
    "_sample_atoms",
    "_apply_morphology_rules",
]
