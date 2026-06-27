"""Shared constants for the ECG beat simulator."""

from ecg_few.findings import LABEL_NAMES as QRS_LABEL_NAMES

ATOM_ORDER = ["P", "Q", "R", "S", "R_prime", "J_elev", "ST1", "ST2", "T1", "T2"]
SOURCE_FAMILIES = ["NORMAL", "RBBB", "ST_ELEVATION", "T_WAVE_INVERSION", "BRUGADA"]
LABEL_NAMES = list(QRS_LABEL_NAMES)
REFERENCE_DURATION_MS = 900.0
REPOLARIZATION_PIVOT_MS = 390.0

__all__ = [
    "ATOM_ORDER",
    "SOURCE_FAMILIES",
    "LABEL_NAMES",
    "REFERENCE_DURATION_MS",
    "REPOLARIZATION_PIVOT_MS",
]
