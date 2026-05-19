"""Synthetic ECG beat simulator integration."""

from .constants import LABEL_NAMES, SOURCE_FAMILIES
from .simulator import generate_beat

__all__ = [
    "LABEL_NAMES",
    "SOURCE_FAMILIES",
    "generate_beat",
]
