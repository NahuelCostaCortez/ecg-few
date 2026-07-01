"""Canonical QRS finding labels and derived Brugada rule."""

from __future__ import annotations

from collections.abc import Mapping

LABEL_NAMES = ("RBBB", "ST_ELEVATION", "T_WAVE_INVERSION")
LABEL_COLUMNS = {
    "RBBB": "label_rbbb",
    "ST_ELEVATION": "label_st_elevation",
    "T_WAVE_INVERSION": "label_t_wave_inversion",
}
NORMAL_TOKEN = "NORMAL"
BRUGADA_DERIVATION_RULE = "RBBB and ST_ELEVATION and T_WAVE_INVERSION"


def normalize_findings(values: Mapping[str, object]) -> dict[str, int]:
    """Read canonical or CSV-column finding keys as 0/1 integers."""
    findings: dict[str, int] = {}
    for label_name in LABEL_NAMES:
        column = LABEL_COLUMNS[label_name]
        if label_name in values:
            raw_value = values[label_name]
        elif column in values:
            raw_value = values[column]
        else:
            raise KeyError(f"Missing QRS finding label: {label_name} / {column}")
        findings[label_name] = _coerce_binary(raw_value)
    return findings


def findings_to_brugada(values: Mapping[str, object]) -> int:
    """Derive the final Brugada decision from the three QRS findings."""
    findings = normalize_findings(values)
    return int(all(bool(findings[label_name]) for label_name in LABEL_NAMES))


def findings_to_text(values: Mapping[str, object]) -> str:
    findings = normalize_findings(values)
    present = [label_name for label_name in LABEL_NAMES if findings[label_name]]
    return NORMAL_TOKEN if not present else "+".join(present)


def _coerce_binary(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return 1
        if normalized in {"0", "false", "no", "n", ""}:
            return 0
    if isinstance(value, bool):
        return int(value)
    if value in {0, 1}:
        return int(bool(value))
    raise ValueError(f"Expected a binary QRS finding value, got {value!r}")


__all__ = [
    "BRUGADA_DERIVATION_RULE",
    "LABEL_COLUMNS",
    "LABEL_NAMES",
    "NORMAL_TOKEN",
    "findings_to_brugada",
    "findings_to_text",
    "normalize_findings",
]
