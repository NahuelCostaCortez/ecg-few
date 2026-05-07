"""Task constants and JSON schemas for ECG visual QA tasks."""

from __future__ import annotations

from ecg_vlm.simulator.constants import LABEL_NAMES

TASK_MULTILABEL = "multilabel_findings"
TASK_BINARY = "binary_finding"

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are evaluating synthetic single-lead ECG beat images for visual morphology "
    "findings. The task is not to diagnose a patient. Return only the requested JSON."
)

SYNTHETIC_RATIONALE_TEXT = {
    "RBBB": "The simulator label marks an RBBB-like terminal QRS morphology.",
    "ST_ELEVATION": "The simulator label marks elevation of the ST/J-point region.",
    "T_WAVE_INVERSION": "The simulator label marks inversion of the repolarization wave.",
}


def multilabel_json_schema() -> dict[str, object]:
    """Return the strict JSON schema for multi-label finding answers."""
    return {
        "type": "object",
        "properties": {label_name: {"type": "boolean"} for label_name in LABEL_NAMES},
        "required": list(LABEL_NAMES),
        "additionalProperties": False,
    }


def multilabel_answer_text(answer: dict[str, object]) -> str:
    """Format an expected multi-label answer as compact JSON text."""
    import json

    return json.dumps(
        {label_name: bool(answer[label_name]) for label_name in LABEL_NAMES},
        sort_keys=True,
    )
