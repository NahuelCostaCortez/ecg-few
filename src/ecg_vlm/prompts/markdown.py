"""Load doctor-editable Markdown prompt templates."""

from __future__ import annotations

from pathlib import Path


def load_markdown_prompt(path: str | Path) -> str:
    """Load a Markdown prompt, stripping simple YAML frontmatter if present."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].strip()
    return text


def render_markdown_prompt(path: str | Path, **values: object) -> str:
    """Load and render a Markdown prompt with Python ``str.format`` variables."""
    return load_markdown_prompt(path).format(**values)
