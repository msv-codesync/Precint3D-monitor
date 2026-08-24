"""Store A: static reference next to the detector. Keyword retrieve, no invented corpus."""

from __future__ import annotations

from pathlib import Path

from paths import ROOT

KNOWLEDGE = ROOT / "knowledge"


def retrieve(class_name: str, *, limit_chars: int = 900) -> str:
    """Return the Store A snippets that match this class (and always the bounds)."""
    if not KNOWLEDGE.is_dir():
        return ""
    needle = (class_name or "").strip().lower().replace(" ", "_")
    parts: list[str] = []
    bounds = KNOWLEDGE / "tool_bounds.md"
    if bounds.exists():
        parts.append(bounds.read_text(encoding="utf-8").strip())
    for path in sorted(KNOWLEDGE.glob("*.md")):
        if path.name == "tool_bounds.md":
            continue
        text = path.read_text(encoding="utf-8")
        if not needle or needle in text.lower():
            parts.append(text.strip())
    blob = "\n\n".join(parts)
    return blob[: int(limit_chars)]
