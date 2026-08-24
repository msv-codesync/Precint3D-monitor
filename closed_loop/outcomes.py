"""Score an action five layers (or frames) later. Writes back into Store B."""

from __future__ import annotations

import json
from pathlib import Path


def load_pending(folder: Path) -> list[dict]:
    path = folder / "pending_outcomes.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_pending(folder: Path, rows: list[dict]) -> None:
    (folder / "pending_outcomes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def queue_action(folder: Path, *, class_name: str, layer: int, frames: int, kind: str) -> None:
    rows = load_pending(folder)
    rows.append(
        {
            "class": class_name,
            "layer": int(layer),
            "frames": int(frames),
            "kind": kind,
        }
    )
    save_pending(folder, rows)


def resolve_due(
    folder: Path,
    *,
    layer: int,
    frames: int,
    current_class: str,
) -> list[dict]:
    """Return pending rows that are 5 layers or 8 frames later, with a quality score."""
    kept: list[dict] = []
    due: list[dict] = []
    for row in load_pending(folder):
        layer_ready = int(layer) >= int(row.get("layer") or 0) + 5
        frame_ready = int(frames) >= int(row.get("frames") or 0) + 8
        if not (layer_ready or frame_ready):
            kept.append(row)
            continue
        if current_class == "good_print":
            quality = 1.0
        elif current_class == row.get("class"):
            quality = 0.0
        else:
            quality = 0.45
        due.append({**row, "outcome_quality": quality, "later_class": current_class})
    save_pending(folder, kept)
    return due
