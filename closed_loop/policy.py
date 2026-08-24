"""Bounded corrective writes from the action map. LLM still cannot send G-code."""

from __future__ import annotations

from actions import pretty
from tool_api.bounds import clip_parameter

# Absolute feedrate targets from the operator action list.
FEEDRATE_FOR = {
    "warping": 70.0,
    "under_extrusion": 80.0,
    "blob_or_zit": 85.0,
    "zits_blobs": 85.0,
    "blob": 85.0,
}

# Relative temperature nudges, clipped at the tool layer.
TEMP_DELTA = {
    "warping": ("bed", 5.0),
    "under_extrusion": ("hotend", 5.0),
    "stringing": ("hotend", -3.0),
}


def planned_nudges(
    class_name: str,
    *,
    hotend: float | None,
    bed: float | None,
) -> list[tuple[str, float]]:
    """Whitelist (param, value) pairs. Empty if we do not have a live base temperature."""
    name = class_name or "good_print"
    if pretty(name)["severity"] != "warning":
        return []
    out: list[tuple[str, float]] = []
    feed = FEEDRATE_FOR.get(name)
    if feed is not None:
        value, _ = clip_parameter("feedrate", feed)
        out.append(("feedrate", value))
    delta = TEMP_DELTA.get(name)
    if delta is not None:
        param, step = delta
        base = hotend if param == "hotend" else bed
        if base is None:
            return out
        value, _ = clip_parameter(param, float(base) + float(step))
        out.append((param, value))
    return out
