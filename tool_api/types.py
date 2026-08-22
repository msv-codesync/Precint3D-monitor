from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Observation:
    timestamp: str
    layer: int | None
    layer_z: float | None
    file_pos: int | None
    progress: float
    gcode_line: str
    hotend: float | None
    bed: float | None
    camera: str | None  # "present" or None — we do not embed JPEG in the log
    filament_id: str | None
    vibration_fft: list[float] | None
    audio_rms: float | None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_log(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("camera", None)
        return data


@dataclass
class Proposal:
    command: str  # none | pause | set_parameter
    param: str | None = None
    value: float | None = None
    reason: str = ""
    source: str = "detector"


@dataclass
class Verdict:
    accept: bool
    label: str  # ALLOW | CLIP | BLOCK
    reasons: list[str]
    command: str
    param: str | None = None
    value: float | None = None
