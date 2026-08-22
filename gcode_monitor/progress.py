"""Where we are in the file: manual, OctoPrint, or Marlin SD print over USB."""

from __future__ import annotations

import re
from dataclasses import dataclass

from gcode_monitor.net import http_url
from gcode_monitor.serial_link import shared_link


@dataclass
class Progress:
    source: str
    file_pos: int | None
    fraction: float
    layer_hint: int | None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    extra: str = ""


def from_manual(layer_index: int, layer_count: int, file_size: int) -> Progress:
    count = max(layer_count, 1)
    frac = (layer_index + 1) / count
    return Progress(
        source="manual",
        file_pos=None,
        fraction=frac,
        layer_hint=layer_index,
        extra=f"layer {layer_index + 1}/{count}",
    )


def from_octoprint(url: str, api_key: str, timeout: float = 4.0) -> Progress:
    import requests

    base = http_url(url)
    resp = requests.get(
        f"{base}/api/job",
        headers={"X-Api-Key": api_key},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    progress = data.get("progress") or {}
    completion = float(progress.get("completion") or 0.0)
    if completion > 1:
        completion = completion / 100.0
    file_pos = progress.get("filepos")
    state = data.get("state") or ""
    return Progress(
        source="octoprint",
        file_pos=int(file_pos) if file_pos is not None else None,
        fraction=max(0.0, min(completion, 1.0)),
        layer_hint=None,
        extra=str(state),
    )


def _parse_m27(lines: list[str], file_size: int | None) -> tuple[int | None, float, str]:
    extra = " ".join(lines[-4:]) if lines else "no reply"
    file_pos = None
    fraction = 0.0
    for line in lines:
        lower = line.lower()
        if "not sd printing" in lower:
            return None, 0.0, line
        if "sd printing byte" not in lower:
            continue
        tail = lower.split("byte", 1)[-1].strip()
        left, _, right = tail.partition("/")
        cur = int("".join(ch for ch in left if ch.isdigit()) or "0")
        total = int("".join(ch for ch in right if ch.isdigit()) or "0")
        file_pos = cur
        if total:
            fraction = cur / total
        elif file_size:
            fraction = cur / file_size
        extra = line
        break
    return file_pos, fraction, extra


def from_marlin_sd(port: str, baud: int = 250000, file_size: int | None = None) -> Progress:
    """Anycubic Mega S: keep the port open and poll M27 / M114."""
    link = shared_link()
    link.connect(port, baud)
    m27 = link.query("M27", seconds=0.8)
    file_pos, fraction, extra = _parse_m27(m27, file_size)

    x = y = z = None
    m114 = link.query("M114", seconds=0.6)
    for line in m114:
        if "X:" in line and "Y:" in line:
            found = re.findall(r"([XYZE]):\s*([-+0-9.]+)", line)
            nums = {k: float(v) for k, v in found}
            x, y, z = nums.get("X"), nums.get("Y"), nums.get("Z")
            extra = extra + " | " + line
            break

    return Progress(
        source="marlin-sd",
        file_pos=file_pos,
        fraction=max(0.0, min(fraction, 1.0)),
        layer_hint=None,
        x=x,
        y=y,
        z=z,
        extra=extra,
    )
