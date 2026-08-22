"""Pause-only printer writes. No LLM-authored G-code."""

from __future__ import annotations

from gcode_monitor.net import http_url
from gcode_monitor.serial_link import shared_link

PAUSE = "M25"  # Marlin SD pause


def pause_octoprint(url: str, api_key: str) -> str:
    import requests

    base = http_url(url)
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
    resp = requests.post(
        f"{base}/api/job",
        json={"command": "pause", "action": "pause"},
        headers=headers,
        timeout=5,
    )
    return f"octoprint pause HTTP {resp.status_code}"


def pause_marlin(port: str, baud: int) -> str:
    link = shared_link()
    link.connect(port, baud)
    link.send(PAUSE)
    return f"marlin sent {PAUSE}"
