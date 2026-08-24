"""OctoPrint / USB writes. Every write goes through ``verify_proposal`` first.

Never touches provider ``tool_api`` internals. Callers still must pass
``confirm=True``. Writable parameters stay hotend / bed / feedrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from actions import is_critical
from gcode_monitor.act import pause_marlin, pause_octoprint
from gcode_monitor.net import http_url
from tool_api.bounds import BOUNDS, clip_parameter
from tool_api.service import verify_proposal
from tool_api.types import Observation, Proposal

PARAM_TO_GCODE = {
    "hotend": lambda v: f"M104 S{int(round(v))}",
    "bed": lambda v: f"M140 S{int(round(v))}",
    "feedrate": lambda v: f"M220 S{int(round(v))}",
}


@dataclass
class ParamResult:
    ok: bool
    message: str
    sent: str = ""
    gate_label: str = ""


def ping_octoprint(url: str, api_key: str, timeout: float = 2.0) -> tuple[bool, str]:
    if not (url or "").strip():
        return False, "NO URL"
    if not (api_key or "").strip():
        return False, "NO API KEY"
    try:
        base = http_url(url)
        resp = requests.get(
            f"{base}/api/version",
            headers={"X-Api-Key": api_key},
            timeout=timeout,
        )
    except (requests.RequestException, ValueError):
        return False, "UNREACHABLE"
    if 200 <= resp.status_code < 300:
        return True, "ONLINE"
    return False, f"HTTP {resp.status_code}"


def read_octoprint_temps(url: str, api_key: str, timeout: float = 3.0) -> tuple[float | None, float | None]:
    """Live hotend / bed from OctoPrint. Missing keys stay None — never invented."""
    if not (url or "").strip() or not (api_key or "").strip():
        return None, None
    try:
        base = http_url(url)
        resp = requests.get(
            f"{base}/api/printer",
            headers={"X-Api-Key": api_key},
            timeout=timeout,
        )
    except (requests.RequestException, ValueError):
        return None, None
    if not (200 <= resp.status_code < 300):
        return None, None
    try:
        data = resp.json()
    except ValueError:
        return None, None
    temps = data.get("temperature") or {}
    tool = temps.get("tool0") or temps.get("tool") or {}
    bed = temps.get("bed") or {}
    hotend = tool.get("target")
    if hotend is None:
        hotend = tool.get("actual")
    bed_t = bed.get("target")
    if bed_t is None:
        bed_t = bed.get("actual")
    try:
        hotend_f = float(hotend) if hotend is not None else None
    except (TypeError, ValueError):
        hotend_f = None
    try:
        bed_f = float(bed_t) if bed_t is not None else None
    except (TypeError, ValueError):
        bed_f = None
    return hotend_f, bed_f


def send_octoprint_command(url: str, api_key: str, command: str, timeout: float = 5.0) -> tuple[int, str]:
    if not (api_key or "").strip():
        raise ValueError("OctoPrint API key is required.")
    base = http_url(url)
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
    resp = requests.post(
        f"{base}/api/printer/command",
        json={"command": command},
        headers=headers,
        timeout=timeout,
    )
    return resp.status_code, resp.text


def apply_parameter(
    *,
    url: str,
    api_key: str,
    param: str,
    value: float,
    confirm: bool,
    observation: Observation | None,
    sticky_ok: bool,
    class_name: str,
) -> ParamResult:
    if param not in PARAM_TO_GCODE:
        return ParamResult(False, f"{param} is not writable.")
    clipped, _changed = clip_parameter(param, value)
    if not confirm:
        return ParamResult(False, f"Would send {PARAM_TO_GCODE[param](clipped)} (needs confirm).")
    if observation is None:
        return ParamResult(False, "No live observation to verify against.")
    gate = verify_proposal(
        observation,
        Proposal(command="set_parameter", param=param, value=clipped, source="operator"),
        sticky_ok=bool(sticky_ok),
        class_name=class_name or "good_print",
        critical=is_critical(class_name or "good_print"),
    )
    if not gate.accept:
        return ParamResult(
            False,
            f"{gate.label}: {'; '.join(gate.reasons)}",
            gate_label=gate.label,
        )
    if not url or not api_key:
        return ParamResult(False, "OctoPrint URL and API key are required.", gate_label=gate.label)
    line = PARAM_TO_GCODE[param](clipped)
    try:
        status, body = send_octoprint_command(url, api_key, line)
    except (requests.RequestException, ValueError) as exc:
        return ParamResult(False, f"Request failed: {exc}", sent=line, gate_label=gate.label)
    if 200 <= status < 300:
        return ParamResult(True, f"OctoPrint accepted ({status}).", sent=line, gate_label=gate.label)
    return ParamResult(False, f"OctoPrint refused ({status}): {body[:120]}", sent=line, gate_label=gate.label)


def request_pause(
    *,
    link: str,
    url: str,
    api_key: str,
    port: str,
    baud: int,
    observation: Observation | None,
    sticky_ok: bool,
    class_name: str,
) -> ParamResult:
    if observation is None:
        return ParamResult(False, "No live observation to verify against.")
    gate = verify_proposal(
        observation,
        Proposal(command="pause", source="operator"),
        sticky_ok=bool(sticky_ok),
        class_name=class_name or "good_print",
        critical=is_critical(class_name or "good_print"),
    )
    if not (gate.accept and gate.command == "pause"):
        return ParamResult(
            False,
            f"{gate.label}: {'; '.join(gate.reasons)}",
            gate_label=gate.label,
        )
    try:
        if link == "OctoPrint":
            msg = pause_octoprint(url, api_key)
        elif link == "Mega S USB":
            msg = pause_marlin(port, int(baud))
        else:
            return ParamResult(False, "No printer link for pause.", gate_label=gate.label)
    except Exception as exc:  # noqa: BLE001 — surface serial/HTTP failures to the operator
        return ParamResult(False, f"Pause failed: {exc}", gate_label=gate.label)
    return ParamResult(True, msg, sent="pause", gate_label=gate.label)


def bounds_for(param: str) -> tuple[float, float]:
    return BOUNDS[param]
