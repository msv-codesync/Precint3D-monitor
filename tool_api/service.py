"""Four functions from the internship abstract. Missing sensors stay None."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tool_api.bounds import ALLOWED_COMMANDS, ALLOWED_PARAMS, clip_parameter
from tool_api.types import Observation, Proposal, Verdict


def read_state(
    *,
    layer: int | None,
    layer_z: float | None,
    file_pos: int | None,
    progress: float,
    gcode_line: str,
    hotend: float | None = None,
    bed: float | None = None,
    camera_present: bool = False,
    filament_id: str | None = None,
) -> Observation:
    return Observation(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        layer=layer,
        layer_z=layer_z,
        file_pos=file_pos,
        progress=float(progress),
        gcode_line=gcode_line[:400],
        hotend=hotend,
        bed=bed,
        camera="present" if camera_present else None,
        filament_id=filament_id,
        vibration_fft=None,
        audio_rms=None,
        extras={
            "imu": "not wired in this software drop",
            "mic": "not wired in this software drop",
            "max31855": "not wired in this software drop",
        },
    )


def set_parameter(name: str, value: float) -> tuple[float, bool]:
    return clip_parameter(name, value)


def verify_proposal(obs: Observation, proposal: Proposal, *, sticky_ok: bool, class_name: str, critical: bool) -> Verdict:
    reasons: list[str] = []
    command = proposal.command
    param = proposal.param
    value = proposal.value

    if command not in ALLOWED_COMMANDS:
        return Verdict(False, "BLOCK", [f"{command} is not on the tool whitelist"], "none")

    if command == "none":
        return Verdict(True, "ALLOW", ["no machine write"], "none")

    if not sticky_ok:
        reasons.append("finding is not held across frames")

    if command == "pause":
        if not critical:
            reasons.append("pause is reserved for critical classes")
        if class_name == "good_print":
            reasons.append("detector says the print looks fine")
        if reasons:
            return Verdict(False, "BLOCK", reasons, "none")
        return Verdict(True, "ALLOW", ["pause after held critical finding"], "pause")

    if command == "set_parameter":
        if param not in ALLOWED_PARAMS:
            return Verdict(False, "BLOCK", [f"{param} is not a writable parameter"], "none")
        if value is None:
            return Verdict(False, "BLOCK", ["no value"], "none")
        clipped, changed = clip_parameter(param, value)
        if changed:
            return Verdict(
                False,
                "CLIP",
                [f"{param}={value} is outside the hard window; refused"],
                "none",
            )
        if reasons:
            return Verdict(False, "BLOCK", reasons, "none")
        return Verdict(True, "ALLOW", [f"set {param}={clipped}"], "set_parameter", param, clipped)

    return Verdict(False, "BLOCK", ["unhandled command"], "none")


def pause_print(reason: str, sink) -> str:
    return sink(reason)


def generate_report(session_dir: Path) -> Path:
    session_dir = Path(session_dir).resolve()
    if not session_dir.is_dir():
        raise ValueError("session folder does not exist")
    if not (session_dir / "events.jsonl").exists() and not (session_dir / "meta.json").exists():
        raise ValueError("not a Precint3D job folder")
    events_path = session_dir / "events.jsonl"
    lines = []
    if events_path.exists():
        lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta = ""
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        meta = meta_path.read_text(encoding="utf-8")
    body = [
        "# Precint3D session record",
        "",
        "Written while the loop ran. This file is the audit trail, not a summary added later.",
        "",
        "## meta.json",
        "```json",
        meta.strip() or "{}",
        "```",
        "",
        f"## events ({len(lines)})",
        "",
    ]
    for ln in lines:
        body.append(f"- `{ln}`")
    out = session_dir / "report.md"
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    return out
