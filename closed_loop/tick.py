"""One perceive → retrieve → reason → verify → act tick. Wraps cycle(); does not replace it."""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from actions import pretty as pretty_action
from agent_loop.cycle import cycle
from closed_loop.outcomes import queue_action, resolve_due
from closed_loop.policy import planned_nudges
from gcode_monitor.pipeline import judge_frame, judge_image
from gcode_monitor.progress import Progress, from_manual, from_marlin_sd, from_octoprint
from gcode_monitor.store import JobStore
from paths import ROOT
from printer_ops import apply_parameter, read_octoprint_temps, request_pause

JOBS_ROOT = ROOT / "jobs"
LLM_NOTE_COOLDOWN_S = 45.0


def progress_for(watch: dict, parsed) -> Progress:
    link = watch.get("link")
    if link == "OctoPrint" and watch.get("octo_url") and watch.get("octo_key"):
        return from_octoprint(watch["octo_url"], watch["octo_key"])
    if link == "Mega S USB" and watch.get("port"):
        file_size = parsed.size if parsed is not None else None
        return from_marlin_sd(watch["port"], int(watch.get("baud") or 250000), file_size=file_size)
    if parsed is not None and parsed.layer_count:
        layer_i = int(watch.get("layer_i") or 0)
        layer_i = max(0, min(layer_i, parsed.layer_count - 1))
        return from_manual(layer_i, parsed.layer_count, parsed.size)
    fraction = 0.0
    if parsed is not None and parsed.layer_count:
        fraction = min(1.0, int(watch.get("frames") or 0) / max(parsed.layer_count, 1))
    return Progress(
        source="camera" if link == "Camera only" else str(link or "camera"),
        file_pos=None,
        fraction=fraction,
        layer_hint=None,
        extra="camera watch",
    )


def _temps(watch: dict) -> tuple[float | None, float | None]:
    if watch.get("link") == "OctoPrint":
        return read_octoprint_temps(watch.get("octo_url") or "", watch.get("octo_key") or "")
    return None, None


def run_tick(
    *,
    watch: dict,
    image: Image.Image,
    model,
    sticky,
    parsed,
    memory,
    runtime: dict,
    conf: float,
    iou: float,
) -> dict:
    """Mutates watch and runtime. Returns a JSON-safe status dict for the console."""
    folder = Path(watch["folder"])
    store = JobStore(JOBS_ROOT)
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False, stream=False)
    try:
        progress = progress_for(watch, parsed)
        runtime["progress_error"] = ""
    except Exception as exc:  # noqa: BLE001
        runtime["progress_error"] = str(exc)[:200]
        progress = Progress(source="camera", file_pos=None, fraction=0.0, layer_hint=None, extra="camera")

    verdict = (
        judge_frame(parsed, progress, image, results, sticky)
        if parsed is not None
        else judge_image(image, results, sticky, progress=progress)
    )
    hotend, bed = _temps(watch)
    now = time.time()
    last_deep = float(runtime.get("last_deep") or 0)
    noted = set(runtime.get("noted_classes") or [])
    first_for_class = verdict.name not in noted
    want_note = (
        verdict.sticky_ok
        and verdict.name != "good_print"
        and (first_for_class or (now - last_deep) >= LLM_NOTE_COOLDOWN_S)
    )
    try:
        outcome = cycle(verdict, memory, deep=want_note, camera=True, hotend=hotend, bed=bed)
    except Exception as exc:  # noqa: BLE001
        outcome = cycle(verdict, memory, deep=False, camera=True, hotend=hotend, bed=bed)
        outcome["record"]["reason"] = pretty_action(verdict.name)["what_to_do"]
        outcome["record"]["note_error"] = str(exc)[:240]
    if want_note:
        runtime["last_deep"] = now
        noted.add(verdict.name)
        runtime["noted_classes"] = sorted(noted)

    watch["frames"] = int(watch.get("frames") or 0) + 1
    record = outcome["record"]
    event = {
        "i": watch["frames"],
        "clock": datetime.now().isoformat(timespec="seconds"),
        "kind": "frame",
        **record,
    }
    try:
        store.append_event(folder, event)
    except OSError:
        pass

    gate = outcome["gate"]
    acted = []
    if (
        watch.get("auto_pause")
        and watch.get("pause_allowed")
        and not watch.get("pause_sent")
        and gate.accept
        and gate.command == "pause"
    ):
        result = request_pause(
            link=watch.get("link") or "",
            url=watch.get("octo_url") or "",
            api_key=watch.get("octo_key") or "",
            port=watch.get("port") or "",
            baud=int(watch.get("baud") or 250000),
            observation=outcome.get("observation"),
            sticky_ok=bool(record.get("sticky")),
            class_name=record.get("class") or "good_print",
        )
        store.append_event(
            folder,
            {
                "t": datetime.now().isoformat(),
                "kind": "pause_requested",
                "via": "auto",
                "ok": result.ok,
                "result": result.message,
                "gate": result.gate_label,
                "class": record.get("class"),
            },
        )
        if result.ok:
            watch["pause_sent"] = True
            queue_action(folder, class_name=str(record.get("class")), layer=int(verdict.layer_index), frames=watch["frames"], kind="pause")
            acted.append("pause")

    if (
        watch.get("auto_nudge")
        and watch.get("pause_allowed")
        and watch.get("link") == "OctoPrint"
        and verdict.sticky_ok
        and pretty_action(verdict.name)["severity"] == "warning"
    ):
        already = set(runtime.get("nudged_classes") or [])
        if verdict.name not in already:
            for param, value in planned_nudges(verdict.name, hotend=hotend, bed=bed):
                result = apply_parameter(
                    url=watch.get("octo_url") or "",
                    api_key=watch.get("octo_key") or "",
                    param=param,
                    value=float(value),
                    confirm=True,
                    observation=outcome.get("observation"),
                    sticky_ok=True,
                    class_name=verdict.name,
                )
                store.append_event(
                    folder,
                    {
                        "t": datetime.now().isoformat(),
                        "kind": "nudge",
                        "via": "auto",
                        "ok": result.ok,
                        "sent": result.sent,
                        "gate": result.gate_label,
                        "class": verdict.name,
                    },
                )
                if result.ok:
                    acted.append(f"{param}={value}")
            already.add(verdict.name)
            runtime["nudged_classes"] = sorted(already)
            if acted:
                queue_action(folder, class_name=verdict.name, layer=int(verdict.layer_index), frames=watch["frames"], kind="nudge")

    for row in resolve_due(folder, layer=int(verdict.layer_index), frames=watch["frames"], current_class=verdict.name):
        memory.append(
            {
                "class": row.get("class"),
                "outcome_quality": row.get("outcome_quality"),
                "later_class": row.get("later_class"),
                "kind": row.get("kind"),
                "verdict": "outcome",
                "reason": "measured after the action",
            }
        )
        store.append_event(folder, {"t": datetime.now().isoformat(), "kind": "outcome", **row})

    marked = verdict.marked_image
    try:
        marked.convert("RGB").save(folder / "live.jpg", quality=82)
    except OSError:
        pass

    obs = outcome.get("observation")
    return {
        "record": record,
        "gate": {
            "accept": bool(gate.accept),
            "label": gate.label,
            "command": gate.command,
            "reasons": list(gate.reasons or []),
        },
        "observation": asdict(obs) if obs is not None else None,
        "progress_source": progress.source,
        "layer": verdict.layer_index,
        "z": verdict.layer_z,
        "frames": watch["frames"],
        "acted": acted,
        "progress_error": runtime.get("progress_error") or "",
        "error": "",
        "updated": datetime.now().isoformat(timespec="seconds"),
        "worker": "running",
    }
