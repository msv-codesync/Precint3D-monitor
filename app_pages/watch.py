"""Watch page. Live camera-first monitoring with a hard printer-write boundary."""

from __future__ import annotations

import base64
import html
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

from actions import pretty as pretty_action
from agent_loop.cycle import cycle
from agent_loop.memory import DecisionMemory
from closed_loop.live import attach_if_running, read_run, read_status, write_run
from closed_loop.worker import loop_alive, start_worker, stop_worker
from gcode_monitor.capture import (
    ProcessWebcam,
    default_camera_index,
    from_esp32,
    kill_stray_camera_workers,
    list_cameras,
)
from gcode_monitor.filter import StickyClass
from gcode_monitor.parse import parse_gcode
from gcode_monitor.pipeline import judge_frame, judge_image
from gcode_monitor.progress import Progress, from_manual, from_marlin_sd, from_octoprint
from gcode_monitor.store import JobStore, list_sessions
from paths import ROOT
from printer_ops import apply_parameter, bounds_for, ping_octoprint, request_pause
from tool_api.types import Observation
from ui_common import load_model


def _image_data_uri(image: Image.Image, quality: int = 82) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render_camera_frame(
    image: Image.Image | None,
    *,
    cam_idx: int,
    conf_pct: int,
    live_str: str,
    cam_name: str | None = None,
) -> None:
    tag = html.escape((cam_name or f"CAM 0{cam_idx}").upper())
    if len(tag) > 28:
        tag = tag[:28]
    overlay = tag
    if image is None:
        st.markdown(
            f"""
            <div class='p3-cam-frame empty'>
              <div class='p3-cam-ov tl'>{overlay}</div>
              <div class='p3-cam-ov tr'>LIVE · {live_str}</div>
              <div class='p3-cam-ov bl'></div>
              <div class='p3-cam-ov br'>{conf_pct}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    uri = _image_data_uri(image)
    st.markdown(
        f"""
        <div class='p3-cam-frame'>
          <img src="{uri}" />
          <div class='p3-cam-ov tl'>{overlay}</div>
          <div class='p3-cam-ov tr'>LIVE · {live_str}</div>
          <div class='p3-cam-ov bl'></div>
          <div class='p3-cam-ov br'>{conf_pct}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

JOBS_ROOT = ROOT / "jobs"
MEMORY_PATH = JOBS_ROOT / "memory.jsonl"
SAMPLE_GCODE = ROOT / "gcode_monitor" / "fixtures" / "sample_square.gcode"
FRAME_INTERVAL_S = 1.5
LLM_NOTE_COOLDOWN_S = 45.0
STICKY_NEED = 3


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

def _read_events(folder: Path) -> list[dict]:
    events_path = folder / "events.jsonl"
    if not events_path.exists():
        return []
    out: list[dict] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _kpis(active: bool) -> dict[str, str]:
    sessions = list_sessions(JOBS_ROOT) if JOBS_ROOT.exists() else []
    critical = 0
    week_count = 0
    hours = 0.0
    now = datetime.now(timezone.utc)
    for row in sessions:
        folder = JOBS_ROOT / row["session"]
        events = _read_events(folder)
        stamps: list[datetime] = []
        for ev in events:
            cls = ev.get("class") or ""
            if cls and pretty_action(cls)["severity"] == "critical" and ev.get("sticky"):
                critical += 1
            raw = ev.get("clock") or ev.get("t")
            if not raw:
                continue
            try:
                stamps.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
            except ValueError:
                continue
        try:
            stamp = datetime.strptime(row["session"].split("_", 1)[0], "%Y%m%d-%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, IndexError):
            continue
        if (now - stamp).days <= 7:
            week_count += 1
        if (now - stamp).days <= 30 and len(stamps) >= 2:
            first, last = min(stamps), max(stamps)
            if first.tzinfo is None:
                first = first.replace(tzinfo=timezone.utc)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            hours += max(0.0, (last - first).total_seconds() / 3600.0)
    return {
        "active": "1" if active else "0",
        "checked": str(week_count),
        "critical": str(critical),
        "uptime": f"{hours:.1f}h" if hours else "0h",
    }


# ---------------------------------------------------------------------------
# Camera picker
# ---------------------------------------------------------------------------

def _camera_label(cam: dict) -> str:
    extra = ""
    if cam.get("preferred"):
        extra = " · iPhone"
    elif "face" in str(cam.get("name") or "").lower() or "mac" in str(cam.get("name") or "").lower():
        extra = " · Mac"
    return f"{cam['name']}{extra}"


def _refresh_cameras(*, probe: bool = False) -> list[dict]:
    catalog = list_cameras(probe=probe)
    st.session_state.cam_catalog = catalog
    return catalog


def _hard_close_camera() -> None:
    """Kill the camera child process so the Mac LED can actually go off."""
    st.session_state["_wp_cam_hold"] = True
    cam = st.session_state.get("_wp_cam")
    if cam is not None:
        try:
            cam.close()
        except Exception:
            pass
    st.session_state.pop("_wp_cam", None)
    kill_stray_camera_workers()


def _pick_camera_ui(
    *,
    key: str,
    current_index: int | None = None,
) -> tuple[int, str]:
    catalog = st.session_state.get("cam_catalog")
    if not catalog:
        catalog = _refresh_cameras(probe=False)
    if not catalog:
        st.warning("No camera listed.")
        return 0, "Camera 0"
    labels = [_camera_label(cam) for cam in catalog]
    pref = default_camera_index(catalog)
    default_pos = next((i for i, cam in enumerate(catalog) if cam["index"] == pref), 0)
    if current_index is not None:
        current_pos = next((i for i, cam in enumerate(catalog) if cam["index"] == int(current_index)), default_pos)
    else:
        current_pos = default_pos
    if key not in st.session_state or st.session_state.get(key) not in labels:
        st.session_state[key] = labels[current_pos]
    choice = st.selectbox("Camera", labels, key=key)
    picked = catalog[labels.index(choice)]
    return int(picked["index"]), str(picked["name"])


# ---------------------------------------------------------------------------
# Dialog: Start a camera watch
# ---------------------------------------------------------------------------

def _reset_watch_runtime():
    _hard_close_camera()
    for k in (
        "_wp_cam",
        "_wp_sticky",
        "_wp_memory",
        "_wp_parsed",
        "_wp_last_verdict",
        "_wp_last_outcome",
        "_wp_last_marked",
        "_wp_last_conf",
        "_wp_last_deep_at",
        "_wp_ping",
        "_wp_last_progress",
        "_wp_progress_error",
        "_wp_cam_hold",
        "_wp_force_cam",
    ):
        st.session_state.pop(k, None)


@st.dialog("Start watch")
def _start_watch_dialog():
    st.session_state.setdefault("new_name", "")
    st.session_state.setdefault("new_link", "Camera only")
    st.session_state.setdefault("new_octo_url", "")
    st.session_state.setdefault("new_octo_key", "")
    st.session_state.setdefault("new_pause", True)
    st.session_state.setdefault("new_auto_pause", True)
    st.session_state.setdefault("new_auto_nudge", True)
    st.session_state.setdefault("new_cam_kind", "Local camera")
    st.session_state.setdefault("new_esp_url", "http://192.168.1.20/capture")
    st.session_state.setdefault("new_port", "/dev/tty.usbmodem1101")
    st.session_state.setdefault("new_baud", 250000)
    st.session_state.setdefault("new_gcode_src", "None")
    st.session_state.setdefault("new_sticky_need", STICKY_NEED)

    name = st.text_input("Print name", key="new_name")

    with st.expander("Camera", expanded=True):
        cam_kind = st.radio(
            "Source",
            ["Local camera", "ESP32"],
            horizontal=True,
            key="new_cam_kind",
        )
        cam_idx, cam_name = 0, "Camera 0"
        if cam_kind == "Local camera":
            cam_idx, cam_name = _pick_camera_ui(key="new_cam_label")
        else:
            st.text_input("ESP32 capture URL", key="new_esp_url")

    with st.expander("Printer"):
        link = st.radio(
            "Printer link",
            ["Camera only", "OctoPrint", "Mega S USB"],
            horizontal=True,
            key="new_link",
        )
        if link == "OctoPrint":
            st.text_input("OctoPrint URL", key="new_octo_url", placeholder="http://octopi.local")
            st.text_input("API key", key="new_octo_key", type="password")
        elif link == "Mega S USB":
            s1, s2 = st.columns(2)
            s1.text_input("Serial port", key="new_port")
            s2.number_input("Baud", key="new_baud", step=1)
        if link != "Camera only":
            st.checkbox("Allow pause", key="new_pause")
            st.checkbox("Auto-pause", key="new_auto_pause")
            st.checkbox("Auto-nudge", key="new_auto_nudge")
        st.slider("Hold after frames", 1, 6, key="new_sticky_need")
        gcode_src = st.radio(
            "G-code",
            ["None", "Sample square", "Upload file"],
            horizontal=True,
            key="new_gcode_src",
        )
        gcode_file = None
        if gcode_src == "Upload file":
            gcode_file = st.file_uploader(
                "G-code",
                type=["gcode", "gco", "txt"],
                key="new_gcode",
            )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Cancel", use_container_width=True, key="dlg_cancel"):
            st.rerun()
    with c2:
        submit = st.button(
            "Start watch",
            type="primary",
            use_container_width=True,
            icon=":material/play_arrow:",
            key="dlg_submit",
        )
    if not submit:
        return

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    store = JobStore(JOBS_ROOT)
    print_name = (name or "camera watch").strip()
    gcode_bytes = b""
    gcode_name = "camera.watch"
    if gcode_src == "Sample square" and SAMPLE_GCODE.exists():
        gcode_bytes = SAMPLE_GCODE.read_bytes()
        gcode_name = SAMPLE_GCODE.name
    elif gcode_file is not None:
        try:
            gcode_bytes = gcode_file.read()
        except (OSError, ValueError):
            gcode_bytes = b""
        gcode_name = gcode_file.name or "print.gcode"
    try:
        folder = store.create(gcode_bytes or b"", gcode_name if gcode_bytes else f"{print_name}.camera")
    except OSError as exc:
        st.error(f"Could not create session folder: {exc}")
        return

    can_write = link in ("OctoPrint", "Mega S USB")
    watch = {
        "name": print_name,
        "started_iso": datetime.now().isoformat(),
        "cam_kind": cam_kind,
        "cam_idx": int(cam_idx),
        "cam_name": cam_name if cam_kind == "Local camera" else "ESP32",
        "esp_url": st.session_state.new_esp_url.strip(),
        "link": link,
        "octo_url": st.session_state.new_octo_url.strip(),
        "octo_key": st.session_state.new_octo_key.strip(),
        "port": str(st.session_state.new_port).strip(),
        "baud": int(st.session_state.new_baud or 250000),
        "pause_allowed": bool(st.session_state.new_pause) and can_write,
        "auto_pause": bool(st.session_state.new_auto_pause) and can_write,
        "auto_nudge": bool(st.session_state.new_auto_nudge) and can_write,
        "sticky_need": int(st.session_state.new_sticky_need or STICKY_NEED),
        "gcode_name": gcode_name if gcode_bytes else "",
        "has_gcode": bool(gcode_bytes),
        "layer_i": 0,
        "folder": str(folder),
        "frames": 0,
        "pause_sent": False,
        "model_path": str(st.session_state.model_path),
        "conf_thresh": float(st.session_state.conf_thresh),
        "iou_thresh": float(st.session_state.iou_thresh),
    }
    stop_worker()
    st.session_state.watch = watch
    st.session_state.watch_folder = str(folder)
    _reset_watch_runtime()
    st.session_state.watch_events = []
    st.session_state.watch_started_at = time.time()
    start_worker(watch)
    st.rerun()


# ---------------------------------------------------------------------------
# Watch runtime helpers
# ---------------------------------------------------------------------------

def _get_camera(watch: dict) -> ProcessWebcam | None:
    if st.session_state.get("_wp_cam_hold"):
        return None
    want = int(watch.get("cam_idx", 0))
    cam = st.session_state.get("_wp_cam")
    if cam is not None and getattr(cam, "index", None) == want and getattr(cam, "cap", None) is not None:
        return cam
    if cam is not None:
        try:
            cam.close()
        except Exception:
            pass
        st.session_state.pop("_wp_cam", None)
    try:
        cam = ProcessWebcam(want)
    except (RuntimeError, ValueError):
        return None
    st.session_state["_wp_cam"] = cam
    return cam


def _get_sticky(watch: dict | None = None) -> StickyClass:
    sticky = st.session_state.get("_wp_sticky")
    need = int((watch or {}).get("sticky_need") or STICKY_NEED)
    if sticky is None or getattr(sticky, "need", None) != need:
        sticky = StickyClass(need=need)
        st.session_state["_wp_sticky"] = sticky
    return sticky


def _get_memory() -> DecisionMemory:
    mem = st.session_state.get("_wp_memory")
    if mem is None:
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        mem = DecisionMemory(MEMORY_PATH)
        st.session_state["_wp_memory"] = mem
    return mem


def _get_parsed(watch: dict):
    if not watch.get("has_gcode"):
        return None
    parsed = st.session_state.get("_wp_parsed")
    if parsed is not None:
        return parsed
    gcode_path = Path(watch["folder"]) / "print.gcode"
    if not gcode_path.exists() or gcode_path.stat().st_size == 0:
        return None
    try:
        parsed = parse_gcode(gcode_path)
    except (ValueError, OSError):
        return None
    st.session_state["_wp_parsed"] = parsed
    return parsed


def _progress_for(watch: dict, parsed) -> Progress:
    link = watch.get("link")
    last = st.session_state.get("_wp_last_progress")
    st.session_state["_wp_progress_error"] = ""
    if link == "OctoPrint" and watch.get("octo_url") and watch.get("octo_key"):
        try:
            progress = from_octoprint(watch["octo_url"], watch["octo_key"])
            st.session_state["_wp_last_progress"] = progress
            return progress
        except Exception as exc:  # noqa: BLE001
            st.session_state["_wp_progress_error"] = f"OctoPrint progress failed: {exc}"
            if last is not None:
                return last
    elif link == "Mega S USB" and watch.get("port"):
        file_size = parsed.size if parsed is not None else None
        try:
            progress = from_marlin_sd(watch["port"], int(watch.get("baud") or 250000), file_size=file_size)
            st.session_state["_wp_last_progress"] = progress
            return progress
        except Exception as exc:  # noqa: BLE001
            st.session_state["_wp_progress_error"] = f"USB progress failed: {exc}"
            if last is not None:
                return last
    if parsed is not None and parsed.layer_count:
        layer_i = int(watch.get("layer_i") or 0)
        layer_i = max(0, min(layer_i, parsed.layer_count - 1))
        return from_manual(layer_i, parsed.layer_count, parsed.size)
    fraction = 0.0
    if parsed is not None and parsed.layer_count:
        fraction = min(1.0, watch["frames"] / max(parsed.layer_count, 1))
    return Progress(
        source="camera" if link == "Camera only" else str(link or "camera"),
        file_pos=None,
        fraction=fraction,
        layer_hint=None,
        extra="camera watch",
    )


def _close_watch(reason: str = "manual stop"):
    watch = st.session_state.get("watch")
    if watch:
        folder = Path(watch["folder"])
        store = JobStore(JOBS_ROOT)
        try:
            store.append_event(folder, {"t": datetime.now().isoformat(), "kind": "watch_stopped", "reason": reason})
        except OSError:
            pass
    stop_worker()
    _hard_close_camera()
    for k in (
        "watch",
        "watch_folder",
        "watch_events",
        "watch_started_at",
        "_wp_cam",
        "_wp_sticky",
        "_wp_memory",
        "_wp_parsed",
        "_wp_last_verdict",
        "_wp_last_outcome",
        "_wp_last_marked",
        "_wp_last_conf",
        "_wp_last_deep_at",
        "_wp_ping",
        "_wp_last_progress",
        "_wp_progress_error",
        "_wp_cam_hold",
        "_wp_force_cam",
    ):
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _hero(active: bool):
    left, right = st.columns([2.2, 1], vertical_alignment="center")
    with left:
        st.markdown("<div class='p3-hero'>Watch</div>", unsafe_allow_html=True)
    with right:
        with st.container(horizontal=True, horizontal_alignment="right"):
            if active:
                if st.button(
                    "Stop watch",
                    key="btn_stop_watch_hero",
                    icon=":material/stop:",
                ):
                    _close_watch("operator stopped")
                    st.rerun()
            else:
                if st.button(
                    "Start watch",
                    type="primary",
                    icon=":material/play_arrow:",
                    key="btn_start_watch_hero",
                ):
                    _start_watch_dialog()


def _kpi_row(k: dict, active: bool):
    labels = [
        ("Active", k["active"], "", active),
        ("Prints", k["checked"], "", False),
        ("Critical", k["critical"], "", False),
        ("Hours", k["uptime"], "", False),
    ]
    with st.container(horizontal=True):
        for eyebrow, value, sub, is_active in labels:
            cls = "p3-panel p3-kpi active" if is_active else "p3-panel p3-kpi"
            st.markdown(
                f"""
                <div class='{cls}' style='flex:1;min-width:200px;'>
                  <div class='p3-kpi-eyebrow'>{eyebrow}</div>
                  <div class='p3-kpi-value'>{value}</div>
                  <div class='p3-kpi-sub'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_evidence_timeline(events: list[dict]):
    frame_events = [e for e in events if e.get("kind") in (None, "frame")]
    with st.container(border=True):
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        with head_l:
            st.markdown("**Frames**")
        with head_r:
            st.markdown(
                f"<div style='text-align:right;'><span class='p3-pill mute'>{len(frame_events)} FRAMES</span></div>",
                unsafe_allow_html=True,
            )
        recent = frame_events[-6:][::-1]
        if not recent:
            st.markdown(
                "<div class='p3-evidence-row'>"
                "<div class='p3-evidence-dot'></div>"
                "<div class='p3-evidence-time'></div>"
                "<div class='p3-evidence-text'>No frames yet</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        for ev in recent:
            action = pretty_action(ev.get("class") or "good_print")
            sev = action["severity"]
            dot_cls = {"ok": "", "warning": "warn", "critical": "crit"}[sev]
            badge_cls = dot_cls
            label = {"ok": "OK", "warning": "WATCH", "critical": "PAUSE"}[sev]
            conf_pct = int(round(float(ev.get("conf") or 0) * 100))
            clock = ev.get("clock") or ev.get("t") or ""
            try:
                clock = datetime.fromisoformat(clock).strftime("%I:%M %p")
            except (ValueError, TypeError):
                pass
            st.markdown(
                f"""
                <div class='p3-evidence-row'>
                  <div class='p3-evidence-dot {dot_cls}'></div>
                  <div class='p3-evidence-time'>{clock} · {conf_pct}%</div>
                  <div class='p3-evidence-text'>{html.escape(action['label'])}</div>
                  <div class='p3-evidence-badge {badge_cls}'>{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _live_outcome() -> dict:
    status = read_status()
    record = status.get("record") or st.session_state.get("_wp_last_verdict") or {}
    gate_d = status.get("gate") or {}
    obs = None
    raw = status.get("observation")
    if isinstance(raw, dict):
        try:
            obs = Observation(**raw)
        except TypeError:
            obs = None
    gate = None
    if gate_d:
        gate = type(
            "Gate",
            (),
            {
                "accept": bool(gate_d.get("accept")),
                "command": gate_d.get("command") or "none",
                "label": gate_d.get("label") or "",
                "reasons": list(gate_d.get("reasons") or []),
            },
        )()
    return {"record": record, "gate": gate, "observation": obs, "status": status}


def _octo_status(watch: dict) -> tuple[bool, str]:
    if watch.get("link") == "Mega S USB":
        return bool(watch.get("port")), "USB" if watch.get("port") else "NO PORT"
    if watch.get("link") != "OctoPrint":
        return False, "OFFLINE"
    now = time.time()
    cache = st.session_state.get("_wp_ping")
    if isinstance(cache, dict) and now - float(cache.get("t") or 0) < 10:
        return bool(cache.get("ok")), str(cache.get("label") or "UNREACHABLE")
    ok, label = ping_octoprint(watch.get("octo_url") or "", watch.get("octo_key") or "")
    if ok:
        label = "ONLINE"
    elif label == "NO API KEY":
        label = "NO API KEY"
    elif label == "NO URL":
        label = "OFFLINE"
    st.session_state["_wp_ping"] = {"t": now, "ok": ok, "label": label}
    return ok, label


def _apply_pause(watch: dict, *, kind: str) -> bool:
    outcome = _live_outcome()
    record = outcome.get("record") or {}
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
    store = JobStore(JOBS_ROOT)
    try:
        store.append_event(
            Path(watch["folder"]),
            {
                "t": datetime.now().isoformat(),
                "kind": "pause_requested",
                "via": kind,
                "ok": result.ok,
                "result": result.message,
                "gate": result.gate_label,
                "class": record.get("class"),
            },
        )
    except OSError:
        pass
    if result.ok:
        watch["pause_sent"] = True
        st.session_state.watch = watch
        write_run(watch)
        st.toast(result.message, icon=":material/pause:")
        return True
    st.error(result.message)
    return False


def _render_printer_boundary(watch: dict):
    reachable, state_label = _octo_status(watch)
    printer_name = html.escape(str(watch.get("printer_name") or watch.get("link") or "Printer"))
    can_write = watch.get("link") in ("OctoPrint", "Mega S USB") and watch.get("pause_allowed")
    octo_ready = watch.get("link") == "OctoPrint" and reachable
    with st.container(border=True):
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        with head_l:
            st.markdown("**Printer**")
        with head_r:
            st.markdown(":material/print:")

        printer_row_l, printer_row_r = st.columns([1, 6], vertical_alignment="center")
        with printer_row_l:
            st.markdown(":material/print:" if reachable else ":material/print_disabled:")
        with printer_row_r:
            st.markdown(
                f"<div style='font-weight:600;color:var(--p3-ink);'>{printer_name}</div>"
                f"<div style='color:var(--p3-muted);font-size:0.78rem;letter-spacing:0.1em;text-transform:uppercase;'>"
                f"{html.escape(state_label)}</div>",
                unsafe_allow_html=True,
            )

        outcome = _live_outcome()
        gate = outcome.get("gate")
        can_pause = (
            can_write
            and gate is not None
            and gate.accept
            and gate.command == "pause"
            and not watch.get("pause_sent")
        )
        if watch.get("link") == "OctoPrint" and not reachable:
            can_pause = False

        if st.button(
            "Pause",
            icon=":material/pause:",
            disabled=not can_pause,
            use_container_width=True,
            key="btn_request_pause",
        ):
            _apply_pause(watch, kind="operator")

        if octo_ready:
            last = outcome.get("record") or {}
            with st.expander("Nudge"):
                p = st.selectbox("Parameter", ["hotend", "bed", "feedrate"], key="np_param")
                lo, hi = bounds_for(p)
                default = (lo + hi) / 2
                value = st.number_input(
                    f"{p} ({lo:g}-{hi:g})",
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=1.0,
                    key=f"np_val_{p}",
                )
                confirm = st.checkbox("Confirm", key=f"np_ok_{p}")
                if st.button(
                    "Send nudge",
                    key="np_send",
                    icon=":material/tune:",
                    disabled=not confirm,
                    use_container_width=True,
                ):
                    result = apply_parameter(
                        url=watch["octo_url"],
                        api_key=watch["octo_key"],
                        param=p,
                        value=float(value),
                        confirm=True,
                        observation=outcome.get("observation"),
                        sticky_ok=bool(last.get("sticky")),
                        class_name=last.get("class") or "good_print",
                    )
                    if result.ok:
                        store = JobStore(JOBS_ROOT)
                        store.append_event(
                            Path(watch["folder"]),
                            {
                                "t": datetime.now().isoformat(),
                                "kind": "nudge",
                                "sent": result.sent,
                                "gate": result.gate_label,
                            },
                        )
                        st.toast(result.message, icon=":material/check:")
                    else:
                        st.error(result.message)


# ---------------------------------------------------------------------------
# Live camera fragment
# ---------------------------------------------------------------------------

def _process_frame(watch: dict, image: Image.Image) -> dict | None:
    """Run detection + agent cycle on one frame; append event, return the record."""
    try:
        model = load_model(st.session_state.model_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None
    conf = float(st.session_state.conf_thresh)
    iou = float(st.session_state.iou_thresh)
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False, stream=False)
    sticky = _get_sticky(watch)
    parsed = _get_parsed(watch)
    progress = _progress_for(watch, parsed)
    verdict = (
        judge_frame(parsed, progress, image, results, sticky)
        if parsed is not None
        else judge_image(image, results, sticky, progress=progress)
    )
    memory = _get_memory()
    now = time.time()
    last_deep = float(st.session_state.get("_wp_last_deep_at") or 0)
    want_note = (
        verdict.sticky_ok
        and verdict.name != "good_print"
        and (now - last_deep) >= LLM_NOTE_COOLDOWN_S
    )
    try:
        outcome = cycle(verdict, memory, deep=want_note, camera=True)
    except Exception as exc:  # noqa: BLE001 — Groq TPM or network must not kill the live loop
        outcome = cycle(verdict, memory, deep=False, camera=True)
        outcome["record"]["reason"] = pretty_action(verdict.name)["what_to_do"]
        outcome["record"]["note_error"] = str(exc)[:240]
    if want_note:
        st.session_state["_wp_last_deep_at"] = now
    record = outcome["record"]

    watch["frames"] = int(watch.get("frames", 0)) + 1
    st.session_state.watch = watch

    event = {
        "i": watch["frames"],
        "clock": datetime.now().isoformat(timespec="seconds"),
        "kind": "frame",
        **record,
    }
    st.session_state.watch_events = (st.session_state.get("watch_events") or []) + [event]
    st.session_state.watch_events = st.session_state.watch_events[-200:]
    try:
        store = JobStore(JOBS_ROOT)
        store.append_event(Path(watch["folder"]), event)
    except OSError:
        pass

    st.session_state["_wp_last_verdict"] = record
    st.session_state["_wp_last_outcome"] = outcome
    st.session_state["_wp_last_marked"] = verdict.marked_image
    st.session_state["_wp_last_conf"] = float(verdict.conf)
    st.session_state["_wp_last_source"] = progress.source
    st.session_state["_wp_last_layer"] = verdict.layer_index
    st.session_state["_wp_last_z"] = verdict.layer_z
    return record


def _live_camera_block():
    watch = st.session_state.get("watch")
    if watch is None:
        return
    fresh = read_run()
    if fresh:
        watch = fresh
        st.session_state.watch = watch
        st.session_state.watch_folder = str(watch.get("folder") or "")
    status = read_status()
    record = status.get("record") or {}
    folder = Path(watch["folder"])
    marked = None
    live_path = folder / "live.jpg"
    if live_path.exists():
        try:
            with Image.open(live_path) as img:
                marked = img.convert("RGB")
        except (OSError, ValueError):
            marked = None

    title = html.escape(str(watch.get("name") or "Untitled"))
    head_l, head_r = st.columns([4, 1], vertical_alignment="center")
    with head_l:
        started = watch.get("started_iso") or ""
        try:
            since = datetime.fromisoformat(started).strftime("%I:%M %p")
        except (ValueError, TypeError):
            since = ""
        sub = f"Since {since}" if since else ""
        st.markdown(
            f"<div class='p3-cam-title'>{title}</div>"
            f"<div class='p3-cam-sub'>{sub}</div>",
            unsafe_allow_html=True,
        )
    with head_r:
        st.markdown("<div style='text-align:right;'><span class='p3-pill'>Watching</span></div>", unsafe_allow_html=True)

    err = str(status.get("error") or "")
    if err:
        st.warning(err)

    _render_camera_frame(
        marked,
        cam_idx=int(watch.get("cam_idx", 0)),
        conf_pct=int(round(float(record.get("conf") or 0) * 100)),
        live_str=datetime.now().strftime("%I:%M %p"),
        cam_name=watch.get("cam_name"),
    )

    action = pretty_action(record.get("class") or "good_print")
    total_frames = int(status.get("frames") or watch.get("frames") or 0)
    status_title = action["label"] if total_frames else "Starting"
    pill_cls = "p3-pill" if action["severity"] == "ok" else (
        "p3-pill warn" if action["severity"] == "warning" else "p3-pill crit"
    )
    pill_label = {"ok": "OK", "warning": "WARN", "critical": "CRIT"}[action["severity"]]
    now_str = datetime.now().strftime("%I:%M %p")
    source = html.escape(str(status.get("progress_source") or "camera"))
    layer_i = status.get("layer")
    layer_z = status.get("z")
    layer_bit = ""
    if layer_i is not None:
        z_txt = f" · Z {float(layer_z):.2f} mm" if layer_z is not None else ""
        layer_bit = f" · layer {int(layer_i)}{z_txt}"
    progress_err = status.get("progress_error") or ""
    extra_err = f" · {html.escape(str(progress_err))}" if progress_err else ""
    st.markdown(
        f"""
        <div class='p3-cam-status'>
          <div class='icon'>&#10003;</div>
          <div class='body'>
            <div class='title'>{html.escape(status_title)}</div>
            <div class='sub'>Last read {now_str} · source: {source}{layer_bit}{extra_err}</div>
          </div>
          <div>&nbsp;<span class='{pill_cls}'>{pill_label}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checks_l, checks_r = st.columns([3, 1], vertical_alignment="center")
    with checks_l:
        st.markdown(
            f"<div style='color:var(--p3-muted);font-size:0.88rem;padding-top:0.3rem;'>"
            f"<b style='color:var(--p3-ink);'>{total_frames}</b> frames"
            f"</div>",
            unsafe_allow_html=True,
        )
    with checks_r:
        if st.button(
            "Save frame",
            icon=":material/photo_camera:",
            use_container_width=True,
            key="btn_record_check",
            disabled=marked is None,
        ):
            try:
                store = JobStore(JOBS_ROOT)
                path = store.save_frame(folder, marked, total_frames)
                store.append_event(
                    folder,
                    {
                        "t": datetime.now().isoformat(),
                        "kind": "manual_check",
                        "frame": path.name,
                        "class": record.get("class"),
                        "conf": record.get("conf"),
                    },
                )
                st.toast(f"Saved {path.name}", icon=":material/save:")
            except OSError as exc:
                st.error(f"Could not save frame: {exc}")


# ---------------------------------------------------------------------------
# Page entrypoint
# ---------------------------------------------------------------------------

def page_watch():
    if st.session_state.get("watch") is None:
        attached = attach_if_running()
        if attached:
            st.session_state.watch = attached
            st.session_state.watch_folder = str(attached.get("folder") or "")
    watch = st.session_state.get("watch")
    active = watch is not None

    _hero(active)
    _kpi_row(_kpis(active), active)

    if not active:
        with st.container(border=True):
            st.markdown(
                "<div style='padding:1.6rem;text-align:center;'>"
                "<div style='font-weight:600;font-size:1.05rem;color:#1A1D23;'>"
                "No watch running</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        return

    if not loop_alive():
        start_worker(watch)

    with st.expander("Camera", expanded=False):
        if watch.get("cam_kind") == "ESP32":
            esp = st.text_input("ESP32 capture URL", value=watch.get("esp_url") or "", key="live_esp_url")
            if esp != watch.get("esp_url"):
                watch["esp_url"] = esp.strip()
                st.session_state.watch = watch
                write_run(watch)
        else:
            idx, cam_name = _pick_camera_ui(
                key="live_cam_label",
                current_index=int(watch.get("cam_idx", 0)),
            )
            if idx != int(watch.get("cam_idx", -1)) or cam_name != watch.get("cam_name"):
                watch["cam_idx"] = idx
                watch["cam_name"] = cam_name
                st.session_state.watch = watch
                write_run(watch)
        parsed = _get_parsed(watch)
        if parsed is not None and parsed.layer_count and watch.get("link") == "Camera only":
            layer_i = st.slider(
                "Layer",
                0,
                parsed.layer_count - 1,
                int(watch.get("layer_i") or 0),
                key="live_layer_i",
            )
            if layer_i != int(watch.get("layer_i") or 0):
                watch["layer_i"] = int(layer_i)
                st.session_state.watch = watch
                write_run(watch)

    _live_dashboard_fragment()


@st.fragment(run_every=FRAME_INTERVAL_S)
def _live_dashboard_fragment():
    watch = st.session_state.get("watch")
    if watch is None:
        return
    try:
        main, side = st.columns([1.55, 1], gap="large")
        with main:
            with st.container(border=True):
                _live_camera_block()
        watch = st.session_state.get("watch")
        if watch is None:
            return
        with side:
            _render_printer_boundary(watch)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        _render_evidence_timeline(_read_events(Path(watch["folder"]))[-30:])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Live watch failed: {exc}")
