"""Job watch: file → frame → clock → camera → watch."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from actions import pretty
from agent_loop.cycle import cycle
from agent_loop.memory import DecisionMemory
from gcode_monitor.act import pause_marlin, pause_octoprint
from gcode_monitor.capture import PersistentWebcam, from_esp32
from gcode_monitor.filter import StickyClass
from gcode_monitor.mask import overlay_guidance, render_path
from gcode_monitor.parse import parse_gcode
from gcode_monitor.pipeline import judge_frame
from gcode_monitor.progress import from_manual, from_marlin_sd, from_octoprint
from gcode_monitor.store import JobStore
from tool_api.service import generate_report

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "gcode_monitor" / "fixtures" / "sample_square.gcode"
JOBS = ROOT / "jobs"
CACHE = JOBS / "_gcode_cache"
MEMORY = DecisionMemory(JOBS / "memory.jsonl")

MEGA_W, MEGA_H = 210.0, 210.0


@st.cache_data(show_spinner="Reading G-code")
def _parse_path(path: str, mtime: float, bed_w: float, bed_h: float):
    return parse_gcode(path, bed_w=bed_w, bed_h=bed_h, keep_source=False)


def _materialise(raw: bytes, name: str) -> Path:
    digest = hashlib.sha256(raw).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:80]
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{digest}_{safe}"
    if not dest.exists():
        dest.write_bytes(raw)
    return dest


def _img(image, caption: str | None = None) -> None:
    st.image(image, caption=caption, width="stretch")


def _progress(kind: str, parsed, layer_i: int, octo_url: str, octo_key: str, port: str, baud: int):
    if kind == "OctoPrint":
        return from_octoprint(octo_url, octo_key)
    if kind == "Mega S USB":
        return from_marlin_sd(port, baud, file_size=parsed.size)
    return from_manual(layer_i, parsed.layer_count, parsed.size)


def render_job_mode(
    *,
    model_path: str,
    conf_thresh: float,
    iou_thresh: float,
    load_model,
    run_inference,
    **_extra,
):
    with st.container(border=True):
        st.markdown("**File**")
        st.caption("Use the same Prusa/Cura file that is on the Mega S. Large files are cached after the first read.")
        file_src = st.pills("G-code", ["Sample square", "Upload file"], default="Sample square") or "Sample square"
        if file_src == "Sample square":
            gcode_bytes = SAMPLE.read_bytes()
            gcode_name = SAMPLE.name
        else:
            uploaded = st.file_uploader("G-code", type=["gcode", "gco", "txt"], label_visibility="collapsed")
            if uploaded is None:
                st.info("Upload the same sliced file the printer is running. Or switch back to Sample square.")
                return
            gcode_bytes = uploaded.getvalue()
            gcode_name = uploaded.name

    gcode_path = _materialise(gcode_bytes, gcode_name)
    parsed = _parse_path(str(gcode_path), gcode_path.stat().st_mtime, MEGA_W, MEGA_H)
    if parsed.layer_count == 0:
        st.error("No XY extrusion moves in that file.")
        return

    with st.container(horizontal=True):
        st.metric("Layers", parsed.layer_count, border=True)
        st.metric("Part X", f"{parsed.min_x:.0f}–{parsed.max_x:.0f} mm", border=True)
        st.metric("Part Y", f"{parsed.min_y:.0f}–{parsed.max_y:.0f} mm", border=True)
        st.metric("Flavour", parsed.flavour.split("-")[0], border=True)

    with st.container(border=True):
        st.markdown("**Frame**")
        st.caption("Close-up if the camera sits on the part. Whole bed if you can see the full Mega S plate.")
        frame = st.pills("Map the path as", ["Close-up", "Whole bed"], default="Close-up") or "Close-up"
        fit = "part" if frame == "Close-up" else "bed"
        layer_i = st.slider("Layer", 0, parsed.layer_count - 1, 0)
        pad = st.slider("Inset", 0.02, 0.30, 0.10, step=0.01, help="Shrink the mapped region if the photo has desk around the bed.")

    with st.container(border=True):
        st.markdown("**Clock**")
        st.caption("The Mega S has no cloud. OctoPrint is the live feed. Opening USB yourself can reset Marlin.")
        progress_kind = st.pills(
            "Progress",
            ["This layer", "OctoPrint", "Mega S USB"],
            default="This layer",
        ) or "This layer"
        octo_url, octo_key, port, baud = "", "", "/dev/tty.usbmodem1101", 250000
        if progress_kind == "OctoPrint":
            o1, o2 = st.columns(2)
            octo_url = o1.text_input("OctoPrint URL", value="http://octopi.local")
            octo_key = o2.text_input("API key", type="password")
        elif progress_kind == "Mega S USB":
            s1, s2 = st.columns(2)
            port = s1.text_input("Serial port", value="/dev/tty.usbmodem1101")
            baud = int(s2.number_input("Baud", value=250000, step=1))
            st.caption("Keep this closed if OctoPrint already owns the USB port.")

    with st.container(border=True):
        st.markdown("**Camera**")
        cam_kind = st.pills("Source", ["Photo", "Laptop", "ESP32"], default="Photo") or "Photo"
        still = None
        cam_index = 1
        esp_url = "http://192.168.1.20/capture"
        align = None
        if cam_kind == "Photo":
            shot = st.file_uploader("Printer photo", type=["jpg", "jpeg", "png"], key="job_still")
            if shot is not None:
                still = Image.open(shot).convert("RGB")
                align = still
        elif cam_kind == "Laptop":
            cam_index = int(st.number_input("Camera index", min_value=0, max_value=5, value=1, step=1))
            trial = st.file_uploader("Optional still to check the mask", type=["jpg", "jpeg", "png"], key="job_align")
            if trial is not None:
                align = Image.open(trial).convert("RGB")
        else:
            esp_url = st.text_input("ESP32 capture URL", value="http://192.168.1.20/capture")
            trial = st.file_uploader("Optional still to check the mask", type=["jpg", "jpeg", "png"], key="job_align_esp")
            if trial is not None:
                align = Image.open(trial).convert("RGB")

    layer = parsed.layer_at(layer_i)
    left, right = st.columns(2, gap="large")
    with left:
        st.caption("Planned path")
        _img(render_path(parsed, layer, width=560, height=560, pad=pad, travel=True, fit=fit))
    with right:
        st.caption("On the photo")
        if align is not None:
            _img(overlay_guidance(align, parsed, layer, pad=pad, fit=fit))
        else:
            st.info("Add a still to see whether the blue path sits on the plastic, not the bed sheet.")

    with st.container(border=True):
        st.markdown("**Watch**")
        w1, w2 = st.columns(2)
        interval = w1.slider("Seconds between logged checks", 1, 30, 4)
        sticky_need = w2.slider("Same finding this many times before a pause", 1, 6, 3)
        allow_pause = st.checkbox("Send pause if a critical finding holds", value=False)
        if allow_pause:
            st.caption("OctoPrint pause, or Marlin M25. Off the table while progress is the layer slider.")

    store = JobStore(JOBS)
    for key, value in (
        ("job_on", False),
        ("job_folder", None),
        ("job_i", 0),
        ("job_prog", None),
        ("job_note", None),
    ):
        if key not in st.session_state:
            st.session_state[key] = value

    with st.container(horizontal=True):
        start = st.button("Start", type="primary", icon=":material/play_arrow:")
        stop = st.button("Stop", icon=":material/stop:")
        one_shot = st.button("Check this photo", icon=":material/search:")
        write_report = st.button("generate_report()", icon=":material/description:")

    if start:
        st.session_state.job_folder = str(store.create(gcode_bytes, gcode_name))
        st.session_state.job_on = True
        st.session_state.job_sticky = StickyClass(need=sticky_need)
        st.session_state.job_i = 0
        st.session_state.job_prog = None
        st.session_state.job_note = None
        st.rerun()
    if stop:
        st.session_state.job_on = False
        st.rerun()
    if write_report and st.session_state.job_folder:
        path = generate_report(Path(st.session_state.job_folder))
        st.success(f"Wrote {path}")

    if not st.session_state.job_on and not one_shot:
        st.caption("Press Start when you press Print on the Mega S. The first logged photo is that moment.")
        return

    try:
        model = load_model(model_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    info_box = st.empty()
    pics = st.columns(3)
    path_box, mask_box, mark_box = pics[0].empty(), pics[1].empty(), pics[2].empty()
    note_box = st.empty()
    live = cam_kind == "Laptop"
    webcam = None

    def grab() -> Image.Image:
        if cam_kind == "Photo":
            if still is None:
                raise RuntimeError("Add a printer photo first.")
            return still
        if cam_kind == "ESP32":
            return from_esp32(esp_url)
        return webcam.grab()

    def show_verdict(verdict, pause_msg: str = "") -> None:
        action = pretty(verdict.name)
        held = "held" if verdict.sticky_ok else "not held yet"
        clock = f"[{verdict.progress_pct:.0f}% · layer {verdict.layer_index}]"
        info_box.markdown(
            f"**{clock}** · **{action['label']}** · {verdict.conf * 100:.0f}% · "
            f"Z {verdict.layer_z:.2f} mm · dropped {verdict.dropped} · {held}\n\n"
            f"{action['what_to_do']}\n\n{pause_msg}"
        )
        path_box.image(verdict.path_image, caption="This layer", width="stretch")
        mask_box.image(verdict.masked_image, caption="Bed dimmed", width="stretch")
        mark_box.image(verdict.marked_image, caption="Kept boxes", width="stretch")

    def once(*, deep: bool) -> None:
        image = grab()
        if deep or st.session_state.job_prog is None:
            st.session_state.job_prog = _progress(
                progress_kind, parsed, layer_i, octo_url, octo_key, port, baud
            )
        sticky = st.session_state.get("job_sticky") or StickyClass(need=sticky_need)
        st.session_state.job_sticky = sticky
        verdict = judge_frame(
            parsed,
            st.session_state.job_prog,
            image,
            run_inference(model, image, conf_thresh, iou_thresh),
            sticky,
            pad=pad,
            fit=fit,
            update_sticky=deep,
        )
        pause_msg = ""
        folder = Path(st.session_state.job_folder) if st.session_state.job_folder else None
        if deep and folder and folder.exists():
            idx = int(st.session_state.job_i)
            store.save_frame(folder, image, idx)
            result = cycle(verdict, MEMORY, deep=True, camera=True)
            rec = result["record"]
            rec["i"] = idx
            rec["progress_note"] = verdict.progress_note
            rec["clock"] = f"{verdict.progress_pct:.0f}% · layer {verdict.layer_index}"
            store.append_event(folder, rec)
            st.session_state.job_i = idx + 1
            st.session_state.job_note = {
                "reasoning": rec["reason"],
                "verification": f"{result['gate'].label}: {'; '.join(result['gate'].reasons)}",
            }
            if (
                allow_pause
                and result["gate"].accept
                and result["gate"].command == "pause"
                and progress_kind != "This layer"
            ):
                try:
                    if progress_kind == "OctoPrint":
                        pause_msg = pause_octoprint(octo_url, octo_key)
                    else:
                        pause_msg = pause_marlin(port, baud)
                except Exception as exc:
                    pause_msg = f"pause failed: {exc}"
        show_verdict(verdict, pause_msg)
        note = st.session_state.job_note
        if note:
            with note_box.container(border=True):
                st.caption("Note")
                st.write(note.get("reasoning", ""))
                st.caption("Check")
                st.write(note.get("verification", ""))

    try:
        if cam_kind == "Laptop":
            webcam = PersistentWebcam(cam_index)
        if one_shot and not st.session_state.job_on:
            try:
                if st.session_state.job_folder is None:
                    st.session_state.job_folder = str(store.create(gcode_bytes, gcode_name))
                st.session_state.job_sticky = StickyClass(need=1)
                once(deep=True)
            except Exception as exc:
                st.error(str(exc))
            return

        last_deep = 0.0
        while st.session_state.job_on:
            now = time.time()
            deep = last_deep == 0.0 or (now - last_deep) >= float(interval)
            try:
                once(deep=deep)
                if deep:
                    last_deep = now
            except Exception as exc:
                st.error(str(exc))
                st.session_state.job_on = False
                break
            time.sleep(0.12 if live else float(interval))
    finally:
        if webcam is not None:
            webcam.close()
