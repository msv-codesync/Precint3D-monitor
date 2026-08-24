"""Inspect page. A deliberate second look on a single frame.

Never touches the printer. Reads through the same detector + filter stack as
the watch, so tolerances (class floors) are visible next to what YOLO returned.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from actions import pretty as pretty_action
from gcode_monitor.capture import ProcessWebcam, default_camera_index, list_cameras
from gcode_monitor.filter import MIN_CONF, keep_box
from gcode_monitor.mask import draw_kept_boxes
from gcode_monitor.pipeline import boxes_from_yolo
from paths import ROOT
from ui_common import load_model

INSPECT_DIR = ROOT / "jobs" / "_inspect"


def _run_still(model, image: Image.Image, conf_thresh: float, iou_thresh: float):
    results = model.predict(
        source=image, conf=conf_thresh, iou=iou_thresh, verbose=False, stream=False
    )
    name, conf, kept, dropped = boxes_from_yolo(results, mask=None, frame_h=image.height)
    marked = draw_kept_boxes(image, kept)

    detail = []
    result = results[0] if results else None
    if result is not None and result.boxes is not None:
        for box in result.boxes:
            cls = result.names[int(box.cls.item())]
            c = float(box.conf.item())
            floor = MIN_CONF.get(cls, 0.18)
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            kept_flag = keep_box(cls, c, xyxy, None, frame_h=image.height)
            detail.append(
                {
                    "class": cls,
                    "confidence": round(c, 3),
                    "floor": floor,
                    "kept": kept_flag,
                    "margin": round(c - floor, 3),
                }
            )
    return name, conf, kept, dropped, marked, detail


def _snapshot_camera() -> Image.Image | None:
    from closed_loop.live import read_run
    from closed_loop.worker import loop_alive

    if loop_alive():
        watch = read_run() or {}
        folder = watch.get("folder")
        if folder:
            live = Path(folder) / "live.jpg"
            if live.exists():
                try:
                    with Image.open(live) as img:
                        return img.convert("RGB")
                except (OSError, ValueError):
                    pass
    idx = int(st.session_state.get("inspect_cam_idx", default_camera_index()))
    cam = None
    try:
        cam = ProcessWebcam(idx)
        return cam.grab()
    except RuntimeError as exc:
        st.warning(str(exc))
        return None
    finally:
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass


def _persist_inspection(reference: str, image: Image.Image, name: str, conf: float, detail: list[dict]):
    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ref_safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (reference or "still"))[:60]
    folder = INSPECT_DIR / f"{stamp}_{ref_safe}"
    folder.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(folder / "frame.jpg", quality=88)
    (folder / "finding.json").write_text(
        json.dumps(
            {
                "reference": reference,
                "class": name,
                "confidence": conf,
                "boxes": detail,
                "recorded": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return folder


def page_inspect():
    st.markdown("<div class='p3-hero'>Inspect</div>", unsafe_allow_html=True)

    left, right = st.columns([1.6, 1], gap="large")

    with right:
        with st.container(border=True):
            st.markdown("**Settings**")
            reference = st.text_input("Reference name", key="inspect_reference")
            st.session_state.setdefault("inspect_conf_thresh", float(st.session_state.get("conf_thresh", 0.15)))
            st.session_state.setdefault("inspect_iou_thresh", float(st.session_state.get("iou_thresh", 0.45)))
            st.slider("Confidence", 0.05, 0.95, key="inspect_conf_thresh", step=0.01)
            st.slider("IoU", 0.10, 0.95, key="inspect_iou_thresh", step=0.01)
            record = st.button(
                "Save",
                type="primary",
                icon=":material/save:",
                use_container_width=True,
                key="inspect_record",
            )

    with left:
        with st.container(border=True):
            st.markdown("**Frame**")
            mode = st.radio(
                "Source",
                ["Upload", "Camera"],
                horizontal=True,
                key="inspect_source",
                label_visibility="collapsed",
            )

            image: Image.Image | None = None
            if mode == "Upload":
                uploaded = st.file_uploader(
                    "Photo",
                    type=["jpg", "jpeg", "png", "bmp"],
                    key="inspect_upload",
                    label_visibility="collapsed",
                )
                if uploaded is not None:
                    try:
                        image = Image.open(uploaded).convert("RGB")
                    except (OSError, ValueError) as exc:
                        st.error(f"Could not read image: {exc}")
                else:
                    st.markdown("**No photo**")
            else:
                catalog = st.session_state.get("cam_catalog")
                if not catalog:
                    catalog = list_cameras(probe=False)
                    st.session_state.cam_catalog = catalog
                if catalog:
                    labels = [
                        f"{c['name']}" + (" · iPhone" if c.get("preferred") else "")
                        for c in catalog
                    ]
                    pref = default_camera_index(catalog)
                    default_pos = next((i for i, c in enumerate(catalog) if c["index"] == pref), 0)
                    if st.session_state.get("inspect_cam_idx") is not None:
                        default_pos = next(
                            (i for i, c in enumerate(catalog) if c["index"] == int(st.session_state.inspect_cam_idx)),
                            default_pos,
                        )
                    if "inspect_cam_label" not in st.session_state or st.session_state.inspect_cam_label not in labels:
                        st.session_state.inspect_cam_label = labels[default_pos]
                    choice = st.selectbox("Camera", labels, key="inspect_cam_label")
                    picked = catalog[labels.index(choice)]
                    st.session_state.inspect_cam_idx = int(picked["index"])
                else:
                    st.warning("No camera listed.")
                    st.session_state.inspect_cam_idx = 0
                if st.button("Snapshot", type="primary", icon=":material/photo_camera:", key="inspect_shot"):
                    snap = _snapshot_camera()
                    if snap is not None:
                        st.session_state["inspect_snap_bytes"] = _image_to_bytes(snap)
                snap_bytes = st.session_state.get("inspect_snap_bytes")
                if snap_bytes:
                    image = Image.open(io.BytesIO(snap_bytes)).convert("RGB")

            if image is not None:
                try:
                    model = load_model(st.session_state.model_path)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    return
                with st.spinner("Inspecting"):
                    name, conf, kept, dropped, marked, detail = _run_still(
                        model,
                        image,
                        float(st.session_state.inspect_conf_thresh),
                        float(st.session_state.inspect_iou_thresh),
                    )
                st.image(marked, width="stretch")

                action = pretty_action(name)
                color = {"ok": "green", "warning": "orange", "critical": "red"}[action["severity"]]
                st.markdown(
                    f":{color}-badge[{action['label'].upper()}] · {int(conf * 100)}%"
                )

                if detail:
                    st.dataframe(
                        [
                            {
                                "Issue": pretty_action(row["class"])["label"],
                                "Confidence": f"{row['confidence']:.2f}",
                                "Floor": f"{row['floor']:.2f}",
                                "Margin": f"{row['margin']:+.2f}",
                                "Kept": "yes" if row["kept"] else "no",
                            }
                            for row in detail
                        ],
                        hide_index=True,
                        width="stretch",
                    )

                if record:
                    folder = _persist_inspection(reference, image, name, conf, detail)
                    st.toast(f"Saved {folder.name}", icon=":material/save:")


def _image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    return buf.getvalue()
