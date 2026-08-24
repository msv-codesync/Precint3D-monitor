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
from gcode_monitor.capture import default_camera_index, list_cameras, open_first_camera
from gcode_monitor.filter import MIN_CONF, keep_box
from gcode_monitor.mask import draw_kept_boxes
from gcode_monitor.pipeline import boxes_from_yolo
from paths import ROOT
from ui_common import load_model, render_top_crumb

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
    try:
        cam, idx = open_first_camera(preferred=int(st.session_state.get("inspect_cam_idx", default_camera_index())))
    except RuntimeError as exc:
        st.warning(str(exc))
        return None
    try:
        return cam.grab()
    except RuntimeError as exc:
        st.warning(str(exc))
        return None
    finally:
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
    render_top_crumb(title="Operator console")

    st.markdown("<div class='p3-eyebrow'>Precint3D / operator view</div>", unsafe_allow_html=True)
    st.markdown("<div class='p3-hero'>Inspect a still.</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='p3-lead'>A deliberate second look, separated from live printer control. "
        "Upload a photo or use the local camera when hardware is available.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.6, 1], gap="large")

    with right:
        with st.container(border=True):
            head_l, head_r = st.columns([4, 1], vertical_alignment="center")
            with head_l:
                st.markdown("**Operator read**")
            with head_r:
                st.markdown(":material/lock:")
            st.caption("No automatic action is available here.")

            reference = st.text_input(
                "Reference name",
                key="inspect_reference",
                placeholder="e.g. bracket after 3h",
            )

            st.markdown("**Tolerances**")
            st.session_state.setdefault("inspect_conf_thresh", float(st.session_state.get("conf_thresh", 0.15)))
            st.session_state.setdefault("inspect_iou_thresh", float(st.session_state.get("iou_thresh", 0.45)))
            st.slider("Confidence floor", 0.05, 0.95, key="inspect_conf_thresh", step=0.01)
            st.slider("Box overlap (IoU)", 0.10, 0.95, key="inspect_iou_thresh", step=0.01)

            record = st.button(
                "Record inspection",
                type="primary",
                icon=":material/save:",
                use_container_width=True,
                key="inspect_record",
            )

        with st.container(border=True):
            st.markdown(":material/warning: **Hard boundary**")
            st.caption(
                "A photo can create evidence, never a pause. Review the image yourself before "
                "making any printer decision."
            )

    with left:
        with st.container(border=True):
            head_l, head_r = st.columns([4, 1], vertical_alignment="center")
            with head_l:
                st.markdown("**Inspection frame**")
            with head_r:
                st.markdown(":material/image:")
            st.caption("Local media stays in this browser until you submit a finding.")

            mode = st.radio(
                "Source",
                ["Upload", "Local camera"],
                horizontal=True,
                key="inspect_source",
                label_visibility="collapsed",
            )

            image: Image.Image | None = None
            if mode == "Upload":
                uploaded = st.file_uploader(
                    "Drop a print photo here",
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
                    with st.container(horizontal_alignment="center"):
                        st.markdown(":material/upload:")
                        st.markdown("**Drop a print photo here**")
                        st.caption(
                            "Or choose a local image. Precint3D never assumes a still "
                            "is permission to control the printer."
                        )
            else:
                catalog = st.session_state.get("cam_catalog")
                if not catalog:
                    catalog = list_cameras()
                    st.session_state.cam_catalog = catalog
                find_l, find_r = st.columns([3, 1], vertical_alignment="bottom")
                with find_r:
                    if st.button("Find cameras", icon=":material/search:", key="inspect_probe"):
                        st.session_state.cam_catalog = list_cameras(probe=True)
                        catalog = st.session_state.cam_catalog
                        st.rerun()
                if catalog:
                    labels = [
                        f"{c['name']}" + (" · iPhone" if c.get("preferred") else "")
                        for c in catalog
                    ]
                    pref = default_camera_index(catalog)
                    default_pos = next((i for i, c in enumerate(catalog) if c["index"] == pref), 0)
                    with find_l:
                        choice = st.selectbox("Camera", labels, index=default_pos, key="inspect_cam_label")
                    picked = catalog[labels.index(choice)]
                    st.session_state.inspect_cam_idx = int(picked["index"])
                    st.caption(
                        "Unlock your iPhone nearby so Continuity Camera appears in this list."
                    )
                else:
                    st.warning("No camera opened. Unlock the iPhone, then click Find cameras.")
                    st.session_state.inspect_cam_idx = 0
                if st.button("Take snapshot", type="primary", icon=":material/photo_camera:", key="inspect_shot"):
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
                with st.spinner("Looking at the photo…"):
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
                    f":{color}-badge[{action['label'].upper()}] · {int(conf * 100)}% confidence · "
                    f"dropped {dropped} box(es) below floor"
                )
                st.caption(action["what_to_do"])

                if detail:
                    st.markdown("<div class='p3-eyebrow' style='margin-top:0.6rem;'>Tolerances</div>", unsafe_allow_html=True)
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
                    st.toast(f"Saved to {folder.name}", icon=":material/save:")

    with st.expander(":material/help: Hardware unavailable?"):
        st.caption(
            "The upload path above is the intended fallback. Camera permission is optional and never blocks still inspection."
        )


def _image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    return buf.getvalue()
