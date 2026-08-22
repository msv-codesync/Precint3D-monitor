from __future__ import annotations

import os

import streamlit as st
from PIL import Image

from actions import pretty as pretty_action
from gcode_monitor.mask import draw_kept_boxes
from gcode_monitor.pipeline import boxes_from_yolo
from paths import ROOT, default_model_path

BADGE = {
    "ok": ("green", "check_circle", "OK"),
    "warning": ("orange", "warning", "Watch"),
    "critical": ("red", "error", "Pause"),
}


def get_default_model_path() -> str:
    return str(default_model_path())


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at {model_path}")
    return YOLO(model_path)


def run_inference(model, image, conf_thresh: float, iou_thresh: float):
    return model.predict(
        source=image,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
        stream=False,
    )


def judge_still(results, image: Image.Image):
    name, conf, kept, dropped = boxes_from_yolo(results, mask=None, frame_h=image.height)
    marked = draw_kept_boxes(image, kept)
    return name, conf, kept, dropped, marked


def inject_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.1rem; padding-bottom: 2.6rem; max-width: 1220px; }
          [data-testid="stHeader"] { background: #FCFCFB; }
          .finding-name { font-size: 1.35rem; font-weight: 650; letter-spacing: -0.02em; margin: 0.1rem 0 0.5rem 0; }
          [data-testid="stMetricValue"] { font-weight: 650; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_finding(name: str, conf: float, extra: str | None = None):
    action = pretty_action(name)
    color, icon, level = BADGE[action["severity"]]
    with st.container(border=True):
        st.caption("Finding")
        st.markdown(f"<p class='finding-name'>{action['label']}</p>", unsafe_allow_html=True)
        st.badge(level, icon=f":material/{icon}:", color=color)
        st.metric("Confidence", f"{conf * 100:.0f}%", border=True)
        if extra:
            st.caption(extra)
        st.markdown("**What to do**")
        st.write(action["what_to_do"])


def live_panel(info_box, image_box, marked, name, conf, extra=None):
    action = pretty_action(name)
    color, icon, level = BADGE[action["severity"]]
    extra_line = f"\n\n{extra}" if extra else ""
    info_box.markdown(
        f"**{action['label']}**\n\n"
        f":{color}-badge[{level}] · {conf * 100:.0f}% sure\n\n"
        f"{action['what_to_do']}{extra_line}"
    )
    image_box.image(marked, width="stretch")
