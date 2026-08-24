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

BRAND = {
    "bg": "#FFFFFF",
    "panel": "#FFFFFF",
    "panel_alt": "#F3F4F6",
    "line": "#E5E7EB",
    "line_soft": "#EEF0F3",
    "ink": "#1A1D23",
    "muted": "#6B7280",
    "signal": "#16856B",
    "sidebar_bg": "#0F1620",
    "sidebar_ink": "#E9E2D3",
    "sidebar_muted": "#8B94A3",
    "sidebar_line": "#1E2A3E",
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
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

          :root {
            --p3-bg: #FFFFFF;
            --p3-panel: #FFFFFF;
            --p3-panel-alt: #F3F4F6;
            --p3-line: #E5E7EB;
            --p3-line-soft: #EEF0F3;
            --p3-ink: #1A1D23;
            --p3-muted: #6B7280;
            --p3-signal: #16856B;
            --p3-signal-soft: rgba(22,133,107,0.10);
            --p3-warn: #C77E00;
            --p3-crit: #B4372A;
            --p3-sb-bg: #0F1620;
            --p3-sb-panel: #182233;
            --p3-sb-ink: #E9E2D3;
            --p3-sb-muted: #8B94A3;
            --p3-sb-line: #1E2A3E;
          }

          html, body, [data-testid="stAppViewContainer"] {
            background: var(--p3-bg) !important;
            color: var(--p3-ink);
          }
          .block-container {
            padding-top: 0.65rem;
            padding-bottom: 2.4rem;
            max-width: 1240px;
          }
          [data-testid="stHeader"] { background: transparent; }
          [data-testid="stDecoration"] { display: none; }
          /* Toolbar stays visible because it holds the "expand sidebar" button.
             Hide only the noisy children we don't want (deploy, running man). */
          [data-testid="stToolbarActions"],
          [data-testid="stStatusWidget"] { display: none !important; }

          /* Expand-sidebar chevron (visible when the sidebar is collapsed).
             stExpandSidebarButton IS the button element — pin it to the top-left
             so it never gets clipped by the header. */
          button[data-testid="stExpandSidebarButton"],
          [data-testid="stSidebarCollapsedControl"] button,
          [data-testid="collapsedControl"] button {
            position: fixed !important;
            top: 0.75rem !important;
            left: 0.75rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 1000000 !important;
            background: var(--p3-panel) !important;
            color: var(--p3-ink) !important;
            border: 1px solid var(--p3-line) !important;
            border-radius: 0.5rem !important;
            width: 2.4rem !important;
            height: 2.4rem !important;
            padding: 0 !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
          }
          button[data-testid="stExpandSidebarButton"]:hover,
          [data-testid="stSidebarCollapsedControl"] button:hover,
          [data-testid="collapsedControl"] button:hover {
            background: var(--p3-panel-alt) !important;
            border-color: var(--p3-signal) !important;
          }
          button[data-testid="stExpandSidebarButton"] span,
          button[data-testid="stExpandSidebarButton"] svg,
          [data-testid="stSidebarCollapsedControl"] svg,
          [data-testid="collapsedControl"] svg {
            color: var(--p3-ink) !important;
            fill: var(--p3-ink) !important;
          }

          /* Sidebar collapse (« inside sidebar) — force always visible */
          section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
          section[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
          section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
          }
          section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
          section[data-testid="stSidebar"] button[kind="headerNoPadding"],
          section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] {
            color: var(--p3-sb-ink) !important;
            background: var(--p3-sb-panel) !important;
            border: 1px solid var(--p3-sb-line) !important;
            border-radius: 0.4rem !important;
            opacity: 1 !important;
            visibility: visible !important;
            width: 2rem !important;
            height: 2rem !important;
          }
          section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg,
          section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] svg {
            color: var(--p3-sb-ink) !important;
            fill: var(--p3-sb-ink) !important;
          }

          /* -------- Sidebar -------- */
          section[data-testid="stSidebar"] {
            background: var(--p3-sb-bg) !important;
            border-right: 1px solid var(--p3-sb-line);
            width: 268px !important;
          }
          section[data-testid="stSidebar"] > div { padding-top: 0.8rem; }
          section[data-testid="stSidebar"] * { color: var(--p3-sb-ink); }

          .p3-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.2rem 0.85rem 0.9rem;
            border-bottom: 1px solid var(--p3-sb-line);
            margin-bottom: 0.9rem;
          }
          .p3-brand .p3-mark {
            font-family: Syne, sans-serif;
            font-weight: 800;
            font-size: 0.95rem;
            letter-spacing: 0.04em;
            background: var(--p3-signal);
            color: #FFFFFF;
            padding: 0.42rem 0.55rem;
            border-radius: 0.5rem;
            line-height: 1;
          }
          .p3-brand .p3-title { display: flex; flex-direction: column; gap: 0.15rem; }
          .p3-brand .p3-name {
            font-family: Syne, sans-serif;
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: -0.02em;
            color: var(--p3-sb-ink);
          }
          .p3-brand .p3-sub {
            font-family: "IBM Plex Sans", sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--p3-sb-muted);
          }
          .p3-nav-label {
            font-family: "IBM Plex Sans", sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--p3-sb-muted);
            padding: 0.2rem 0.95rem 0.4rem;
          }

          section[data-testid="stSidebar"] .stButton { padding: 0 0.55rem; }
          section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: transparent !important;
            color: var(--p3-sb-muted) !important;
            border: 1px solid transparent !important;
            border-radius: 0.55rem !important;
            justify-content: flex-start !important;
            text-align: left !important;
            padding: 0.55rem 0.75rem !important;
            font-weight: 500 !important;
            min-height: 2.55rem;
            box-shadow: none !important;
            position: relative;
          }
          section[data-testid="stSidebar"] .stButton > button:hover {
            background: var(--p3-sb-panel) !important;
            color: var(--p3-sb-ink) !important;
            border-color: var(--p3-sb-line) !important;
          }
          section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: var(--p3-sb-panel) !important;
            color: var(--p3-sb-ink) !important;
            border: 1px solid var(--p3-sb-line) !important;
            font-weight: 600 !important;
          }
          section[data-testid="stSidebar"] .stButton > button[kind="primary"]::before {
            content: "";
            position: absolute;
            left: -0.55rem;
            top: 0.5rem;
            bottom: 0.5rem;
            width: 3px;
            background: var(--p3-signal);
            border-radius: 3px;
          }
          section[data-testid="stSidebar"] .stButton > button p {
            font-size: 0.98rem !important;
            font-weight: inherit !important;
          }

          /* -------- Typography -------- */
          h1, h2, h3, .p3-hero, .p3-finding {
            font-family: Syne, sans-serif !important;
            letter-spacing: -0.02em;
            color: var(--p3-ink);
          }
          .p3-hero {
            font-size: clamp(1.85rem, 3.8vw, 2.55rem);
            font-weight: 800;
            line-height: 1.05;
            margin: 0.4rem 0 0.35rem;
          }
          .p3-hero .accent { color: var(--p3-signal); }
          .p3-eyebrow {
            font-family: "IBM Plex Sans", sans-serif;
            font-size: 0.7rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--p3-muted);
          }
          .p3-lead {
            color: #4B5259;
            font-size: 0.98rem;
            max-width: 46rem;
            margin: 0 0 1.1rem;
          }
          .p3-crumbs {
            display: flex; gap: 0.5rem; align-items: center;
            font-family: "IBM Plex Sans", sans-serif;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--p3-muted);
          }
          .p3-status-dot {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.82rem;
            color: var(--p3-muted);
          }
          .p3-status-dot::before {
            content: "";
            display: inline-block;
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--p3-signal);
            box-shadow: 0 0 0 3px var(--p3-signal-soft);
          }

          /* -------- Panels / cards -------- */
          div[data-testid="stExpander"], div[data-testid="stForm"] {
            background: var(--p3-panel);
            border: 1px solid var(--p3-line);
            border-radius: 0.75rem;
          }
          .p3-panel {
            background: var(--p3-panel);
            border: 1px solid var(--p3-line);
            border-radius: 0.75rem;
            padding: 1rem 1.1rem;
          }
          .p3-kpi-eyebrow {
            font-family: "IBM Plex Sans", sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--p3-muted);
          }
          .p3-kpi-value {
            font-family: Syne, sans-serif;
            font-size: 2.1rem;
            font-weight: 800;
            color: var(--p3-ink);
            line-height: 1;
            margin: 0.4rem 0 0.35rem;
          }
          .p3-kpi-sub { color: var(--p3-muted); font-size: 0.86rem; }
          .p3-kpi.active {
            background: var(--p3-panel);
            border-color: var(--p3-signal);
            box-shadow: 0 0 0 3px var(--p3-signal-soft);
          }
          .p3-kpi.active .p3-kpi-value { color: var(--p3-signal); }

          .p3-cam-head {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 0.55rem;
          }
          .p3-cam-title { font-weight: 600; color: var(--p3-ink); }
          .p3-cam-sub { font-size: 0.82rem; color: var(--p3-muted); }

          /* Framed camera image with corner overlays */
          .p3-cam-frame {
            position: relative;
            border-radius: 0.6rem;
            overflow: hidden;
            background: #0F1620;
            border: 1px solid var(--p3-line);
          }
          .p3-cam-frame img {
            display: block;
            width: 100%;
            height: auto;
            border-radius: 0;
          }
          .p3-cam-frame.empty {
            aspect-ratio: 16 / 9;
            background:
              linear-gradient(#EDF3F1, #E4EDEA);
          }
          .p3-cam-ov {
            position: absolute;
            padding: 0.32rem 0.55rem;
            font-family: "IBM Plex Sans", sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #FFFFFF;
            background: rgba(15,22,32,0.72);
            border-radius: 0.35rem;
            backdrop-filter: blur(4px);
            white-space: nowrap;
          }
          .p3-cam-ov.tl {
            top: 0.55rem; left: 0.55rem;
            color: var(--p3-signal);
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--p3-signal);
          }
          .p3-cam-ov.tr {
            top: 0.55rem; right: 0.55rem;
            color: var(--p3-ink);
            background: rgba(255,255,255,0.92);
          }
          .p3-cam-ov.bl {
            bottom: 0.55rem; left: 0.55rem;
            color: #F3F0EA;
            display: inline-flex; align-items: center; gap: 0.4rem;
          }
          .p3-cam-ov.bl::before {
            content: ""; width: 7px; height: 7px; border-radius: 50%;
            background: #E5484D;
            box-shadow: 0 0 0 3px rgba(229,72,77,0.28);
          }
          .p3-cam-ov.br {
            bottom: 0.55rem; right: 0.55rem;
            color: var(--p3-ink);
            background: rgba(255,255,255,0.92);
          }

          /* Status strip beneath the camera frame */
          .p3-cam-status {
            display: flex; align-items: center; gap: 0.7rem;
            padding: 0.75rem 0.15rem 0.25rem;
          }
          .p3-cam-status .icon {
            width: 22px; height: 22px; border-radius: 50%;
            background: var(--p3-signal-soft);
            color: var(--p3-signal);
            display: inline-flex; align-items: center; justify-content: center;
            font-size: 14px;
          }
          .p3-cam-status .body { flex: 1; }
          .p3-cam-status .body .title { color: var(--p3-ink); font-weight: 600; font-size: 0.95rem; }
          .p3-cam-status .body .sub { color: var(--p3-muted); font-size: 0.8rem; }
          .p3-pill {
            font-family: "IBM Plex Sans", sans-serif;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.24rem 0.6rem;
            border-radius: 0.4rem;
            background: var(--p3-signal-soft);
            color: var(--p3-signal);
            border: 1px solid rgba(22,133,107,0.28);
          }
          .p3-pill.warn { background: rgba(199,126,0,0.10); color: var(--p3-warn); border-color: rgba(199,126,0,0.30); }
          .p3-pill.crit { background: rgba(180,55,42,0.10); color: var(--p3-crit); border-color: rgba(180,55,42,0.30); }
          .p3-pill.mute { background: rgba(107,114,128,0.10); color: var(--p3-muted); border-color: rgba(107,114,128,0.30); }

          /* Nested "sub-card" — used inside Printer boundary */
          .p3-subcard {
            background: var(--p3-panel-alt);
            border: 1px solid var(--p3-line);
            border-radius: 0.55rem;
            padding: 0.75rem 0.85rem;
          }
          .p3-subcard .title { color: var(--p3-ink); font-weight: 600; font-size: 0.92rem; }
          .p3-subcard .sub { color: var(--p3-muted); font-size: 0.82rem; margin-top: 0.15rem; }

          .p3-evidence-row {
            display: flex; align-items: center; gap: 0.8rem;
            padding: 0.7rem 0.9rem;
            border-top: 1px solid var(--p3-line);
          }
          .p3-evidence-row:first-child { border-top: 0; }
          .p3-evidence-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--p3-signal); }
          .p3-evidence-dot.warn { background: var(--p3-warn); }
          .p3-evidence-dot.crit { background: var(--p3-crit); }
          .p3-evidence-time { font-size: 0.78rem; color: var(--p3-muted); min-width: 12ch; }
          .p3-evidence-text { color: var(--p3-ink); font-size: 0.95rem; }
          .p3-evidence-badge {
            margin-left: auto;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            padding: 0.18rem 0.55rem;
            border-radius: 0.4rem;
            background: var(--p3-signal-soft);
            color: var(--p3-signal);
            border: 1px solid rgba(22,133,107,0.28);
            text-transform: uppercase;
          }
          .p3-evidence-badge.warn {
            background: rgba(199,126,0,0.10);
            color: var(--p3-warn);
            border-color: rgba(199,126,0,0.28);
          }
          .p3-evidence-badge.crit {
            background: rgba(180,55,42,0.10);
            color: var(--p3-crit);
            border-color: rgba(180,55,42,0.28);
          }

          /* Primary teal button */
          .stButton > button[kind="primary"] {
            background: var(--p3-signal) !important;
            color: #FFFFFF !important;
            border: 0 !important;
            font-weight: 600 !important;
          }
          .stButton > button[kind="primary"]:hover {
            background: #10715C !important;
            color: #FFFFFF !important;
          }

          img { border-radius: 0.55rem; }

          @media (max-width: 900px) {
            .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
            .p3-hero { font-size: 1.75rem; }
            div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
            div[data-testid="stHorizontalBlock"] > div {
              min-width: 100% !important;
              flex: 1 1 100% !important;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.markdown(
        """
        <div class='p3-brand'>
          <span class='p3-mark'>P3D</span>
          <span class='p3-title'>
            <span class='p3-name'>Precint3D</span>
            <span class='p3-sub'>Operator console</span>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_crumb(bay: str = "Bay 01", title: str = "Operator console") -> None:
    st.markdown(
        f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    margin:0 0 0.55rem;padding-bottom:0.55rem;border-bottom:1px solid var(--p3-line);'>
          <div>
            <div class='p3-crumbs'>Print floor / {bay}</div>
            <div style='font-weight:600;font-size:0.95rem;color:var(--p3-ink);margin-top:0.15rem;'>{title}</div>
          </div>
          <div class='p3-status-dot'>System ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_finding(name: str, conf: float, extra: str | None = None):
    action = pretty_action(name)
    color, icon, level = BADGE[action["severity"]]
    with st.container(border=True):
        st.caption("Finding")
        st.markdown(f"<p class='p3-finding' style='font-size:1.5rem;font-weight:800;margin:0.1rem 0 0.5rem;'>{action['label']}</p>", unsafe_allow_html=True)
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
