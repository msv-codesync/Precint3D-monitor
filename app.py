"""Precint3D operator console. Sidebar nav: Watch, Inspect, History."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import streamlit as st

from ui_common import get_default_model_path, inject_css, render_sidebar_brand

_ROOT = Path(__file__).resolve().parent


def _load_page(modname: str, relpath: str):
    """Load a page module from a file path.

    Streamlit's file watcher can drop ``app_pages`` from ``sys.modules`` mid-import
    (KeyError: 'app_pages' / 'app_pages.history'). Loading by path with a stable
    module name avoids that race.
    """
    existing = sys.modules.get(modname)
    if existing is not None:
        return existing
    path = _ROOT / relpath
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


_history = _load_page("p3_history", "app_pages/history.py")
_inspect = _load_page("p3_inspect", "app_pages/inspect_still.py")
_watch = _load_page("p3_watch", "app_pages/watch.py")
page_history = _history.page_history
page_inspect = _inspect.page_inspect
page_watch = _watch.page_watch

st.set_page_config(
    page_title="Precint3D — Operator console",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ---- session defaults ------------------------------------------------------
_defaults = {
    "model_path": get_default_model_path(),
    "conf_thresh": 0.15,
    "iou_thresh": 0.45,
    "route": "watch",
    "watch": None,          # dict describing the active watch, or None
    "watch_folder": None,   # path str
    "history_selected": None,
}
for _k, _v in _defaults.items():
    st.session_state.setdefault(_k, _v)

# ---- sidebar (brand + 3 nav buttons) --------------------------------------
NAV = [
    ("watch", "Watch", "dashboard"),
    ("inspect", "Inspect", "center_focus_strong"),
    ("history", "History", "history"),
]

with st.sidebar:
    render_sidebar_brand()
    st.markdown("<div class='p3-nav-label'>Workspace</div>", unsafe_allow_html=True)
    for _key, _label, _icon in NAV:
        _selected = st.session_state.route == _key
        if st.button(
            _label,
            key=f"nav_{_key}",
            icon=f":material/{_icon}:",
            type="primary" if _selected else "secondary",
            use_container_width=True,
        ):
            if st.session_state.route != _key:
                st.session_state.route = _key
                st.rerun()
    st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='position:fixed;bottom:0.9rem;left:0.95rem;font-size:0.72rem;color:#8B94A3;'>"
        "<div style='color:#4ECBA3;'>● Safety mode enabled</div>"
        "<div>Camera-first · No auto actions</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ---- route to page --------------------------------------------------------
_route = st.session_state.route
if _route == "inspect":
    page_inspect()
elif _route == "history":
    page_history()
else:
    page_watch()
