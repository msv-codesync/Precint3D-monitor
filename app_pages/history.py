"""Watch history. Read-only view of past sessions and their evidence."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from actions import pretty as pretty_action
from gcode_monitor.store import resolve_session_dir
from paths import ROOT
from tool_api.service import generate_report

JOBS_ROOT = ROOT / "jobs"


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


def _summarise(folder: Path, active_folder: str | None) -> dict:
    events = _read_events(folder)
    frame_events = [e for e in events if e.get("kind") in (None, "frame") and e.get("class")]
    last = frame_events[-1] if frame_events else {}
    finding = pretty_action(last.get("class") or "good_print")
    meta_path = folder / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    started_display = folder.name
    stem = folder.name.split("_", 1)[0]
    try:
        started_dt = datetime.strptime(stem, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
        started_display = started_dt.astimezone().strftime("%b %d, %Y · %I:%M %p")
    except ValueError:
        pass
    is_active = active_folder and str(folder) == active_folder
    return {
        "folder": folder,
        "name": meta.get("filename") or folder.name,
        "started": started_display,
        "frames": len(frame_events),
        "finding_label": finding["label"] if frame_events else "None",
        "finding_sev": finding["severity"] if frame_events else "ok",
        "confidence": last.get("conf", 0.0),
        "status": "WATCHING" if is_active else "STOPPED",
    }


def _list_sessions(active_folder: str | None) -> list[dict]:
    if not JOBS_ROOT.exists():
        return []
    rows = []
    for folder in sorted(
        (p for p in JOBS_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")),
        reverse=True,
    ):
        rows.append(_summarise(folder, active_folder))
    return rows


def _row_html(row: dict) -> str:
    sev = row["finding_sev"]
    dot = {"ok": "#16856B", "warning": "#C77E00", "critical": "#B4372A"}[sev]
    status_bg = {
        "WATCHING": "rgba(22,133,107,0.12)",
        "STOPPED": "rgba(107,114,128,0.10)",
    }[row["status"]]
    status_fg = {
        "WATCHING": "#16856B",
        "STOPPED": "#6B7280",
    }[row["status"]]
    conf_pct = int(round(float(row.get("confidence") or 0) * 100))
    return f"""
    <div style='display:grid;grid-template-columns:2.4fr 1.6fr 1fr 1fr 1fr;
                gap:1rem;padding:0.9rem 1rem;border-top:1px solid #E5E7EB;align-items:center;'>
      <div>
        <div style='font-weight:600;color:#1A1D23;'>{html.escape(str(row['name']))}</div>
      </div>
      <div>
        <div style='color:#1A1D23;display:flex;align-items:center;gap:0.5rem;'>
          <span style='width:8px;height:8px;border-radius:50%;background:{dot};display:inline-block;'></span>
          {html.escape(str(row['finding_label']))}
        </div>
        <div style='color:#6B7280;font-size:0.82rem;'>{conf_pct}%</div>
      </div>
      <div style='color:#4B5259;'>{html.escape(str(row['frames']))} frames</div>
      <div style='color:#4B5259;'>{html.escape(str(row['started']))}</div>
      <div>
        <span style='background:{status_bg};color:{status_fg};padding:0.22rem 0.6rem;
                     border-radius:0.4rem;font-size:0.72rem;letter-spacing:0.06em;'>
          {html.escape(str(row['status']))}
        </span>
      </div>
    </div>
    """


def page_history():
    left, right = st.columns([2.4, 1], vertical_alignment="center")
    with left:
        st.markdown("<div class='p3-hero'>History</div>", unsafe_allow_html=True)
    with right:
        with st.container(horizontal=True, horizontal_alignment="right"):
            if st.button(
                "Inspect",
                icon=":material/center_focus_strong:",
                key="hist_go_inspect",
            ):
                st.session_state.route = "inspect"
                st.rerun()

    rows = _list_sessions(st.session_state.get("watch_folder"))
    if not rows:
        with st.container(border=True):
            st.markdown(
                "<div style='padding:1.5rem;text-align:center;color:#6B7280;'>"
                "No sessions</div>",
                unsafe_allow_html=True,
            )
        return

    with st.container(border=True):
        st.markdown(
            "<div style='padding:0.9rem 1rem 0.7rem;'>"
            f"<div style='font-weight:600;color:#1A1D23;'>{len(rows)} sessions</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='display:grid;grid-template-columns:2.4fr 1.6fr 1fr 1fr 1fr;"
            "gap:1rem;padding:0.55rem 1rem;color:#6B7280;font-size:0.72rem;"
            "letter-spacing:0.14em;text-transform:uppercase;border-top:1px solid #E5E7EB;'>"
            "<div>Print</div><div>Finding</div><div>Frames</div><div>Started</div><div>Status</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        for row in rows:
            st.markdown(_row_html(row), unsafe_allow_html=True)
            c1, c2, _c3 = st.columns([1, 1, 5])
            with c1:
                if st.button(
                    "Open",
                    icon=":material/arrow_forward:",
                    key=f"open_{row['folder'].name}",
                ):
                    st.session_state.history_selected = str(row["folder"])
                    st.rerun()
            with c2:
                if st.button(
                    "Save summary",
                    icon=":material/description:",
                    key=f"rep_{row['folder'].name}",
                ):
                    try:
                        out = generate_report(row["folder"])
                        st.toast(f"Wrote {out.name}", icon=":material/check:")
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))

    selected = st.session_state.get("history_selected")
    if not selected:
        return
    try:
        folder = resolve_session_dir(JOBS_ROOT, selected)
    except ValueError:
        st.session_state.history_selected = None
        return

    st.markdown("<div class='p3-eyebrow' style='margin-top:1.4rem;'>Session</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"**{html.escape(folder.name)}**")
        events = _read_events(folder)
        frame_events = [e for e in events if e.get("kind") in (None, "frame") and e.get("class")]
        pause_events = [e for e in events if e.get("kind") == "pause_requested"]
        nudges = [e for e in events if e.get("kind") == "nudge"]

        with st.container(horizontal=True):
            st.metric("Frames", len(frame_events), border=True)
            st.metric("Pauses", len(pause_events), border=True)
            st.metric("Nudges", len(nudges), border=True)

        if frame_events:
            st.dataframe(
                [
                    {
                        "clock": e.get("clock"),
                        "layer": e.get("layer"),
                        "class": e.get("class"),
                        "conf": e.get("conf"),
                        "sticky": e.get("sticky"),
                        "verdict": e.get("verdict"),
                    }
                    for e in frame_events[-100:]
                ],
                hide_index=True,
                width="stretch",
            )
        report_path = folder / "report.md"
        if report_path.exists():
            with st.expander("Saved summary"):
                st.markdown(report_path.read_text(encoding="utf-8"))
