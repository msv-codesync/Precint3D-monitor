import json
from pathlib import Path

import streamlit as st

from gcode_monitor.store import list_sessions
from tool_api.service import generate_report
from ui_common import ROOT

JOBS = ROOT / "jobs"


def page_records():
    if not JOBS.exists():
        st.info("No jobs yet.")
        return

    overview = list_sessions(JOBS)
    if not overview:
        st.info("No jobs yet.")
        return

    with st.container(border=True):
        st.markdown("**Sessions**")
        st.dataframe(overview, width="stretch", hide_index=True)

    names = [row["session"] for row in overview]
    choice = st.selectbox("Open", names)
    folder = JOBS / choice
    meta_path = folder / "meta.json"
    events_path = folder / "events.jsonl"
    report_path = folder / "report.md"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with st.container(horizontal=True):
            st.metric("File", meta.get("filename", "—"), border=True)
            st.metric("Bytes", meta.get("bytes", "—"), border=True)

    events = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"raw": line})

    st.metric("Events", len(events), border=True)
    if events:
        slim = [
            {
                "i": e.get("i"),
                "clock": e.get("clock"),
                "layer": e.get("layer"),
                "class": e.get("class"),
                "conf": e.get("conf"),
                "sticky": e.get("sticky"),
                "dropped": e.get("dropped"),
                "verdict": e.get("verdict"),
                "reason": e.get("reason"),
            }
            for e in events
            if "raw" not in e
        ]
        st.dataframe(slim or events, width="stretch", hide_index=True)
    else:
        st.warning("This folder has G-code but no events. The watch loop did not log a cycle.")

    if st.button("generate_report()", icon=":material/description:"):
        path = generate_report(folder)
        st.success(f"Wrote {path.name}")
        st.session_state["last_report"] = str(path)

    show = Path(st.session_state.get("last_report") or report_path)
    if show.exists() and show.parent == folder:
        st.markdown(show.read_text(encoding="utf-8"))

    mem = JOBS / "memory.jsonl"
    with st.expander("Store B (decision memory)"):
        if mem.exists() and mem.stat().st_size:
            st.code(mem.read_text(encoding="utf-8")[-4000:], language="json")
        else:
            st.caption("Empty until a Job cycle writes a record.")
