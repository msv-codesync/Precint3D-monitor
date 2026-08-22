"""Operator home: status, last sessions, and the four tools."""

from pathlib import Path

import streamlit as st

from gcode_monitor.store import list_sessions
from paths import ROOT, default_model_path
from tool_api.bounds import BOUNDS


def page_desk():
    weights = Path(default_model_path())
    sessions = list_sessions(ROOT / "jobs")
    events = sum(int(row["events"]) for row in sessions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weights", weights.name if weights.exists() else "missing", border=True)
    c2.metric("Hotend window", f"{int(BOUNDS['hotend'][0])}–{int(BOUNDS['hotend'][1])} °C", border=True)
    c3.metric("Sessions on disk", len(sessions), border=True)
    c4.metric("Logged events", events, border=True)

    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Pages**")
        st.write("**Inspect** — diagnose a still, the laptop camera, or your sample frames. Nothing is sent to the printer.")
        st.write("**Job** — load the G-code the Mega S is printing. Boxes off the planned path are dropped. Pause is OctoPrint or Marlin M25, and only after the same critical class holds.")
        st.write("**Bench** — six stills you already named. Top-1 against the filename, then whether a pause would survive the verifier.")
        st.write("**Records** — the session folder written while Job ran. generate_report() is the audit file, not a later summary.")
    with b.container(border=True):
        st.markdown("**Honest limits**")
        st.write("IMU, microphone, extra thermocouple — specified, not wired. They stay None.")
        st.write("Store B is JSONL by class, not a Chroma index.")
        st.write("This camera is a room or bed view. Nozzle-close correction (CAXTON) is out of scope.")
        st.write("The Python console does not run on Vercel. The public page is only the door.")
        st.write("A textured bed sheet still fools warping if you skip G-code. Use Job, not Inspect, for that.")

    with st.container(border=True):
        st.markdown("**Tool layer**")
        st.code(
            "read_state()        observation: layer, Z, progress, snippet, temps if known\n"
            "set_parameter()     hotend 180–260, bed 40–90, feedrate 50–120 — clip or refuse\n"
            "pause_print()       OctoPrint pause or Marlin M25\n"
            "generate_report()   jobs/<session>/report.md from events.jsonl",
            language="text",
        )

    with st.container(border=True):
        st.markdown("**Recent sessions**")
        if sessions:
            st.dataframe(sessions, width="stretch", hide_index=True)
        else:
            st.caption("Empty until you press Start on Job.")
