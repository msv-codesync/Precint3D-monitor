import streamlit as st

from eval.bench import run_bench, summarise
from ui_common import load_model


def page_bench():
    conf = st.session_state.conf_thresh
    iou = st.session_state.iou_thresh
    if st.button("Run bench", type="primary", icon=":material/science:"):
        try:
            model = load_model(st.session_state.model_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        with st.spinner("Running the detector on the labelled stills…"):
            rows = run_bench(model, conf, iou)
        st.session_state.bench_rows = rows

    rows = st.session_state.get("bench_rows")
    if not rows:
        st.info("Run the bench to see raw top-1 against your filenames, then whether a pause would survive the tool verifier.")
        return

    summary = summarise(rows)
    with st.container(horizontal=True):
        st.metric("Stills scored", summary["n"], border=True)
        st.metric("Top-1 vs filename", f"{summary['top1']}/{summary['n']}", border=True)
        st.metric("Top-1 %", f"{summary['top1_pct']}%" if summary["top1_pct"] is not None else "—", border=True)
    st.caption(summary["note"])
    if summary["missing_files"]:
        st.warning("Missing: " + ", ".join(summary["missing_files"]))

    table = [
        {
            "file": r["file"],
            "filename label": r["label"],
            "detector": r["raw"],
            "conf": r["raw_conf"],
            "match": r["match_raw"],
            "held ×3": r["sticky_held"],
            "verifier": r["gate"],
            "why": r["gate_reasons"],
        }
        for r in rows
        if r["present"]
    ]
    with st.container(border=True):
        st.markdown("**Rows**")
        st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "A pause is ALLOW only for a critical class that held on three identical frames. "
        "Repeating one still is a check of the gate, not a claim about a live print."
    )
