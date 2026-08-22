import streamlit as st
from PIL import Image

from agentic_workflow import run_agentic_reasoning
from ui_common import (
    ROOT,
    judge_still,
    live_panel,
    load_model,
    render_finding,
    run_inference,
    pretty_action,
)


def _run_still(model, image, conf_thresh, iou_thresh):
    results = run_inference(model, image, conf_thresh, iou_thresh)
    name, conf, kept, dropped, marked = judge_still(results, image)
    note = run_agentic_reasoning(
        [{"class": d["class"], "confidence": d["confidence"]} for d in kept],
        name,
        conf,
        None,
    )
    return name, conf, kept, dropped, marked, note


def page_inspect():
    conf_thresh = st.session_state.conf_thresh
    iou_thresh = st.session_state.iou_thresh
    model_path = st.session_state.model_path

    mode = st.pills("Source", ["Photo", "Camera", "Samples"], default="Photo") or "Photo"

    if mode == "Photo":
        uploaded = st.file_uploader(
            "Printer photo",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded is None:
            st.info("A still from the printer. Low-confidence boxes and warping on the frame edge are dropped before the finding.")
            return
        uploaded.seek(0)
        image = Image.open(uploaded).convert("RGB")
        try:
            model = load_model(model_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        with st.spinner("Looking at the photo…"):
            name, conf, kept, dropped, marked, note = _run_still(model, image, conf_thresh, iou_thresh)
        left, right = st.columns([1.05, 1.15], gap="large")
        with left:
            render_finding(name, conf, extra=f"Dropped {dropped} box(es) below floor or on the frame edge.")
            with st.container(border=True):
                st.caption("Note")
                st.write(note.get("reasoning", "No note this time."))
                st.caption("Check")
                st.write(note.get("verification", "The detector still chose the class. The note does not move the printer."))
            if kept:
                table_rows = [
                    {"Issue": pretty_action(item["class"])["label"], "Confidence": item["confidence"]}
                    for item in kept
                ]
                st.dataframe(table_rows, width="stretch", hide_index=True)
        with right:
            st.caption("Kept boxes only")
            st.image(marked, width="stretch")
        return

    if mode == "Samples":
        import time

        frames = sorted(
            p
            for p in (ROOT / "demo_frames").glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not frames:
            st.error("No stills in demo_frames.")
            return
        if "demo_on" not in st.session_state:
            st.session_state.demo_on = False
        with st.container(horizontal=True):
            if st.button("Start", type="primary", icon=":material/play_arrow:"):
                st.session_state.demo_on = True
                st.rerun()
            if st.button("Stop", icon=":material/stop:"):
                st.session_state.demo_on = False
                st.rerun()
        if not st.session_state.demo_on:
            st.info("Walks the intern's own stills. Labels live on Bench.")
            return
        try:
            model = load_model(model_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        left, right = st.columns([1.05, 1.15], gap="large")
        info_box, image_box = left.empty(), right.empty()
        idx = 0
        while st.session_state.demo_on:
            frame_path = frames[idx % len(frames)]
            image = Image.open(frame_path).convert("RGB")
            results = run_inference(model, image, conf_thresh, iou_thresh)
            name, conf, kept, dropped, marked = judge_still(results, image)
            live_panel(info_box, image_box, marked, name, conf, extra=frame_path.name)
            idx += 1
            time.sleep(1.2)
        return

    import time
    import cv2

    cam_index = st.number_input("Camera", min_value=0, max_value=5, value=1, step=1)
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False
    with st.container(horizontal=True):
        if st.button("Start", type="primary", icon=":material/play_arrow:"):
            st.session_state.cam_on = True
            st.rerun()
        if st.button("Stop", icon=":material/stop:"):
            st.session_state.cam_on = False
            st.rerun()
    if not st.session_state.cam_on:
        st.info("Index 0 is usually the Mac webcam. 1 is often Continuity Camera. Allow Camera for Terminal or Cursor.")
        return
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error("Camera did not open. Allow Camera for Terminal or Cursor, then try index 1.")
        st.session_state.cam_on = False
        return
    try:
        model = load_model(model_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        cap.release()
        return
    left, right = st.columns([1.05, 1.15], gap="large")
    info_box, image_box = left.empty(), right.empty()
    misses = 0
    while st.session_state.cam_on:
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 10:
                st.error("The camera stream stopped.")
                st.session_state.cam_on = False
                break
            time.sleep(0.5)
            continue
        misses = 0
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = run_inference(model, image, conf_thresh, iou_thresh)
        name, conf, kept, dropped, marked = judge_still(results, image)
        live_panel(info_box, image_box, marked, name, conf)
        time.sleep(0.05)
    cap.release()
