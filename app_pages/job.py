from job_mode import render_job_mode
from ui_common import load_model, run_inference
import streamlit as st


def page_job():
    render_job_mode(
        model_path=st.session_state.model_path,
        conf_thresh=st.session_state.conf_thresh,
        iou_thresh=st.session_state.iou_thresh,
        load_model=load_model,
        run_inference=run_inference,
    )
