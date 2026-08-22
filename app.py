import streamlit as st

from app_pages.bench import page_bench
from app_pages.desk import page_desk
from app_pages.inspect import page_inspect
from app_pages.job import page_job
from app_pages.records import page_records
from ui_common import get_default_model_path, inject_css

st.set_page_config(
    page_title="Precint3D",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

if "model_path" not in st.session_state:
    st.session_state.model_path = get_default_model_path()

CAPTIONS = {
    "Desk": "What this console is, and what it will not pretend to be.",
    "Inspect": "One photo or a camera. Perception only — no printer writes.",
    "Job": "Same G-code as the Mega S. Path mask. Pause only if the verifier allows.",
    "Bench": "Your six stills against the detector. Not a hidden test set.",
    "Records": "Events written while the loop ran. generate_report() turns them into report.md.",
}

with st.sidebar:
    st.markdown("**Detector**")
    st.session_state.conf_thresh = st.slider("YOLO floor", 0.05, 0.95, 0.15, step=0.01)
    st.session_state.iou_thresh = st.slider("Box overlap", 0.10, 0.95, 0.45, step=0.01)
    st.session_state.model_path = st.text_input("Weights", value=st.session_state.model_path)
    st.caption("Each class also has its own floor. Warping on the frame edge is dropped.")
    st.divider()
    st.caption("read_state · set_parameter · pause_print · generate_report")

page = st.navigation(
    {
        "Operate": [
            st.Page(page_desk, title="Desk", icon=":material/home:"),
            st.Page(page_inspect, title="Inspect", icon=":material/image:"),
            st.Page(page_job, title="Job", icon=":material/precision_manufacturing:"),
        ],
        "Evidence": [
            st.Page(page_bench, title="Bench", icon=":material/science:"),
            st.Page(page_records, title="Records", icon=":material/folder:"),
        ],
    },
    position="top",
)

st.title(page.title)
st.caption(CAPTIONS.get(page.title, ""))
page.run()
