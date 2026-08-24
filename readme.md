Precint3D
=========

Operator console for an FDM print. Camera or still + the same G-code the Anycubic Mega S is printing. YOLO boxes that sit on the planned path are kept. Pause only if a code verifier allows it. The language model does not write G-code.

Read **HANDBOOK.md** for the internship/thesis write-up (loop, dataset, tools, what not to claim).

Pages
-----
- Pipeline — five nodes, how to test Loop.
- Loop — G-code path mask, OctoPrint or Marlin progress, Check this photo / Start loop, optional pause (M25 / OctoPrint).
- Look — still, laptop camera, ESP32, or intern Samples. No printer writes.
- Log — events.jsonl; generate_report() writes report.md. Bench tab on six intern-named stills.

Run it
------
Python 3.10+ (on this Mac: Anaconda). Weights in `Models/best.pt`.

    pip install -r requirements.txt
    streamlit run app.py

Open the URL it prints (usually http://127.0.0.1:8501).

    /opt/anaconda3/bin/python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1

Groq is optional. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Without a key the detector still works.

Anycubic Mega S
---------------
Marlin over USB. No cloud API. Cura / Anycubic Host will not feed live progress. Use OctoPrint, or print from SD and poll `M27`. Upload the same G-code file the printer is running.

Put it online
-------------
The Python app will not run as a Vercel Serverless function. Vercel serves `web/` only (`vercel.json`).

1. Push this repo (keep `.streamlit/secrets.toml` and `jobs/` out of git).
2. Host the console on Streamlit Community Cloud. Main file: `app.py`. You must provide `Models/best.pt` (gitignored). Add `GROQ_API_KEY` under Secrets if you want the note.
3. `npx vercel login` then `npx vercel --prod --yes` from this folder for the static door. Point “Open the console” at Streamlit once you have a URL, or leave `http://127.0.0.1:8501` for local demos.

`packages.txt` is for Streamlit Cloud (OpenCV).

Tests
-----
    python -m pytest tests -q

License
-------
Use it. This tree does not include other projects' source. The monitor is an original integration around Ultralytics YOLO.
