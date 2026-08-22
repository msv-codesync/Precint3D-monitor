Precint3D
=========

Operator console for an FDM print. You give it a photo or a camera, and the same G-code the Anycubic Mega S is printing. It keeps YOLO boxes that sit on the planned path, and it will pause only if a code verifier allows it. The language model does not write G-code.

Pages
-----
- Desk — what is wired, last sessions, tool bounds.
- Inspect — one still, laptop camera, or the intern stills. No printer writes.
- Job — G-code path mask, OctoPrint or Marlin progress, optional pause (M25 / OctoPrint).
- Bench — six intern-named stills. Not a hidden test set.
- Records — events.jsonl written while the loop ran; generate_report() writes report.md.

Run it
------
Python 3.10+, weights in `Models/best.pt` (the app also accepts `best_8hr.pt`).

    pip install -r requirements.txt
    streamlit run app.py

Open the URL it prints (usually http://127.0.0.1:8501).

The short written note uses Groq. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Without a key the detector still works.

Anycubic Mega S
---------------
Marlin over USB. No cloud API. Cura / Anycubic Host will not feed live progress. Use OctoPrint, or print from SD and poll `M27`. Upload the same G-code file the printer is running.

Put it online
-------------
This Python app will not run as a Vercel Serverless function.

1. Push this repo (keep `.streamlit/secrets.toml` and `jobs/` out of git).
2. Host the console on Streamlit Community Cloud. Main file: `app.py`. Add `GROQ_API_KEY` under Secrets if you want the note.
3. The page in `web/` is the public door. Vercel should use `web` as the output (see root `vercel.json`). Point “Open the console” at the Streamlit URL once you have one.

`packages.txt` is for Streamlit Cloud (OpenCV).

Tests
-----
    python -m pytest tests -q

License
-------
Use it. This tree does not include other projects' source (CAXTON, Obico, Print Doctor). The monitor is an original integration around Ultralytics YOLO.
