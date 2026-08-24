Precint3D
=========

Operator console for an FDM print. Camera or still, optional G-code path mask. YOLO boxes on the planned path are kept. Pause and parameter nudges go through `verify_proposal`. The language model does not write G-code.

Pages
-----
- Watch — live camera, optional OctoPrint / USB / ESP32.
- Inspect — one still. No printer writes.
- History — session events and `generate_report()`.

Run it
------
Python 3.10+. Weights in `Models/best.pt`.

    pip install -r requirements.txt
    streamlit run app.py

Open http://127.0.0.1:8501

Groq is optional. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Without a key the detector still works.

Vercel
------
This Python app will not run as a Vercel Serverless function. Vercel serves `web/` only (`vercel.json` at the repo root).

    npx vercel --prod --yes

Point “Open the app” at a Streamlit host if you have one, or leave `http://127.0.0.1:8501` for local use.

Tests
-----
    python -m pytest tests -q
