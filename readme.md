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

Hosted console (Streamlit Community Cloud)
------------------------------------------
Vercel cannot run a Python server, so the console itself is hosted on Streamlit Community Cloud and the Vercel page only links to it.

1. Sign in at https://share.streamlit.io with the GitHub account that owns this repo.
2. New app → repository `msv-codesync/Precint3D-monitor`, branch `main`, main file `app.py`.
3. Advanced settings → Python 3.11. App URL: `precint3d` (gives https://precint3d.streamlit.app, which `web/index.html` already points to).
4. Optional: paste `GROQ_API_KEY = "..."` into Secrets for the operator note. Without it the action-map sentence is used.

`requirements.txt` pins CPU-only torch via the PyTorch extra index so the cloud build stays small. The trained weights `Models/best.pt` are in git, so no download step is needed.

On the hosted console: Inspect and History work fully. Watch has no camera on the server; the live loop, Continuity Camera, USB and OctoPrint pause only work when the console runs on the machine next to the printer.

Vercel (landing page only)
--------------------------
Vercel serves `web/` only (`vercel.json` at the repo root).

    npx vercel --prod --yes

Tests
-----
    python -m pytest tests -q
