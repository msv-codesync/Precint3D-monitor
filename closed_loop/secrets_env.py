"""Load GROQ_API_KEY into the environment for the worker. Never print the value."""

from __future__ import annotations

import os
from pathlib import Path

from paths import ROOT


def load_groq_key() -> str:
    existing = os.getenv("GROQ_API_KEY", "").strip()
    if existing:
        return existing
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.is_file():
        return ""
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    key = str(data.get("GROQ_API_KEY") or "").strip()
    if key:
        os.environ["GROQ_API_KEY"] = key
    return key
