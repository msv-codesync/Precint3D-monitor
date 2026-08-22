"""Project root. Never hard-code a username or Desktop folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "Models"


def default_model_path() -> Path:
    for name in ("best.pt", "best_8hr.pt"):
        path = MODELS_DIR / name
        if path.exists():
            return path
    return MODELS_DIR / "best.pt"
