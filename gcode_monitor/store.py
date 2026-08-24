"""Local job folders. Nothing is sent to Azure or any other cloud store."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, gcode_bytes: bytes, filename: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in filename)[:80]
        folder = self.root / f"{stamp}_{safe}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "print.gcode").write_bytes(gcode_bytes)
        (folder / "frames").mkdir()
        meta = {
            "created": stamp,
            "filename": filename,
            "bytes": len(gcode_bytes),
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (folder / "events.jsonl").write_text("", encoding="utf-8")
        return folder

    def append_event(self, folder: Path, event: dict) -> None:
        line = json.dumps(event, default=str)
        with (folder / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def save_frame(self, folder: Path, image: Image.Image, index: int) -> Path:
        path = folder / "frames" / f"{index:05d}.jpg"
        image.convert("RGB").save(path, quality=88)
        return path

    def copy_gcode(self, folder: Path, dest: Path) -> None:
        shutil.copy2(folder / "print.gcode", dest)


def resolve_session_dir(root: Path, selected: str | Path) -> Path:
    """Return a job folder only if it sits under ``root`` and is a real session."""
    root = Path(root).resolve()
    folder = Path(selected).expanduser().resolve()
    try:
        folder.relative_to(root)
    except ValueError as exc:
        raise ValueError("session is not under jobs/") from exc
    if folder == root or folder.name.startswith("_") or not folder.is_dir():
        raise ValueError("not a Precint3D session folder")
    return folder


def list_sessions(root: Path) -> list[dict]:
    root = Path(root)
    if not root.exists():
        return []
    rows = []
    for folder in sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name,
        reverse=True,
    ):
        events_path = folder / "events.jsonl"
        n = 0
        last = ""
        if events_path.exists():
            lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            n = len(lines)
            if lines:
                try:
                    last = json.loads(lines[-1]).get("class", "")
                except json.JSONDecodeError:
                    last = ""
        meta = {}
        meta_path = folder / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        rows.append(
            {
                "session": folder.name,
                "file": meta.get("filename", "—"),
                "events": n,
                "last class": last or "—",
            }
        )
    return rows
