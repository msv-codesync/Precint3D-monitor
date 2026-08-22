"""Offline perception bench on intern-labelled stills. No synthetic scores."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from actions import is_critical, pretty
from gcode_monitor.filter import StickyClass, MIN_CONF
from gcode_monitor.pipeline import boxes_from_yolo
from tool_api.service import read_state, verify_proposal
from tool_api.types import Proposal

ROOT = Path(__file__).resolve().parent.parent
LABELS = Path(__file__).resolve().parent / "labels.json"
FRAMES = ROOT / "demo_frames"


def load_labels() -> list[dict]:
    return json.loads(LABELS.read_text(encoding="utf-8"))


def run_bench(model, conf_thresh: float = 0.15, iou_thresh: float = 0.45) -> list[dict]:
    rows = []
    for item in load_labels():
        path = FRAMES / item["file"]
        row = {
            "file": item["file"],
            "label": item["label"],
            "present": path.exists(),
            "raw": None,
            "raw_conf": None,
            "match_raw": None,
            "min_conf": MIN_CONF.get(item["label"], 0.18),
            "sticky_held": None,
            "gate": None,
            "gate_reasons": None,
        }
        if not path.exists():
            rows.append(row)
            continue
        image = Image.open(path).convert("RGB")
        results = model.predict(source=image, conf=conf_thresh, iou=iou_thresh, verbose=False)
        name, conf, _, _ = boxes_from_yolo(results, mask=None, frame_h=image.height)
        row["raw"] = name
        row["raw_conf"] = round(conf, 3)
        row["match_raw"] = name == item["label"]

        sticky = StickyClass(need=3)
        held = False
        for _ in range(3):
            _, held = sticky.push(name)
        row["sticky_held"] = held

        proposal = Proposal(command="pause" if is_critical(name) else "none", source="detector")
        obs = read_state(
            layer=0,
            layer_z=0.2,
            file_pos=None,
            progress=0.0,
            gcode_line="",
            camera_present=True,
        )
        gate = verify_proposal(
            obs,
            proposal,
            sticky_ok=held,
            class_name=name,
            critical=is_critical(name),
        )
        row["gate"] = gate.label
        row["gate_reasons"] = "; ".join(gate.reasons)
        row["pretty"] = pretty(name)["label"]
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["match_raw"] is not None]
    n = len(scored)
    hits = sum(1 for r in scored if r["match_raw"])
    return {
        "n": n,
        "top1": hits,
        "top1_pct": round(100.0 * hits / n, 1) if n else None,
        "missing_files": [r["file"] for r in rows if not r["present"]],
        "note": (
            "Labels are the intern's filenames for these stills, not a hidden gold set. "
            "n=6 and the files were chosen as examples of those classes — this is not a test-set accuracy. "
            "G-code masking is not applied here: these photos are not registered to a toolpath."
        ),
    }
