"""Background closed loop in a thread so it can own ProcessWebcam.

A daemon multiprocessing.Process cannot spawn the camera child
(daemonic processes are not allowed to have children). That killed both cameras.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from closed_loop.live import read_run, request_stop, stop_requested, write_run, write_status
from closed_loop.secrets_env import load_groq_key
from closed_loop.store_b import OutcomeMemory
from closed_loop.tick import run_tick
from gcode_monitor.capture import ProcessWebcam, from_esp32, kill_stray_camera_workers
from gcode_monitor.filter import StickyClass
from gcode_monitor.parse import parse_gcode
from paths import ROOT, default_model_path
from tool_api.service import generate_report

JOBS = ROOT / "jobs"
MEMORY = JOBS / "memory.jsonl"
INTERVAL_S = 1.5

_lock = threading.Lock()
_thread: threading.Thread | None = None


def loop_alive() -> bool:
    t = _thread
    return t is not None and t.is_alive()


def _load_model(path: str):
    from ultralytics import YOLO

    if not Path(path).exists():
        raise FileNotFoundError(f"No model at {path}")
    return YOLO(path)


def _parsed(watch: dict):
    if not watch.get("has_gcode"):
        return None
    gcode_path = Path(watch["folder"]) / "print.gcode"
    if not gcode_path.exists() or gcode_path.stat().st_size == 0:
        return None
    try:
        return parse_gcode(gcode_path)
    except (ValueError, OSError):
        return None


def run_loop(watch: dict) -> None:
    load_groq_key()
    folder = Path(watch["folder"])
    model_path = str(watch.get("model_path") or default_model_path())
    conf = float(watch.get("conf_thresh") or 0.15)
    iou = float(watch.get("iou_thresh") or 0.45)
    sticky = StickyClass(need=int(watch.get("sticky_need") or 3))
    parsed = _parsed(watch)
    memory = OutcomeMemory(MEMORY)
    runtime: dict = {}
    cam = None
    cam_idx = None
    try:
        model = _load_model(model_path)
    except Exception as exc:  # noqa: BLE001
        write_status({"error": str(exc), "worker": "dead"})
        return
    write_status({"error": "", "worker": "running", "frames": 0})
    try:
        while not stop_requested():
            latest = read_run() or watch
            watch = latest
            try:
                if watch.get("cam_kind") == "ESP32":
                    if cam is not None:
                        cam.close()
                        cam = None
                        cam_idx = None
                    image = from_esp32(watch.get("esp_url") or "")
                else:
                    want = int(watch.get("cam_idx") or 0)
                    if cam is None or cam_idx != want:
                        if cam is not None:
                            cam.close()
                        cam = ProcessWebcam(want)
                        cam_idx = want
                    image = cam.grab()
                status = run_tick(
                    watch=watch,
                    image=image,
                    model=model,
                    sticky=sticky,
                    parsed=parsed,
                    memory=memory,
                    runtime=runtime,
                    conf=conf,
                    iou=iou,
                )
                write_run(watch)
                write_status(status)
            except Exception as exc:  # noqa: BLE001
                if cam is not None:
                    try:
                        cam.close()
                    except Exception:
                        pass
                    cam = None
                    cam_idx = None
                write_status(
                    {
                        "error": str(exc)[:240],
                        "worker": "running",
                        "frames": watch.get("frames") or 0,
                    }
                )
            time.sleep(INTERVAL_S)
    finally:
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
        kill_stray_camera_workers()
        try:
            generate_report(folder)
        except Exception:
            pass
        from closed_loop.live import read_status

        write_status({**read_status(), "worker": "stopped"})


def start_worker(watch: dict) -> None:
    global _thread
    with _lock:
        if loop_alive():
            write_run(watch)
            return
        from closed_loop.live import STOP

        STOP.unlink(missing_ok=True)
        kill_stray_camera_workers()
        write_run(watch)
        write_status({"error": "", "worker": "starting", "frames": 0})
        _thread = threading.Thread(target=run_loop, args=(dict(watch),), daemon=True, name="p3-loop")
        _thread.start()


def stop_worker() -> None:
    global _thread
    request_stop()
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout=8.0)
    _thread = None
    kill_stray_camera_workers()
    from closed_loop.live import STOP, PID

    STOP.unlink(missing_ok=True)
    PID.unlink(missing_ok=True)
