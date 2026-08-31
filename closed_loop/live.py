"""Start / stop the background watch process. State lives under jobs/_live."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

from paths import ROOT

LIVE = ROOT / "jobs" / "_live"
RUN = LIVE / "run.json"
PID = LIVE / "worker.pid"
STOP = LIVE / "stop"
STATUS = LIVE / "status.json"


def _ensure() -> None:
    LIVE.mkdir(parents=True, exist_ok=True)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def running_pid() -> int | None:
    if not PID.exists():
        return None
    try:
        pid = int(PID.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return None
    return pid if pid_alive(pid) else None


def write_run(watch: dict) -> None:
    _ensure()
    STOP.unlink(missing_ok=True)
    RUN.write_text(json.dumps(watch, default=str), encoding="utf-8")


def read_run() -> dict | None:
    if not RUN.exists():
        return None
    try:
        data = json.loads(RUN.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_status(payload: dict) -> None:
    _ensure()
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    tmp.replace(STATUS)


def read_status() -> dict:
    if not STATUS.exists():
        return {}
    try:
        data = json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def request_stop() -> None:
    _ensure()
    STOP.write_text("1", encoding="utf-8")


def clear_live_state() -> None:
    """Remove persisted live-watch state after the camera loop is stopped."""
    _ensure()
    for path in (RUN, PID, STOP, STATUS):
        path.unlink(missing_ok=True)


def stop_requested() -> bool:
    return STOP.exists()


def write_pid(pid: int) -> None:
    _ensure()
    PID.write_text(str(int(pid)), encoding="utf-8")


def kill_worker(timeout: float = 3.0) -> None:
    request_stop()
    pid = running_pid()
    if pid is None:
        clear_live_state()
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            break
        time.sleep(0.15)
    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        time.sleep(0.4)
    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    clear_live_state()


def attach_if_running() -> dict | None:
    from closed_loop.worker import loop_alive

    if not loop_alive():
        return None
    return read_run()
