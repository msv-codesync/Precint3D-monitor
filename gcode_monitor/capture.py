"""Grab one still: ESP32 HTTP, webcam index, or an already-open image."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO

from PIL import Image

from gcode_monitor.net import http_url

# Continuity Camera / Desk View show up as extra AVFoundation devices on macOS.
# FaceTime is almost always index 0; the iPhone is usually 1 once it is unlocked nearby.
_IPHONE_HINTS = ("iphone", "continuity", "desk view", "center stage")
_MAC_HINTS = ("facetime", "built-in", "macbook", "imac")


def _av_backend():
    import cv2

    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return int(cv2.CAP_AVFOUNDATION)
    return int(getattr(cv2, "CAP_ANY", 0))


def _macos_camera_names() -> list[str]:
    if sys.platform != "darwin":
        return []
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPCameraDataType", "-json"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return []
    names: list[str] = []
    for item in data.get("SPCameraDataType") or []:
        name = item.get("_name") or item.get("spcamera_model-id")
        if name:
            names.append(str(name))
    return names


def is_iphone_camera(name: str) -> bool:
    text = (name or "").lower()
    if any(h in text for h in _MAC_HINTS):
        return False
    return any(h in text for h in _IPHONE_HINTS)


def _silence_cv2() -> None:
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    try:
        import cv2

        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
        utils = getattr(cv2, "utils", None)
        logging = getattr(utils, "logging", None) if utils is not None else None
        silent = getattr(logging, "LOG_LEVEL_SILENT", None) if logging is not None else None
        if logging is not None and silent is not None and hasattr(logging, "setLogLevel"):
            logging.setLogLevel(silent)
    except Exception:
        pass


def list_cameras(max_index: int = 6, *, probe: bool = False) -> list[dict]:
    """Cameras as ``{index, name, preferred}``. Preferred = iPhone / Continuity.

    Default is names from macOS, no OpenCV open — probing 0–6 fights Continuity Camera.
    Pass ``probe=True`` from Find cameras; only indices that match the name list (or 0–2) are opened, without grabbing a frame.
    """
    names = _macos_camera_names()
    if names and not probe:
        return [
            {"index": i, "name": name, "preferred": is_iphone_camera(name)}
            for i, name in enumerate(names)
        ]

    _silence_cv2()
    import cv2

    backend = _av_backend()
    if names:
        last = min(int(max_index), max(len(names) - 1, 0))
    else:
        last = min(int(max_index), 2)
    found: list[dict] = []
    for index in range(last + 1):
        cap = cv2.VideoCapture(index, backend)
        ok = bool(cap.isOpened())
        cap.release()
        if not ok:
            continue
        name = names[index] if index < len(names) else f"Camera {index}"
        found.append(
            {
                "index": index,
                "name": name,
                "preferred": is_iphone_camera(name),
            }
        )
    return found


def default_camera_index(cameras: list[dict] | None = None) -> int:
    cams = cameras if cameras is not None else list_cameras()
    for cam in cams:
        if cam.get("preferred"):
            return int(cam["index"])
    indices = [int(cam["index"]) for cam in cams]
    if 1 in indices:
        return 1
    return indices[0] if indices else 0


def from_esp32(url: str, timeout: float = 8.0) -> Image.Image:
    import requests

    resp = requests.get(http_url(url), timeout=timeout)
    resp.raise_for_status()
    with Image.open(BytesIO(resp.content)) as img:
        return img.convert("RGB")


def from_webcam(index: int) -> Image.Image:
    cam = PersistentWebcam(index)
    try:
        return cam.grab()
    finally:
        cam.close()


def probe_cameras(max_index: int = 5) -> list[int]:
    """Indices that OpenCV can open. Call from a button — opening every device is slow on macOS."""
    return [int(cam["index"]) for cam in list_cameras(max_index=max_index, probe=True)]


class PersistentWebcam:
    """Keep the capture open while watching. Opening every tick drops frames on macOS."""

    def __init__(self, index: int):
        import cv2

        self._cv2 = cv2
        self.index = int(index)
        self.cap = cv2.VideoCapture(self.index, _av_backend())
        if not self.cap.isOpened():
            raise RuntimeError(
                "That camera did not open. Unlock your iPhone nearby (Continuity Camera), "
                "or click Find cameras and pick it by name. Allow Camera for Terminal or Cursor in System Settings."
            )

    def grab(self) -> Image.Image:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera returned no frame.")
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_first_camera(preferred: int | None = None) -> tuple[PersistentWebcam, int]:
    """Open a camera. Prefers iPhone / Continuity Camera, then index 1, then the rest."""
    cams = list_cameras()
    order: list[int] = []
    if preferred is not None:
        order.append(int(preferred))
    order.append(default_camera_index(cams))
    order.extend(int(cam["index"]) for cam in cams if cam.get("preferred"))
    order.extend(int(cam["index"]) for cam in cams)
    order.extend([1, 0, 2, 3])
    seen: set[int] = set()
    last_err: Exception | None = None
    for idx in order:
        if idx in seen:
            continue
        seen.add(idx)
        try:
            cam = PersistentWebcam(idx)
            return cam, idx
        except RuntimeError as exc:
            last_err = exc
    raise RuntimeError(str(last_err) if last_err else "No camera opened.")
