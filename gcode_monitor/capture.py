"""Grab one still: network camera (ESP32 / phone), webcam index, or an open image."""

from __future__ import annotations

import atexit
import gc
import json
import multiprocessing
import os
import re
import subprocess
import sys
import time
import weakref
from io import BytesIO
from urllib.parse import urlparse

from PIL import Image

from gcode_monitor.net import http_url

# Continuity Camera / Desk View show up as extra AVFoundation devices on macOS.
# Name lists (system_profiler / ffmpeg) usually put FaceTime first. OpenCV's
# CAP_AVFOUNDATION often puts the iPhone at 0 when Continuity is the default.
_IPHONE_HINTS = ("iphone", "continuity", "desk view", "center stage")
_MAC_HINTS = ("facetime", "built-in", "macbook", "imac", "integrated", "webcam")

# A phone should reach the console over the LAN, not by hijacking the local
# capture device. Anything wireless goes through from_network_camera().
DEFAULT_ESP32_PATH = "/capture"
_MJPEG_MARKERS = ("multipart/x-mixed-replace", "mjpeg", "mjpg")

LOCAL_SOURCE = "Laptop camera"
PHONE_SOURCE = "Phone (Wi-Fi)"
ESP32_SOURCE = "ESP32-CAM"
CAMERA_SOURCES = (LOCAL_SOURCE, PHONE_SOURCE, ESP32_SOURCE)
# Sessions saved by earlier builds still say "Local camera" / "ESP32".
_LOCAL_ALIASES = frozenset({"", "local camera", "laptop camera", "local", "laptop", "usb"})


def is_network_source(kind: str) -> bool:
    """True when frames arrive over HTTP instead of from an attached capture device."""
    return str(kind or "").strip().lower() not in _LOCAL_ALIASES


def _av_backend():
    import cv2

    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return int(cv2.CAP_AVFOUNDATION)
    return int(getattr(cv2, "CAP_ANY", 0))


def parse_avfoundation_list(text: str) -> list[tuple[int, str]]:
    """Device names from ``ffmpeg -f avfoundation -list_devices`` (not OpenCV order)."""
    names: list[tuple[int, str]] = []
    in_video = False
    for line in (text or "").splitlines():
        lower = line.lower()
        if "avfoundation video devices" in lower:
            in_video = True
            continue
        if in_video and "avfoundation audio devices" in lower:
            break
        if not in_video:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if match:
            names.append((int(match.group(1)), match.group(2).strip()))
    return names


def _ffmpeg_camera_names() -> list[str]:
    if sys.platform != "darwin":
        return []
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    pairs = parse_avfoundation_list(text)
    if not pairs:
        return []
    last = max(i for i, _ in pairs)
    out = [f"Camera {i}" for i in range(last + 1)]
    for index, name in pairs:
        if 0 <= index < len(out):
            out[index] = name
    return out


def _macos_camera_names() -> list[str]:
    if sys.platform != "darwin":
        return []
    ffmpeg_names = _ffmpeg_camera_names()
    if ffmpeg_names:
        return ffmpeg_names
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


def release_capture(cap, cv2_mod=None) -> None:
    """Drop an AVFoundation device. ``release()`` alone often leaves the Mac LED on."""
    if cap is None:
        return
    try:
        cap.release()
    except Exception:
        pass
    if cv2_mod is not None:
        try:
            cv2_mod.destroyAllWindows()
        except Exception:
            pass
    gc.collect()
    time.sleep(0.25)


def is_iphone_camera(name: str) -> bool:
    text = (name or "").lower()
    if any(h in text for h in _MAC_HINTS):
        return False
    return any(h in text for h in _IPHONE_HINTS)


def is_mac_camera(name: str) -> bool:
    text = (name or "").lower()
    return any(h in text for h in _MAC_HINTS)


def is_builtin_camera(name: str) -> bool:
    """True for the camera physically attached to this machine."""
    return is_mac_camera(name) or not is_iphone_camera(name)


def opencv_capture_index(listed_index: int, names: list[str] | None = None, *, macos: bool | None = None) -> int:
    """Map a labeled camera row to the OpenCV device index on macOS.

    The dropdown is labeled from the system name list (FaceTime, then iPhone).
    OpenCV often opens those two in the opposite order. Swap only that pair.
    """
    listed = int(listed_index)
    if macos is None:
        macos = sys.platform == "darwin"
    if not macos:
        return listed
    if names is None:
        names = _macos_camera_names()
    if len(names) < 2:
        return listed
    mac_at = next((i for i, name in enumerate(names) if is_mac_camera(name)), None)
    phone_at = next((i for i, name in enumerate(names) if is_iphone_camera(name)), None)
    if mac_at is None or phone_at is None or mac_at >= phone_at:
        return listed
    if listed == mac_at:
        return phone_at
    if listed == phone_at:
        return mac_at
    return listed


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
    """Cameras as ``{index, name, preferred, phone}``. Preferred = this machine's own camera.

    Default lists AVFoundation names without opening devices. Probing fights Continuity
    if another capture is already held, so pass ``probe=True`` only after releasing the
    live camera. A paired phone is still listed, but it is never chosen for you.
    """
    names = _macos_camera_names()
    if names and not probe:
        return [
            {
                "index": i,
                "name": name,
                "preferred": is_builtin_camera(name),
                "phone": is_iphone_camera(name),
            }
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
        release_capture(cap, cv2)
        if not ok:
            continue
        name = names[index] if index < len(names) else f"Camera {index}"
        found.append(
            {
                "index": index,
                "name": name,
                "preferred": is_builtin_camera(name),
                "phone": is_iphone_camera(name),
            }
        )
    return found


def default_camera_index(cameras: list[dict] | None = None) -> int:
    """This machine's own camera. Never auto-select a paired phone."""
    cams = cameras if cameras is not None else list_cameras()
    for cam in cams:
        if cam.get("preferred"):
            return int(cam["index"])
    indices = [int(cam["index"]) for cam in cams]
    return indices[0] if indices else 0


def preferred_camera_choice(cameras: list[dict] | None = None) -> tuple[int, str]:
    """Name and index of the camera the console should open. Does not open a capture."""
    cams = cameras if cameras is not None else list_cameras(probe=False)
    for cam in cams:
        if cam.get("preferred"):
            return int(cam["index"]), str(cam["name"])
    names = {int(cam["index"]): str(cam["name"]) for cam in cams}
    if names:
        idx = default_camera_index(cams)
        return idx, names.get(idx, f"Camera {idx}")
    return 0, "Camera 0"


# ---------------------------------------------------------------------------
# Network cameras: ESP32-CAM boards and phones running an IP-camera app
# ---------------------------------------------------------------------------

def normalize_camera_url(text: str, *, path: str = DEFAULT_ESP32_PATH) -> str:
    """Accept ``192.168.1.20``, ``192.168.1.20:81`` or a full URL and return a usable one.

    Operators paste whatever the board or the phone app shows them. Only the
    scheme and a default path are filled in; the host is never guessed.
    """
    raw = (text or "").strip().strip("/")
    if not raw:
        raise ValueError("Enter the camera IP address shown on your phone or board.")
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if not parsed.path or parsed.path == "/":
        raw = f"{raw.rstrip('/')}{path}"
    return http_url(raw)


def _frame_from_mjpeg(resp) -> Image.Image:
    """Pull the first complete JPEG out of a multipart MJPEG stream."""
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buf.extend(chunk)
        start = buf.find(b"\xff\xd8")
        end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
        if start >= 0 and end > start:
            with Image.open(BytesIO(bytes(buf[start:end + 2]))) as img:
                return img.convert("RGB")
        if len(buf) > 4_000_000:
            break
    raise RuntimeError("No complete frame arrived from that camera stream.")


def from_network_camera(url: str, timeout: float = 8.0) -> Image.Image:
    """One frame from an ESP32-CAM or a phone IP-camera app.

    Handles both a plain JPEG snapshot endpoint and an MJPEG stream, because
    Android IP Webcam serves ``/video`` while ESP32-CAM serves ``/capture``.
    """
    import requests

    safe = http_url(url)
    resp = requests.get(safe, timeout=timeout, stream=True)
    resp.raise_for_status()
    content_type = str(resp.headers.get("Content-Type", "")).lower()
    if any(marker in content_type for marker in _MJPEG_MARKERS):
        try:
            return _frame_from_mjpeg(resp)
        finally:
            resp.close()
    with Image.open(BytesIO(resp.content)) as img:
        return img.convert("RGB")


def from_esp32(url: str, timeout: float = 8.0) -> Image.Image:
    return from_network_camera(url, timeout=timeout)


def from_webcam(index: int) -> Image.Image:
    cam = ProcessWebcam(index)
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
        self.cap = cv2.VideoCapture(opencv_capture_index(self.index), _av_backend())
        if not self.cap.isOpened():
            raise RuntimeError("That camera did not open.")

    def grab(self) -> Image.Image:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera returned no frame.")
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def close(self) -> None:
        cap = self.cap
        self.cap = None
        release_capture(cap, self._cv2)


def _camera_worker(index: int, frame_q, stop_evt) -> None:
    """Child process owns AVFoundation. Killing this process is what turns the LED off."""
    import time as _time

    import cv2

    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backend = int(cv2.CAP_AVFOUNDATION)
    else:
        backend = int(getattr(cv2, "CAP_ANY", 0))
    cap = cv2.VideoCapture(int(index), backend)
    if not cap.isOpened():
        try:
            stop_evt.set()
        except Exception:
            pass
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        while not stop_evt.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                _time.sleep(0.04)
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok2:
                continue
            payload = buf.tobytes()
            try:
                frame_q.put_nowait(payload)
            except Exception:
                try:
                    frame_q.get_nowait()
                except Exception:
                    pass
                try:
                    frame_q.put_nowait(payload)
                except Exception:
                    pass
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _stop_process(proc) -> None:
    if proc is None:
        return
    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.6)
    except Exception:
        pass
    try:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=0.5)
    except Exception:
        pass


def kill_stray_camera_workers() -> None:
    """Terminate leftover p3-cam-* children if Streamlit dropped the wrapper object."""
    for proc in multiprocessing.active_children():
        name = str(getattr(proc, "name", "") or "")
        if name.startswith("p3-cam-"):
            _stop_process(proc)


atexit.register(kill_stray_camera_workers)


def _close_mp_queue(q) -> None:
    if q is None:
        return
    try:
        q.close()
    except Exception:
        pass
    try:
        q.cancel_join_thread()
    except Exception:
        pass


class ProcessWebcam:
    """Live camera in a child process so Stop can actually release the Mac camera."""

    def __init__(self, index: int):
        self.index = int(index)
        ctx = multiprocessing.get_context("spawn")
        self._q = ctx.Queue(maxsize=1)
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=_camera_worker,
            args=(opencv_capture_index(self.index), self._q, self._stop),
            daemon=True,
            name=f"p3-cam-{self.index}",
        )
        self._finalizer = None
        self._closed = False
        self._proc.start()
        self._finalizer = weakref.finalize(self, _shutdown_camera_child, self._proc, self._stop, self._q)
        first = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not self._proc.is_alive():
                self.close()
                raise RuntimeError("That camera did not open.")
            try:
                first = self._q.get(timeout=0.15)
                break
            except Exception:
                continue
        if first is None:
            self.close()
            raise RuntimeError("That camera did not open.")
        try:
            self._q.put_nowait(first)
        except Exception:
            pass

    @property
    def cap(self):
        proc = self._proc
        if proc is not None and proc.is_alive():
            return proc
        return None

    def grab(self) -> Image.Image:
        try:
            data = self._q.get(timeout=2.8)
        except Exception as exc:
            raise RuntimeError("Camera returned no frame.") from exc
        with Image.open(BytesIO(data)) as img:
            return img.convert("RGB")

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            try:
                finalizer.detach()
            except Exception:
                pass
            self._finalizer = None
        try:
            stop = getattr(self, "_stop", None)
            if stop is not None:
                stop.set()
        except Exception:
            pass
        proc = getattr(self, "_proc", None)
        self._proc = None
        had_proc = proc is not None
        _stop_process(proc)
        _close_mp_queue(getattr(self, "_q", None))
        if had_proc:
            time.sleep(0.3)


def _shutdown_camera_child(proc, stop_evt, q) -> None:
    try:
        stop_evt.set()
    except Exception:
        pass
    _stop_process(proc)
    _close_mp_queue(q)


def open_first_camera(preferred: int | None = None) -> tuple[PersistentWebcam, int]:
    """Open a camera. Tries this machine's own camera first, then the rest."""
    cams = list_cameras()
    order: list[int] = []
    if preferred is not None:
        order.append(int(preferred))
    order.append(default_camera_index(cams))
    order.extend(int(cam["index"]) for cam in cams if cam.get("preferred"))
    order.extend(int(cam["index"]) for cam in cams)
    order.extend([0, 1, 2, 3])
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
