"""Grab one still: ESP32 HTTP, webcam index, or an already-open image."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from gcode_monitor.net import http_url


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


class PersistentWebcam:
    """Keep the capture open while Job is watching. Opening every tick drops frames on macOS."""

    def __init__(self, index: int):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(int(index))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera {index} did not open. On a Mac try index 1 (Continuity Camera) "
                "and allow Camera for Terminal or Cursor."
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
