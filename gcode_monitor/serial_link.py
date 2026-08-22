"""Keep one USB session open. Re-opening Marlin often pulses DTR and resets the print."""

from __future__ import annotations

import time

_link = None


class MarlinLink:
    def __init__(self):
        self.ser = None
        self.port = None
        self.baud = None

    def connect(self, port: str, baud: int) -> None:
        import serial

        if self.ser is not None and self.ser.is_open and self.port == port and self.baud == baud:
            return
        self.close()
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = int(baud)
        ser.timeout = 0.4
        ser.dsrdtr = False
        ser.rtscts = False
        ser.open()
        try:
            ser.dtr = False
            ser.rts = False
        except Exception:
            pass
        time.sleep(0.25)
        ser.reset_input_buffer()
        self.ser = ser
        self.port = port
        self.baud = int(baud)

    def query(self, command: str, seconds: float = 0.9) -> list[str]:
        if self.ser is None or not self.ser.is_open:
            return []
        self.ser.reset_input_buffer()
        self.ser.write((command.strip() + "\n").encode("ascii"))
        lines: list[str] = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            lines.append(raw.decode("latin-1", errors="replace").strip())
        return lines

    def send(self, command: str) -> None:
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("serial is not open")
        self.ser.write((command.strip() + "\n").encode("ascii"))
        time.sleep(0.15)

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.port = None
        self.baud = None


def shared_link() -> MarlinLink:
    global _link
    if _link is None:
        _link = MarlinLink()
    return _link
