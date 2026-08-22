"""Parse FDM G-code into layers. Handles Cura ;LAYER:n and Prusa ;LAYER_CHANGE.

Stores a subsampled extrusion path per layer, not every G1. A 10 MB Prusa file
cannot live as hundreds of thousands of Python move objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

WORD = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")
LAYER_INDEX = re.compile(r";\s*LAYER\s*[:=]\s*(-?\d+)\b", re.IGNORECASE)
LAYER_CHANGE = re.compile(r";\s*LAYER_CHANGE\b", re.IGNORECASE)
Z_COMMENT = re.compile(r";\s*Z:([-+0-9.]+)")


@dataclass
class Layer:
    index: int
    z: float
    start_pos: int = 0
    end_pos: int = 0
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 0.0
    max_y: float = 0.0
    points: list[tuple[float, float]] = field(default_factory=list)
    n_extrude: int = 0
    n_travel: int = 0
    _seeded: bool = field(default=False, repr=False)

    def add_point(self, x: float, y: float, *, keep_every_mm: float = 0.45, cap: int = 400) -> None:
        if not self._seeded:
            self.min_x = self.max_x = x
            self.min_y = self.max_y = y
            self._seeded = True
        else:
            self.min_x = min(self.min_x, x)
            self.max_x = max(self.max_x, x)
            self.min_y = min(self.min_y, y)
            self.max_y = max(self.max_y, y)
        if not self.points:
            self.points.append((x, y))
            return
        if len(self.points) >= cap:
            return
        lx, ly = self.points[-1]
        if (x - lx) ** 2 + (y - ly) ** 2 >= keep_every_mm ** 2:
            self.points.append((x, y))


@dataclass
class ParsedPrint:
    name: str
    size: int
    layers: list[Layer]
    bed_w: float
    bed_h: float
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    path: str = ""
    source: str = ""
    flavour: str = ""

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def layer_at(self, index: int) -> Layer:
        if not self.layers:
            raise IndexError("no layers")
        i = max(0, min(int(index), len(self.layers) - 1))
        return self.layers[i]

    def layer_for_filepos(self, file_pos: int) -> Layer:
        if not self.layers:
            raise IndexError("no layers")
        chosen = self.layers[0]
        for layer in self.layers:
            if layer.start_pos <= file_pos:
                chosen = layer
            else:
                break
        return chosen

    def snippet(self, layer: Layer, max_chars: int = 700) -> str:
        data = b""
        if self.path:
            data = Path(self.path).read_bytes()
        elif self.source:
            data = self.source.encode("utf-8")
        if not data:
            return ""
        chunk = data[layer.start_pos : layer.end_pos].decode("utf-8", errors="replace").strip()
        if len(chunk) > max_chars:
            return chunk[: max_chars // 2] + "\n…\n" + chunk[-max_chars // 2 :]
        return chunk


def _words(line: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for match in WORD.finditer(line):
        out[match.group(1).upper()] = float(match.group(2))
    return out


def parse_gcode(
    source: str | Path | bytes,
    *,
    name: str = "print.gcode",
    bed_w: float = 210.0,
    bed_h: float = 210.0,
    keep_source: bool = False,
) -> ParsedPrint:
    path_str = ""
    if isinstance(source, bytes):
        raw = source.decode("utf-8", errors="replace")
        label = name
    else:
        path = Path(source)
        raw = path.read_text(encoding="utf-8", errors="replace")
        label = path.name
        path_str = str(path)

    has_layer_comments = bool(LAYER_CHANGE.search(raw) or LAYER_INDEX.search(raw))
    flavour = "prusa-or-cura-layers" if has_layer_comments else "z-hop-fallback"

    abs_xyz = True
    abs_e = True
    x = y = z = e = 0.0
    known_xy = False
    layers: list[Layer] = []
    current: Layer | None = None

    def close_current(end: int) -> None:
        if current is not None:
            current.end_pos = end

    def open_layer(index: int, z_val: float, pos: int) -> Layer:
        layer = Layer(index=index, z=z_val, start_pos=pos)
        layers.append(layer)
        return layer

    lines = raw.splitlines(keepends=True)
    pos = 0
    for line_no, line in enumerate(lines, start=1):
        start = pos
        pos += len(line.encode("utf-8"))
        stripped = line.strip()
        if not stripped:
            continue

        if LAYER_CHANGE.match(stripped):
            close_current(start)
            current = open_layer(len(layers), z, start)
            continue

        indexed = LAYER_INDEX.search(stripped)
        if indexed:
            close_current(start)
            current = open_layer(max(int(indexed.group(1)), 0), z, start)
            continue

        z_hit = Z_COMMENT.match(stripped)
        if z_hit and current is not None:
            current.z = float(z_hit.group(1))
            z = current.z
            continue

        code = stripped.split(";", 1)[0].strip()
        if not code:
            continue
        cmd = code.split()[0].upper()
        args = _words(code)

        if cmd == "G90":
            abs_xyz = True
            continue
        if cmd == "G91":
            abs_xyz = False
            continue
        if cmd == "M82":
            abs_e = True
            continue
        if cmd == "M83":
            abs_e = False
            continue
        if cmd == "G92":
            if "X" in args:
                x = args["X"]
            if "Y" in args:
                y = args["Y"]
            if "X" in args or "Y" in args:
                known_xy = True
            if "Z" in args:
                z = args["Z"]
            if "E" in args:
                e = args["E"]
            continue
        if cmd == "G28":
            if not any(k in args for k in ("X", "Y", "Z")):
                x = y = z = 0.0
                known_xy = True
            else:
                if "X" in args:
                    x = 0.0
                    known_xy = True
                if "Y" in args:
                    y = 0.0
                    known_xy = True
                if "Z" in args:
                    z = 0.0
            continue
        if cmd not in {"G0", "G00", "G1", "G01"}:
            continue

        x1, y1, z1, e1 = x, y, z, e
        has_xy = "X" in args or "Y" in args
        if "X" in args:
            x1 = args["X"] if abs_xyz else x + args["X"]
            known_xy = True
        if "Y" in args:
            y1 = args["Y"] if abs_xyz else y + args["Y"]
            known_xy = True
        if "Z" in args:
            z1 = args["Z"] if abs_xyz else z + args["Z"]
        if "E" in args:
            e1 = args["E"] if abs_e else e + args["E"]

        e_delta = e1 - e
        extrude = e_delta > 1e-6 and known_xy and has_xy

        if current is None:
            if has_layer_comments or not has_xy:
                x, y, z, e = x1, y1, z1, e1
                continue
            current = open_layer(0, z1, start)
        elif not has_layer_comments and z1 - current.z > 0.04:
            close_current(start)
            current = open_layer(len(layers), z1, start)

        current.z = z1
        if extrude:
            current.n_extrude += 1
            current.add_point(x1, y1)
        elif has_xy:
            current.n_travel += 1
        x, y, z, e = x1, y1, z1, e1

    if current is not None and current.end_pos == 0:
        current.end_pos = len(raw.encode("utf-8"))

    xs: list[float] = []
    ys: list[float] = []
    for layer in layers:
        if layer._seeded:
            xs.extend([layer.min_x, layer.max_x])
            ys.extend([layer.min_y, layer.max_y])
    if not xs:
        min_x = min_y = 0.0
        max_x, max_y = bed_w, bed_h
    else:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

    return ParsedPrint(
        name=label,
        size=len(raw.encode("utf-8")),
        layers=layers,
        bed_w=bed_w,
        bed_h=bed_h,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        path=path_str,
        source=raw if keep_source else "",
        flavour=flavour,
    )
