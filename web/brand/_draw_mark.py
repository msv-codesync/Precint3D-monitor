"""Clean Precint3D mark: one open frame, centered, no stray ticks."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
NAVY = (15, 22, 32, 255)
MINT = (78, 203, 163, 255)
TEAL = (22, 133, 107, 255)
WHITE = (255, 255, 255, 255)


def rounded_open(cx, cy, w, h, r, t0, t1, n=240):
    hw, hh = w / 2, h / 2
    steps = [
        ("v", cx + hw, cy - hh + r, cy + hh - r),
        ("a", cx + hw - r, cy + hh - r, 0.0, math.pi / 2),
        ("h", cy + hh, cx + hw - r, cx - hw + r),
        ("a", cx - hw + r, cy + hh - r, math.pi / 2, math.pi),
        ("v", cx - hw, cy + hh - r, cy - hh + r),
        ("a", cx - hw + r, cy - hh + r, math.pi, 1.5 * math.pi),
        ("h", cy - hh, cx - hw + r, cx + hw - r),
        ("a", cx + hw - r, cy - hh + r, 1.5 * math.pi, 2 * math.pi),
    ]
    lengths = []
    for kind, *rest in steps:
        lengths.append(abs(rest[2] - rest[1]) if kind in ("h", "v") else r * abs(rest[3] - rest[2]))
    total = sum(lengths) or 1.0
    pts = []
    for i in range(n):
        t = t0 + (t1 - t0) * (i / (n - 1))
        d = (t % 1.0) * total
        acc = 0.0
        x = y = 0.0
        for (kind, *rest), L in zip(steps, lengths):
            if d <= acc + L + 1e-9:
                u = 0 if L == 0 else min(1.0, max(0.0, (d - acc) / L))
                if kind == "h":
                    y0, x0, x1 = rest
                    x, y = x0 + (x1 - x0) * u, y0
                elif kind == "v":
                    x0, y0, y1 = rest
                    x, y = x0, y0 + (y1 - y0) * u
                else:
                    ax, ay, a0, a1 = rest
                    a = a0 + (a1 - a0) * u
                    x, y = ax + r * math.cos(a), ay + r * math.sin(a)
                break
            acc += L
        pts.append((x, y))
    return pts


def path_d(pts) -> str:
    if not pts:
        return ""
    bits = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    for x, y in pts[1:]:
        bits.append(f"L {x:.2f} {y:.2f}")
    return " ".join(bits)


def paint(pts, bead, *, bg, color, size=512, stroke=0.072):
    img = Image.new("RGBA", (size, size), bg)
    dr = ImageDraw.Draw(img)
    s = size / 80.0
    rad = size * stroke / 2
    for x, y in pts:
        px, py = x * s, y * s
        dr.ellipse((px - rad, py - rad, px + rad, py + rad), fill=color)
    br = rad * 1.15
    bx, by = bead[0] * s, bead[1] * s
    dr.ellipse((bx - br, by - br, bx + br, by + br), fill=color)
    return img


def main():
    # Open at the far corner only. No second stroke, no tick.
    pts = rounded_open(40, 40, 44, 44, 13, 0.02, 0.86, n=280)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    dx = 40 - (min(xs) + max(xs)) / 2
    dy = 40 - (min(ys) + max(ys)) / 2
    pts = [(x + dx, y + dy) for x, y in pts]
    bead = pts[0]

    svg_paths = f"""  <path d="{path_d(pts)}" stroke="{{stroke}}" stroke-width="5.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <circle cx="{bead[0]:.2f}" cy="{bead[1]:.2f}" r="3.4" fill="{{stroke}}"/>"""

    (OUT / "mark.svg").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" fill="none" role="img" aria-label="Precint3D">\n'
        + svg_paths.format(stroke="#16856B")
        + "\n</svg>\n",
        encoding="utf-8",
    )
    (OUT / "favicon.svg").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80">\n'
        '  <rect width="80" height="80" fill="#0F1620"/>\n'
        + svg_paths.format(stroke="#4ECBA3")
        + "\n</svg>\n",
        encoding="utf-8",
    )
    (OUT / "mark-current.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" fill="none" aria-hidden="true">\n'
        + svg_paths.format(stroke="currentColor")
        + "\n</svg>\n",
        encoding="utf-8",
    )

    # Opaque bitmaps so CSS radius does not pick up a gray halo.
    paint(pts, bead, bg=WHITE, color=TEAL, size=640, stroke=0.070).save(OUT / "mark.png")
    icon = paint(pts, bead, bg=NAVY, color=MINT, size=640, stroke=0.070)
    icon.save(OUT / "favicon.png")
    icon.resize((64, 64), Image.Resampling.LANCZOS).save(OUT / "favicon-64.png")
    web = OUT.parent
    icon.resize((256, 256), Image.Resampling.LANCZOS).save(web / "favicon.png")
    icon.resize((256, 256), Image.Resampling.LANCZOS).save(
        web / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)]
    )


if __name__ == "__main__":
    main()
