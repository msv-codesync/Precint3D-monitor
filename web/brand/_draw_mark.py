"""Precint3D mark. Two same-size open contours stacked in Z. Gate on the far corner."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
TEAL = (22, 133, 107, 255)
MINT = (78, 203, 163, 255)
NAVY = (15, 22, 32, 255)


def contour(cx, cy, w, h, r, t0, t1, n=48, wobble=0.42, seed=2.4, shear=0.09):
    hw, hh = w / 2, h / 2
    steps = [
        ("h", cy + hh, cx, cx - hw + r),
        ("a", cx - hw + r, cy + hh - r, math.pi / 2, math.pi),
        ("v", cx - hw, cy + hh - r, cy - hh + r),
        ("a", cx - hw + r, cy - hh + r, math.pi, 1.5 * math.pi),
        ("h", cy - hh, cx - hw + r, cx + hw - r),
        ("a", cx + hw - r, cy - hh + r, 1.5 * math.pi, 2 * math.pi),
        ("v", cx + hw, cy - hh + r, cy + hh - r),
        ("a", cx + hw - r, cy + hh - r, 0.0, math.pi / 2),
        ("h", cy + hh, cx + hw - r, cx),
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
        phase = t * math.tau * 1.9 + seed
        amp = wobble * (0.45 + 0.55 * math.sin(t * math.tau * 2.2 + seed))
        x += amp * math.cos(phase * 1.35)
        y += amp * math.sin(phase * 1.05) * 0.75
        x = x + shear * (50 - y)
        pts.append((x, y))
    return pts


def path_d(pts) -> str:
    if not pts:
        return ""
    out = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    for i in range(1, len(pts) - 1):
        x, y = pts[i]
        nx, ny = pts[i + 1]
        out.append(f"Q {x:.2f} {y:.2f} {(x + nx) / 2:.2f} {(y + ny) / 2:.2f}")
    out.append(f"L {pts[-1][0]:.2f} {pts[-1][1]:.2f}")
    return " ".join(out)


def ink_png(paths, beads, size=720, bg=(255, 255, 255, 255), color=TEAL, weights=None):
    img = Image.new("RGBA", (size, size), bg)
    dr = ImageDraw.Draw(img)
    s = size / 80.0
    if weights is None:
        weights = [0.04] * len(paths)
    for pts, w in zip(paths, weights):
        rad = max(1.8, size * w / 2)
        for x, y in pts:
            px, py = x * s, y * s
            dr.ellipse((px - rad, py - rad, px + rad, py + rad), fill=color)
    br = size * 0.027
    for x, y in beads:
        px, py = x * s, y * s
        dr.ellipse((px - br, py - br, px + br, py + br), fill=color)
    return img


def group(L1, L2, tick, bead, color):
    return f"""  <path d="{path_d(L1)}" stroke="{color}" stroke-width="3.35" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="{path_d(L2)}" stroke="{color}" stroke-width="2.55" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="{path_d(tick)}" stroke="{color}" stroke-width="2.7" stroke-linecap="round"/>
  <circle cx="{bead[0]:.2f}" cy="{bead[1]:.2f}" r="3.1" fill="{color}"/>"""


def main():
    # Same part, two layers. Open far corner. Not a letter C (right side exists on L1).
    L1 = contour(39.5, 45.5, 45, 38, 12.2, 0.00, 0.88, n=180, wobble=0.72, seed=2.4, shear=0.10)
    # Second layer is only the risen top rail, not a second stem (avoids reading as a letter).
    L2 = contour(39.5, 33.0, 45, 38, 12.2, 0.30, 0.58, n=100, wobble=0.55, seed=5.1, shear=0.10)
    L1s = L1[::4] + [L1[-1]]
    L2s = L2[::4] + [L2[-1]]
    tick_end = [(63.2, 22.8), (65.1, 18.6), (66.4, 14.8)]
    tick = []
    for i in range(24):
        u = i / 23
        x = tick_end[0][0] * (1 - u) * (1 - u) + tick_end[1][0] * 2 * (1 - u) * u + tick_end[2][0] * u * u
        y = tick_end[0][1] * (1 - u) * (1 - u) + tick_end[1][1] * 2 * (1 - u) * u + tick_end[2][1] * u * u
        tick.append((x, y))
    tick = [(63.2, 22.8), (65.1, 18.6), (66.4, 14.8)]
    bead = L1[0]
    print("L1", L1[0], "->", L1[-1])
    print("L2", L2[0], "->", L2[-1])

    (OUT / "mark.svg").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" fill="none" role="img" aria-label="Precint3D">\n'
        f"{group(L1s, L2s, tick, bead, '#16856B')}\n</svg>\n",
        encoding="utf-8",
    )
    (OUT / "favicon.svg").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" fill="none">\n'
        '  <rect width="80" height="80" rx="18" fill="#0F1620"/>\n'
        f"{group(L1s, L2s, tick, bead, '#4ECBA3')}\n</svg>\n",
        encoding="utf-8",
    )
    (OUT / "mark-current.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" fill="none" aria-hidden="true">\n'
        f"{group(L1s, L2s, tick, bead, 'currentColor')}\n</svg>\n",
        encoding="utf-8",
    )
    ink_png([L1, L2, tick], [bead], color=TEAL, weights=[0.040, 0.030, 0.032]).save(OUT / "mark.png")
    navy = ink_png([L1, L2, tick], [bead], bg=NAVY, color=MINT, weights=[0.040, 0.030, 0.032])
    navy.save(OUT / "mark-on-navy.png")
    navy.resize((160, 160), Image.Resampling.LANCZOS).save(OUT / "mark-160.png")


if __name__ == "__main__":
    main()
