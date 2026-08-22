"""Draw the planned path and a part mask in image space."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter

from gcode_monitor.parse import Layer, ParsedPrint


def _mapper(
    parsed: ParsedPrint,
    width: int,
    height: int,
    pad: float = 0.10,
    fit: str = "bed",
):
    if fit == "part":
        min_x, min_y = parsed.min_x, parsed.min_y
        max_x, max_y = parsed.max_x, parsed.max_y
    else:
        min_x = min_y = 0.0
        max_x, max_y = parsed.bed_w, parsed.bed_h
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    inner_w = width * (1.0 - 2 * pad)
    inner_h = height * (1.0 - 2 * pad)
    scale = min(inner_w / span_x, inner_h / span_y)
    ox = (width - span_x * scale) / 2 - min_x * scale
    oy = (height + span_y * scale) / 2 + min_y * scale

    def xy(x: float, y: float) -> tuple[float, float]:
        return x * scale + ox, oy - y * scale

    return xy, scale


def render_path(
    parsed: ParsedPrint,
    layer: Layer,
    *,
    width: int = 640,
    height: int = 640,
    pad: float = 0.10,
    travel: bool = False,
    fit: str = "bed",
) -> Image.Image:
    img = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(img)
    xy, scale = _mapper(parsed, width, height, pad, fit=fit)
    nozzle_px = max(2.0, 0.4 * scale)
    draw.rectangle([8, 8, width - 9, height - 9], outline=(228, 224, 216), width=1)
    pts = [xy(x, y) for x, y in layer.points]
    if len(pts) >= 2:
        draw.line(pts, fill=(24, 95, 165), width=max(2, int(nozzle_px)), joint="curve")
    elif len(pts) == 1:
        cx, cy = pts[0]
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(24, 95, 165))
    if pts:
        cx, cy = pts[-1]
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(180, 35, 24))
    _ = travel
    return img


def render_mask(
    parsed: ParsedPrint,
    layer: Layer,
    *,
    width: int,
    height: int,
    pad: float = 0.10,
    grow: int = 8,
    fit: str = "bed",
) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    xy, scale = _mapper(parsed, width, height, pad, fit=fit)
    pts = [xy(x, y) for x, y in layer.points]
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = max(6.0, 4.0 * scale * 0.4)
        draw.rectangle(
            [min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin],
            fill=255,
        )
        if len(pts) >= 2:
            width_px = max(4, int(0.8 * scale) + max(2, grow // 2))
            draw.line(pts, fill=255, width=width_px, joint="curve")
    if grow >= 3:
        size = grow if grow % 2 else grow + 1
        mask = mask.filter(ImageFilter.MaxFilter(size=size))
    return mask


def apply_mask(image: Image.Image, mask: Image.Image, dim: int = 55) -> Image.Image:
    rgb = image.convert("RGB")
    m = mask.resize(rgb.size, Image.Resampling.NEAREST)
    faded = rgb.point(lambda p: int(p * dim / 255))
    return Image.composite(rgb, faded, m)


def overlay_guidance(
    image: Image.Image,
    parsed: ParsedPrint,
    layer: Layer,
    *,
    pad: float = 0.10,
    fit: str = "part",
) -> Image.Image:
    rgb = image.convert("RGB")
    mask = render_mask(parsed, layer, width=rgb.width, height=rgb.height, pad=pad, fit=fit)
    guided = apply_mask(rgb, mask, dim=72)
    draw = ImageDraw.Draw(guided)
    xy, scale = _mapper(parsed, rgb.width, rgb.height, pad, fit=fit)
    pts = [xy(x, y) for x, y in layer.points]
    if len(pts) >= 2:
        draw.line(pts, fill=(24, 95, 165), width=max(2, int(0.45 * scale)))
    if pts:
        cx, cy = pts[-1]
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(180, 35, 24))
    return guided


def draw_kept_boxes(image: Image.Image, kept: list[dict]) -> Image.Image:
    from actions import pretty

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    colors = {
        "ok": (6, 118, 71),
        "warning": (181, 71, 8),
        "critical": (180, 35, 24),
    }
    for item in kept:
        x1, y1, x2, y2 = item["xyxy"]
        color = colors[pretty(item["class"])["severity"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{pretty(item['class'])['label']} {item['confidence'] * 100:.0f}%"
        draw.rectangle([x1, max(0, y1 - 16), x1 + 8 * len(label), y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=(255, 255, 255))
    return out


def box_on_mask(mask: Image.Image, xyxy: tuple[float, float, float, float], min_frac: float = 0.35) -> bool:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, x2 = sorted((max(0, x1), min(mask.width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(mask.height - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = mask.crop((x1, y1, x2, y2))
    hist = crop.histogram()
    on = sum(hist[128:])
    return on / max(crop.width * crop.height, 1) >= min_frac
