"""One frame: G-code layer + mask + YOLO + sticky class."""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from actions import severity_rank
from gcode_monitor.filter import StickyClass, keep_box
from gcode_monitor.mask import apply_mask, draw_kept_boxes, render_mask, render_path
from gcode_monitor.parse import ParsedPrint
from gcode_monitor.progress import Progress


@dataclass
class FrameVerdict:
    layer_index: int
    layer_z: float
    progress_pct: float
    progress_note: str
    name: str
    conf: float
    sticky_ok: bool
    kept: list[dict]
    dropped: int
    path_image: Image.Image
    masked_image: Image.Image
    marked_image: Image.Image
    snippet: str
    extra: str = ""
    raw: list[dict] = field(default_factory=list)


def resolve_layer(parsed: ParsedPrint, progress: Progress):
    if progress.file_pos is not None and parsed.size:
        return parsed.layer_for_filepos(progress.file_pos)
    if progress.layer_hint is not None:
        return parsed.layer_at(progress.layer_hint)
    idx = int(progress.fraction * max(parsed.layer_count - 1, 0))
    return parsed.layer_at(idx)


def boxes_from_yolo(results, mask=None, frame_h=None) -> tuple[str, float, list[dict], int]:
    """Apply class floors and optional path/edge filters. YOLO plot is not the verdict."""
    kept: list[dict] = []
    dropped = 0
    result = results[0] if results else None
    if result is not None and result.boxes is not None:
        for box in result.boxes:
            name = result.names[int(box.cls.item())]
            conf = float(box.conf.item())
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            item = {"class": name, "confidence": round(conf, 3), "xyxy": xyxy}
            if keep_box(name, conf, xyxy, mask, frame_h=frame_h):
                kept.append(item)
            else:
                dropped += 1
    if not kept:
        return "good_print", 0.0, kept, dropped
    kept.sort(key=lambda it: (severity_rank(it["class"]), it["confidence"]), reverse=True)
    return kept[0]["class"], kept[0]["confidence"], kept, dropped


def judge_frame(
    parsed: ParsedPrint,
    progress: Progress,
    image: Image.Image,
    results,
    sticky: StickyClass,
    *,
    pad: float = 0.10,
    fit: str = "part",
    update_sticky: bool = True,
) -> FrameVerdict:
    image = image.convert("RGB")
    layer = resolve_layer(parsed, progress)
    mask = render_mask(parsed, layer, width=image.width, height=image.height, pad=pad, fit=fit)
    path_image = render_path(parsed, layer, width=640, height=640, pad=pad, travel=True, fit=fit)
    masked = apply_mask(image, mask)

    name, conf, kept, dropped = boxes_from_yolo(results, mask=mask, frame_h=image.height)

    if update_sticky:
        shown, sticky_ok = sticky.push(name)
    else:
        shown, sticky_ok = name, False

    snippet = parsed.snippet(layer)
    return FrameVerdict(
        layer_index=layer.index,
        layer_z=layer.z,
        progress_pct=progress.fraction * 100.0,
        progress_note=progress.extra,
        name=shown,
        conf=conf,
        sticky_ok=sticky_ok,
        kept=kept,
        dropped=dropped,
        path_image=path_image,
        masked_image=masked,
        marked_image=draw_kept_boxes(masked, kept),
        snippet=snippet,
        extra=progress.source,
        raw=kept,
    )


def should_ask_llm(name: str, sticky_ok: bool) -> bool:
    return bool(sticky_ok) and name != "good_print"


def llm_context(verdict: FrameVerdict) -> str:
    held = "yes" if verdict.sticky_ok else "no — do not authorise a printer command"
    return (
        f"Layer {verdict.layer_index}, Z={verdict.layer_z:.2f} mm, "
        f"{verdict.progress_pct:.0f}% through the file ({verdict.extra}). "
        f"Off-path boxes dropped: {verdict.dropped}. "
        f"Same finding held across frames: {held}. "
        f"You do not send G-code. Recommend in words only.\n"
        f"G-code slice:\n{verdict.snippet[:700]}"
    )
