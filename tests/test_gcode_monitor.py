from pathlib import Path

from PIL import Image, ImageDraw

from gcode_monitor.filter import StickyClass, keep_box
from gcode_monitor.mask import box_on_mask, render_mask, render_path
from gcode_monitor.parse import parse_gcode
from gcode_monitor.pipeline import judge_frame
from gcode_monitor.progress import from_manual

FIXTURE = Path(__file__).resolve().parents[1] / "gcode_monitor" / "fixtures" / "sample_square.gcode"


class _Val:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Xy:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return list(self._vals)


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = _Val(cls_id)
        self.conf = _Val(conf)
        self.xyxy = [_Xy(xyxy)]


class _Result:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


def test_parse_layers_and_bbox():
    parsed = parse_gcode(FIXTURE, bed_w=210, bed_h=210)
    assert parsed.layer_count == 3
    assert parsed.min_x == 80
    assert parsed.max_x == 120
    assert parsed.min_y == 80
    assert parsed.max_y == 120
    assert abs(parsed.layers[0].z - 0.20) < 1e-6
    assert parsed.layers[0].n_extrude > 0
    assert len(parsed.layers[0].points) >= 2


def test_filepos_maps_to_later_layer():
    parsed = parse_gcode(FIXTURE)
    last = parsed.layers[-1]
    got = parsed.layer_for_filepos(last.start_pos + 10)
    assert got.index == last.index


def test_path_and_mask_draw():
    parsed = parse_gcode(FIXTURE)
    layer = parsed.layers[0]
    path = render_path(parsed, layer, width=200, height=200)
    mask = render_mask(parsed, layer, width=200, height=200)
    assert path.size == (200, 200)
    assert mask.getextrema()[1] > 0


def test_center_of_part_is_on_mask():
    parsed = parse_gcode(FIXTURE)
    mask = render_mask(parsed, parsed.layers[0], width=210, height=210, grow=3, pad=0.0)
    # 100,100 mm on a 210 mm bed, Y-up → image y = 110
    assert box_on_mask(mask, (90, 100, 110, 120), min_frac=0.1) is True


def test_bed_corner_is_not_on_path():
    parsed = parse_gcode(FIXTURE)
    mask = render_mask(parsed, parsed.layers[0], width=200, height=200)
    assert box_on_mask(mask, (0, 0, 12, 12), min_frac=0.3) is False
    assert keep_box("warping", 0.9, (0, 0, 12, 12), mask) is False
    assert keep_box("warping", 0.2, (80, 80, 120, 120), mask) is False


def test_sticky_needs_repeats():
    sticky = StickyClass(need=3)
    _, ok = sticky.push("warping")
    assert ok is False
    sticky.push("warping")
    name, ok = sticky.push("warping")
    assert name == "warping" and ok is True
    _, ok = sticky.push("spaghetti_failure")
    assert ok is False


def test_judge_drops_off_path_warping():
    parsed = parse_gcode(FIXTURE)
    img = Image.new("RGB", (200, 200), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.line((0, 190, 200, 190), fill=(80, 80, 80), width=3)
    results = [_Result({0: "warping"}, [_Box(0, 0.9, (0, 180, 40, 199))])]
    verdict = judge_frame(
        parsed,
        from_manual(0, 3, parsed.size),
        img,
        results,
        StickyClass(need=1),
        fit="bed",
    )
    assert verdict.dropped >= 1
    assert verdict.name == "good_print"
    assert verdict.layer_index == 0


def test_relative_e_counts_as_extrusion():
    text = """
G90
M83
G1 X10 Y10 F3000
G1 X20 Y10 E0.8 F1200
;LAYER:1
G1 Z0.4
G1 X20 Y20 E0.8
"""
    parsed = parse_gcode(text.encode(), name="rel.gcode")
    assert parsed.layer_count >= 1
    assert parsed.layers[0].n_extrude > 0


def test_job_store_stays_on_disk(tmp_path):
    from gcode_monitor.store import JobStore

    store = JobStore(tmp_path)
    folder = store.create(FIXTURE.read_bytes(), "sample_square.gcode")
    assert (folder / "print.gcode").exists()
    store.append_event(folder, {"name": "good_print"})
    lines = (folder / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_octoprint_maps_percent(monkeypatch):
    from gcode_monitor.progress import from_octoprint

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"progress": {"completion": 47.0, "filepos": 1200}, "state": "Printing"}

    monkeypatch.setattr("requests.get", lambda *a, **k: Resp())
    progress = from_octoprint("http://octopi.local", "key")
    assert progress.file_pos == 1200
    assert abs(progress.fraction - 0.47) < 1e-6
    assert progress.source == "octoprint"


def test_overlay_matches_photo_size():
    from gcode_monitor.mask import overlay_guidance

    parsed = parse_gcode(FIXTURE)
    photo = Image.new("RGB", (320, 240), (180, 180, 180))
    guided = overlay_guidance(photo, parsed, parsed.layers[0], pad=0.1, fit="part")
    assert guided.size == photo.size


def test_llm_only_on_held_defects():
    from gcode_monitor.pipeline import should_ask_llm

    assert should_ask_llm("good_print", True) is False
    assert should_ask_llm("spaghetti_failure", False) is False
    assert should_ask_llm("spaghetti_failure", True) is True


def test_m27_byte_line():
    from gcode_monitor.progress import _parse_m27

    pos, frac, extra = _parse_m27(["SD printing byte 250/1000"], 1000)
    assert pos == 250
    assert abs(frac - 0.25) < 1e-9
    pos, frac, extra = _parse_m27(["Not SD printing"], 1000)
    assert pos is None and frac == 0.0


def test_warping_on_frame_edge_is_dropped():
    assert keep_box("warping", 0.9, (10, 180, 40, 199), None, frame_h=200) is False
    assert keep_box("warping", 0.9, (80, 80, 120, 120), None, frame_h=200) is True


def test_http_url_rejects_file_scheme():
    from gcode_monitor.net import http_url
    import pytest

    with pytest.raises(ValueError):
        http_url("file:///etc/passwd")
    assert http_url("http://octopi.local") == "http://octopi.local"


def test_prusa_layer_change_on_real_file():
    path = Path(__file__).resolve().parents[1] / "jobs" / "20260821-215410_Nose_Pad.gcode" / "print.gcode"
    if not path.exists():
        return
    parsed = parse_gcode(path, keep_source=False)
    assert parsed.flavour == "prusa-or-cura-layers"
    assert parsed.layer_count == 15
    assert parsed.layers[0].n_extrude > 0
    assert parsed.layers[0].points

