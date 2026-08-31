import sys
from pathlib import Path

import pytest
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


def test_avfoundation_list_order():
    from gcode_monitor.capture import parse_avfoundation_list

    text = """
[AVFoundation indev @ 0x0] AVFoundation video devices:
[AVFoundation indev @ 0x0] [0] FaceTime HD Camera
[AVFoundation indev @ 0x0] [1] Continuity Camera
[AVFoundation indev @ 0x0] AVFoundation audio devices:
[AVFoundation indev @ 0x0] [0] MacBook Pro Microphone
"""
    names = parse_avfoundation_list(text)
    assert names[0] == (0, "FaceTime HD Camera")
    assert names[1] == (1, "Continuity Camera")
    from gcode_monitor.capture import is_iphone_camera

    assert is_iphone_camera("Continuity Camera") is True
    assert is_iphone_camera("FaceTime HD Camera") is False


def test_opencv_capture_index_swaps_facetime_and_iphone():
    from gcode_monitor.capture import opencv_capture_index

    names = ["FaceTime HD Camera", "iPhone (2) Camera"]
    assert opencv_capture_index(0, names, macos=True) == 1
    assert opencv_capture_index(1, names, macos=True) == 0
    assert opencv_capture_index(0, names, macos=False) == 0
    aligned = ["iPhone (2) Camera", "FaceTime HD Camera"]
    assert opencv_capture_index(0, aligned, macos=True) == 0
    assert opencv_capture_index(1, aligned, macos=True) == 1


def test_default_camera_never_picks_a_paired_phone():
    from gcode_monitor.capture import default_camera_index, list_cameras

    cams = [
        {"index": 0, "name": "FaceTime HD Camera", "preferred": True, "phone": False},
        {"index": 1, "name": "iPhone (2) Camera", "preferred": False, "phone": True},
    ]
    assert default_camera_index(cams) == 0
    assert default_camera_index([]) == 0
    for cam in list_cameras(probe=False):
        assert "phone" in cam


def test_normalize_camera_url_completes_a_bare_ip():
    from gcode_monitor.capture import normalize_camera_url

    assert normalize_camera_url("192.168.1.20") == "http://192.168.1.20/capture"
    assert normalize_camera_url("192.168.1.42:8080/shot.jpg") == "http://192.168.1.42:8080/shot.jpg"
    assert normalize_camera_url("https://cam.local/capture") == "https://cam.local/capture"
    with pytest.raises(ValueError):
        normalize_camera_url("   ")
    with pytest.raises(ValueError):
        normalize_camera_url("file:///etc/passwd")


def test_network_source_covers_legacy_session_names():
    from gcode_monitor.capture import ESP32_SOURCE, LOCAL_SOURCE, PHONE_SOURCE, is_network_source

    assert is_network_source(LOCAL_SOURCE) is False
    assert is_network_source("Local camera") is False
    assert is_network_source(None) is False
    assert is_network_source(PHONE_SOURCE) is True
    assert is_network_source(ESP32_SOURCE) is True
    assert is_network_source("ESP32") is True


def test_from_network_camera_reads_an_mjpeg_stream(monkeypatch):
    import io as _io

    from PIL import Image as _Image

    from gcode_monitor import capture

    buf = _io.BytesIO()
    _Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="JPEG")
    jpeg = buf.getvalue()
    body = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    class FakeResponse:
        headers = {"Content-Type": "multipart/x-mixed-replace; boundary=frame"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=4096):
            for i in range(0, len(body), chunk_size):
                yield body[i:i + chunk_size]

        def close(self):
            return None

    fake_requests = type("R", (), {"get": staticmethod(lambda *a, **k: FakeResponse())})
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    image = capture.from_network_camera("http://192.168.1.42:8080/video")
    assert image.size == (8, 8)


def test_esp32_sketch_embeds_the_operator_network():
    from gcode_monitor.esp32_sketch import BOARDS, sketch_for

    code = sketch_for("PrinterLab", "hunter2", "AI-Thinker ESP32-CAM")
    assert '"PrinterLab"' in code
    assert '"hunter2"' in code
    assert BOARDS["AI-Thinker ESP32-CAM"] in code
    assert "/capture" in code
    blank = sketch_for("", "")
    assert "YOUR_WIFI_NAME" in blank


def test_preferred_camera_choice_skips_facetime():
    from gcode_monitor.capture import preferred_camera_choice

    cams = [
        {"index": 0, "name": "FaceTime HD Camera", "preferred": False},
        {"index": 1, "name": "Continuity Camera", "preferred": True},
    ]
    assert preferred_camera_choice(cams) == (1, "Continuity Camera")
    assert preferred_camera_choice(
        [{"index": 0, "name": "FaceTime HD Camera", "preferred": False}]
    ) == (0, "FaceTime HD Camera")


def test_process_webcam_close_kills_child(monkeypatch):
    from gcode_monitor.capture import ProcessWebcam

    monkeypatch.setattr("gcode_monitor.capture.time.sleep", lambda *_a, **_k: None)

    class FakeProc:
        def __init__(self):
            self._alive = True
            self.terminate_called = 0

        def is_alive(self):
            return self._alive

        def terminate(self):
            self.terminate_called += 1
            self._alive = False

        def kill(self):
            self._alive = False

        def join(self, timeout=None):
            return

    class FakeQ:
        def close(self):
            return

        def cancel_join_thread(self):
            return

    cam = ProcessWebcam.__new__(ProcessWebcam)
    cam.index = 0
    cam._closed = False
    cam._finalizer = None
    cam._stop = type("E", (), {"set": lambda self: None})()
    proc = FakeProc()
    cam._proc = proc
    cam._q = FakeQ()
    cam.close()
    cam.close()
    assert cam._proc is None
    assert proc.terminate_called == 1


def test_http_url_rejects_file_scheme():
    from gcode_monitor.net import http_url
    import pytest

    with pytest.raises(ValueError):
        http_url("file:///etc/passwd")
    assert http_url("http://octopi.local") == "http://octopi.local"
    assert http_url("http://192.168.1.20/capture") == "http://192.168.1.20/capture"


def test_http_url_rejects_cloud_metadata():
    from gcode_monitor.net import http_url
    import pytest

    with pytest.raises(ValueError, match="not a printer"):
        http_url("http://169.254.169.254/latest/meta-data")
    with pytest.raises(ValueError, match="not a printer"):
        http_url("http://metadata.google.internal/")


def test_resolve_session_dir_stays_under_jobs(tmp_path):
    from gcode_monitor.store import resolve_session_dir
    import pytest

    jobs = tmp_path / "jobs"
    session = jobs / "20260824-120000_print"
    session.mkdir(parents=True)
    assert resolve_session_dir(jobs, session) == session.resolve()
    with pytest.raises(ValueError):
        resolve_session_dir(jobs, tmp_path / "outside")
    hidden = jobs / "_inspect"
    hidden.mkdir()
    with pytest.raises(ValueError):
        resolve_session_dir(jobs, hidden)


def test_prusa_layer_change_on_real_file():
    path = Path(__file__).resolve().parents[1] / "jobs" / "20260821-215410_Nose_Pad.gcode" / "print.gcode"
    if not path.exists():
        return
    parsed = parse_gcode(path, keep_source=False)
    assert parsed.flavour == "prusa-or-cura-layers"
    assert parsed.layer_count == 15
    assert parsed.layers[0].n_extrude > 0
    assert parsed.layers[0].points


def test_judge_image_without_gcode():
    from gcode_monitor.pipeline import judge_image

    image = Image.new("RGB", (200, 200), (40, 40, 40))
    names = {0: "spaghetti_failure"}
    results = [_Result(names, [_Box(0, 0.8, (40, 40, 80, 80))])]
    sticky = StickyClass(need=1)
    verdict = judge_image(image, results, sticky, update_sticky=True)
    assert verdict.name == "spaghetti_failure"
    assert verdict.sticky_ok is True
    assert verdict.marked_image.size == (200, 200)

