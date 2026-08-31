from pathlib import Path

from closed_loop.outcomes import queue_action, resolve_due
from closed_loop.policy import planned_nudges
from closed_loop.store_a import retrieve
from closed_loop.store_b import OutcomeMemory


def test_store_a_mentions_pause_for_spaghetti():
    text = retrieve("spaghetti_failure")
    assert "Pause" in text
    assert "180" in retrieve("good_print")


def test_planned_nudges_clip_and_skip_missing_temps():
    assert planned_nudges("spaghetti_failure", hotend=210, bed=60) == []
    nudges = planned_nudges("warping", hotend=210, bed=60)
    names = [n for n, _ in nudges]
    assert "feedrate" in names
    assert "bed" in names
    assert planned_nudges("stringing", hotend=None, bed=60) == []
    hot = dict(planned_nudges("stringing", hotend=210, bed=60))
    assert 180 <= hot["hotend"] <= 260


def test_outcome_memory_ranks_wins_first(tmp_path):
    path = tmp_path / "memory.jsonl"
    mem = OutcomeMemory(path)
    mem.append({"class": "warping", "outcome_quality": 0.0, "verdict": "loss"})
    mem.append({"class": "warping", "outcome_quality": 1.0, "verdict": "win"})
    mem.append({"class": "warping", "outcome_quality": 0.5, "verdict": "mid"})
    hits = mem.like("warping", limit=3)
    assert hits[0]["verdict"] == "win"
    assert hits[1]["verdict"] == "loss"


def test_resolve_due_after_five_layers(tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    queue_action(folder, class_name="warping", layer=1, frames=2, kind="nudge")
    still = resolve_due(folder, layer=3, frames=4, current_class="warping")
    assert still == []
    due = resolve_due(folder, layer=7, frames=4, current_class="good_print")
    assert due[0]["outcome_quality"] == 1.0


def test_llm_context_includes_store_a():
    from gcode_monitor.pipeline import FrameVerdict, llm_context
    from PIL import Image

    img = Image.new("RGB", (8, 8), (0, 0, 0))
    verdict = FrameVerdict(
        layer_index=2,
        layer_z=0.4,
        progress_pct=10.0,
        progress_note="test",
        name="warping",
        conf=0.8,
        sticky_ok=True,
        kept=[],
        dropped=0,
        path_image=img,
        masked_image=img,
        marked_image=img,
        snippet="G1 X1",
    )
    text = llm_context(verdict)
    assert "Store A:" in text
    assert "You do not send G-code" in text


def test_clear_live_state_removes_persisted_watch_files(tmp_path, monkeypatch):
    from closed_loop import live

    live_dir = tmp_path / "_live"
    monkeypatch.setattr(live, "LIVE", live_dir)
    monkeypatch.setattr(live, "RUN", live_dir / "run.json")
    monkeypatch.setattr(live, "PID", live_dir / "worker.pid")
    monkeypatch.setattr(live, "STOP", live_dir / "stop")
    monkeypatch.setattr(live, "STATUS", live_dir / "status.json")

    live.write_run({"folder": str(tmp_path), "cam_idx": 0})
    live.write_pid(12345)
    live.request_stop()
    live.write_status({"worker": "running"})

    live.clear_live_state()

    assert not live.RUN.exists()
    assert not live.PID.exists()
    assert not live.STOP.exists()
    assert not live.STATUS.exists()
