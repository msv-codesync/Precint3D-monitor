from tool_api.bounds import clip_parameter
from tool_api.service import read_state, verify_proposal
from tool_api.types import Proposal


def test_hotend_clips_and_is_refused_when_out_of_range():
    value, changed = clip_parameter("hotend", 400)
    assert changed is True
    assert value == 260.0
    value, changed = clip_parameter("hotend", 210)
    assert changed is False


def test_verifier_blocks_pause_until_held():
    obs = read_state(layer=3, layer_z=0.6, file_pos=100, progress=0.2, gcode_line="G1 X10")
    proposal = Proposal(command="pause")
    blocked = verify_proposal(obs, proposal, sticky_ok=False, class_name="spaghetti_failure", critical=True)
    assert blocked.accept is False
    assert blocked.label == "BLOCK"
    allowed = verify_proposal(obs, proposal, sticky_ok=True, class_name="spaghetti_failure", critical=True)
    assert allowed.accept is True
    assert allowed.command == "pause"


def test_verifier_blocks_pause_on_good_print():
    obs = read_state(layer=3, layer_z=0.6, file_pos=100, progress=0.2, gcode_line="G1 X10")
    gate = verify_proposal(
        obs,
        Proposal(command="pause"),
        sticky_ok=True,
        class_name="good_print",
        critical=False,
    )
    assert gate.accept is False


def test_unknown_command_blocked():
    obs = read_state(layer=0, layer_z=0.2, file_pos=0, progress=0, gcode_line="")
    gate = verify_proposal(
        obs,
        Proposal(command="M104 S999"),
        sticky_ok=True,
        class_name="warping",
        critical=False,
    )
    assert gate.accept is False
    assert gate.command == "none"


def test_set_parameter_out_of_window_is_clip_not_write():
    obs = read_state(layer=0, layer_z=0.2, file_pos=0, progress=0, gcode_line="")
    gate = verify_proposal(
        obs,
        Proposal(command="set_parameter", param="hotend", value=400),
        sticky_ok=True,
        class_name="under_extrusion",
        critical=False,
    )
    assert gate.accept is False
    assert gate.label == "CLIP"


def test_generate_report_needs_a_job_folder(tmp_path):
    from tool_api.service import generate_report

    empty = tmp_path / "not_a_job"
    empty.mkdir()
    try:
        generate_report(empty)
        assert False, "should refuse"
    except ValueError:
        pass
    job = tmp_path / "session"
    job.mkdir()
    (job / "events.jsonl").write_text("", encoding="utf-8")
    out = generate_report(job)
    assert out.name == "report.md"
    assert out.exists()
