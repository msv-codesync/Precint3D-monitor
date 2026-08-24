from unittest.mock import patch

import pytest

from printer_ops import apply_parameter, ping_octoprint, request_pause
from tool_api.service import read_state


def _obs():
    return read_state(layer=1, layer_z=0.2, file_pos=10, progress=0.1, gcode_line="G1 X10")


def test_nudge_blocked_without_sticky():
    with patch("printer_ops.send_octoprint_command") as send:
        result = apply_parameter(
            url="http://octopi.local",
            api_key="k",
            param="bed",
            value=60,
            confirm=True,
            observation=_obs(),
            sticky_ok=False,
            class_name="warping",
        )
    assert result.ok is False
    assert result.gate_label == "BLOCK"
    send.assert_not_called()


def test_nudge_sends_when_verifier_allows():
    with patch("printer_ops.send_octoprint_command", return_value=(200, "ok")) as send:
        result = apply_parameter(
            url="http://octopi.local",
            api_key="k",
            param="bed",
            value=60,
            confirm=True,
            observation=_obs(),
            sticky_ok=True,
            class_name="warping",
        )
    assert result.ok is True
    send.assert_called_once()


def test_pause_blocked_on_good_print():
    with patch("printer_ops.pause_octoprint") as pause:
        result = request_pause(
            link="OctoPrint",
            url="http://octopi.local",
            api_key="k",
            port="",
            baud=250000,
            observation=_obs(),
            sticky_ok=True,
            class_name="good_print",
        )
    assert result.ok is False
    pause.assert_not_called()


def test_pause_sends_after_held_critical():
    with patch("printer_ops.pause_octoprint", return_value="octoprint pause HTTP 204") as pause:
        result = request_pause(
            link="OctoPrint",
            url="http://octopi.local",
            api_key="k",
            port="",
            baud=250000,
            observation=_obs(),
            sticky_ok=True,
            class_name="spaghetti_failure",
        )
    assert result.ok is True
    pause.assert_called_once()


def test_pause_octoprint_rejects_http_error():
    from gcode_monitor.act import pause_octoprint

    class _Resp:
        status_code = 401
        text = "unauthorized"

    with patch("requests.post", return_value=_Resp()):
        with pytest.raises(RuntimeError, match="401"):
            pause_octoprint("http://octopi.local", "key")


def test_pause_octoprint_requires_api_key():
    from gcode_monitor.act import pause_octoprint

    with pytest.raises(ValueError, match="API key"):
        pause_octoprint("http://octopi.local", "  ")


def test_ping_octoprint_unreachable():
    import requests

    with patch("printer_ops.requests.get", side_effect=requests.ConnectionError("down")):
        ok, label = ping_octoprint("http://octopi.local", "k")
    assert ok is False
    assert label == "UNREACHABLE"
