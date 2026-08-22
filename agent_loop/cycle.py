"""perceive → retrieve → reason → verify → act.

Verify and act are code. Reason may call Groq; it cannot open the tool layer.
"""

from __future__ import annotations

from actions import is_critical, pretty
from agent_loop.memory import DecisionMemory
from gcode_monitor.pipeline import FrameVerdict, llm_context, should_ask_llm
from tool_api.service import read_state, verify_proposal
from tool_api.types import Observation, Proposal, Verdict


def perceive(verdict: FrameVerdict, *, hotend=None, bed=None, camera=False) -> Observation:
    return read_state(
        layer=verdict.layer_index,
        layer_z=verdict.layer_z,
        file_pos=None,
        progress=verdict.progress_pct / 100.0,
        gcode_line=verdict.snippet,
        hotend=hotend,
        bed=bed,
        camera_present=camera,
    )


def retrieve(memory: DecisionMemory, class_name: str) -> list[dict]:
    return memory.like(class_name)


def propose_from_detector(class_name: str) -> Proposal:
    action = pretty(class_name)
    if action["severity"] == "critical":
        return Proposal(command="pause", reason=action["what_to_do"], source="detector")
    if class_name == "good_print":
        return Proposal(command="none", reason=action["what_to_do"], source="detector")
    return Proposal(command="none", reason=action["what_to_do"] + " Logged only until it holds.", source="detector")


def reason_text(verdict: FrameVerdict, past: list[dict], ask_llm: bool) -> str:
    if not ask_llm:
        return pretty(verdict.name)["what_to_do"]
    from agentic_workflow import run_agentic_reasoning

    past_txt = "\n".join(
        f"- prior {item.get('class')} → {item.get('verdict')} ({item.get('reason', '')})"
        for item in past
    ) or "- no prior cases of this class"
    extra = llm_context(verdict) + "\n\nStore B (same class, last cases):\n" + past_txt
    note = run_agentic_reasoning(
        [{"class": d["class"], "confidence": d["confidence"]} for d in verdict.kept],
        verdict.name,
        verdict.conf,
        None,
        extra_context=extra,
    )
    return str(note.get("reasoning") or pretty(verdict.name)["what_to_do"])


def cycle(
    verdict: FrameVerdict,
    memory: DecisionMemory,
    *,
    deep: bool,
    camera: bool = True,
    hotend=None,
    bed=None,
) -> dict:
    obs = perceive(verdict, hotend=hotend, bed=bed, camera=camera)
    past = retrieve(memory, verdict.name)
    proposal = propose_from_detector(verdict.name)
    ask = deep and should_ask_llm(verdict.name, verdict.sticky_ok)
    text = reason_text(verdict, past, ask)
    gate = verify_proposal(
        obs,
        proposal,
        sticky_ok=verdict.sticky_ok,
        class_name=verdict.name,
        critical=is_critical(verdict.name),
    )
    record = {
        "layer": verdict.layer_index,
        "class": verdict.name,
        "conf": verdict.conf,
        "sticky": verdict.sticky_ok,
        "dropped": verdict.dropped,
        "proposal": proposal.command,
        "verdict": gate.label,
        "accept": gate.accept,
        "reasons": gate.reasons,
        "reason": text,
        "past_cases": len(past),
        "sensors_present": {"camera": camera, "hotend": hotend is not None, "bed": bed is not None, "imu": False, "mic": False},
    }
    if deep:
        memory.append(record)
    return {"observation": obs, "proposal": proposal, "gate": gate, "record": record, "past": past}
