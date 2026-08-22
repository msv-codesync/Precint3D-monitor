import os
from typing import Dict, List, TypedDict

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - fallback for environments without compatible packages
    HumanMessage = SystemMessage = None  # type: ignore[assignment]
    ChatGroq = None  # type: ignore[assignment]
    END = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]

class AgentState(TypedDict, total=False):
    detections: List[Dict[str, object]]
    top_detection: str
    top_confidence: float
    reasoning: str
    verification: str
    extra_context: str
    final_action: str
    status: str


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        secret = st.secrets.get("GROQ_API_KEY", "")
        return str(secret or "").strip()
    except Exception:
        return ""


def build_reasoning_prompt(
    detections: List[Dict[str, object]],
    top_detection: str,
    top_confidence: float,
    extra_context: str = "",
) -> str:
    detection_text = "\n".join(
        f"- {item['class']}: {item['confidence']:.2f}" for item in detections
    ) or "- no detections"
    extra = f"\n\nPrint context:\n{extra_context}" if extra_context else ""
    return f"""
You help a 3D printer operator. Keep it short and plain.

Current detections:
{detection_text}

Top detection: {top_detection} ({top_confidence:.2f})
{extra}
Task:
1. Explain the most likely root cause.
2. Recommend one practical corrective action.
3. Do not invent a different defect than the top detection.
4. Keep your answer concise.
""".strip()


def build_verification_prompt(
    detections: List[Dict[str, object]],
    reasoning: str,
    extra_context: str = "",
) -> str:
    detection_text = "\n".join(
        f"- {item['class']}: {item['confidence']:.2f}" for item in detections
    ) or "- no detections"
    extra = f"\n\nPrint context:\n{extra_context}" if extra_context else ""
    return f"""
You are the verification agent.
Review the reasoning below. You cannot send printer commands.

Detections:
{detection_text}

Reasoning:
{reasoning}
{extra}
Return a short verdict with a safety label: SAFE, CAUTION, or BLOCK.
If the finding is not held across frames, label BLOCK for acting on the printer.
If the top detection is good_print, do not invent a defect.
""".strip()


def format_final_action(reasoning: str, verification: str) -> str:
    return f"Reasoning:\n{reasoning}\n\nVerification:\n{verification}"


def build_agentic_graph(api_key: str | None = None):
    if ChatGroq is None or StateGraph is None or HumanMessage is None or SystemMessage is None:
        def fallback_graph(state: AgentState) -> AgentState:
            detections = state.get("detections", []) or []
            top_detection = state.get("top_detection") or "good_print"
            top_confidence = float(state.get("top_confidence") or 0.0)
            reasoning = (
                f"Detected {top_detection.replace('_', ' ')} with {top_confidence:.2f} confidence. "
                f"Review the print and confirm the condition before continuing."
            )
            return {
                "reasoning": reasoning,
                "verification": "CAUTION: fallback reasoning used because the agent dependencies are unavailable.",
                "final_action": f"Reasoning:\n{reasoning}\n\nVerification:\nCAUTION: fallback reasoning used because the agent dependencies are unavailable.",
                "status": "completed",
            }

        def fallback_compile() -> callable:
            return fallback_graph

        return fallback_compile()

    key = api_key or get_groq_api_key()
    if not key:
        def missing_key_graph(state: AgentState) -> AgentState:
            top_detection = state.get("top_detection") or "good_print"
            top_confidence = float(state.get("top_confidence") or 0.0)
            reasoning = (
                f"Saw {top_detection.replace('_', ' ')} at {top_confidence:.0%} confidence. "
                "Set GROQ_API_KEY if you want a written note."
            )
            return {
                "reasoning": reasoning,
                "verification": "No language-model check. The detector still chose the action.",
                "final_action": reasoning,
                "status": "completed",
            }

        return missing_key_graph

    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=key, temperature=0.2)

    def perceive(state: AgentState) -> AgentState:
        detections = state.get("detections", []) or []
        top_detection = state.get("top_detection") or "good_print"
        top_confidence = float(state.get("top_confidence") or 0.0)
        if not detections:
            summary = "No defect candidates were detected above threshold."
        else:
            summary = ", ".join(f"{item['class']} ({item['confidence']:.2f})" for item in detections)
        return {
            "detections": detections,
            "top_detection": top_detection,
            "top_confidence": top_confidence,
            "status": f"Perceived: {summary}",
        }

    def reason(state: AgentState) -> AgentState:
        prompt = build_reasoning_prompt(
            state.get("detections", []),
            state.get("top_detection", "good_print"),
            float(state.get("top_confidence") or 0.0),
            extra_context=str(state.get("extra_context") or ""),
        )
        response = llm.invoke([
            SystemMessage(content="You help a 3D printer operator. Be short and plain."),
            HumanMessage(content=prompt),
        ])
        text = getattr(response, "content", str(response))
        return {"reasoning": str(text)}

    def verify(state: AgentState) -> AgentState:
        prompt = build_verification_prompt(
            state.get("detections", []),
            state.get("reasoning", ""),
            extra_context=str(state.get("extra_context") or ""),
        )
        response = llm.invoke([
            SystemMessage(content="You are the verification agent. Decide whether the plan is safe and practical."),
            HumanMessage(content=prompt),
        ])
        text = getattr(response, "content", str(response))
        return {"verification": str(text)}

    def finalize(state: AgentState) -> AgentState:
        final_action = format_final_action(state.get("reasoning", ""), state.get("verification", ""))
        return {"final_action": final_action, "status": "completed"}

    workflow = StateGraph(AgentState)
    workflow.add_node("perceive", perceive)
    workflow.add_node("reason", reason)
    workflow.add_node("verify", verify)
    workflow.add_node("finalize", finalize)
    workflow.set_entry_point("perceive")
    workflow.add_edge("perceive", "reason")
    workflow.add_edge("reason", "verify")
    workflow.add_edge("verify", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()


def run_agentic_reasoning(
    detections: List[Dict[str, object]],
    top_detection: str,
    top_confidence: float,
    api_key: str | None = None,
    extra_context: str = "",
) -> Dict[str, object]:
    graph = build_agentic_graph(api_key)
    state: AgentState = {
        "detections": detections,
        "top_detection": top_detection,
        "top_confidence": float(top_confidence),
        "extra_context": extra_context or "",
    }
    if hasattr(graph, "invoke"):
        result = graph.invoke(state)
    else:
        result = graph(state)
    return result
