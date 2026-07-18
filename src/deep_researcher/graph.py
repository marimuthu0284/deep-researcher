"""LangGraph assembly.

    START -> query_planner -> retriever -> triage
             triage --(Send per article)--> debate_article  (parallel map)
             debate_article -> judge  (reduce; runs once after all branches)
             judge -> report_builder -> dispatcher -> END

The per-article fan-out uses the Send API; results accumulate through the
additive reducer on `analyses`. A bounded re-retrieval lives inside
debate_article, so the graph itself stays acyclic and deterministic.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents.debate import debate_article
from .agents.dispatcher import dispatcher
from .agents.judge import judge
from .agents.query_planner import query_planner
from .agents.report_builder import report_builder
from .agents.retriever import retriever
from .agents.triage import triage
from .state import ResearchState
from .telemetry import timed


async def _gate(state: ResearchState) -> dict:
    """Passthrough node that is the human-in-the-loop interrupt point.

    Interrupting *before* this node (after triage) lets a human review/edit
    `bundles`; the fan-out below reads the possibly-edited list on resume, so
    approval actually changes which articles get debated.
    """
    from .agents.common import event

    bundles = state.get("bundles", [])
    return {"status_log": [event("Gate", f"proceeding with {len(bundles)} articles")]}


def _fan_out(state: ResearchState):
    """Emit one Send per approved article; skip straight to judge if none."""
    bundles = state.get("bundles", [])
    if not bundles:
        return "judge"
    return [Send("debate_article", {"article": b}) for b in bundles]


def build_graph(checkpointer=None, interrupt_before: list[str] | None = None):
    builder = StateGraph(ResearchState)

    builder.add_node("query_planner", timed("query_planner")(query_planner))
    builder.add_node("retriever", timed("retriever")(retriever))
    builder.add_node("triage", timed("triage")(triage))
    builder.add_node("gate", _gate)
    builder.add_node("debate_article", timed("debate_article")(debate_article))
    builder.add_node("judge", timed("judge")(judge))
    builder.add_node("report_builder", timed("report_builder")(report_builder))
    builder.add_node("dispatcher", timed("dispatcher")(dispatcher))

    builder.add_edge(START, "query_planner")
    builder.add_edge("query_planner", "retriever")
    builder.add_edge("retriever", "triage")
    builder.add_edge("triage", "gate")
    builder.add_conditional_edges("gate", _fan_out, ["debate_article", "judge"])
    builder.add_edge("debate_article", "judge")
    builder.add_edge("judge", "report_builder")
    builder.add_edge("report_builder", "dispatcher")
    builder.add_edge("dispatcher", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
