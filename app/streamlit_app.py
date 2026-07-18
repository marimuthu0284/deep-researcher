"""Streamlit demo UI: a live agent-status board where the wait *is* the demo.

Shows the typed state object mutating as each agent completes, then the
rejected-claims list, then the final report. A replay toggle streams a cached
run at realistic speed so the demo survives a dead wifi connection.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow running via `streamlit run app/streamlit_app.py` without installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_researcher.cache import hydrate_state, load_run  # noqa: E402
from deep_researcher.config import reload_settings  # noqa: E402
from deep_researcher.eval import evaluate_state  # noqa: E402
from deep_researcher.pipeline import arun  # noqa: E402
from deep_researcher.telemetry import summarize_metrics  # noqa: E402

st.set_page_config(page_title="Deep Researcher", page_icon="[R]", layout="wide")

AGENT_ORDER = [
    "Query Planner",
    "Retriever",
    "Triage",
    "Debate",
    "Critical Analysis",
    "Judge",
    "Report Builder",
    "Dispatcher",
]


def _render_board(board, state) -> None:
    log = state.get("status_log", [])
    done_agents = {e.get("agent") for e in log}
    cols = board.columns(len(AGENT_ORDER))
    for col, agent in zip(cols, AGENT_ORDER):
        active = agent in done_agents
        icon = "[x]" if active else "[ ]"
        col.markdown(f"**{icon} {agent}**")
    board.markdown("---")
    for e in log[-14:]:
        board.markdown(f"`{e.get('agent', '?')}` — {e.get('message', '')}")


def _render_charts(container, state) -> None:
    judged = state.get("judged")
    if not judged:
        return
    verdicts = getattr(judged, "verdicts", [])
    findings = getattr(judged, "findings", [])
    if verdicts:
        container.subheader("Per-article confidence")
        df = pd.DataFrame(
            {"confidence": [v.confidence_score for v in verdicts]},
            index=[v.article_id for v in verdicts],
        )
        container.bar_chart(df)
    if findings:
        container.subheader("Finding confidence")
        df = pd.DataFrame(
            {"confidence": [f.confidence for f in findings]},
            index=[f.statement[:40] + "..." for f in findings],
        )
        container.bar_chart(df)


def _render_telemetry(container, state) -> None:
    stats = summarize_metrics(state.get("metrics", []))
    if not stats["total_seconds"]:
        return
    with container.expander(f"Run telemetry — {stats['total_seconds']}s total", expanded=False):
        rows = [
            {"node": node, "calls": agg["calls"], "seconds": agg["total_seconds"]}
            for node, agg in sorted(
                stats["per_node"].items(),
                key=lambda kv: kv[1]["total_seconds"],
                reverse=True,
            )
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_state_inspector(container, state) -> None:
    plan = state.get("search_plan")
    snapshot = {
        "topic": state.get("topic"),
        "filters": state.get("filters"),
        "queries": [
            {"q": q.query_str, "source_type": q.source_type, "facet": q.facet}
            for q in (plan.queries if plan else [])
        ],
        "n_bundles": len(state.get("bundles", [])),
        "n_analyses": len(state.get("analyses", [])),
        "delivery_status": state.get("delivery_status", {}),
        "errors": state.get("errors", []),
    }
    with container.expander("State inspector (typed pipeline state)", expanded=False):
        st.json(snapshot)
        try:
            st.subheader("Evaluation metrics")
            st.json(evaluate_state(state))
        except Exception as exc:  # noqa: BLE001
            st.caption(f"eval unavailable: {exc}")


def _render_report(container, state) -> None:
    judged = state.get("judged")
    if judged and getattr(judged, "findings", None):
        container.subheader("Findings")
        for f in judged.findings:
            container.markdown(
                f"- **[{f.band} · {f.confidence:.0f}]** {f.statement}"
            )
    _render_charts(container, state)
    rejected = []
    for a in state.get("analyses", []):
        rejected += getattr(a.critique, "uncited_claims_rejected", [])
    if rejected:
        container.subheader("Rejected claims (failed citation audit)")
        container.write(rejected)
    _render_telemetry(container, state)
    _render_state_inspector(container, state)
    html = state.get("report_html")
    if html:
        container.subheader("Report")
        st.components.v1.html(html, height=700, scrolling=True)
        container.download_button(
            "Download report.html", html, file_name="report.html", mime="text/html"
        )


def main() -> None:
    reload_settings()
    st.title("Multi-Agent Deep Researcher")
    st.caption(
        "Two agents argue over every source, a critic audits their citations, "
        "and a judge converts the surviving evidence into inspectable confidence scores."
    )

    with st.sidebar:
        st.header("Run")
        topic = st.text_input("Topic", value="Ozempic")
        filters = st.text_input(
            "Filters", value="peer-reviewed papers and major news, last 30 days"
        )
        email = st.text_input("Email report to (optional)", value="")
        use_cache = st.checkbox("Use cached run if available", value=True)
        replay = st.checkbox("Replay cached run (offline demo)", value=False)
        go = st.button("Run research", type="primary")

    board = st.container()
    report_area = st.container()

    if not go:
        st.info("Enter a topic and press Run research.")
        return

    if replay:
        payload = load_run(topic, filters)
        if not payload:
            st.error("No cached run found for that topic/filters.")
            return
        state = hydrate_state(payload)
        shown: list = []
        for e in payload.get("status_log", []):
            shown.append(e)
            _render_board(board, {"status_log": shown})
            time.sleep(0.35)
        _render_report(report_area, state)
        return

    async def on_event(state) -> None:
        _render_board(board, state)

    with st.spinner("Agents collaborating..."):
        state = asyncio.run(
            arun(
                topic,
                filters,
                recipient_email=email or None,
                use_cache=use_cache,
                on_event=on_event,
            )
        )
    _render_board(board, state)
    _render_report(report_area, state)
    st.success(f"Delivery: {state.get('delivery_status', {})}")


if __name__ == "__main__":
    main()
