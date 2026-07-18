"""Command-line entrypoint: `deep-researcher "topic" --filters "..."`."""

from __future__ import annotations

import argparse
import asyncio
import time

from .cache import load_run
from .pipeline import arun
from .state import ResearchState


def _print_event(seen: set):
    async def cb(state: ResearchState):
        for entry in state.get("status_log", []):
            key = (entry.get("ts"), entry.get("message"))
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{entry.get('agent', '?'):<18}] {entry.get('message', '')}")

    return cb


def _approval_cb():
    async def cb(bundles):
        if not bundles:
            return None
        print("\n--- Review triaged articles (human-in-the-loop) ---")
        for i, b in enumerate(bundles):
            print(f"  [{i}] {b.source_type:<13} {b.source_name:<20} {b.title[:70]}")
        raw = await asyncio.to_thread(
            input,
            "Keep which? comma-separated indices, blank = keep all, 'q' = drop all: ",
        )
        raw = (raw or "").strip()
        if raw.lower() == "q":
            return []
        if not raw:
            return None
        try:
            keep = {int(x) for x in raw.split(",") if x.strip() != ""}
            return [b for i, b in enumerate(bundles) if i in keep]
        except ValueError:
            print("  (unparseable input; keeping all)")
            return None

    return cb


def _replay(topic: str, filters: str, speed: float) -> int:
    payload = load_run(topic, filters)
    if not payload:
        print("No cached run found for that topic/filters.")
        return 1
    print(f"Replaying cached run for: {topic}\n")
    for entry in payload.get("status_log", []):
        print(f"  [{entry.get('agent', '?'):<18}] {entry.get('message', '')}")
        time.sleep(max(0.0, speed))
    print("\n--- Report (markdown) ---\n")
    print(payload.get("report_md", "(no report)"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent Deep Researcher")
    parser.add_argument("topic", nargs="?", default=None, help="Research topic")
    parser.add_argument("--filters", default="", help="Filter string (dates, source types, geo)")
    parser.add_argument("--email", default=None, help="Recipient email for the report")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cached run")
    parser.add_argument("--replay", action="store_true", help="Stream a cached run")
    parser.add_argument("--speed", type=float, default=0.4, help="Replay delay per event (s)")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Human-in-the-loop: review/trim triaged articles before the debate",
    )
    parser.add_argument("--doctor", action="store_true", help="Run a preflight check and exit")
    args = parser.parse_args()

    if args.doctor:
        from .preflight import print_report

        return print_report()

    if not args.topic:
        parser.error("topic is required (unless using --doctor)")

    if args.replay:
        return _replay(args.topic, args.filters, args.speed)

    print(f"Researching: {args.topic}\nFilters: {args.filters or 'none'}\n")
    seen: set = set()
    state = asyncio.run(
        arun(
            args.topic,
            args.filters,
            recipient_email=args.email,
            use_cache=not args.no_cache,
            on_event=_print_event(seen),
            on_approve=_approval_cb() if args.approve else None,
        )
    )

    delivery = state.get("delivery_status", {})
    print("\n--- Done ---")
    print(f"Findings: {len(state.get('judged').findings) if state.get('judged') else 0}")
    print(f"Delivery: {delivery}")

    from .telemetry import summarize_metrics

    stats = summarize_metrics(state.get("metrics", []))
    if stats["total_seconds"]:
        print(f"Wall-clock (sum of node time): {stats['total_seconds']}s")
        for node, agg in sorted(
            stats["per_node"].items(), key=lambda kv: kv[1]["total_seconds"], reverse=True
        ):
            print(f"  {node:<16} {agg['total_seconds']:>7.2f}s  ({agg['calls']} calls)")
    if state.get("errors"):
        print(f"Non-fatal errors: {len(state['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
