"""Live eval suite: proves the pipeline answers different questions differently.

Runs a small fixed set of distinct topics through the real pipeline (live
OpenRouter + Tavily calls, no cache) and checks, per run:

  1. no unhandled errors in state["errors"]
  2. synthesis is NOT degraded (LLM calls actually succeeded)
  3. calibration_checks all pass (eval.py structural invariants)
  4. the report has non-trivial content (findings + executive summary)

...and across all runs:

  5. no two topics produced the same report (the original bug this guards
     against: every question returning identical boilerplate output)

Usage:
    python scripts/eval_suite.py            # all 5 built-in topics
    python scripts/eval_suite.py --top-n 3   # cap articles/topic for speed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TOPICS: list[tuple[str, str]] = [
    ("Ozempic weight loss", "peer-reviewed papers and major news, last 90 days"),
    ("Electric vehicle battery recycling", "major news and industry reports, last 6 months"),
    ("Quantum computing error correction", "peer-reviewed papers, last 12 months"),
    ("Remote work productivity", "major news and industry reports, last 12 months"),
    ("Coral reef bleaching", "peer-reviewed papers and major news, last 12 months"),
]


@dataclass
class RunResult:
    topic: str
    ok: bool = False
    degraded: bool = True
    calibration_passed: bool = False
    errors: list[str] = field(default_factory=list)
    report_md: str = ""
    n_findings: int = 0
    seconds: float = 0.0
    detail: str = ""


async def _run_one(topic: str, filters: str) -> RunResult:
    from deep_researcher.eval import evaluate_state
    from deep_researcher.pipeline import arun

    result = RunResult(topic=topic)
    start = time.monotonic()
    try:
        state = await arun(topic, filters, use_cache=False)
    except Exception as exc:  # noqa: BLE001
        result.detail = f"arun raised: {exc}"
        result.seconds = time.monotonic() - start
        return result
    result.seconds = time.monotonic() - start

    result.errors = list(state.get("errors", []))
    judged = state.get("judged")
    result.degraded = bool(getattr(judged, "synthesis_degraded", True)) if judged else True
    result.n_findings = len(getattr(judged, "findings", []) or [])
    result.report_md = state.get("report_md", "")

    try:
        report = evaluate_state(state)
        result.calibration_passed = bool(report["calibration_passed"])
    except Exception as exc:  # noqa: BLE001
        result.detail = f"evaluate_state raised: {exc}"
        return result

    reasons = []
    if result.errors:
        reasons.append(f"errors={result.errors}")
    if result.degraded:
        reasons.append("synthesis_degraded=True")
    if not result.calibration_passed:
        reasons.append("calibration_failed")
    if result.n_findings == 0:
        reasons.append("no_findings")
    if not result.report_md.strip():
        reasons.append("empty_report")

    result.ok = not reasons
    result.detail = "; ".join(reasons) if reasons else "ok"
    return result


async def main_async(top_n: int | None) -> int:
    if top_n:
        os.environ["DR_TOP_N"] = str(top_n)
        from deep_researcher.config import reload_settings

        reload_settings()

    from deep_researcher.preflight import can_run_live

    if not can_run_live():
        print("OPENROUTER_API_KEY is not set - cannot run the live eval suite.")
        return 2

    results: list[RunResult] = []
    print(f"Running {len(TOPICS)} live eval cases (this hits real OpenRouter + Tavily APIs)...\n")
    for i, (topic, filters) in enumerate(TOPICS, 1):
        print(f"[{i}/{len(TOPICS)}] {topic!r} ...", flush=True)
        r = await _run_one(topic, filters)
        results.append(r)
        status = "PASS" if r.ok else "FAIL"
        print(f"    {status} in {r.seconds:.1f}s - findings={r.n_findings} detail={r.detail}\n")

    # Cross-check: no two distinct topics produced an identical report.
    print("=" * 60)
    duplicate_pairs: list[tuple[str, str]] = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            if a.report_md and a.report_md == b.report_md:
                duplicate_pairs.append((a.topic, b.topic))

    n_pass = sum(1 for r in results if r.ok)
    print(f"Per-topic checks: {n_pass}/{len(results)} passed")
    for r in results:
        mark = "OK " if r.ok else "-- "
        print(f"  [{mark}] {r.topic}")

    distinct_ok = not duplicate_pairs
    print(f"\nDistinct-answers check: {'PASS' if distinct_ok else 'FAIL'}")
    if duplicate_pairs:
        for a, b in duplicate_pairs:
            print(f"    IDENTICAL report for {a!r} and {b!r}")

    all_ok = distinct_ok and n_pass == len(results)
    print("\n" + ("ALL EVALS PASSED" if all_ok else "EVAL SUITE FAILED"))

    from deep_researcher.config import get_settings

    settings = get_settings()
    settings.ensure_dirs()
    summary_path = settings.reports_dir / "eval_suite_results.json"
    summary = {
        "all_passed": all_ok,
        "distinct_answers": distinct_ok,
        "duplicate_pairs": duplicate_pairs,
        "results": [
            {k: v for k, v in asdict(r).items() if k != "report_md"} for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")

    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Live 5-topic eval suite")
    parser.add_argument("--top-n", type=int, default=3, help="articles per topic (default 3, for speed/cost)")
    args = parser.parse_args()
    return asyncio.run(main_async(args.top_n))


if __name__ == "__main__":
    raise SystemExit(main())
