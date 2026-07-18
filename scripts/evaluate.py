"""Evaluate a completed run (cached or live) and print quality metrics.

Usage:
    python scripts/evaluate.py "Ozempic" --filters "...last 30 days"
    python scripts/evaluate.py "Ozempic" --filters "..." --run   # force a live run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_researcher.cache import hydrate_state, load_run  # noqa: E402
from deep_researcher.eval import evaluate_state  # noqa: E402
from deep_researcher.pipeline import arun  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Deep Researcher run")
    parser.add_argument("topic")
    parser.add_argument("--filters", default="")
    parser.add_argument("--run", action="store_true", help="Run live instead of using cache")
    args = parser.parse_args()

    if args.run:
        state = asyncio.run(arun(args.topic, args.filters, use_cache=False))
    else:
        payload = load_run(args.topic, args.filters)
        if not payload:
            print("No cached run found; pass --run to execute a live run.")
            return 1
        state = hydrate_state(payload)

    report = evaluate_state(state)
    print(json.dumps(report, indent=2))
    print(
        "\nCalibration:",
        "PASS" if report["calibration_passed"] else "FAIL",
    )
    return 0 if report["calibration_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
