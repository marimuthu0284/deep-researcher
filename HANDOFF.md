# Deep Researcher — Handoff / Continuation Guide

A working session summary so you can pick up the build in a fresh terminal or
agent session without re-discovering context. Pair this with `README.md` (user
docs) and `.cursor/plans/deep_researcher_implementation_*.plan.md` (original plan).

Last updated: after degraded-run hardening. Status: **fully working end-to-end,
live-verified, 50/50 tests passing.**

---

## 1. What this is

A stateful, graph-orchestrated multi-agent research pipeline (LangGraph). A user
gives a topic + filters; the system plans queries, retrieves real sources, runs
an adversarial advocate/skeptic debate per article, cross-examines it, synthesizes
a confidence-scored report, and emails it. Built from
`~/Downloads/deep-researcher-architecture (2).md`.

Core thesis: **structured disagreement as a quality mechanism** — two agents argue
opposite readings of the same evidence; a critic audits citations; a judge resolves
and scores.

---

## 2. Current status

- Core pipeline (all 10 agents), confidence scoring, report, email, Streamlit UI: DONE.
- Phase 2 (observability, eval, offline demo seed, reliability, UI polish): DONE.
- Phase 3 (concurrency cap, human-in-the-loop approval, doctor, CI/Docker/Make): DONE.
- Live-tested against OpenRouter + Tavily; produces genuine synthesis + calibrated scores.
- Tests: `pytest` -> **42 passed** (incl. offline end-to-end graph run).
- Git: branch `cursor/multi-agent-deep-researcher`, working tree clean, all committed.
  **Not pushed** (no remote auth configured; remote `origin` points at
  `https://github.com/jar11/Outskill-Hackathon-Batch6.git`).

---

## 3. Setup (fresh terminal)

```bash
cd ~/Outskill-Hackathon
python -m venv .venv && source .venv/bin/activate   # if .venv missing
pip install -e ".[dev]"                              # if not installed
cp .env.example .env                                 # if .env missing
# edit .env: set OPENROUTER_API_KEY (required for live). TAVILY_API_KEY + RESEND_API_KEY optional.
```

Python note: developed on **Python 3.14** (all deps install fine). CI runs 3.11/3.12.

Quick sanity:
```bash
deep-researcher --doctor          # shows which capabilities are live vs degraded
pytest -q                         # 42 tests, offline, no keys needed
```

Run it:
```bash
# fast live run (cap articles for speed/cost):
DR_TOP_N=3 deep-researcher "Ozempic" --filters "peer-reviewed papers and major news, last 90 days" --no-cache
# offline demo (no keys): seed once, then replay
python scripts/seed_demo.py
deep-researcher "Ozempic" --filters "peer-reviewed papers and major news, last 30 days" --replay
# UI
streamlit run app/streamlit_app.py
# evaluate a completed (cached or live) run
python scripts/evaluate.py "Ozempic" --filters "peer-reviewed papers and major news, last 90 days"
```

Report output: `data/reports/report.{md,html}`. Run cache: `data/cache/`.

---

## 4. Architecture map

```
START -> query_planner -> retriever -> triage -> gate
         gate --(Send per approved article)--> debate_article  (parallel map)
         debate_article -> judge  (reduce; runs once after all branches)
         judge -> report_builder -> dispatcher -> END
```

- `gate` is a passthrough node that is the human-in-the-loop interrupt point
  (`--approve`). Editing `bundles` during the pause changes which articles get
  debated, because the fan-out reads `bundles` on resume.
- Per-article bounded re-retrieval (one shot) lives INSIDE `debate_article`, so the
  graph stays acyclic/deterministic.

### File map (`src/deep_researcher/`)
- `config.py` — env settings, model routing, fallback chain, tracing toggle.
- `models.py` — Pydantic inter-agent contracts (the typed protocol).
- `state.py` — LangGraph `ResearchState` (TypedDict) + additive reducers
  (`analyses`, `status_log`, `errors`, `metrics`).
- `llm.py` — OpenRouter gateway: `structured(role, schema, system, user)`,
  per-role candidate chain + timeout + retry, concurrency semaphore.
- `prompts.py` — all agent system prompts (from doc section 4).
- `scoring.py` — rubric total, noisy-OR finding aggregation, calibration bands.
- `telemetry.py` — `timed()` node decorator -> `metrics` channel; `summarize_metrics`.
- `eval.py` — citation/confidence metrics + calibration checks.
- `preflight.py` — the `--doctor` capability report.
- `cache.py` — run cache (topic, filters) + hydrate for replay.
- `pipeline.py` — `arun(...)` orchestration: stream, cache, checkpointer, HITL.
- `graph.py` — StateGraph assembly + `Send` fan-out.
- `cli.py` — `deep-researcher` entrypoint.
- `agents/` — one module per node: query_planner, retriever, triage, perspective
  (advocate+skeptic), critical_analysis, debate (map orchestrator), judge,
  report_builder, dispatcher, common.
- `tools/` — `search.py` (Tavily/arXiv/S2/GDELT/Crossref, fault-tolerant + cached),
  `extract.py` (trafilatura + chunking).
- `report/templates.py` — deterministic Markdown + email-safe HTML.
- `app/streamlit_app.py` — live board, charts, telemetry, state inspector, replay.
- `scripts/` — `seed_demo.py`, `evaluate.py`.

---

## 5. Model routing (IMPORTANT gotcha)

OpenRouter is the single gateway. Per-role defaults in `config.py`:
- planner/glue: `meta-llama/llama-3.1-8b-instruct`
- perspective A/B: `meta-llama/llama-3.3-70b-instruct`
- critic: `mistralai/mistral-small-3.2-24b-instruct`
- judge: `google/gemini-2.5-flash`
- fallback chain: gemini-2.5-flash -> mistral-small-3.2 -> gpt-4o-mini

**Model slugs go stale.** Verify current slugs before trusting them:
```bash
python -c "import requests; ids=[m['id'] for m in requests.get('https://openrouter.ai/api/v1/models').json()['data']]; print([i for i in ids if 'gemini' in i and 'flash' in i])"
```
Override without code changes via env: `DR_MODEL_JUDGE=...`, `DR_FALLBACK_CHAIN=a,b,c`.

---

## 6. Bugs already found & fixed (do NOT reintroduce)

These were caught only by live testing; regression tests now guard them.

1. **Judge called `structured()` without `user`** -> instant TypeError swallowed
   into fallback ("judge model unavailable"). All `structured()` calls must pass
   `user`. `user` now has a default; `tests/test_judge.py` requires it.
2. **Stale model slugs** (`gemini-2.0-flash-001`, `mistral-small-latest`) 404/400 on
   OpenRouter. Fixed to valid slugs; verify per section 5.
3. **`source_type: "web"`** emitted by planner inside `api_params` crashed
   ArticleBundle validation. `run_query` now forces the query's validated
   `source_type`; retriever coerces unknowns to `report`. See `tests/test_retriever.py`.
4. **Checkpointer cross-run contamination** — stable `thread_id` made re-runs resume
   and merge. `pipeline.arun` now uses a unique `thread_id` per invocation.
5. **Silent degradation produced an off-topic, misleading report** (the "Dubai
   Realestate" run returned 8 unrelated arXiv AI/physics preprints with
   confidence-30 "judge model unavailable" findings, no warning). Three root
   causes + fixes:
   - `DR_API_TIMEOUT` default was **15s**, which tripped the slow planner/judge
     structured-output call (the manual-JSON fallback path is slow) into the
     degraded fallback. Raised default to **40s** (`config.py`, `.env`,
     `.env.example`). Observed planner node time swings 8s vs 71s — the tight
     timeout was the likely trigger.
   - `_fallback_plan` (query_planner) always led with `peer_reviewed` (arXiv)
     queries, so a "major news" topic returned irrelevant preprints. Now
     **filter-aware**: news/empty filters route to news+web; academic terms
     route to scholarly sources; always keeps a criticism query.
     `tests/test_query_planner.py` guards this.
   - Degraded runs were rendered as if valid. `JudgedState` now carries
     `synthesis_degraded` + `degraded_articles` (set in `judge.py`), and the
     report (md + HTML) shows a prominent **"Degraded run — NOT a full analysis"**
     banner. `tests/test_report.py` + `tests/test_judge.py` guard this.
   Net effect: even in a total LLM outage, retrieval stays on-topic and the
   report is loudly flagged instead of masquerading as real analysis.

Other gotchas:
- Python 3.14: use `asyncio.get_running_loop()` (not `get_event_loop()`).
- Silent degradation is by design (graceful fallback); when debugging, temporarily
  surface exceptions (wrap `structured` in agents, or check `state["errors"]`).
- `data/` is gitignored; the seeded demo cache is NOT committed — run
  `python scripts/seed_demo.py` in each fresh clone to enable `--replay`.

---

## 7. Testing

```bash
pytest -q                     # all
pytest tests/test_graph_smoke.py -q   # offline end-to-end (LLM + net mocked)
```
Coverage: models, scoring, triage dedup, report render, llm fallback chain,
telemetry, eval, preflight, retriever coercion, judge (no-fallback), HITL, and a
full offline graph run. When adding a node/feature, mirror the mock pattern in
`tests/test_graph_smoke.py` and add a regression test.

---

## 8. Suggested next steps (backlog)

Not yet built — good candidates to continue with:
- Human-in-the-loop approval in the **Streamlit UI** (currently CLI `--approve` only;
  needs session-state pause/resume).
- Additional source adapters behind keys: Brave Search, NewsAPI (keys not present).
- Full-text extraction cache (retriever refetches per run; `tools/search.py` caches
  search results but not trafilatura extractions).
- "Compare two topics" / diff mode.
- Real email demo (add `RESEND_API_KEY`, use `--email`).
- Prompt tuning for the planner (it occasionally emits odd `api_params`).
- LangSmith trace screenshot flow for demo (set `LANGSMITH_API_KEY`).
- Tune `DR_TOP_N` / `DR_MAX_CONCURRENCY` for the demo machine; debate dominates latency
  (~90-150s for 3 articles).

---

## 9. Env vars reference (see `.env.example`)

Required for live: `OPENROUTER_API_KEY`.
Recommended: `TAVILY_API_KEY`, `RESEND_API_KEY` (+ `RESEND_FROM`).
Optional tuning: `DR_TOP_N`, `DR_MAX_CONCURRENCY`, `DR_API_TIMEOUT`,
`DR_MODEL_{PLANNER,PERSPECTIVE,CRITIC,JUDGE,FALLBACK}`, `DR_FALLBACK_CHAIN`,
`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `DR_CACHE_DIR`, `DR_REPORTS_DIR`,
`DR_CHECKPOINT_DB`.
