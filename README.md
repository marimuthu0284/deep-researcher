# Multi-Agent AI Deep Researcher

A stateful, graph-orchestrated multi-agent pipeline that retrieves sources,
runs an **adversarial perspective debate** over each one, cross-examines the
debate, and synthesizes a **confidence-scored** report delivered by email.

The thesis: **structured disagreement as a quality mechanism.** Two agents
argue opposite interpretations of the *same evidence*, a critic audits their
citations and classifies their disagreements, and a judge converts the
surviving evidence into inspectable confidence scores.

## Architecture

```
User (topic + filters)
  -> Query Planner        decompose into 4-8 faceted queries
  -> Retriever            Tavily / arXiv / Semantic Scholar / GDELT / Crossref
  -> Triage & Dedup       dedup + syndication accounting, rank, keep top-N
  -> [map, parallel per article]
        Advocate  \
        Skeptic    >  Critical Analysis (citation audit, contradictions)
  -> Judge / Synthesis    resolve, score confidence, cross-source synthesis
  -> Report Builder       Markdown + email-safe HTML with rubric breakdowns
  -> Dispatcher           Resend email, falls back to a local report link
```

Built on **LangGraph** (`StateGraph` + `Send` map-reduce), with **OpenRouter**
as the single LLM gateway routing different jobs to different model families.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in keys
```

Required key: `OPENROUTER_API_KEY`. Recommended: `TAVILY_API_KEY` (web search)
and `RESEND_API_KEY` (email). arXiv, Semantic Scholar, GDELT, and Crossref need
no key. Everything degrades gracefully when a key is missing.

## Usage

CLI:

```bash
deep-researcher "Ozempic" --filters "peer-reviewed papers and major news, last 30 days" --email you@example.com
# stream a cached run (offline demo):
deep-researcher "Ozempic" --filters "peer-reviewed papers and major news, last 30 days" --replay
# human-in-the-loop: review/trim the triaged articles before the debate:
deep-researcher "Ozempic" --filters "..." --approve
# preflight: see which capabilities are live vs degraded:
deep-researcher --doctor
```

Streamlit UI (live agent-status board):

```bash
streamlit run app/streamlit_app.py
```

## Model routing (OpenRouter)

| Role | Default model | Override env |
| --- | --- | --- |
| Planner / glue | `meta-llama/llama-3.1-8b-instruct` | `DR_MODEL_PLANNER` |
| Perspective A & B | `meta-llama/llama-3.3-70b-instruct` | `DR_MODEL_PERSPECTIVE` |
| Critical Analyst | `mistralai/mistral-small-3.2-24b-instruct` | `DR_MODEL_CRITIC` |
| Judge / Synthesis | `google/gemini-2.5-flash` | `DR_MODEL_JUDGE` |
| Fallback | `openai/gpt-4o-mini` | `DR_MODEL_FALLBACK` |

Perspectives A and B share one model on purpose (differences come from the
stance prompt); the critic runs on a different family to decorrelate blind
spots; the judge gets the longest-context model.

## Confidence scoring

Per-article, 0-100: the LLM scores five rubric components (0-10 each, with a
required justification) and code computes the total:

`confidence = Σ(componentᵢ × weightᵢ) × 10`

Weights: source credibility 25%, evidence strength 25%, corroboration 20%,
internal consistency 15%, recency & relevance 15%. Finding-level confidence is
aggregated via noisy-OR damped by source independence, so two independent
70-confidence sources beat one 85.

## Offline demo (no keys needed)

Seed a coherent cached run once, then replay it instantly with no network:

```bash
python scripts/seed_demo.py
deep-researcher "Ozempic" --filters "peer-reviewed papers and major news, last 30 days" --replay
# or tick "Replay cached run" in the Streamlit sidebar
```

## Evaluation harness

Score a completed run (cached or live) for citation-audit survival, confidence
distribution, and calibration invariants:

```bash
python scripts/evaluate.py "Ozempic" --filters "peer-reviewed papers and major news, last 30 days"
python scripts/evaluate.py "Ozempic" --filters "..." --run   # force a live run first
```

## Observability

- Per-node timing is captured automatically and shown in the CLI summary, the
  Streamlit "Run telemetry" panel, and the state inspector.
- Set `LANGSMITH_API_KEY` to enable live LangChain/LangSmith tracing (put the
  trace on the projector as your agent-collaboration evidence).

## Reliability / fallback

Each role tries its primary model, then walks the `DR_FALLBACK_CHAIN`
(default `gemini-2.5-flash -> mistral-small-3.2 -> gpt-4o-mini`) until one
succeeds. Every LLM attempt is bounded by a hard timeout so a hung provider
can't stall the pipeline. A concurrency semaphore (`DR_MAX_CONCURRENCY`) caps
simultaneous in-flight LLM calls so the parallel debate stays under provider
rate limits. A bounded human-in-the-loop checkpoint (`--approve`) can pause
after triage to let a human curate the article set before the expensive debate.

## Deployment

```bash
docker build -t deep-researcher .
docker run -p 8501:8501 --env-file .env deep-researcher   # Streamlit UI
```

CI runs the test suite on every push (`.github/workflows/ci.yml`). A `Makefile`
wraps the common commands (`make install`, `make test`, `make doctor`,
`make seed`, `make ui`).

## Testing

```bash
pytest
```

Includes a fully offline end-to-end graph smoke test (LLM + network mocked),
plus unit coverage for scoring, dedup, fallback chain, telemetry, and eval.

## Reliability notes

- Every search source is fault-tolerant (returns empty rather than crashing).
- Per-query and full-run caching keyed on `(topic, filters)` for repeatable demos.
- Optional SQLite checkpointing for resume-after-crash.
- Bounded re-retrieval (one shot) inside the debate node keeps runs deterministic.
