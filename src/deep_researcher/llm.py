"""OpenRouter LLM gateway.

A single `llm(role)` router returns a configured chat model for a role, so the
whole system talks to one provider (OpenRouter) while still routing different
cognitive jobs to different model families. `structured()` is the workhorse the
agents call: it asks for schema-validated output and transparently falls back
to a second model on failure.
"""

from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import OPENROUTER_BASE_URL, get_settings

T = TypeVar("T", bound=BaseModel)


# One semaphore per event loop, so the parallel debate fan-out never exceeds
# the configured number of simultaneous in-flight LLM calls (rate-limit safety).
_semaphores: dict[int, asyncio.Semaphore] = {}


def _get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _semaphores.get(key)
    if sem is None:
        sem = asyncio.Semaphore(get_settings().max_concurrency)
        _semaphores[key] = sem
    return sem


@lru_cache(maxsize=32)
def _client(model: str, temperature: float) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=settings.api_timeout,
        max_retries=0,  # we manage retries/fallback ourselves
        api_key=settings.require_openrouter(),
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/deep-researcher",
            "X-Title": "Deep Researcher",
        },
    )


def llm(role: str, temperature: float = 0.2) -> ChatOpenAI:
    """Return the chat model configured for a pipeline role."""
    settings = get_settings()
    return _client(settings.model_for(role), temperature)


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from a model response."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    brace = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if brace:
        return brace.group(1)
    return text


class _Transient(Exception):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_exception_type(_Transient),
)
async def _attempt(
    model_name: str,
    schema: type[T],
    system: str,
    user: str,
    temperature: float,
) -> T:
    """One model's attempt (with an internal retry), guarded by a hard timeout."""
    settings = get_settings()
    # Overall wall-clock budget per attempt: a little more than the HTTP timeout
    # so a hung connection can't stall the whole pipeline.
    budget = settings.api_timeout * 2 + 5
    try:
        return await asyncio.wait_for(
            _invoke(model_name, schema, system, user, temperature), timeout=budget
        )
    except asyncio.TimeoutError as exc:
        raise _Transient(f"timeout after {budget}s on {model_name}") from exc


async def _invoke(
    model_name: str,
    schema: type[T],
    system: str,
    user: str,
    temperature: float,
) -> T:
    client = _client(model_name, temperature)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]

    async with _get_semaphore():
        # Primary path: provider-native structured output.
        try:
            structured_client = client.with_structured_output(schema)
            result = await structured_client.ainvoke(messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except Exception:  # noqa: BLE001 - fall back to manual JSON parsing
            pass

        # Fallback path: ask for raw JSON and validate ourselves.
        schema_hint = json.dumps(schema.model_json_schema())
        raw = await client.ainvoke(
            [
                SystemMessage(
                    content=system
                    + "\n\nRespond ONLY with a single JSON object that validates "
                    "against this JSON schema:\n" + schema_hint
                ),
                HumanMessage(content=user),
            ]
        )
    text = raw.content if isinstance(raw.content, str) else str(raw.content)
    try:
        payload = json.loads(_extract_json(text))
        return schema.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise _Transient(f"structured output failed for {model_name}: {exc}") from exc


async def structured(
    role: str,
    schema: type[T],
    system: str,
    user: str = "Produce the required output as valid JSON.",
    temperature: float = 0.2,
) -> T:
    """Produce schema-validated output for a role, with a fallback chain.

    Tries the role's primary model, then each model in the configured fallback
    chain in order, until one succeeds. This is the graceful-degradation story:
    the pipeline keeps moving even when one provider is flaky or rate-limited.
    """
    settings = get_settings()
    candidates = settings.candidate_models(role)
    last_exc: Exception | None = None
    for model_name in candidates:
        try:
            return await _attempt(model_name, schema, system, user, temperature)
        except Exception as exc:  # noqa: BLE001 - try the next provider
            last_exc = exc
            continue
    raise RuntimeError(
        f"all models failed for role '{role}' ({candidates}): {last_exc}"
    ) from last_exc


async def freeform(role: str, system: str, user: str, temperature: float = 0.3) -> str:
    """Return plain text from a role's model (used for prose synthesis bits)."""
    client = llm(role, temperature)
    resp = await client.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return resp.content if isinstance(resp.content, str) else str(resp.content)
