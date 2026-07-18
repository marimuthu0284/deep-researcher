import asyncio

import pytest

import deep_researcher.llm as llm_mod
from deep_researcher.config import get_settings


@pytest.mark.asyncio
async def test_semaphore_uses_configured_value(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_concurrency", 3)
    llm_mod._semaphores.clear()
    sem = llm_mod._get_semaphore()
    assert sem._value == 3


@pytest.mark.asyncio
async def test_semaphore_caps_concurrency(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_concurrency", 2)
    llm_mod._semaphores.clear()

    active = {"now": 0, "peak": 0}

    async def worker():
        async with llm_mod._get_semaphore():
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.02)
            active["now"] -= 1

    await asyncio.gather(*[worker() for _ in range(8)])
    assert active["peak"] <= 2
