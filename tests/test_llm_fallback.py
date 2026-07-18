import pytest
from pydantic import BaseModel

import deep_researcher.llm as llm_mod
from deep_researcher.config import get_settings


class Foo(BaseModel):
    x: int


def test_candidate_models_dedup_and_order():
    settings = get_settings()
    cands = settings.candidate_models("planner")
    assert cands[0] == settings.model_for("planner")
    assert len(cands) == len(set(cands))  # de-duplicated
    assert len(cands) >= 2  # primary + fallback chain


@pytest.mark.asyncio
async def test_fallback_chain_uses_later_model(monkeypatch):
    calls = []

    async def fake_attempt(model, schema, system, user, temperature):
        calls.append(model)
        if len(calls) < 2:
            raise RuntimeError("first model down")
        return schema(x=len(calls))

    monkeypatch.setattr(llm_mod, "_attempt", fake_attempt)
    result = await llm_mod.structured("planner", Foo, "sys", "user")
    assert result.x == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_all_models_fail_raises(monkeypatch):
    async def fake_attempt(model, schema, system, user, temperature):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm_mod, "_attempt", fake_attempt)
    with pytest.raises(RuntimeError):
        await llm_mod.structured("planner", Foo, "sys", "user")
