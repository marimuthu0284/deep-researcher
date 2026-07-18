from deep_researcher.preflight import Capability, can_run_live, check


def test_check_returns_capabilities():
    caps = check()
    assert caps and all(isinstance(c, Capability) for c in caps)
    names = {c.name for c in caps}
    assert "LLM gateway (OpenRouter)" in names
    assert any("Retrieval" in n for n in names)


def test_can_run_live_is_bool():
    assert isinstance(can_run_live(), bool)


def test_llm_gateway_reflects_key(monkeypatch):
    from deep_researcher.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    llm_cap = next(c for c in check() if c.name == "LLM gateway (OpenRouter)")
    assert llm_cap.ok is False

    monkeypatch.setattr(settings, "openrouter_api_key", "sk-test")
    llm_cap = next(c for c in check() if c.name == "LLM gateway (OpenRouter)")
    assert llm_cap.ok is True
