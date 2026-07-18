from deep_researcher.agents.retriever import _build_bundle, _coerce_source_type


def test_coerce_valid_source_types_pass_through():
    for st in ["peer_reviewed", "preprint", "news", "report", "blog"]:
        assert _coerce_source_type(st) == st


def test_coerce_invalid_source_type_falls_back():
    # An LLM can emit "web" inside api_params; it must not crash bundle building.
    assert _coerce_source_type("web") == "report"
    assert _coerce_source_type(None) == "report"


def test_build_bundle_with_bad_source_type_does_not_raise():
    r = {
        "title": "T",
        "url": "https://example.com/a",
        "source_name": "example.com",
        "source_type": "web",  # invalid literal
        "snippet": "some snippet text that is reasonably long " * 3,
        "full_text": "word " * 100,
        "published_at": None,
        "citation_count": None,
    }
    bundle = _build_bundle(0, r, timeout=1)
    assert bundle is not None
    assert bundle.source_type == "report"
