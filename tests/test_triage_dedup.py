from deep_researcher.agents.triage import _dedup, _heuristic_select, _title_cosine
from deep_researcher.models import ArticleBundle


def _mk(aid, title, url, source_type="news"):
    return ArticleBundle(
        article_id=aid, title=title, url=url, source_name="x", source_type=source_type
    )


def test_title_cosine_identical():
    assert _title_cosine("Ozempic cuts heart risk", "Ozempic cuts heart risk") == 1.0


def test_title_cosine_unrelated():
    assert _title_cosine("cats are nice", "quantum chromodynamics") < 0.2


def test_dedup_exact_url():
    docs = [
        _mk("a0", "One", "https://x.com/1"),
        _mk("a1", "Two", "https://x.com/1"),  # duplicate url
    ]
    kept, dups = _dedup(docs)
    assert len(kept) == 1


def test_dedup_syndication_count():
    docs = [
        _mk("a0", "Ozempic cuts cardiovascular risk in new trial", "https://a.com/1"),
        _mk("a1", "Ozempic cuts cardiovascular risk in new trial", "https://b.com/1"),
        _mk("a2", "Ozempic cuts cardiovascular risk in new trial", "https://c.com/1"),
    ]
    kept, dups = _dedup(docs)
    assert len(kept) == 1
    assert kept[0].syndication_count == 3
    assert dups == 2


def test_heuristic_select_caps_to_n():
    docs = [_mk(f"a{i}", f"Title {i}", f"https://x.com/{i}") for i in range(20)]
    chosen = _heuristic_select(docs, 8)
    assert len(chosen) == 8
