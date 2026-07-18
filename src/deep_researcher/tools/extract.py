"""Full-text extraction and chunking.

`trafilatura` pulls clean article text from a URL when possible; otherwise we
keep the abstract and mark full_text=False. Chunking is a simple whitespace
token approximation (~800 tokens, 100 overlap) which is plenty for citation
granularity and avoids a tokenizer dependency.
"""

from __future__ import annotations

from ..models import Chunk

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100
# Rough tokens->words ratio; keeps us tokenizer-free but in the right ballpark.
_WORDS_PER_CHUNK = int(CHUNK_TOKENS * 0.75)
_WORDS_OVERLAP = int(CHUNK_OVERLAP * 0.75)


def fetch_full_text(url: str, timeout: int = 15) -> str | None:
    """Return extracted main text for a URL, or None on failure."""
    try:
        import trafilatura
    except Exception:  # pragma: no cover - optional at import time
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        return text or None
    except Exception:
        return None


def chunk_text(text: str, article_id: str) -> list[Chunk]:
    """Split text into overlapping chunks with stable {article_id}-c{n} ids."""
    words = text.split()
    if not words:
        return []
    chunks: list[Chunk] = []
    step = max(1, _WORDS_PER_CHUNK - _WORDS_OVERLAP)
    n = 0
    for start in range(0, len(words), step):
        window = words[start : start + _WORDS_PER_CHUNK]
        if not window:
            break
        chunks.append(
            Chunk(chunk_id=f"{article_id}-c{n}", text=" ".join(window))
        )
        n += 1
        if start + _WORDS_PER_CHUNK >= len(words):
            break
    return chunks
