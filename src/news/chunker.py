from __future__ import annotations

import hashlib
import re

from .fetcher import canonicalize_article_url


def _article_identity(article: dict) -> str:
    canonical_url = str(article.get("canonical_url") or "").strip()
    if canonical_url:
        return canonical_url
    url = str(article.get("url") or "").strip()
    if url:
        return canonicalize_article_url(url) or url
    return "|".join((
        str(article.get("ticker") or "").strip().upper(),
        re.sub(r"\s+", " ", str(article.get("title") or "")).strip().lower(),
        str(article.get("published_at") or ""),
    ))


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 120) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_article(article: dict) -> list[dict]:
    text = " ".join(
        part for part in (article.get("title"), article.get("summary"), article.get("content")) if part
    )
    results = []
    canonical_url = str(article.get("canonical_url") or "").strip()
    if not canonical_url:
        canonical_url = canonicalize_article_url(str(article.get("url") or ""))
    article_id = hashlib.sha256(_article_identity(article).encode("utf-8")).hexdigest()
    for index, chunk in enumerate(split_text_into_chunks(text)):
        chunk_id = hashlib.sha256(f"{article_id}:{index}:{chunk}".encode("utf-8")).hexdigest()
        results.append({
            "article_id": article_id,
            "ticker": article.get("ticker", ""),
            "title": article.get("title", "").strip(),
            "source": article.get("source", "Unknown"),
            "published_at": article.get("published_at"),
            "observed_at": article.get("observed_at"),
            "url": article.get("url") or "",
            "canonical_url": canonical_url,
            "duplicate_sources": list(article.get("duplicate_sources", [])),
            "duplicate_urls": list(article.get("duplicate_urls", [])),
            "chunk_index": index,
            "chunk_id": chunk_id,
            "chunk_text": chunk,
        })
    return results
