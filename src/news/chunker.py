from __future__ import annotations

import re
from typing import List


def split_text_into_chunks(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    if not text or not text.strip():
        return []

    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_article(article: dict) -> List[dict]:
    text_parts = [
        article.get("title", ""),
        article.get("summary", ""),
        article.get("content", ""),
    ]
    combined = " ".join(p for p in text_parts if p).strip()
    chunks = split_text_into_chunks(combined, chunk_size=500, overlap=120)
    results = []
    for idx, chunk in enumerate(chunks):
        results.append({
            "title": article.get("title", "").strip(),
            "source": article.get("source", "Unknown"),
            "published_at": article.get("published_at"),
            "url": article.get("url") or "",
            "chunk_index": idx,
            "chunk_text": chunk,
        })
    return results
