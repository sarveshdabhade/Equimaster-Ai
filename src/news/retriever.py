from __future__ import annotations

from typing import Any
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import chunk_article
from .embeddings import EmbeddingRetriever
from config import NEWS_RECENCY_DAYS


class NewsRetriever:
    """Retrieve evidence with lightweight TF-IDF, with no model download required."""

    def __init__(self, top_k: int = 5, embedding_cache_dir: str | None = None):
        self.top_k = top_k
        self.embedding_cache_dir = embedding_cache_dir

    def retrieve(
        self,
        articles: list[dict[str, Any]],
        query: str,
        recency_days: int | None = NEWS_RECENCY_DAYS,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        now = as_of or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if recency_days is not None or as_of is not None:
            cutoff = now - timedelta(days=max(0, recency_days)) if recency_days is not None else datetime.min.replace(tzinfo=timezone.utc)
            recent_articles = []
            for article in articles:
                try:
                    published = datetime.fromisoformat(str(article.get("published_at", "")).replace("Z", "+00:00"))
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)
                    observed_at = article.get("observed_at")
                    if as_of is not None:
                        if not observed_at:
                            continue
                        observed = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
                        if observed.tzinfo is None:
                            observed = observed.replace(tzinfo=timezone.utc)
                        if observed > now:
                            continue
                    if cutoff <= published <= now:
                        recent_articles.append(article)
                except (TypeError, ValueError):
                    continue
            articles = recent_articles
        chunks = [chunk for article in articles for chunk in chunk_article(article)]
        if not chunks or not query:
            return []
        texts = [item["chunk_text"] for item in chunks]
        metadata = [{
            "article_id": item.get("article_id"),
            "ticker": item.get("ticker", ""),
            "source": item.get("source", "Unknown"),
            "published_at": item.get("published_at"),
            "observed_at": item.get("observed_at"),
            "url": item.get("url", ""),
            "canonical_url": item.get("canonical_url", ""),
            "chunk_id": item.get("chunk_id"),
            "duplicate_sources": item.get("duplicate_sources", []),
            "duplicate_urls": item.get("duplicate_urls", []),
            "sentiment": item.get("sentiment", "neutral"),
            "score_confidence": item.get("confidence", 0.0),
            "source_weight": 1.0,
        } for item in chunks]
        ticker = str(metadata[0].get("ticker") or "unknown").replace(".", "_")
        dates = sorted(str(item.get("published_at") or "unknown")[:10].replace("-", "") for item in metadata)
        date_scope = f"{dates[0]}_{dates[-1]}" if dates else "unknown"
        persist_dir = Path(self.embedding_cache_dir or Path.cwd() / ".cache" / "news" / "embeddings") / ticker / date_scope
        embedder = EmbeddingRetriever(persist_dir=str(persist_dir))
        if embedder.available:
            loaded = embedder.load_index()
            if not loaded or embedder.documents != texts or embedder.metadata != metadata:
                embedder.build_index(texts, metadata)
            hits = embedder.search(query, k=min(self.top_k, len(chunks)))
            if hits:
                scored = []
                for hit in hits:
                    idx = int(hit.get("index", -1))
                    if idx < 0 or idx >= len(chunks):
                        continue
                    item = dict(chunks[idx])
                    item["relevance"] = round(max(0.0, float(hit["score"])), 3)
                    item["embedding_metadata"] = hit.get("metadata", {})
                    scored.append(item)
                if scored:
                    return sorted(scored, key=lambda item: item["relevance"], reverse=True)[: self.top_k]

        try:
            matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts + [query])
            scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
        except ValueError:
            scores = [0.0] * len(chunks)
        results = []
        for item, score in zip(chunks, scores):
            result = dict(item)
            result["relevance"] = round(float(score), 3)
            results.append(result)
        return sorted(results, key=lambda item: item["relevance"], reverse=True)[: self.top_k]
