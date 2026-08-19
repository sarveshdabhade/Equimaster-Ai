from __future__ import annotations

from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import chunk_article
from .embeddings import EmbeddingRetriever


class NewsRetriever:
    """Retrieval layer using sentence embeddings when available, otherwise TF-IDF fallback."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def _fallback_tfidf(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        texts = [item["chunk_text"] for item in chunks]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(texts + [query])
        similarity = cosine_similarity(matrix[-1], matrix[:-1])[0]

        scored = []
        for idx, item in enumerate(chunks):
            item_copy = dict(item)
            item_copy["relevance"] = float(similarity[idx])
            scored.append(item_copy)

        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[: self.top_k]

    def retrieve(self, articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        if not articles or not query:
            return []

        chunks: List[Dict[str, Any]] = []
        for article in articles:
            for item in chunk_article(article):
                chunks.append(item)

        if not chunks:
            return []

        embedder = EmbeddingRetriever()
        if embedder.available:
            texts = [item["chunk_text"] for item in chunks]
            embedder.build_index(texts)
            hits = embedder.search(query, k=min(self.top_k, len(chunks)))
            if hits:
                scored = []
                for hit in hits:
                    idx = hit["index"]
                    item_copy = dict(chunks[idx])
                    item_copy["relevance"] = max(0.0, float(hit["score"]))
                    scored.append(item_copy)
                scored.sort(key=lambda x: x["relevance"], reverse=True)
                return scored[: self.top_k]

        return self._fallback_tfidf(chunks, query)
