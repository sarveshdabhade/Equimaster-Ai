from __future__ import annotations

from typing import Any, List, Sequence

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - runtime fallback when dependency is absent
    SentenceTransformer = None

try:
    import faiss
except Exception:  # pragma: no cover - fallback for environments without faiss
    faiss = None


class EmbeddingRetriever:
    """Lightweight embedding-backed retriever with an in-memory FAISS index when available."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents: List[str] = []
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None and faiss is not None

    def encode(self, texts: Sequence[str]):
        if self.model is None:
            raise RuntimeError("SentenceTransformer is unavailable; cannot encode text.")
        return self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)

    def build_index(self, texts: Sequence[str]):
        self.documents = list(texts)
        if not self.available or not self.documents:
            return None

        vectors = self.encode(self.documents).astype("float32")
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        self.index = index
        return index

    def search(self, query: str, k: int = 5) -> List[dict]:
        if not self.available or self.index is None or not self.documents:
            return []

        query_vector = self.encode([query]).astype("float32")
        scores, indices = self.index.search(query_vector, min(k, len(self.documents)))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append({"index": int(index), "score": float(score)})
        return results
