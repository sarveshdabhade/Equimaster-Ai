from __future__ import annotations

import json
from pathlib import Path
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
    """Embedding-backed retriever with an optional persistent FAISS index on disk."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str | None = None,
    ):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents: List[str] = []
        self.persist_dir = Path(persist_dir) if persist_dir else Path.cwd() / ".cache" / "news" / "embeddings"
        self.index_path = self.persist_dir / "faiss.index"
        self.metadata_path = self.persist_dir / "metadata.json"

        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None and faiss is not None

    def _ensure_storage(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def encode(self, texts: Sequence[str]):
        if self.model is None:
            raise RuntimeError("SentenceTransformer is unavailable; cannot encode text.")
        return self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)

    def save_index(self) -> None:
        if not self.available or not self.documents:
            return
        self._ensure_storage()
        try:
            faiss.write_index(self.index, str(self.index_path))
        except Exception:
            # best-effort; don't fail the caller on save errors
            pass
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump({"documents": self.documents}, handle)

    def load_index(self) -> bool:
        if not self.available or not self.index_path.exists() or not self.metadata_path.exists():
            return False
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.documents = list(payload.get("documents", []))
            return bool(self.documents)
        except Exception:
            self.index = None
            self.documents = []
            return False

    def build_index(self, texts: Sequence[str]):
        """Build a full index from texts and persist it to disk."""
        self.documents = list(texts)
        if not self.available or not self.documents:
            return None

        vectors = self.encode(self.documents).astype("float32")
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        self.index = index
        self.save_index()
        return index

    def add_documents(self, texts: Sequence[str]):
        """Incrementally encode and add new texts to the existing index and metadata.

        If no index exists, this creates a fresh index identical to build_index.
        """
        new_docs = list(texts)
        if not self.available or not new_docs:
            return None

        # Ensure model is loaded
        vectors = self.encode(new_docs).astype("float32")
        if self.index is None:
            # create new index
            dimension = vectors.shape[1]
            idx = faiss.IndexFlatIP(dimension)
            idx.add(vectors)
            self.index = idx
            self.documents = new_docs
        else:
            # append vectors
            try:
                self.index.add(vectors)
                self.documents.extend(new_docs)
            except Exception:
                # fallback: rebuild full index including new docs
                combined = list(self.documents) + new_docs
                return self.build_index(combined)

        # persist
        self.save_index()
        return self.index

    def search(self, query: str, k: int = 5) -> List[dict]:
        if self.index is None and self.load_index():
            pass

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
