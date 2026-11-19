from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from config import NEWS_CACHE_PATH_ENV

try:
    if os.getenv("EQUIMASTER_EMBEDDINGS", "0") == "1":
        from sentence_transformers import SentenceTransformer
    else:
        SentenceTransformer = None
except Exception:  # pragma: no cover - runtime fallback when dependency is absent
    SentenceTransformer = None

try:
    import faiss
except Exception:  # pragma: no cover - fallback for environments without faiss
    faiss = None


class EmbeddingRetriever:
    """Embedding-backed retriever with an optional persistent FAISS index on disk.

    This implementation uses stable document IDs (sha1 of the text) so incremental
    additions can detect duplicates and support out-of-order inserts.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str | None = None,
    ):
        self.model_name = model_name
        self.model = None
        self.index = None
        # documents holds the texts in index order
        self.documents: List[str] = []
        # ids holds the sha1 ids in index order
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        # mapping id -> text is stored in metadata
        default_dir = Path(os.getenv(NEWS_CACHE_PATH_ENV, str(Path.cwd() / ".cache" / "news"))) / "embeddings"
        self.persist_dir = Path(persist_dir) if persist_dir else default_dir
        self.index_path = self.persist_dir / "faiss.index"
        self.metadata_path = self.persist_dir / "metadata.json"

        if SentenceTransformer is not None and os.getenv("EQUIMASTER_EMBEDDINGS", "0") == "1":
            try:
                self.model = SentenceTransformer(model_name, local_files_only=os.getenv("EQUIMASTER_ALLOW_MODEL_DOWNLOADS", "0") != "1")
            except Exception:
                self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None and faiss is not None

    @staticmethod
    def compute_id(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

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
        # persist metadata: ordered ids and mapping id->text
        meta: Dict[str, Any] = {"documents": self.ids, "doc_texts": {}, "metadata": self.metadata}
        for _id, text in zip(self.ids, self.documents):
            meta["doc_texts"][_id] = text
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

    def load_index(self) -> bool:
        if not self.available or not self.index_path.exists() or not self.metadata_path.exists():
            return False
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.ids = list(payload.get("documents", []))
            doc_texts = payload.get("doc_texts", {})
            self.metadata = list(payload.get("metadata", [{} for _ in self.ids]))
            if len(self.metadata) != len(self.ids):
                self.metadata = [{} for _ in self.ids]
            # rebuild documents list in order of ids
            self.documents = [doc_texts.get(_id, "") for _id in self.ids]
            return bool(self.documents)
        except Exception:
            self.index = None
            self.documents = []
            self.ids = []
            self.metadata = []
            return False

    def build_index(self, texts: Sequence[str], metadata: Sequence[Dict[str, Any]] | None = None):
        """Build a full index from texts and persist it to disk."""
        self.documents = list(texts)
        self.metadata = list(metadata or [{} for _ in self.documents])
        if len(self.metadata) != len(self.documents):
            raise ValueError("Embedding metadata must match document count.")
        if not self.available or not self.documents:
            return None
        # compute ids
        self.ids = [self.compute_id(t) for t in self.documents]

        vectors = self.encode(self.documents).astype("float32")
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        self.index = index
        self.save_index()
        return index

    def add_documents(self, texts: Sequence[str], metadata: Sequence[Dict[str, Any]] | None = None):
        """Incrementally encode and add new texts to the existing index and metadata.

        Uses stable ids (sha1) to avoid duplicates and supports out-of-order inserts.
        If no index exists, this creates a fresh index identical to build_index.
        """
        new_docs = list(texts)
        new_metadata = list(metadata or [{} for _ in new_docs])
        if len(new_metadata) != len(new_docs):
            raise ValueError("Embedding metadata must match document count.")
        if not self.available or not new_docs:
            return None

        new_ids = [self.compute_id(t) for t in new_docs]
        # determine which docs are actually new
        existing_ids_set = set(self.ids) if self.ids else set()
        to_add: List[Tuple[str, str]] = []
        for position, (_id, txt) in enumerate(zip(new_ids, new_docs)):
            if _id in existing_ids_set:
                continue
            to_add.append((_id, txt, new_metadata[position]))

        if not to_add and self.index is not None:
            # nothing to do
            return self.index

        vectors = self.encode([txt for (_id, txt, _meta) in to_add]).astype("float32") if to_add else None
        if self.index is None:
            # create new index from all documents (existing empty + to_add)
            all_texts = list(self.documents) + [txt for (_id, txt, _meta) in to_add]
            all_metadata = list(self.metadata) + [meta for (_id, _txt, meta) in to_add]
            return self.build_index(all_texts, all_metadata)

        # append vectors
        try:
            if vectors is not None and len(vectors) > 0:
                self.index.add(vectors)
                for (_id, txt, meta) in to_add:
                    self.ids.append(_id)
                    self.documents.append(txt)
                    self.metadata.append(meta)
        except Exception:
            # fallback: rebuild full index including new docs
            combined = list(self.documents) + [txt for (_id, txt, _meta) in to_add]
            combined_metadata = list(self.metadata) + [meta for (_id, _txt, meta) in to_add]
            return self.build_index(combined, combined_metadata)

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
            results.append({"index": int(index), "score": float(score), "metadata": self.metadata[index]})
        return results
