"""
Vector Store Abstraction for MedRAG.
Supports persistent local ChromaDB, in-memory FAISS, and NumPy matrix multiplication.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .retriever import Document

logger = logging.getLogger("medrag.vector_store")


class BaseVectorStore:
    """Abstract base class for vector database implementations."""

    def index(self, documents: List[Document], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """Returns list of (doc_id, similarity_score) tuples."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    """
    Persistent ChromaDB vector database with HNSW indexing and cosine similarity.
    Saves embeddings and metadata to disk so textbooks and documents are not re-embedded.
    """

    def __init__(
        self,
        collection_name: str = "medrag_corpus",
        persist_dir: Optional[str] = "./data/chroma_db",
    ) -> None:
        import chromadb
        # Sanitize collection name for ChromaDB (3-63 characters, alphanumeric, underscores, hyphens)
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name).strip("_")
        if len(clean_name) < 3:
            clean_name = f"medrag_{clean_name}"
        clean_name = clean_name[:63]

        self.collection_name = clean_name
        self.persist_dir = persist_dir

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized ChromaVectorStore (collection='{self.collection_name}', persist_dir='{persist_dir}')"
        )

    def has_ids(self, doc_ids: List[str]) -> bool:
        """Returns True if all given doc_ids are already present in the collection."""
        if not doc_ids:
            return True
        try:
            res = self._collection.get(ids=doc_ids)
            return len(res.get("ids", [])) == len(doc_ids)
        except Exception:
            return False

    def index(self, documents: List[Document], embeddings: np.ndarray) -> None:
        if not documents:
            return
        if len(documents) != len(embeddings):
            raise ValueError(f"Document count ({len(documents)}) does not match embeddings count ({len(embeddings)})")

        # Check existing IDs in Chroma to avoid redundant embedding insertions
        existing_ids = set()
        try:
            sample_ids = [d.doc_id for d in documents[:1000]]
            check_res = self._collection.get(ids=sample_ids)
            if check_res and "ids" in check_res:
                existing_ids = set(check_res["ids"])
        except Exception:
            existing_ids = set()

        to_add_ids = []
        to_add_docs = []
        to_add_embs = []
        to_add_meta = []

        for doc, emb in zip(documents, embeddings):
            if doc.doc_id in existing_ids:
                continue
            to_add_ids.append(doc.doc_id)
            to_add_docs.append(doc.full_text)
            to_add_embs.append(emb.tolist())
            to_add_meta.append({
                "title": str(doc.title or "")[:200],
                "source": str(doc.source or "")[:100],
                "year": int(doc.year or 2024),
                "quality": float(doc.quality_score or 0.5),
            })

        if not to_add_ids:
            logger.info(f"All {len(documents)} documents already cached in ChromaDB collection '{self.collection_name}'.")
            return

        # Add in batches of 500 to ChromaDB
        batch_size = 500
        total = len(to_add_ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            self._collection.add(
                ids=to_add_ids[i:end_idx],
                documents=to_add_docs[i:end_idx],
                embeddings=to_add_embs[i:end_idx],
                metadatas=to_add_meta[i:end_idx],
            )
        logger.info(f"Indexed {len(to_add_ids)} new documents into ChromaDB collection '{self.collection_name}'.")

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        if self._collection.count() == 0 or top_k <= 0:
            return []

        q_list = query_embedding.tolist()
        if isinstance(q_list[0], (int, float)):
            q_list = [q_list]

        n_res = min(top_k, self._collection.count())
        res = self._collection.query(query_embeddings=q_list, n_results=n_res)

        out: List[Tuple[str, float]] = []
        if res and "ids" in res and res["ids"]:
            ret_ids = res["ids"][0]
            ret_dists = res.get("distances", [[]])[0] if "distances" in res else [0.0] * len(ret_ids)
            for doc_id, dist in zip(ret_ids, ret_dists):
                # In Chroma with cosine space, distance is (1 - cosine_similarity).
                sim_score = max(0.0, min(1.0, 1.0 - float(dist)))
                out.append((doc_id, sim_score))
        return out

    def count(self) -> int:
        return self._collection.count()


class FAISSVectorStore(BaseVectorStore):
    """In-memory FAISS flat inner-product vector store."""

    def __init__(self, dim: int = 384) -> None:
        import faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._doc_ids: List[str] = []

    def index(self, documents: List[Document], embeddings: np.ndarray) -> None:
        import faiss
        if not documents:
            return
        xb = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(xb)
        self._index.add(xb)
        self._doc_ids.extend([d.doc_id for d in documents])

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        import faiss
        if self._index.ntotal == 0 or top_k <= 0:
            return []
        xq = np.asarray(query_embedding, dtype="float32")
        if xq.ndim == 1:
            xq = np.expand_dims(xq, axis=0)
        faiss.normalize_L2(xq)
        n_res = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(xq, n_res)
        out = []
        for sc, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._doc_ids):
                out.append((self._doc_ids[idx], float(sc)))
        return out

    def count(self) -> int:
        return self._index.ntotal


class NumpyVectorStore(BaseVectorStore):
    """In-memory NumPy matrix multiplication vector store."""

    def __init__(self) -> None:
        self._embeddings: Optional[np.ndarray] = None
        self._doc_ids: List[str] = []

    def index(self, documents: List[Document], embeddings: np.ndarray) -> None:
        self._doc_ids = [d.doc_id for d in documents]
        self._embeddings = np.asarray(embeddings, dtype=float)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        if self._embeddings is None or len(self._doc_ids) == 0 or top_k <= 0:
            return []
        q = np.asarray(query_embedding, dtype=float)
        if q.ndim == 2:
            q = q[0]
        scores = (q @ self._embeddings.T)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in indices]

    def count(self) -> int:
        return len(self._doc_ids)
