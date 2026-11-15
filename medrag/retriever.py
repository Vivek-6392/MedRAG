"""
Phase 2 — Evidence-Aware Hybrid Retrieval
BM25 + Dense Retrieval → Reciprocal Rank Fusion → Cross-Encoder Reranking
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Document ──────────────────────────────────────────────────────────────────

@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""
    year: int = 2020
    quality_score: float = 0.5          # 0–1 journal / source quality
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class RetrievedDocument:
    document: Document
    score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


# ── BM25 ──────────────────────────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self) -> None:
        self._bm25 = None
        self._docs: List[Document] = []

    def index(self, documents: List[Document]) -> None:
        self._docs = documents
        if not documents:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        tokenised = [doc.full_text.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenised)

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        if self._bm25 is None or top_k <= 0 or not query or not query.strip():
            return []
        tokens = query.lower().split()
        if not tokens:
            return []
        scores: np.ndarray = self._bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedDocument(
                document=self._docs[i],
                score=float(scores[i]),
                bm25_rank=rank + 1,
                bm25_score=float(scores[i]),
            )
            for rank, i in enumerate(indices)
        ]


# ── Dense ─────────────────────────────────────────────────────────────────────

class DenseRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_backend: str = "chromadb",
        collection_name: str = "medrag_corpus",
        persist_dir: Optional[str] = "./data/chroma_db",
    ) -> None:
        self.model_name = model_name
        self.vector_backend = (vector_backend or "chromadb").lower()
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._docs: List[Document] = []
        self._docs_by_id: Dict[str, Document] = {}
        self._store = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                # Cache-first / offline-friendly loading to prevent network timeouts
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name)

    def _init_store(self) -> None:
        if self._store is not None:
            return
        if self.vector_backend == "chromadb":
            try:
                from .vector_store import ChromaVectorStore
                self._store = ChromaVectorStore(
                    collection_name=self.collection_name,
                    persist_dir=self.persist_dir,
                )
            except Exception:
                from .vector_store import NumpyVectorStore
                self._store = NumpyVectorStore()
        elif self.vector_backend == "faiss":
            try:
                from .vector_store import FAISSVectorStore
                self._store = FAISSVectorStore(dim=384)
            except Exception:
                from .vector_store import NumpyVectorStore
                self._store = NumpyVectorStore()
        else:
            from .vector_store import NumpyVectorStore
            self._store = NumpyVectorStore()

    def index(self, documents: List[Document]) -> None:
        self._docs = documents
        self._docs_by_id = {doc.doc_id: doc for doc in documents}
        if not documents:
            self._embeddings = None
            return
        self._load()
        self._init_store()

        # If ChromaDB collection already contains all documents, skip redundant re-embedding
        if self.vector_backend == "chromadb" and self._store is not None and hasattr(self._store, "has_ids"):
            try:
                sample_check = [documents[0].doc_id, documents[-1].doc_id]
                if self._store.count() >= len(documents) and self._store.has_ids(sample_check):
                    print(f"  [DenseRetriever] ChromaDB collection '{self.collection_name}' already indexed ({self._store.count()} vectors cached). Reusing index.")
                    return
            except Exception:
                pass

        texts = [doc.full_text for doc in documents]
        self._embeddings = self._model.encode(
            texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        )
        if self._store is not None:
            self._store.index(documents, self._embeddings)

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        if not self._docs or top_k <= 0 or not query or not query.strip():
            return []
        self._load()
        self._init_store()
        q_emb = self._model.encode([query.strip()], normalize_embeddings=True)
        if self._store is not None:
            scored_ids = self._store.search(q_emb, top_k=top_k)
            results = []
            for rank, (doc_id, score) in enumerate(scored_ids):
                doc = self._docs_by_id.get(doc_id)
                if doc is not None:
                    results.append(
                        RetrievedDocument(
                            document=doc,
                            score=float(score),
                            dense_rank=rank + 1,
                            dense_score=float(score),
                        )
                    )
            return results

        if self._embeddings is None:
            return []
        scores: np.ndarray = (q_emb @ self._embeddings.T)[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedDocument(
                document=self._docs[i],
                score=float(scores[i]),
                dense_rank=rank + 1,
                dense_score=float(scores[i]),
            )
            for rank, i in enumerate(indices)
        ]


# ── Cross-Encoder Reranker ────────────────────────────────────────────────────

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            try:
                # Cache-first / offline-friendly loading to prevent network timeouts
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        if not documents or top_k <= 0:
            return []
        self._load()
        pairs = [(query, rdoc.document.full_text[:512]) for rdoc in documents]
        scores = self._model.predict(pairs)
        for rdoc, sc in zip(documents, scores):
            rdoc.rerank_score = float(sc)
        reranked = sorted(documents, key=lambda x: x.rerank_score or 0.0, reverse=True)
        return reranked[:top_k]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: List[RetrievedDocument],
    dense_results: List[RetrievedDocument],
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
    k: int = 60,
) -> List[RetrievedDocument]:
    """
    Weighted Reciprocal Rank Fusion (RRF) combining BM25 and Dense candidate rankings.

    Guarantees:
    - Merges documents retrieved by both modalities.
    - Preserves both `bm25_rank` and `dense_rank` accurately on merged records.
    - Preserves raw component scores (`bm25_score`, `dense_score`).
    - Deterministic tie-breaking on identical RRF scores.
    - Removes internal duplicate document IDs.
    - Handles empty BM25 or Dense sets gracefully.
    """
    if not bm25_results and not dense_results:
        return []

    doc_map: Dict[str, Document] = {}
    bm25_ranks: Dict[str, int] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_scores: Dict[str, float] = {}
    dense_scores: Dict[str, float] = {}
    rrf_scores: Dict[str, float] = {}

    for rank, rdoc in enumerate(bm25_results):
        did = rdoc.document.doc_id
        if did not in doc_map:
            doc_map[did] = rdoc.document
        # Use recorded rank if present, else 1-based iteration rank
        b_rank = rdoc.bm25_rank if rdoc.bm25_rank is not None else (rank + 1)
        bm25_ranks[did] = b_rank
        bm25_scores[did] = rdoc.bm25_score if rdoc.bm25_score is not None else rdoc.score
        rrf_scores[did] = rrf_scores.get(did, 0.0) + bm25_weight / (k + b_rank)

    for rank, rdoc in enumerate(dense_results):
        did = rdoc.document.doc_id
        if did not in doc_map:
            doc_map[did] = rdoc.document
        d_rank = rdoc.dense_rank if rdoc.dense_rank is not None else (rank + 1)
        dense_ranks[did] = d_rank
        dense_scores[did] = rdoc.dense_score if rdoc.dense_score is not None else rdoc.score
        rrf_scores[did] = rrf_scores.get(did, 0.0) + dense_weight / (k + d_rank)

    # Sort deterministically: highest RRF score first; break ties by best individual rank, then doc_id
    def _sort_key(did: str):
        best_rank = min(bm25_ranks.get(did, 999999), dense_ranks.get(did, 999999))
        return (-rrf_scores[did], best_rank, did)

    sorted_ids = sorted(doc_map.keys(), key=_sort_key)

    return [
        RetrievedDocument(
            document=doc_map[did],
            score=round(rrf_scores[did], 6),
            bm25_rank=bm25_ranks.get(did),
            dense_rank=dense_ranks.get(did),
            bm25_score=bm25_scores.get(did),
            dense_score=dense_scores.get(did),
        )
        for did in sorted_ids
    ]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Phase 2: BM25 + Dense → RRF Fusion → Cross-Encoder Reranking
    """

    def __init__(
        self,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        use_reranker: bool = True,
        vector_backend: str = "chromadb",
        collection_name: str = "medrag_corpus",
        persist_dir: Optional[str] = "./data/chroma_db",
    ) -> None:
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(
            model_name=dense_model,
            vector_backend=vector_backend,
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        self.reranker = CrossEncoderReranker(reranker_model) if use_reranker else None
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.use_reranker = use_reranker

    def index(self, documents: List[Document]) -> None:
        print(f"  Indexing {len(documents)} documents with BM25 + Dense...")
        self.bm25.index(documents)
        self.dense.index(documents)
        print("  Indexing complete.")

    def retrieve(self, query: str) -> List[RetrievedDocument]:
        if (
            not query
            or not query.strip()
            or self.top_k_retrieve <= 0
            or self.top_k_rerank <= 0
        ):
            return []
        bm25_hits = self.bm25.retrieve(query, top_k=self.top_k_retrieve)
        dense_hits = self.dense.retrieve(query, top_k=self.top_k_retrieve)

        merged = reciprocal_rank_fusion(
            bm25_hits, dense_hits, self.bm25_weight, self.dense_weight
        )

        if self.use_reranker and self.reranker:
            candidates = merged[: self.top_k_retrieve]
            return self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)

        return merged[: self.top_k_rerank]
