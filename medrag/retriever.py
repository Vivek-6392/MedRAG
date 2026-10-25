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


# ── BM25 ──────────────────────────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self) -> None:
        self._bm25 = None
        self._docs: List[Document] = []

    def index(self, documents: List[Document]) -> None:
        from rank_bm25 import BM25Okapi
        self._docs = documents
        tokenised = [doc.full_text.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenised)

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        if self._bm25 is None:
            return []
        scores: np.ndarray = self._bm25.get_scores(query.lower().split())
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedDocument(
                document=self._docs[i],
                score=float(scores[i]),
                bm25_rank=rank + 1,
            )
            for rank, i in enumerate(indices)
        ]


# ── Dense ─────────────────────────────────────────────────────────────────────

class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._docs: List[Document] = []

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def index(self, documents: List[Document]) -> None:
        self._load()
        self._docs = documents
        texts = [doc.full_text for doc in documents]
        self._embeddings = self._model.encode(
            texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        if self._embeddings is None:
            return []
        self._load()
        q_emb = self._model.encode([query], normalize_embeddings=True)
        scores: np.ndarray = (q_emb @ self._embeddings.T)[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedDocument(
                document=self._docs[i],
                score=float(scores[i]),
                dense_rank=rank + 1,
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
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        if not documents:
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
    """Weighted RRF combining BM25 and Dense rankings."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, RetrievedDocument] = {}

    for rank, rdoc in enumerate(bm25_results):
        did = rdoc.document.doc_id
        scores[did] = scores.get(did, 0.0) + bm25_weight / (k + rank + 1)
        doc_map[did] = rdoc

    for rank, rdoc in enumerate(dense_results):
        did = rdoc.document.doc_id
        scores[did] = scores.get(did, 0.0) + dense_weight / (k + rank + 1)
        doc_map.setdefault(did, rdoc)

    return [
        RetrievedDocument(
            document=doc_map[did].document,
            score=sc,
            bm25_rank=doc_map[did].bm25_rank,
            dense_rank=doc_map[did].dense_rank,
        )
        for did, sc in sorted(scores.items(), key=lambda x: -x[1])
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
    ) -> None:
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(dense_model)
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
        bm25_hits = self.bm25.retrieve(query, top_k=self.top_k_retrieve)
        dense_hits = self.dense.retrieve(query, top_k=self.top_k_retrieve)

        merged = reciprocal_rank_fusion(
            bm25_hits, dense_hits, self.bm25_weight, self.dense_weight
        )

        if self.use_reranker and self.reranker:
            candidates = merged[: self.top_k_retrieve]
            return self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)

        return merged[: self.top_k_rerank]
