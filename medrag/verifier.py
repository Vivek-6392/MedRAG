"""
Phase 4 — Claim-Level Verification
NLI-based scoring of every atomic claim against retrieved documents.
Produces a structured EvidenceState.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .retriever import RetrievedDocument
from .evidence_state import ClaimEvidence, EvidenceState


class NLIVerifier:
    """
    Scores each atomic claim against every retrieved document using NLI.

    Model: cross-encoder/nli-MiniLM2-L6-H768
    Label order: [contradiction, entailment, neutral]
    """

    # Label indices for cross-encoder/nli-MiniLM2-L6-H768
    IDX_CONTRADICTION = 0
    IDX_ENTAILMENT = 1
    IDX_NEUTRAL = 2

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    # ── model loading ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is None:
            print(f"  Loading NLI model: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, num_labels=3)

    # ── NLI scoring ──────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _score_pairs(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[float, float, float]]:
        """
        Returns list of (entailment, contradiction, neutral) probability tuples.
        """
        self._load()
        results = []
        for i in range(0, len(pairs), self.batch_size):
            batch = [(p[:512], h[:256]) for p, h in pairs[i : i + self.batch_size]]
            raw_scores = self._model.predict(batch)
            for score_vec in raw_scores:
                probs = self._softmax(np.array(score_vec, dtype=float))
                ent = float(probs[self.IDX_ENTAILMENT])
                con = float(probs[self.IDX_CONTRADICTION])
                neu = float(probs[self.IDX_NEUTRAL])
                results.append((ent, con, neu))
        return results

    # ── public API ───────────────────────────────────────────────────────────

    def verify_claims(
        self,
        claims: List[str],
        retrieved_docs: List[RetrievedDocument],
        clinical_risk: float = 0.5,
    ) -> EvidenceState:
        """
        Verify each claim against all retrieved documents.

        For every claim c and document d, we run NLI(d.text, c).
        We take:
          support_score  = max entailment across all docs
          conflict_score = max contradiction across all docs
          best_source    = document with highest entailment
        """
        if not retrieved_docs:
            # No docs at all → all claims missing
            evidences = [
                ClaimEvidence(
                    claim=c,
                    support_score=0.0,
                    conflict_score=0.0,
                    missing=True,
                    clinical_risk=clinical_risk,
                )
                for c in claims
            ]
            return EvidenceState(claim_evidences=evidences)

        claim_evidences: List[ClaimEvidence] = []

        # Build all (premise=doc, hypothesis=claim) pairs in one flat list
        # then reshape results back to [claim][doc]
        flat_pairs: List[Tuple[str, str]] = []
        for claim in claims:
            for rdoc in retrieved_docs:
                flat_pairs.append((rdoc.document.full_text, claim))

        all_scores = self._score_pairs(flat_pairs)

        n_docs = len(retrieved_docs)
        for ci, claim in enumerate(claims):
            start = ci * n_docs
            claim_scores = all_scores[start : start + n_docs]

            entailments = [s[0] for s in claim_scores]
            contradictions = [s[1] for s in claim_scores]

            max_ent = max(entailments)
            max_con = max(contradictions)
            best_idx = int(np.argmax(entailments))
            best_doc = retrieved_docs[best_idx].document

            claim_evidences.append(
                ClaimEvidence(
                    claim=claim,
                    support_score=max_ent,
                    conflict_score=max_con,
                    best_source=best_doc.title,
                    recency_year=best_doc.year,
                    source_quality=best_doc.quality_score,
                    clinical_risk=clinical_risk,
                    missing=(max_ent < 0.30),
                    best_snippet=best_doc.text[:350],
                )
            )

        return EvidenceState(claim_evidences=claim_evidences)
