"""
Phase 4 — Claim-Level Verification
NLI-based multi-granularity scoring of every atomic claim against retrieved documents.
Produces a structured EvidenceState preserving supporting, contradicting, and neutral passages.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

from .retriever import Document, RetrievedDocument
from .evidence_state import ClaimEvidence, EvidencePassage, EvidenceState
from .generator import _extract_specific_keywords


class NLIVerifier:
    """
    Scores each atomic claim against every retrieved document using NLI.

    Model: cross-encoder/nli-MiniLM2-L6-H768
    Label order: [contradiction, entailment, neutral]
    """

    IDX_CONTRADICTION = 0
    IDX_ENTAILMENT = 1
    IDX_NEUTRAL = 2

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    # ── model loading ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is None:
            print(f"  Loading NLI model: {self.model_name}")
            from sentence_transformers import CrossEncoder
            try:
                # Cache-first / offline-friendly loading to prevent network timeouts
                self._model = CrossEncoder(self.model_name, num_labels=3, local_files_only=True)
            except Exception:
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
        if not pairs:
            return []
        self._load()
        results = []
        for i in range(0, len(pairs), self.batch_size):
            batch = [(p[:1000], h[:512]) for p, h in pairs[i : i + self.batch_size]]
            raw_scores = np.asarray(self._model.predict(batch), dtype=float)
            if raw_scores.shape != (len(batch), 3) or not np.isfinite(raw_scores).all():
                raise ValueError("NLI model returned malformed or non-finite scores.")
            for score_vec in raw_scores:
                probs = self._softmax(np.array(score_vec, dtype=float))
                ent = float(probs[self.IDX_ENTAILMENT])
                con = float(probs[self.IDX_CONTRADICTION])
                neu = float(probs[self.IDX_NEUTRAL])
                results.append((ent, con, neu))
        return results

    @staticmethod
    def _extract_premises(doc: Document) -> List[Tuple[str, Document]]:
        """
        Extracts multi-granularity premises: full document plus context-aware sentences.
        MiniLM NLI performs best on sentence-level inputs with subject context.
        Returns list of (premise_text, document) tuples.
        """
        premises: List[Tuple[str, Document]] = [(doc.full_text, doc)]
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.text) if len(s.strip()) > 15]
        for s in sents:
            p_text = f"{doc.title}. {s}" if doc.title else s
            premises.append((p_text, doc))
        return premises

    # ── public API ───────────────────────────────────────────────────────────

    def verify_claims(
        self,
        claims: List[str],
        retrieved_docs: List[RetrievedDocument],
        clinical_risk: float = 0.5,
        support_threshold: float = 0.70,
        conflict_threshold: float = 0.35,
        missing_threshold: float = 0.35,
        round_number: int = 1,
        clinical_confidence_threshold: float = 0.65,
    ) -> EvidenceState:
        """
        Verify each claim against all retrieved documents.

        Evaluates multi-granularity premises per document (full text + context-aware sentences).
        Guarantees:
        1. Support and contradiction are evaluated INDEPENDENTLY. A supporting passage does not
           hide contradictory evidence elsewhere.
        2. If no actual support exists, best_source is NOT set to an arbitrary document.
        3. All supporting, contradicting, and neutral passages are preserved with full metadata.
        """
        if any(not isinstance(claim, str) for claim in claims):
            raise TypeError("claims must contain only strings.")
        claims = [claim.strip() for claim in claims if claim.strip()]
        if not claims:
            return EvidenceState(
                claim_evidences=[],
                round_number=round_number,
                support_threshold=support_threshold,
                conflict_threshold=conflict_threshold,
                clinical_confidence_threshold=clinical_confidence_threshold,
            )

        if not retrieved_docs:
            evidences = [
                ClaimEvidence(
                    claim=c,
                    support_score=0.0,
                    conflict_score=0.0,
                    best_source="",
                    best_snippet="No documents retrieved to evaluate this claim.",
                    missing=True,
                    clinical_risk=clinical_risk,
                    support_threshold=support_threshold,
                    conflict_threshold=conflict_threshold,
                    missing_threshold=missing_threshold,
                    clinical_confidence_threshold=clinical_confidence_threshold,
                )
                for c in claims
            ]
            return EvidenceState(
                claim_evidences=evidences,
                round_number=round_number,
                support_threshold=support_threshold,
                conflict_threshold=conflict_threshold,
                clinical_confidence_threshold=clinical_confidence_threshold,
            )

        # Pre-extract all premises across all retrieved documents
        all_doc_premises: List[Tuple[str, Document]] = []
        for rdoc in retrieved_docs:
            all_doc_premises.extend(self._extract_premises(rdoc.document))

        # Build flat (premise, claim) pairs
        flat_pairs: List[Tuple[str, str]] = []
        claim_ranges: List[Tuple[int, int]] = []

        for claim in claims:
            start = len(flat_pairs)
            for p_text, _ in all_doc_premises:
                flat_pairs.append((p_text, claim))
            end = len(flat_pairs)
            claim_ranges.append((start, end))

        all_scores = self._score_pairs(flat_pairs)

        claim_evidences: List[ClaimEvidence] = []

        for ci, claim in enumerate(claims):
            c_start, c_end = claim_ranges[ci]
            c_scores = all_scores[c_start:c_end]
            # First pass: collect all candidate supporting passages
            temp_passages: List[Tuple[EvidencePassage, float, float, float]] = []

            for (p_text, doc), (ent, con, neu) in zip(all_doc_premises, c_scores):
                q_score = getattr(doc, "quality_score", None)
                is_default_q = q_score is None or not (0.0 <= float(q_score) <= 1.0)
                final_q = float(q_score) if not is_default_q else 0.50
                passage = EvidencePassage(
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    source=doc.source,
                    year=doc.year,
                    quality_score=final_q,
                    text=p_text,
                    entailment=ent,
                    contradiction=con,
                    neutral=neu,
                    illustrative=bool(doc.metadata.get("illustrative", False)),
                    is_default_quality=is_default_q,
                )
                temp_passages.append((passage, ent, con, neu))

            supporting_passages: List[EvidencePassage] = []
            contradicting_passages: List[EvidencePassage] = []
            neutral_passages: List[EvidencePassage] = []

            claim_kw = set(_extract_specific_keywords(claim))
            for passage, ent, con, neu in temp_passages:
                # Apply the same configured thresholds used for claim status.
                is_supporting = ent >= support_threshold

                # A passage only contradicts if it shares topical relevance/entity with the claim
                has_overlap = True
                if claim_kw:
                    p_tokens = set(re.findall(r"\b[a-zA-Z0-9-]{3,}\b", passage.text.lower()))
                    has_overlap = bool(claim_kw & p_tokens)

                is_contradicting = (
                    has_overlap and con >= conflict_threshold and con > ent and con > neu
                )
                if is_supporting:
                    supporting_passages.append(passage)

                # Evaluate contradiction independently, including passages from a
                # document that contains other supporting passages.
                if is_contradicting:
                    contradicting_passages.append(passage)

                if not is_supporting and not is_contradicting:
                    neutral_passages.append(passage)

            # Sort passages by respective confidence
            supporting_passages.sort(key=lambda p: -p.entailment)
            contradicting_passages.sort(key=lambda p: -p.contradiction)
            neutral_passages.sort(key=lambda p: -p.neutral)

            # Independent maximum scores
            max_support = max((ent for _, ent, _, _ in temp_passages), default=0.0)
            max_conflict = contradicting_passages[0].contradiction if contradicting_passages else 0.0

            # Assign best supporting source ONLY when actual credible support exists
            is_missing = (
                max_support < missing_threshold
                and max_conflict < conflict_threshold
            )
            if supporting_passages:
                top_sup = supporting_passages[0]
                best_source = top_sup.doc_title
                recency_year = top_sup.year
                source_quality = top_sup.quality_score
                best_snippet = top_sup.text[:350]
            else:
                best_source = ""
                best_snippet = "No supporting evidence found in retrieved literature."
                recency_year = 2024
                source_quality = 0.5

            source_label = best_source or "none"
            evidence = ClaimEvidence(
                    claim=claim,
                    support_score=max_support,
                    conflict_score=max_conflict,
                    best_source=best_source,
                    recency_year=recency_year,
                    source_quality=source_quality,
                    clinical_risk=clinical_risk,
                    missing=is_missing,
                    best_snippet=best_snippet,
                    supporting_passages=supporting_passages,
                    contradicting_passages=contradicting_passages,
                    neutral_passages=neutral_passages,
                    support_threshold=support_threshold,
                    conflict_threshold=conflict_threshold,
                    missing_threshold=missing_threshold,
                    clinical_confidence_threshold=clinical_confidence_threshold,
                )
            print(
                f"  [Verifier] {evidence.status.value}: {claim[:90]!r} "
                f"(support={max_support:.2f}/{support_threshold:.2f}, "
                f"clinical confidence={evidence.clinical_confidence:.2f}/"
                f"{clinical_confidence_threshold:.2f}, "
                f"conflict={max_conflict:.2f}/{conflict_threshold:.2f}, "
                f"supporting source={source_label})"
            )
            claim_evidences.append(evidence)

        return EvidenceState(
            claim_evidences=claim_evidences,
            round_number=round_number,
            support_threshold=support_threshold,
            conflict_threshold=conflict_threshold,
            clinical_confidence_threshold=clinical_confidence_threshold,
        )
