"""
Evidence State — Structured per-claim evidence assessment
Support • Contradiction • Uncertainty • Provenance • Source Quality • Clinical Risk
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONFLICTED = "conflicted"
    MISSING = "missing"
    UNCERTAIN = "uncertain"


@dataclass
class EvidencePassage:
    """Individual passage evaluation with provenance metadata."""
    doc_id: str
    doc_title: str
    source: str
    year: int
    quality_score: float
    text: str
    entailment: float
    contradiction: float
    neutral: float
    illustrative: bool = False
    is_default_quality: bool = False

    def reliability(
        self,
        reference_year: Optional[int] = None,
        quality_weight: float = 0.6,
        annual_recency_decay: float = 0.02,
    ) -> float:
        """
        Explicit source reliability combining normalized quality score and temporal recency.
        Does not use arbitrary hidden floors.
        """
        ref_year = reference_year or datetime.now().year
        age = max(0, ref_year - self.year)
        recency = max(0.0, 1.0 - (annual_recency_decay * age))
        q = max(0.0, min(1.0, self.quality_score if self.quality_score is not None else 0.5))
        w = max(0.0, min(1.0, quality_weight))
        return round(w * q + (1.0 - w) * recency, 4)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source": self.source,
            "year": self.year,
            "quality_score": round(self.quality_score, 2),
            "text": self.text,
            "entailment": round(self.entailment, 3),
            "contradiction": round(self.contradiction, 3),
            "neutral": round(self.neutral, 3),
            "illustrative": self.illustrative,
            "is_default_quality": self.is_default_quality,
        }


@dataclass
class ClaimEvidence:
    """
    Evidence assessment for a single atomic clinical claim.
    Separates raw NLI support from source reliability and contradiction.
    """
    claim: str
    support_score: float = 0.0          # Max NLI entailment (0–1) across supporting evidence
    conflict_score: float = 0.0         # Max NLI contradiction (0–1) across material contradictions
    best_source: str = ""               # title of strongest supporting doc (empty if no support)
    recency_year: int = 2024
    source_quality: float = 0.5         # 0–1 journal / guideline quality
    clinical_risk: float = 0.5          # 0–1 risk level associated with this claim
    missing: bool = False               # True when no document provides credible support
    best_snippet: str = ""              # verified excerpt from the supporting document
    supporting_passages: List[EvidencePassage] = field(default_factory=list)
    contradicting_passages: List[EvidencePassage] = field(default_factory=list)
    neutral_passages: List[EvidencePassage] = field(default_factory=list)
    support_threshold: float = 0.70
    conflict_threshold: float = 0.35
    missing_threshold: float = 0.35
    clinical_confidence_threshold: float = 0.65

    # Configurable scoring parameters (no undocumented hardcoded constants)
    reference_year: Optional[int] = None
    quality_weight: float = 0.6
    annual_recency_decay: float = 0.02
    conflict_penalty_weight: float = 0.5

    # ── derived properties ──────────────────────────────────────────────────

    @property
    def status(self) -> ClaimStatus:
        """
        Determines claim status using unified configurable thresholds.
        Contradictions take precedence: a claim cannot be marked SUPPORTED if material
        contradictions exist in the literature.
        """
        if self.conflict_score >= self.conflict_threshold:
            # CONFLICTED means the literature disagrees (support AND contradiction).
            # Contradiction with no credible support is just an unverified claim.
            if self.support_score >= self.support_threshold:
                return ClaimStatus.CONFLICTED
            return ClaimStatus.UNCERTAIN
        if (
            self.support_score >= self.support_threshold
            and any(not passage.illustrative for passage in self.supporting_passages)
            and self.heuristic_confidence >= self.clinical_confidence_threshold
        ):
            return ClaimStatus.SUPPORTED
        if self.support_score < self.missing_threshold:
            return ClaimStatus.MISSING
        return ClaimStatus.UNCERTAIN

    @property
    def verification_status(self) -> str:
        """
        Explicit 4-way claim verification status:
        - SUPPORTED: credible supporting evidence without material contradiction
        - CONTRADICTED: credible contradicting evidence without sufficient support
        - CONFLICTED: contradictory signals in the literature (both support and contradiction)
        - INSUFFICIENT_EVIDENCE: lack of credible supporting or contradicting evidence
        """
        has_sup = (
            self.support_score >= self.support_threshold
            and any(not p.illustrative for p in self.supporting_passages)
        )
        has_con = (
            self.conflict_score >= self.conflict_threshold
            and any(not p.illustrative for p in self.contradicting_passages)
        )
        if has_sup and has_con:
            return "CONFLICTED"
        if has_con:
            return "CONTRADICTED"
        if has_sup:
            return "SUPPORTED"
        return "INSUFFICIENT_EVIDENCE"

    @property
    def has_support(self) -> bool:
        return (
            self.support_score >= self.support_threshold
            and any(not passage.illustrative for passage in self.supporting_passages)
        )

    @property
    def has_conflict(self) -> bool:
        return self.conflict_score >= self.conflict_threshold

    @property
    def source_reliability(self) -> float:
        """Aggregate source reliability across non-illustrative supporting passages."""
        valid = [p for p in self.supporting_passages if not p.illustrative]
        if not valid:
            return 0.0
        return max(
            p.reliability(
                reference_year=self.reference_year,
                quality_weight=self.quality_weight,
                annual_recency_decay=self.annual_recency_decay,
            )
            for p in valid
        )

    @property
    def heuristic_confidence(self) -> float:
        """
        Heuristic confidence index combining NLI entailment with explicit source reliability
        and contradiction penalty.

        NOTE: This is an empirical heuristic index (0.0 to 1.0) for ranking and stopping,
        NOT a statistically calibrated clinical probability.
        """
        if self.support_score <= 0.0 or not self.supporting_passages:
            return 0.0

        passage_scores = []
        for passage in self.supporting_passages:
            if passage.illustrative:
                continue
            r = passage.reliability(
                reference_year=self.reference_year,
                quality_weight=self.quality_weight,
                annual_recency_decay=self.annual_recency_decay,
            )
            passage_scores.append(passage.entailment * r)

        if not passage_scores:
            return 0.0

        base_score = max(passage_scores)
        penalty = self.conflict_score * self.conflict_penalty_weight
        return max(0.0, min(1.0, round(base_score - penalty, 3)))

    @property
    def clinical_confidence(self) -> float:
        """Backward-compatible alias for heuristic_confidence."""
        return self.heuristic_confidence

    @property
    def effective_support(self) -> float:
        """Support adjusted for conflict severity and source recency."""
        ref_year = self.reference_year or datetime.now().year
        age_penalty = max(0.0, (ref_year - self.recency_year) * self.annual_recency_decay)
        conflict_penalty = self.conflict_score * 0.4
        return max(0.0, self.support_score - conflict_penalty - age_penalty)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "support": round(self.support_score, 3),
            "conflict": round(self.conflict_score, 3),
            "heuristic_confidence": self.heuristic_confidence,
            "clinical_confidence": self.clinical_confidence,
            "source_reliability": round(self.source_reliability, 3),
            "clinical_confidence_threshold": self.clinical_confidence_threshold,
            "effective_support": round(self.effective_support, 3),
            "recency_year": self.recency_year,
            "source_quality": round(self.source_quality, 2),
            "clinical_risk": round(self.clinical_risk, 2),
            "best_source": self.best_source,
            "best_snippet": self.best_snippet,
            "n_supporting": len(self.supporting_passages),
            "n_contradicting": len(self.contradicting_passages),
            "supporting_passages": [p.to_dict() for p in self.supporting_passages],
            "contradicting_passages": [p.to_dict() for p in self.contradicting_passages],
            "neutral_passages": [p.to_dict() for p in self.neutral_passages],
        }
        return {
            "claim": self.claim,
            "status": self.status.value,
            "support": round(self.support_score, 3),
            "conflict": round(self.conflict_score, 3),
            "clinical_confidence": self.clinical_confidence,
            "clinical_confidence_threshold": self.clinical_confidence_threshold,
            "effective_support": round(self.effective_support, 3),
            "recency_year": self.recency_year,
            "source_quality": round(self.source_quality, 2),
            "clinical_risk": round(self.clinical_risk, 2),
            "best_source": self.best_source,
            "best_snippet": self.best_snippet,
            "n_supporting": len(self.supporting_passages),
            "n_contradicting": len(self.contradicting_passages),
            "supporting_passages": [p.to_dict() for p in self.supporting_passages],
            "contradicting_passages": [p.to_dict() for p in self.contradicting_passages],
            "neutral_passages": [p.to_dict() for p in self.neutral_passages],
        }

    def __repr__(self) -> str:
        return (
            f"ClaimEvidence({self.claim[:50]!r}, "
            f"status={self.status.value}, "
            f"support={self.support_score:.2f}, "
            f"conflict={self.conflict_score:.2f}, "
            f"conf={self.clinical_confidence:.2f})"
        )


@dataclass
class EvidenceState:
    """
    Aggregated evidence state across all atomic claims for one QA round.
    Drives the STOP / PARTIAL / CONFLICTED / SEARCH / ABSTAIN controller decisions.
    """
    claim_evidences: List[ClaimEvidence] = field(default_factory=list)
    round_number: int = 1
    support_threshold: float = 0.70
    conflict_threshold: float = 0.35
    clinical_confidence_threshold: float = 0.65

    @property
    def claims(self) -> List[ClaimEvidence]:
        """Alias for claim_evidences."""
        return self.claim_evidences

    # ── aggregate statistics ────────────────────────────────────────────────

    @property
    def mean_support(self) -> float:
        if not self.claim_evidences:
            return 0.0
        return sum(c.support_score for c in self.claim_evidences) / len(self.claim_evidences)

    @property
    def max_conflict(self) -> float:
        if not self.claim_evidences:
            return 0.0
        return max(c.conflict_score for c in self.claim_evidences)

    @property
    def mean_confidence(self) -> float:
        if not self.claim_evidences:
            return 0.0
        return sum(c.clinical_confidence for c in self.claim_evidences) / len(self.claim_evidences)

    @property
    def max_clinical_risk(self) -> float:
        if not self.claim_evidences:
            return 0.0
        return max(c.clinical_risk for c in self.claim_evidences)

    @property
    def supported_claims(self) -> List[ClaimEvidence]:
        return [c for c in self.claim_evidences if c.status == ClaimStatus.SUPPORTED]

    @property
    def conflicted_claims(self) -> List[ClaimEvidence]:
        return [c for c in self.claim_evidences if c.status == ClaimStatus.CONFLICTED]

    @property
    def missing_claims(self) -> List[ClaimEvidence]:
        return [c for c in self.claim_evidences if c.status == ClaimStatus.MISSING]

    @property
    def uncertain_claims(self) -> List[ClaimEvidence]:
        return [c for c in self.claim_evidences if c.status == ClaimStatus.UNCERTAIN]

    @property
    def weakest_claims(self) -> List[ClaimEvidence]:
        """Claims that still require more evidence or clarification (missing | conflicted | uncertain)."""
        return [c for c in self.claim_evidences if c.status != ClaimStatus.SUPPORTED]

    # ── sufficiency check ───────────────────────────────────────────────────

    def is_sufficient(
        self,
        support_threshold: Optional[float] = None,
        conflict_threshold: Optional[float] = None,
        clinical_confidence_threshold: Optional[float] = None,
    ) -> bool:
        """True iff every claim meets support, conflict, and confidence criteria."""
        if not self.claim_evidences:
            return False
        sup_t = support_threshold if support_threshold is not None else self.support_threshold
        con_t = conflict_threshold if conflict_threshold is not None else self.conflict_threshold
        conf_t = (
            clinical_confidence_threshold
            if clinical_confidence_threshold is not None
            else self.clinical_confidence_threshold
        )
        all_supported = all(
            c.support_score >= sup_t
            and c.clinical_confidence >= conf_t
            and any(not passage.illustrative for passage in c.supporting_passages)
            for c in self.claim_evidences
        )
        no_conflict = all(c.conflict_score < con_t for c in self.claim_evidences)
        return all_supported and no_conflict

    # ── serialisation ───────────────────────────────────────────────────────

    def summary_dict(self) -> dict:
        return {
            "round": self.round_number,
            "total_claims": len(self.claim_evidences),
            "supported": len(self.supported_claims),
            "conflicted": len(self.conflicted_claims),
            "missing": len(self.missing_claims),
            "uncertain": len(self.uncertain_claims),
            "mean_support": round(self.mean_support, 3),
            "max_conflict": round(self.max_conflict, 3),
            "mean_confidence": round(self.mean_confidence, 3),
            "max_clinical_risk": round(self.max_clinical_risk, 3),
            "clinical_confidence_threshold": self.clinical_confidence_threshold,
        }

    def detailed_dict(self) -> dict:
        return {
            **self.summary_dict(),
            "claims": [c.to_dict() for c in self.claim_evidences],
        }
