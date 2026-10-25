"""
Evidence State — Structured per-claim evidence assessment
Support • Conflict • Missing • Recency • Source Quality • Clinical Risk
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONFLICTED = "conflicted"
    MISSING = "missing"
    UNCERTAIN = "uncertain"


@dataclass
class ClaimEvidence:
    """Evidence assessment for a single atomic claim."""
    claim: str
    support_score: float            # NLI entailment (0–1)
    conflict_score: float           # NLI contradiction (0–1)
    best_source: str = ""           # title of strongest supporting doc
    recency_year: int = 2020
    source_quality: float = 0.5     # 0–1 journal quality
    clinical_risk: float = 0.5      # 0–1 risk level of this claim
    missing: bool = False           # True when no document provides support

    # ── derived properties ──────────────────────────────────────────────────

    @property
    def status(self) -> ClaimStatus:
        if self.support_score >= 0.65 and self.conflict_score < 0.35:
            return ClaimStatus.SUPPORTED
        if self.conflict_score >= 0.50:
            return ClaimStatus.CONFLICTED
        if self.support_score < 0.35:
            return ClaimStatus.MISSING
        return ClaimStatus.UNCERTAIN

    @property
    def effective_support(self) -> float:
        """Support adjusted for conflict severity and source recency."""
        age_penalty = max(0.0, (2026 - self.recency_year) * 0.015)
        conflict_penalty = self.conflict_score * 0.4
        return max(0.0, self.support_score - conflict_penalty - age_penalty)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "support": round(self.support_score, 3),
            "conflict": round(self.conflict_score, 3),
            "effective_support": round(self.effective_support, 3),
            "recency_year": self.recency_year,
            "source_quality": round(self.source_quality, 2),
            "clinical_risk": round(self.clinical_risk, 2),
            "best_source": self.best_source,
        }

    def __repr__(self) -> str:
        return (
            f"ClaimEvidence({self.claim[:50]!r}, "
            f"status={self.status.value}, "
            f"support={self.support_score:.2f}, "
            f"conflict={self.conflict_score:.2f})"
        )


@dataclass
class EvidenceState:
    """
    Aggregated evidence state across all atomic claims for one QA round.
    Drives the SEARCH / STOP / ABSTAIN controller decision.
    """
    claim_evidences: List[ClaimEvidence] = field(default_factory=list)
    round_number: int = 0

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
    def max_clinical_risk(self) -> float:
        if not self.claim_evidences:
            return 0.0
        return max(c.clinical_risk for c in self.claim_evidences)

    @property
    def missing_claims(self) -> List[ClaimEvidence]:
        return [c for c in self.claim_evidences if c.status == ClaimStatus.MISSING]

    @property
    def conflicted_claims(self) -> List[ClaimEvidence]:
        return [c for c in self.claim_evidences if c.status == ClaimStatus.CONFLICTED]

    @property
    def uncertain_claims(self) -> List[ClaimEvidence]:
        return [c for c in self.claim_evidences if c.status == ClaimStatus.UNCERTAIN]

    @property
    def weakest_claims(self) -> List[ClaimEvidence]:
        """Claims that still need more evidence (missing | conflicted | uncertain)."""
        return [
            c for c in self.claim_evidences
            if c.status in (ClaimStatus.MISSING, ClaimStatus.CONFLICTED, ClaimStatus.UNCERTAIN)
        ]

    # ── sufficiency check ───────────────────────────────────────────────────

    def is_sufficient(
        self,
        support_threshold: float = 0.70,
        conflict_threshold: float = 0.35,
    ) -> bool:
        """True iff every claim is well-supported and no claim has significant conflict."""
        if not self.claim_evidences:
            return False
        all_supported = all(
            c.support_score >= support_threshold for c in self.claim_evidences
        )
        no_conflict = all(
            c.conflict_score < conflict_threshold for c in self.claim_evidences
        )
        return all_supported and no_conflict

    # ── serialisation ───────────────────────────────────────────────────────

    def summary_dict(self) -> dict:
        return {
            "round": self.round_number,
            "total_claims": len(self.claim_evidences),
            "supported": len([c for c in self.claim_evidences if c.status == ClaimStatus.SUPPORTED]),
            "conflicted": len(self.conflicted_claims),
            "missing": len(self.missing_claims),
            "uncertain": len(self.uncertain_claims),
            "mean_support": round(self.mean_support, 3),
            "max_conflict": round(self.max_conflict, 3),
            "max_clinical_risk": round(self.max_clinical_risk, 3),
        }

    def detailed_dict(self) -> dict:
        return {
            **self.summary_dict(),
            "claims": [c.to_dict() for c in self.claim_evidences],
        }
