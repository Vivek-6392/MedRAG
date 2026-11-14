"""
Phase 6 — Next-Action Decision Controller
Implements STOP / PARTIAL / CONFLICTED / SEARCH / ABSTAIN decision policy driven by EvidenceState.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .evidence_state import EvidenceState


class Action(str, Enum):
    STOP = "STOP"              # All claims verified with sufficient evidence, no material conflict
    VERIFIED = "VERIFIED"      # Explicit alias for STOP
    PARTIAL = "PARTIAL"        # Some claims verified, others remain missing or uncertain
    CONFLICTED = "CONFLICTED"  # Material contradictions detected across retrieved literature
    SEARCH = "SEARCH"          # Evidence gaps remain and retrieval budget allows further search
    ABSTAIN = "ABSTAIN"        # Evidence is insufficient, unreliable, or unsafe for a clinical recommendation


@dataclass
class ControllerDecision:
    action: Action
    reason: str
    confidence: float = 0.0     # clinical confidence of current evidence state
    gap_focus: str = ""         # target evidence gaps to resolve (SEARCH only)


class EvidenceController:
    """
    Evaluates EvidenceState and selects the next clinical action.

    Decision Outcomes:
    ──────────────────
    1. STOP / VERIFIED  — All claims meet the configured evidence criteria with no material conflict.
    2. SEARCH           — Rounds remain (< max_search_rounds) and unresolved claims exist.
    3. CONFLICTED       — Retrieved evidence contains material contradictions.
    4. PARTIAL          — Some claims verified; others remain uncertain/missing, and clinical risk permits
                          qualified disclosure.
    5. ABSTAIN          — Evidence remains insufficient, conflicting under high risk, or unsafe for recommendations.
    """

    def __init__(
        self,
        support_threshold: float = 0.70,
        conflict_threshold: float = 0.35,
        max_search_rounds: int = 3,
        high_risk_threshold: float = 0.75,
        high_risk_penalty: float = 0.15,
        clinical_confidence_threshold: float = 0.65,
    ) -> None:
        self.support_threshold = support_threshold
        self.conflict_threshold = conflict_threshold
        self.clinical_confidence_threshold = clinical_confidence_threshold
        self.max_search_rounds = max_search_rounds
        self.high_risk_threshold = high_risk_threshold
        self.high_risk_penalty = high_risk_penalty

    def decide(
        self,
        state: EvidenceState,
        current_round: Optional[int] = None,
        max_rounds: Optional[int] = None,
    ) -> ControllerDecision:
        rnd = current_round if current_round is not None else state.round_number
        budget = max_rounds if max_rounds is not None else self.max_search_rounds

        # 0. Edge case: empty claims
        if not state.claim_evidences:
            return ControllerDecision(
                action=Action.ABSTAIN,
                reason="No verifiable clinical claims were provided or extracted.",
                confidence=0.0,
            )

        from .risk_policy import ClinicalRiskPolicy

        eff_support, eff_conflict, allow_partial = ClinicalRiskPolicy.compute_effective_thresholds(
            base_support=self.support_threshold,
            base_conflict=self.conflict_threshold,
            risk_score=state.max_clinical_risk,
            high_risk_threshold=self.high_risk_threshold,
        )

        # ── 1. STOP / VERIFIED ────────────────────────────────────────────────
        if state.is_sufficient(
            eff_support,
            eff_conflict,
            self.clinical_confidence_threshold,
        ):
            return ControllerDecision(
                action=Action.STOP,
                reason=(
                    f"All {len(state.claims)} claims meet configured evidence criteria "
                    f"(mean support={state.mean_support:.2f}, max conflict={state.max_conflict:.2f}, "
                    f"effective support threshold={eff_support:.2f}, "
                    f"clinical confidence threshold={self.clinical_confidence_threshold:.2f}, "
                    f"clinical risk={state.max_clinical_risk:.2f})."
                ),
                confidence=state.mean_confidence,
            )

        # ── 2. SEARCH (if rounds remain) ─────────────────────────────────────
        if rnd < budget:
            weak = state.weakest_claims
            gap_focus = "; ".join(c.claim[:60] for c in weak[:3])
            return ControllerDecision(
                action=Action.SEARCH,
                reason=(
                    f"Round {rnd} of {budget}: {len(weak)} claim(s) need more evidence "
                    f"({len(state.missing_claims)} missing, {len(state.conflicted_claims)} conflicted, "
                    f"{len(state.uncertain_claims)} uncertain). Launching targeted retrieval."
                ),
                confidence=state.mean_confidence,
                gap_focus=gap_focus,
            )

        # ── 3. Final Determination (rounds exhausted: rnd >= budget) ─────────

        # 3a. CONFLICTED — Material contradictions in literature
        if len(state.conflicted_claims) > 0:
            return ControllerDecision(
                action=Action.CONFLICTED,
                reason=(
                    f"Retrieval budget reached ({budget} rounds). Material contradictions detected "
                    f"across retrieved evidence for {len(state.conflicted_claims)} claim(s). "
                    "Reporting the disagreement without selecting a clinical recommendation."
                ),
                confidence=state.mean_confidence,
            )

        # 3b. PARTIAL — Some claims verified, others missing/uncertain, risk permitting
        if len(state.supported_claims) > 0 and allow_partial:
            return ControllerDecision(
                action=Action.PARTIAL,
                reason=(
                    f"Retrieval budget reached ({budget} rounds). {len(state.supported_claims)} of "
                    f"{len(state.claims)} claims verified. {len(state.missing_claims) + len(state.uncertain_claims)} "
                    "claim(s) remain unresolved. Providing qualified partial answer."
                ),
                confidence=state.mean_confidence,
            )

        # 3c. ABSTAIN — Insufficient or unsafe evidence
        reason_detail = (
            f"Clinical risk tier ({state.max_clinical_risk:.2f} >= {self.high_risk_threshold:.2f}) "
            "strictly prohibits issuing incomplete recommendations."
            if not allow_partial and len(state.supported_claims) > 0
            else f"Insufficient evidence after {rnd} retrieval round(s) "
                 f"({len(state.missing_claims)} missing, {len(state.uncertain_claims)} uncertain, "
                 f"clinical risk={state.max_clinical_risk:.2f})."
        )
        return ControllerDecision(
            action=Action.ABSTAIN,
            reason=f"{reason_detail} Withholding clinical recommendation for patient safety.",
            confidence=state.mean_confidence,
        )
