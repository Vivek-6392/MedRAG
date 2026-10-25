"""
Phase 6 — Next-Action Decision Controller
Implements the SEARCH / STOP / ABSTAIN logic driven by EvidenceState.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .evidence_state import EvidenceState


class Action(str, Enum):
    STOP = "STOP"          # Evidence is sufficient — produce final answer
    ABSTAIN = "ABSTAIN"    # Evidence is still insufficient after max rounds
    SEARCH = "SEARCH"      # Retrieve again with a targeted gap query


@dataclass
class ControllerDecision:
    action: Action
    reason: str
    confidence: float = 0.0     # mean support of the current evidence state
    gap_focus: str = ""         # description of what to search (SEARCH only)


class EvidenceController:
    """
    Evaluates the current EvidenceState and decides what to do next.

    Decision tree
    ─────────────
    1. STOP    — all claims supported above threshold AND no major conflict
    2. ABSTAIN — retrieval round limit reached
               — OR: very high clinical risk + unresolvable conflicts (round ≥ 2)
    3. SEARCH  — evidence gaps / conflicts remain and rounds are available
    """

    def __init__(
        self,
        support_threshold: float = 0.70,
        conflict_threshold: float = 0.35,
        max_search_rounds: int = 3,
        high_risk_threshold: float = 0.75,
        high_risk_penalty: float = 0.15,
    ) -> None:
        self.support_threshold = support_threshold
        self.conflict_threshold = conflict_threshold
        self.max_search_rounds = max_search_rounds
        self.high_risk_threshold = high_risk_threshold
        self.high_risk_penalty = high_risk_penalty

    def decide(self, state: EvidenceState) -> ControllerDecision:
        s = state.summary_dict()
        rnd = state.round_number

        # Tighten threshold for high-risk queries
        eff_threshold = self.support_threshold
        if state.max_clinical_risk >= self.high_risk_threshold:
            eff_threshold += self.high_risk_penalty

        # ── 1. STOP ─────────────────────────────────────────────────────────
        if state.is_sufficient(eff_threshold, self.conflict_threshold):
            return ControllerDecision(
                action=Action.STOP,
                reason=(
                    f"All {s['total_claims']} claims are sufficiently supported "
                    f"(mean support={s['mean_support']:.2f}, "
                    f"max conflict={s['max_conflict']:.2f}). "
                    f"Threshold={eff_threshold:.2f}."
                ),
                confidence=state.mean_support,
            )

        # ── 2a. ABSTAIN — round limit ────────────────────────────────────────
        if rnd >= self.max_search_rounds:
            weak_n = s["missing"] + s["conflicted"] + s["uncertain"]
            return ControllerDecision(
                action=Action.ABSTAIN,
                reason=(
                    f"Retrieval limit reached ({self.max_search_rounds} rounds). "
                    f"{weak_n} claim(s) remain unresolved "
                    f"(missing={s['missing']}, conflicted={s['conflicted']}, "
                    f"uncertain={s['uncertain']}). "
                    "Insufficient evidence for a reliable clinical statement."
                ),
                confidence=state.mean_support,
            )

        # ── 2b. ABSTAIN — high-risk + persistent conflict ────────────────────
        if (
            state.max_clinical_risk >= 0.90
            and len(state.conflicted_claims) > 0
            and rnd >= 2
        ):
            return ControllerDecision(
                action=Action.ABSTAIN,
                reason=(
                    f"Very high clinical risk ({state.max_clinical_risk:.2f}) with "
                    f"{len(state.conflicted_claims)} unresolved conflicting claim(s) "
                    "after multiple retrieval rounds. Abstaining for patient safety."
                ),
                confidence=0.10,
            )

        # ── 3. SEARCH ────────────────────────────────────────────────────────
        weak = state.weakest_claims
        gap_focus = "; ".join(c.claim[:50] for c in weak[:2])
        return ControllerDecision(
            action=Action.SEARCH,
            reason=(
                f"Round {rnd + 1}: {len(weak)} claim(s) need more evidence "
                f"(missing={s['missing']}, conflicted={s['conflicted']}, "
                f"uncertain={s['uncertain']}). Launching targeted retrieval."
            ),
            confidence=state.mean_support,
            gap_focus=gap_focus,
        )
