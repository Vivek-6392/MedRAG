"""
Clinical Risk Policy and Verification Requirements Engine.
Defines risk tiers, deterministic risk classification with word-boundary analysis,
safe handling of malformed risk inputs, and risk-adjusted decision thresholds.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple


class RiskTier(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    tier: RiskTier
    score: float
    factors: List[str]
    is_fallback: bool = False
    warning: Optional[str] = None


class ClinicalRiskPolicy:
    """
    Evaluates clinical question risk and produces risk-adjusted verification thresholds.
    Ensures high-risk clinical scenarios enforce stricter support requirements and
    conservative abstention.
    """

    # Compiled regex with word boundaries to prevent substring collisions (e.g. 'renal' in 'adrenal')
    CRITICAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b(anaphylaxis|anaphylactic)\b", re.I), "anaphylaxis"),
        (re.compile(r"\b(cardiac arrest|asystole|ventricular fibrillation)\b", re.I), "cardiac_arrest"),
        (re.compile(r"\b(overdose|acute poisoning|toxicity)\b", re.I), "overdose"),
        (re.compile(r"\b(sepsis|septic shock)\b", re.I), "sepsis"),
        (re.compile(r"\b(status epilepticus)\b", re.I), "status_epilepticus"),
        (re.compile(r"\b(icu|intensive care|critically ill)\b", re.I), "critical_care"),
        (re.compile(r"\b(fatal|life-threatening|emergency)\b", re.I), "emergency"),
    ]

    HIGH_RISK_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b(pregnan(cy|t)?|lactat(ion|ing)|breastfeed(ing)?)\b", re.I), "maternal_fetal"),
        (re.compile(r"\b(pediatric|paediatric|infant|neonate|neonatal|child|toddler)\b", re.I), "pediatric"),
        (re.compile(r"\b(renal (failure|impairment)|kidney (disease|failure)|ckd|egfr)\b", re.I), "renal_impairment"),
        (re.compile(r"\b(hepatic (failure|impairment)|cirrhosis|liver failure)\b", re.I), "hepatic_impairment"),
        (re.compile(r"\b(anticoagula(nt|tion)|warfarin|doac|apixaban|rivaroxaban|dabigatran|heparin)\b", re.I), "anticoagulation"),
        (re.compile(r"\b(chemotherap(y|eutic)|cytotoxic|immunosuppress(ive|ion))\b", re.I), "immunosuppression"),
        (re.compile(r"\b(elderly|geriatric|older adults?|aged 65\+|aged 60\+)\b", re.I), "geriatric"),
        (re.compile(r"\b(contraindicat(ed|ion)|black box warning|adverse drug reaction)\b", re.I), "contraindication"),
    ]

    MODERATE_RISK_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b(dos(e|age|ing)|titrat(e|ion)|frequency|milligram|mg)\b", re.I), "dosing"),
        (re.compile(r"\b(drug interaction|interaction between)\b", re.I), "drug_interaction"),
        (re.compile(r"\b(adverse effect|side effect|tolerability)\b", re.I), "side_effects"),
    ]

    @classmethod
    def sanitize_risk_score(cls, raw_risk: Any, default: float = 0.50) -> float:
        """
        Guarantees that a risk score is a finite float bounded between 0.0 and 1.0.
        Falls back safely on malformed, NaN, infinite, or non-numeric values.
        """
        if raw_risk is None:
            return default
        try:
            val = float(raw_risk)
            if not math.isfinite(val):
                return default
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return default

    @classmethod
    def score_to_tier(cls, score: float) -> RiskTier:
        s = cls.sanitize_risk_score(score)
        if s >= 0.90:
            return RiskTier.CRITICAL
        if s >= 0.70:
            return RiskTier.HIGH
        if s >= 0.40:
            return RiskTier.MODERATE
        return RiskTier.LOW

    @classmethod
    def assess_query(cls, question: str, intent: str = "general") -> RiskAssessment:
        """
        Assesses question text using multi-tiered regex patterns and intent context.
        """
        factors: List[str] = []

        # 1. Critical tier check
        for pattern, factor in cls.CRITICAL_PATTERNS:
            if pattern.search(question):
                factors.append(factor)
        if factors:
            return RiskAssessment(
                tier=RiskTier.CRITICAL,
                score=1.00,
                factors=factors,
            )

        # 2. High risk tier check
        for pattern, factor in cls.HIGH_RISK_PATTERNS:
            if pattern.search(question):
                factors.append(factor)
        if factors:
            return RiskAssessment(
                tier=RiskTier.HIGH,
                score=0.80,
                factors=factors,
            )

        # 3. Moderate tier check
        for pattern, factor in cls.MODERATE_RISK_PATTERNS:
            if pattern.search(question):
                factors.append(factor)
        
        intent_lower = (intent or "").lower()
        if intent_lower in ("dosage", "safety"):
            factors.append(f"intent_{intent_lower}")

        if factors:
            return RiskAssessment(
                tier=RiskTier.MODERATE,
                score=0.60,
                factors=factors,
            )

        # 4. Low risk tier (informational / physiology)
        return RiskAssessment(
            tier=RiskTier.LOW,
            score=0.30,
            factors=["general_inquiry"],
        )

    @classmethod
    def compute_effective_thresholds(
        cls,
        base_support: float,
        base_conflict: float,
        risk_score: float,
        high_risk_threshold: float = 0.75,
    ) -> Tuple[float, float, bool]:
        """
        Adjusts support and conflict thresholds dynamically based on risk level.
        Returns: (effective_support_threshold, effective_conflict_threshold, allow_partial)
        """
        sanitized_risk = cls.sanitize_risk_score(risk_score)
        is_high_risk = sanitized_risk >= high_risk_threshold

        if is_high_risk:
            # Conservative safety policy:
            # - Boost required support (stricter evidence threshold)
            # - Reduce conflict tolerance (zero/lower tolerance for contradiction)
            # - Strictly disallow partial answers
            eff_support = min(0.95, base_support + 0.05)
            eff_conflict = max(0.20, base_conflict - 0.10)
            allow_partial = False
        else:
            eff_support = base_support
            eff_conflict = base_conflict
            allow_partial = True

        return (round(eff_support, 3), round(eff_conflict, 3), allow_partial)
