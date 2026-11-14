"""
Regression tests for Part D: Clinical Risk Policy & Threshold Adaptation.
Covers:
- Risk tier classifications: LOW, MODERATE, HIGH, CRITICAL.
- Word-boundary matching to prevent false substring triggers.
- Safe handling of malformed/NaN/negative risk inputs.
- Effective threshold adaptation based on risk level.
- Conservative abstention policy under high-risk scenarios.
"""
import unittest

from medrag.risk_policy import ClinicalRiskPolicy, RiskTier
from medrag.evidence_state import EvidenceState, ClaimEvidence, EvidencePassage
from medrag.controller import EvidenceController, Action


class TestClinicalRiskPolicy(unittest.TestCase):

    def test_word_boundary_prevents_substring_false_positive(self):
        # "adrenal" contains "renal", but should NOT trigger renal impairment high risk!
        assessment = ClinicalRiskPolicy.assess_query("Does the adrenal gland produce cortisol?")
        self.assertEqual(assessment.tier, RiskTier.LOW)
        self.assertNotIn("renal_impairment", assessment.factors)

        # "renal failure" must trigger high risk
        assessment_renal = ClinicalRiskPolicy.assess_query("Is this safe in renal failure?")
        self.assertIn(assessment_renal.tier, [RiskTier.HIGH, RiskTier.CRITICAL])
        self.assertIn("renal_impairment", assessment_renal.factors)

    def test_critical_risk_tier_triggers(self):
        assessment = ClinicalRiskPolicy.assess_query("What is the acute protocol for anaphylaxis in the ICU?")
        self.assertEqual(assessment.tier, RiskTier.CRITICAL)
        self.assertEqual(assessment.score, 1.0)
        self.assertIn("anaphylaxis", assessment.factors)

    def test_malformed_risk_score_fallback(self):
        # NaN, infinity, string, None, negative numbers must be safely sanitized
        self.assertEqual(ClinicalRiskPolicy.sanitize_risk_score(float("nan"), default=0.5), 0.5)
        self.assertEqual(ClinicalRiskPolicy.sanitize_risk_score(float("inf"), default=0.5), 0.5)
        self.assertEqual(ClinicalRiskPolicy.sanitize_risk_score("invalid_string", default=0.5), 0.5)
        self.assertEqual(ClinicalRiskPolicy.sanitize_risk_score(None, default=0.5), 0.5)
        self.assertEqual(ClinicalRiskPolicy.sanitize_risk_score(-0.2), 0.0)
        self.assertEqual(ClinicalRiskPolicy.sanitize_risk_score(1.5), 1.0)

    def test_effective_thresholds_stricter_for_high_risk(self):
        # Low risk: thresholds remain base
        sup_low, con_low, allow_partial_low = ClinicalRiskPolicy.compute_effective_thresholds(
            base_support=0.70, base_conflict=0.35, risk_score=0.30
        )
        self.assertEqual(sup_low, 0.70)
        self.assertEqual(con_low, 0.35)
        self.assertTrue(allow_partial_low)

        # High risk (risk=0.85): stricter support (+0.05), lower conflict tolerance (-0.10), NO partial
        sup_high, con_high, allow_partial_high = ClinicalRiskPolicy.compute_effective_thresholds(
            base_support=0.70, base_conflict=0.35, risk_score=0.85
        )
        self.assertEqual(sup_high, 0.75)
        self.assertEqual(con_high, 0.25)
        self.assertFalse(allow_partial_high)

    def test_controller_abstains_on_unresolved_claims_when_high_risk(self):
        controller = EvidenceController(
            support_threshold=0.70,
            conflict_threshold=0.35,
            high_risk_threshold=0.75,
            max_search_rounds=1,
        )
        # 1 supported claim, 1 missing claim, but clinical risk is HIGH (0.85)
        p1 = EvidencePassage(
            doc_id="doc1",
            doc_title="Title",
            source="NEJM",
            year=2024,
            quality_score=0.9,
            text="Documented evidence supporting claim",
            entailment=0.95,
            contradiction=0.01,
            neutral=0.04,
        )
        c1 = ClaimEvidence(
            "Supported finding",
            support_score=0.90,
            clinical_risk=0.85,
            supporting_passages=[p1],
            reference_year=2024,
        )
        c2 = ClaimEvidence("Missing finding", support_score=0.10, missing=True, clinical_risk=0.85)
        state = EvidenceState([c1, c2], round_number=1)

        # At round 1 (budget exhausted), normal low-risk would allow PARTIAL,
        # but high-risk policy MUST ABSTAIN
        decision = controller.decide(state, current_round=1, max_rounds=1)
        self.assertEqual(decision.action, Action.ABSTAIN)
        self.assertIn("Clinical risk tier", decision.reason)


if __name__ == "__main__":
    unittest.main()

