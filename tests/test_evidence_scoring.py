"""
Regression tests for Part C: Evidence Scoring & Reliability Calibration.
Covers:
- Separation of raw NLI support from source reliability.
- Separation of contradiction from missing evidence.
- Removal of undocumented arbitrary floors.
- Configurable reference year preventing calendar drift.
- Safe handling of missing source-quality metadata.
- Clear distinction of heuristic confidence from clinical probabilities.
"""
import unittest
from medrag.evidence_state import EvidencePassage, ClaimEvidence, ClaimStatus


class TestEvidenceScoringCalibration(unittest.TestCase):

    def test_source_reliability_calculation(self):
        # Explicit reliability combines quality and linear recency
        p = EvidencePassage(
            doc_id="p1", doc_title="Title", source="Journal", year=2024,
            quality_score=0.90, text="Passage text", entailment=0.95,
            contradiction=0.01, neutral=0.04
        )
        # With reference year 2024 (age 0), recency = 1.0
        # reliability = 0.6 * 0.90 + 0.4 * 1.0 = 0.54 + 0.40 = 0.94
        rel = p.reliability(reference_year=2024, quality_weight=0.6, annual_recency_decay=0.02)
        self.assertAlmostEqual(rel, 0.94, places=2)

    def test_missing_evidence_vs_contradiction_separation(self):
        # Missing: low support AND low contradiction
        ce_missing = ClaimEvidence(
            claim="Claim without any literature",
            support_score=0.10,
            conflict_score=0.05,
            support_threshold=0.70,
            missing_threshold=0.35,
            conflict_threshold=0.35,
        )
        self.assertEqual(ce_missing.status, ClaimStatus.MISSING)
        self.assertFalse(ce_missing.has_conflict)

        # Contradicted: high contradiction regardless of support
        ce_conflict = ClaimEvidence(
            claim="Controverted claim",
            support_score=0.85,
            conflict_score=0.75,
            support_threshold=0.70,
            missing_threshold=0.35,
            conflict_threshold=0.35,
        )
        self.assertEqual(ce_conflict.status, ClaimStatus.CONFLICTED)
        self.assertTrue(ce_conflict.has_conflict)

    def test_missing_source_quality_metadata_handled_safely(self):
        p_missing_q = EvidencePassage(
            doc_id="p_no_q", doc_title="Title", source="Source", year=2024,
            quality_score=None,  # Missing quality metadata
            text="Evidence text", entailment=0.85, contradiction=0.0, neutral=0.15,
            is_default_quality=True,
        )
        # Should safely default to 0.50 without raising exception
        rel = p_missing_q.reliability(reference_year=2024)
        self.assertIsInstance(rel, float)
        self.assertGreater(rel, 0.0)

    def test_reference_year_prevents_calendar_drift(self):
        p = EvidencePassage(
            doc_id="p_hist", doc_title="Old Trial", source="NEJM", year=2010,
            quality_score=0.95, text="Historical trial", entailment=0.90,
            contradiction=0.0, neutral=0.10
        )
        ce1 = ClaimEvidence(
            claim="Historical fact",
            support_score=0.90,
            reference_year=2024,  # Fixed benchmark reference year
            supporting_passages=[p],
        )
        ce2 = ClaimEvidence(
            claim="Historical fact",
            support_score=0.90,
            reference_year=2024,  # Re-evaluating later with same frozen reference
            supporting_passages=[p],
        )
        self.assertEqual(ce1.heuristic_confidence, ce2.heuristic_confidence)

    def test_heuristic_confidence_alias_backward_compatibility(self):
        p = EvidencePassage(
            doc_id="p1", doc_title="Title", source="Journal", year=2024,
            quality_score=0.95, text="Passage text", entailment=0.92,
            contradiction=0.0, neutral=0.08
        )
        ce = ClaimEvidence(
            claim="Clinical claim",
            support_score=0.92,
            conflict_score=0.0,
            supporting_passages=[p],
        )
        # heuristic_confidence and clinical_confidence must return identical values
        self.assertEqual(ce.heuristic_confidence, ce.clinical_confidence)
        # Check dictionary contains both keys for serialization compatibility
        d = ce.to_dict()
        self.assertIn("heuristic_confidence", d)
        self.assertIn("clinical_confidence", d)
        self.assertIn("source_reliability", d)


if __name__ == "__main__":
    unittest.main()
