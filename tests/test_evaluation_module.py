"""
Comprehensive Test Suite for Research-Grade Evaluation and Statistical Analysis Module.
Tests:
- PART A: Answer Quality (MedQA per-option metrics, PubMedQA per-class metrics, confusion matrices)
- PART B: Retrieval Quality (Recall@k, MRR, graded nDCG, MedQA guardrail check)
- PART C: Claim-Level Faithfulness (ClaimAnnotation schema, 4 labels, false-support, unsupported rates, citations)
- PART D: Abstention & Selective Answering (Coverage, selective risk, risk-coverage curve, AURC, validation tuning)
- PART E: Calibration (Brier score, ECE, MCE, reliability bins, temperature scaling)
- PART F: Statistical Reporting (Wilson score interval, bootstrap CIs, ComprehensiveStatisticalReport)
- PART G: Visualizations (Rendering publication figures in publication/figures/ as PNG and PDF)
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import tempfile
import unittest
import numpy as np

from medrag.benchmarks import (
    PredictionRecord,
    MetricRegistry,
    EvaluationMetrics,
    ClaimEvidenceLabel,
    ClaimAnnotation,
    ClaimEvaluationMetrics,
    evaluate_claim_faithfulness,
    load_claim_annotations,
    save_claim_annotations,
    ThresholdTuningResult,
    compute_risk_coverage_curve,
    tune_abstention_thresholds,
    CalibrationBin,
    CalibrationMetrics,
    TemperatureScaler,
    compute_brier_score,
    compute_expected_calibration_error,
    ComprehensiveStatisticalReport,
    compute_wilson_score_interval,
    generate_comprehensive_statistical_report,
    generate_all_publication_figures,
)


class TestAnswerEvaluationPartA(unittest.TestCase):
    """PART A: MedQA and PubMedQA Answer Evaluation Tests."""

    def test_medqa_per_option_performance(self):
        records = [
            PredictionRecord(
                example_id="q1", dataset="medqa", split="test",
                question="Q1", gold_answer="A", predicted_answer="A",
                normalized_answer="A", is_correct=True, abstained=False,
            ),
            PredictionRecord(
                example_id="q2", dataset="medqa", split="test",
                question="Q2", gold_answer="B", predicted_answer="B",
                normalized_answer="B", is_correct=True, abstained=False,
            ),
            PredictionRecord(
                example_id="q3", dataset="medqa", split="test",
                question="Q3", gold_answer="A", predicted_answer="B",
                normalized_answer="B", is_correct=False, abstained=False,
            ),
            PredictionRecord(
                example_id="q4", dataset="medqa", split="test",
                question="Q4", gold_answer="C", predicted_answer=None,
                normalized_answer=None, is_correct=False, abstained=True,
            ),
        ]
        metrics: EvaluationMetrics = MetricRegistry.compute_all(records)

        self.assertEqual(metrics.total_examples, 4)
        self.assertEqual(metrics.evaluated_count, 3)
        self.assertEqual(metrics.abstained_count, 1)
        self.assertAlmostEqual(metrics.accuracy, 0.50, places=2)
        self.assertAlmostEqual(metrics.selective_accuracy, 2 / 3, places=2)

        # Per-option metrics for MedQA options
        self.assertIn("A", metrics.per_option_performance)
        self.assertIn("B", metrics.per_option_performance)
        perf_a = metrics.per_option_performance["A"]
        self.assertEqual(perf_a["tp"], 1)
        self.assertEqual(perf_a["fn"], 1)
        self.assertEqual(perf_a["support"], 2)

        # Specificity
        self.assertIn("A", metrics.per_class_specificity)
        self.assertIn("B", metrics.per_class_f1)
        self.assertIsNotNone(metrics.macro_f1)

        # Confusion matrix checks
        matrix = metrics.confusion_matrix
        self.assertEqual(matrix["A"]["A"], 1)
        self.assertEqual(matrix["A"]["B"], 1)
        self.assertEqual(matrix["C"]["ABSTAIN"], 1)

    def test_pubmedqa_per_class_performance(self):
        records = [
            PredictionRecord(
                example_id="101", dataset="pubmedqa", split="test",
                question="P1", gold_answer="yes", predicted_answer="yes",
                normalized_answer="yes", is_correct=True, abstained=False,
            ),
            PredictionRecord(
                example_id="102", dataset="pubmedqa", split="test",
                question="P2", gold_answer="no", predicted_answer="no",
                normalized_answer="no", is_correct=True, abstained=False,
            ),
            PredictionRecord(
                example_id="103", dataset="pubmedqa", split="test",
                question="P3", gold_answer="maybe", predicted_answer="yes",
                normalized_answer="yes", is_correct=False, abstained=False,
            ),
        ]
        metrics: EvaluationMetrics = MetricRegistry.compute_all(records)
        self.assertIn("yes", metrics.per_class_precision)
        self.assertIn("no", metrics.per_class_recall)
        self.assertIn("maybe", metrics.per_class_specificity)
        self.assertIsNotNone(metrics.macro_f1)


class TestRetrievalEvaluationPartB(unittest.TestCase):
    """PART B: Retrieval Recall@k, MRR, Graded nDCG, and Guardrail Tests."""

    def test_pubmedqa_retrieval_and_mrr(self):
        records = [
            PredictionRecord(
                example_id="pmid_12345", dataset="pubmedqa", split="test",
                question="Retrieval query", gold_answer="yes",
                retrieved_doc_ids=["pmid_99999", "pmid_12345", "pmid_88888"],
                metadata={"protocol": "pubmedqa_retrieval"},
            ),
            PredictionRecord(
                example_id="pmid_67890", dataset="pubmedqa", split="test",
                question="Retrieval query 2", gold_answer="no",
                retrieved_doc_ids=["pmid_67890", "pmid_11111"],
                metadata={"protocol": "pubmedqa_retrieval"},
            ),
        ]
        metrics: EvaluationMetrics = MetricRegistry.compute_all(records)

        self.assertIsNotNone(metrics.retrieval_recall_at_k)
        recalls = metrics.retrieval_recall_at_k
        self.assertEqual(recalls["Recall@1"], 0.50)  # item 2 at rank 1, item 1 at rank 2
        self.assertEqual(recalls["Recall@5"], 1.00)  # both in top 5
        self.assertAlmostEqual(metrics.retrieval_mrr, (0.5 + 1.0) / 2.0, places=2)  # MRR = 0.75

    def test_graded_ndcg_at_k(self):
        records = [
            PredictionRecord(
                example_id="graded_q1", dataset="synthetic", split="test",
                question="Query with graded relevance", gold_answer="A",
                retrieved_doc_ids=["doc_med", "doc_high", "doc_low"],
                metadata={
                    "passage_relevance_grades": {"doc_high": 2.0, "doc_med": 1.0, "doc_low": 0.0}
                },
            ),
        ]
        metrics: EvaluationMetrics = MetricRegistry.compute_all(records)
        self.assertIsNotNone(metrics.retrieval_ndcg_at_k)
        self.assertTrue(metrics.retrieval_ndcg_at_k["nDCG@5"] > 0.0)

    def test_medqa_retrieval_guardrail_omits_when_no_passage_annotations(self):
        records = [
            PredictionRecord(
                example_id="medqa_001", dataset="medqa", split="test",
                question="USMLE question without passage annotations", gold_answer="A",
                predicted_answer="A", normalized_answer="A", is_correct=True,
                retrieved_doc_ids=["book1_p1", "book2_p2"],
                metadata={},  # No independent passage annotations!
            ),
        ]
        metrics: EvaluationMetrics = MetricRegistry.compute_all(records)
        # Guardrail: Must NOT invent retrieval metrics for MedQA from answer correctness!
        self.assertIsNone(metrics.retrieval_recall_at_k)
        self.assertIsNone(metrics.retrieval_mrr)
        self.assertIn("medqa_retrieval_evaluation_note", metrics.custom_metrics)


class TestClaimLevelEvaluationPartC(unittest.TestCase):
    """PART C: Claim-Level Faithfulness and Attribution Tests."""

    def test_claim_annotation_schema_and_serialization(self):
        ann = ClaimAnnotation(
            example_id="ex_1",
            claim_id="c1",
            claim="Metformin reduces HbA1c.",
            evidence_id="doc_1",
            label=ClaimEvidenceLabel.SUPPORTED,
            annotator_id="physician_1",
            adjudication_status="adjudicated",
        )
        self.assertEqual(ann.label, ClaimEvidenceLabel.SUPPORTED)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "annotations.jsonl")
            save_claim_annotations([ann], path)
            loaded = load_claim_annotations(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].claim, ann.claim)
            self.assertEqual(loaded[0].label, ClaimEvidenceLabel.SUPPORTED)

    def test_claim_faithfulness_metrics_calculation(self):
        annotations = [
            ClaimAnnotation(
                example_id="ex_1", claim_id="c1", claim="Metformin reduces HbA1c.",
                evidence_id="doc_1", label=ClaimEvidenceLabel.SUPPORTED,
            ),
            ClaimAnnotation(
                example_id="ex_1", claim_id="c2", claim="Metformin is safe in ESRD.",
                evidence_id="doc_2", label=ClaimEvidenceLabel.CONTRADICTED,
            ),
        ]

        predictions = [
            PredictionRecord(
                example_id="ex_1", dataset="medqa", split="test",
                question="Is metformin safe?", gold_answer="A",
                retrieved_doc_ids=["doc_1", "doc_2"],
                evidence_state={
                    "claims": [
                        {"claim_id": "c1", "claim": "Metformin reduces HbA1c.", "status": "supported", "cited_doc_ids": ["doc_1"]},
                        {"claim_id": "c2", "claim": "Metformin is safe in ESRD.", "status": "conflicted", "cited_doc_ids": ["doc_2"]},
                    ]
                },
            )
        ]

        metrics: ClaimEvaluationMetrics = evaluate_claim_faithfulness(predictions, annotations)
        self.assertEqual(metrics.annotated_subset_evaluated, 2)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.macro_f1, 1.0)
        self.assertEqual(metrics.false_support_rate, 0.0)
        self.assertEqual(metrics.citation_precision, 1.0)
        self.assertEqual(metrics.citation_completeness, 1.0)
        self.assertIn("manually annotated subset", metrics.annotation_subset_warning)


class TestAbstentionEvaluationPartD(unittest.TestCase):
    """PART D: Abstention, Selective Risk, and Validation Threshold Tuning Tests."""

    def test_risk_coverage_curve_and_aurc(self):
        records = [
            PredictionRecord(
                example_id="q1", dataset="synthetic", split="test", question="Q1",
                gold_answer="A", is_correct=True, heuristic_evidence_score=0.95,
            ),
            PredictionRecord(
                example_id="q2", dataset="synthetic", split="test", question="Q2",
                gold_answer="A", is_correct=True, heuristic_evidence_score=0.85,
            ),
            PredictionRecord(
                example_id="q3", dataset="synthetic", split="test", question="Q3",
                gold_answer="A", is_correct=False, heuristic_evidence_score=0.40,
            ),
            PredictionRecord(
                example_id="q4", dataset="synthetic", split="test", question="Q4",
                gold_answer="A", is_correct=False, heuristic_evidence_score=0.30,
            ),
        ]
        points, aurc = compute_risk_coverage_curve(records)
        self.assertTrue(len(points) > 0)
        self.assertIsNotNone(aurc)
        # Highest confidence items (q1, q2) are correct, so risk at lower coverage is 0.0
        self.assertEqual(points[0].selective_risk, 0.0)

    def test_validation_threshold_tuning(self):
        val_records = [
            PredictionRecord(example_id="v1", dataset="synthetic", split="val", question="Q", gold_answer="A", is_correct=True, heuristic_evidence_score=0.90),
            PredictionRecord(example_id="v2", dataset="synthetic", split="val", question="Q", gold_answer="A", is_correct=True, heuristic_evidence_score=0.80),
            PredictionRecord(example_id="v3", dataset="synthetic", split="val", question="Q", gold_answer="A", is_correct=False, heuristic_evidence_score=0.50),
            PredictionRecord(example_id="v4", dataset="synthetic", split="val", question="Q", gold_answer="A", is_correct=False, heuristic_evidence_score=0.20),
        ]
        tuning: ThresholdTuningResult = tune_abstention_thresholds(
            val_records, target_risk=0.00, min_coverage=0.50
        )
        self.assertIsNotNone(tuning.optimal_threshold)
        self.assertGreaterEqual(tuning.optimal_threshold, 0.70)
        self.assertEqual(tuning.val_selective_risk, 0.0)
        self.assertIn("validation data", tuning.methodology_note)


class TestCalibrationPartE(unittest.TestCase):
    """PART E: Confidence Calibration, ECE, Brier Score, and Temperature Scaling Tests."""

    def test_brier_and_ece_calculation(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        brier = compute_brier_score(y_true, y_prob)
        self.assertTrue(0.0 <= brier <= 0.05)

        ece, mce, bins = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
        self.assertTrue(0.0 <= ece <= 0.20)
        self.assertEqual(len(bins), 5)

    def test_temperature_scaling_optimization(self):
        val_probs = np.array([0.95, 0.90, 0.85, 0.70])
        # Overconfident: actually only 50% accurate
        val_labels = np.array([1, 0, 1, 0])

        scaler = TemperatureScaler()
        scaler.fit(val_probs, val_labels)
        self.assertTrue(scaler.fitted)
        self.assertGreater(scaler.temperature, 1.0)  # Temperature softens overconfident logits

        scaled_p = scaler.transform(np.array([0.95]))
        self.assertLess(scaled_p[0], 0.95)  # Softened probability


class TestStatisticalReportingPartF(unittest.TestCase):
    """PART F: Wilson Score CIs and Comprehensive Statistical Reporting Tests."""

    def test_wilson_score_interval(self):
        ci_low, ci_high = compute_wilson_score_interval(successes=80, total=100)
        self.assertTrue(0.70 <= ci_low <= 0.75)
        self.assertTrue(0.85 <= ci_high <= 0.90)

    def test_comprehensive_statistical_report_generation(self):
        records = [
            PredictionRecord(
                example_id=f"ex_{i}", dataset="medqa", split="test",
                question=f"Question {i}", gold_answer="A",
                is_correct=(i % 2 == 0), abstained=(i == 3), latency_seconds=0.5,
            )
            for i in range(10)
        ]
        rep: ComprehensiveStatisticalReport = generate_comprehensive_statistical_report(
            records=records, experiment_id="test_stat_report"
        )
        self.assertEqual(rep.total_examples, 10)
        self.assertIn("accuracy", rep.metrics)
        self.assertIn("coverage", rep.metrics)
        self.assertIn("selective_accuracy", rep.metrics)

        md = rep.to_markdown()
        self.assertIn("Statistical Analysis Report", md)
        self.assertIn("Wilson score interval", md)


class TestVisualizationsPartG(unittest.TestCase):
    """PART G: Publication-Quality Plots Rendering in publication/figures/."""

    def test_generate_all_publication_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = EvaluationMetrics(
                total_examples=10, accuracy=0.80, selective_accuracy=0.90,
                mean_latency_seconds=1.2, retrieval_recall_at_k={"Recall@1": 0.6, "Recall@5": 0.9, "Recall@10": 1.0},
                confusion_matrix={"A": {"A": 5, "B": 1}, "B": {"B": 4, "A": 0}},
            )
            m2 = EvaluationMetrics(
                total_examples=10, accuracy=0.60, selective_accuracy=0.70,
                mean_latency_seconds=0.8,
            )

            metrics_by_system = {"full_medrag": m1, "baseline_dense": m2}
            records_by_system = {
                "full_medrag": [
                    PredictionRecord(example_id="1", dataset="synthetic", split="test", question="Q", gold_answer="A", is_correct=True, heuristic_evidence_score=0.9, rounds=1, latency_seconds=1.0),
                    PredictionRecord(example_id="2", dataset="synthetic", split="test", question="Q", gold_answer="A", is_correct=True, heuristic_evidence_score=0.8, rounds=2, latency_seconds=1.4),
                ],
                "baseline_dense": [
                    PredictionRecord(example_id="1", dataset="synthetic", split="test", question="Q", gold_answer="A", is_correct=False, heuristic_evidence_score=0.5, rounds=1, latency_seconds=0.8),
                ]
            }

            claim_metrics = ClaimEvaluationMetrics(
                total_annotated_claims=4,
                macro_f1=0.85,
                per_label_precision={"supported": 0.9, "contradicted": 0.8},
                per_label_recall={"supported": 0.85, "contradicted": 0.8},
                per_label_f1={"supported": 0.87, "contradicted": 0.8},
            )

            calib_metrics = CalibrationMetrics(
                total_samples=10, brier_score=0.12, expected_calibration_error=0.08,
                maximum_calibration_error=0.15,
                bins=[
                    CalibrationBin(bin_idx=1, bin_lower=0.0, bin_upper=0.5, mean_confidence=0.3, empirical_accuracy=0.35, calibration_error=0.05, sample_count=5, weight=0.5),
                    CalibrationBin(bin_idx=2, bin_lower=0.5, bin_upper=1.0, mean_confidence=0.8, empirical_accuracy=0.75, calibration_error=0.05, sample_count=5, weight=0.5),
                ],
            )

            fig_map = generate_all_publication_figures(
                metrics_by_system=metrics_by_system,
                records_by_system=records_by_system,
                claim_metrics=claim_metrics,
                calib_metrics=calib_metrics,
                output_dir=tmpdir,
            )

            # Ensure plots exist and are non-empty
            for name, path in fig_map.items():
                self.assertTrue(os.path.exists(path), f"Figure {name} was not generated at {path}")
                self.assertGreater(os.path.getsize(path), 0, f"Figure {name} is empty")

                # Verify companion vector PDF also generated
                pdf_path = path.replace(".png", ".pdf")
                self.assertTrue(os.path.exists(pdf_path), f"Companion PDF {pdf_path} was not generated")


if __name__ == "__main__":
    unittest.main()
