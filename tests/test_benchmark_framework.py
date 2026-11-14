"""
Unit and Integration Tests for Reusable Benchmark Evaluation Framework.
Tests:
- BenchmarkExample and PredictionRecord schemas
- Common DatasetAdapter contract and concrete adapters (synthetic, medqa, pubmedqa)
- BenchmarkRegistry registration, lookup, and error handling
- ExperimentManifest provenance and prompt hashing
- PredictionRecorder streaming, artifact layout, and resume capability
- MetricRegistry evaluation calculations (accuracy, selective accuracy, abstention, AURC)
- EvaluationRunner end-to-end execution and resume recovery with synthetic fixtures
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from medrag.benchmarks import (
    BenchmarkExample,
    BenchmarkRegistry,
    EvaluationRunner,
    ExperimentConfig,
    MetricRegistry,
    PredictionRecord,
    PredictionRecorder,
    RetrievalTrace,
    SyntheticAdapter,
    MedQAAdapter,
    PubMedQAAdapter,
)
from medrag.evidence_state import ClaimEvidence, EvidenceState
from medrag.pipeline import PipelineResult
from medrag.retriever import Document, RetrievedDocument


class TestBenchmarkSchemasAndAdapters(unittest.TestCase):

    def test_benchmark_example_formatted_input(self):
        ex = BenchmarkExample(
            example_id="q1",
            dataset="medqa",
            split="test",
            question="What is the first-line treatment for hypertension in a diabetic patient?",
            options={"A": "ACE inhibitor", "B": "Beta-blocker", "C": "Loop diuretic"},
            gold_answer="A",
        )
        prompt = ex.formatted_input()
        self.assertIn("Question: What is the first-line treatment", prompt)
        self.assertIn("A. ACE inhibitor", prompt)
        self.assertIn("B. Beta-blocker", prompt)
        self.assertIn("C. Loop diuretic", prompt)

    def test_synthetic_adapter_normalization_and_answer_extraction(self):
        adapter = SyntheticAdapter()
        examples = adapter.load_split("test", limit=3)
        self.assertEqual(len(examples), 3)

        ex0 = examples[0]
        self.assertEqual(ex0.example_id, "syn_001")
        self.assertEqual(ex0.gold_answer, "A")
        self.assertIn("Metformin", ex0.options["A"])

        # Answer normalization
        self.assertEqual(adapter.normalize_answer("The correct option is A.", ex0), "A")
        self.assertEqual(adapter.normalize_answer("Answer: B", ex0), "B")
        self.assertEqual(adapter.normalize_answer("Metformin is contraindicated", ex0), "A")
        self.assertIsNone(adapter.normalize_answer("Insufficient evidence to conclude.", ex0))

        # Evaluation
        self.assertTrue(adapter.evaluate_prediction("A", "A", ex0))
        self.assertFalse(adapter.evaluate_prediction("B", "A", ex0))
        self.assertFalse(adapter.evaluate_prediction(None, "A", ex0))

    def test_medqa_adapter_fallback_and_options(self):
        adapter = MedQAAdapter()
        examples = adapter.load_split("test")
        self.assertGreater(len(examples), 0)
        ex = examples[0]
        self.assertEqual(ex.dataset, "medqa")
        self.assertIn(ex.gold_answer, ["A", "B", "C", "D"])

        # Multiple choice extraction
        self.assertEqual(adapter.normalize_answer("**A.** Occult bleeding", ex), "A")
        self.assertEqual(adapter.normalize_answer("Choice (B) is the right answer", ex), "B")

    def test_pubmedqa_adapter_normalization_and_evaluation(self):
        adapter = PubMedQAAdapter()
        examples = adapter.load_split("test")
        self.assertGreater(len(examples), 0)
        ex0 = examples[0]
        self.assertEqual(ex0.dataset, "pubmedqa")
        self.assertIn(ex0.gold_answer, ["yes", "no", "maybe"])

        # Normalization
        self.assertEqual(adapter.normalize_answer("Yes, preoperative statins help.", ex0), "yes")
        self.assertEqual(adapter.normalize_answer("No, routine drainage is not needed.", ex0), "no")
        self.assertEqual(adapter.normalize_answer("The result is maybe.", ex0), "maybe")
        self.assertEqual(adapter.normalize_answer("Option A", ex0), "yes")

        # Evaluation
        self.assertTrue(adapter.evaluate_prediction("yes", "yes", ex0))
        self.assertFalse(adapter.evaluate_prediction("no", "yes", ex0))


class TestBenchmarkRegistry(unittest.TestCase):

    def test_registered_adapters_lookup(self):
        available = BenchmarkRegistry.list_available()
        self.assertIn("synthetic", available)
        self.assertIn("medqa", available)
        self.assertIn("pubmedqa", available)

        adapter = BenchmarkRegistry.get("synthetic")
        self.assertIsInstance(adapter, SyntheticAdapter)

    def test_unknown_adapter_raises_key_error(self):
        with self.assertRaises(KeyError) as ctx:
            BenchmarkRegistry.get("nonexistent_dataset")
        self.assertIn("Available benchmarks", str(ctx.exception))


class TestPredictionRecorderAndResume(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_recorder_streaming_and_resume_detection(self):
        recorder = PredictionRecorder(run_dir=self.test_dir, experiment_id="test_exp_01")
        self.assertTrue(os.path.exists(recorder.run_dir))
        self.assertTrue(os.path.exists(recorder.logs_dir))

        # Initial state: empty
        self.assertEqual(len(recorder.completed_ids), 0)
        self.assertFalse(recorder.is_completed("ex_1"))

        # Write prediction record
        rec1 = PredictionRecord(
            example_id="ex_1",
            dataset="synthetic",
            split="test",
            question="Question 1?",
            gold_answer="A",
            predicted_answer="A",
            raw_model_response="Answer: A",
            normalized_answer="A",
            is_correct=True,
            abstained=False,
            latency_seconds=0.45,
        )
        recorder.record_prediction(rec1)
        self.assertTrue(recorder.is_completed("ex_1"))

        # Write retrieval trace
        trace = RetrievalTrace(
            example_id="ex_1",
            round_number=1,
            search_query="Question 1",
            retrieved_doc_ids=["doc_1"],
            retrieval_scores={"doc_1": 0.88},
            new_docs_count=1,
            total_docs_count=1,
        )
        recorder.record_retrieval_traces([trace])

        # Save config and metrics
        cfg = ExperimentConfig(
            experiment_id="test_exp_01",
            benchmark_name="synthetic",
            split="test",
            output_dir=self.test_dir,
        )
        recorder.save_config(cfg)
        self.assertTrue(os.path.exists(recorder.config_path))

        recorder.save_metrics({"accuracy": 1.0})
        self.assertTrue(os.path.exists(recorder.metrics_path))

        # Re-instantiate recorder to verify resume detection from disk
        resumed_recorder = PredictionRecorder(run_dir=self.test_dir, experiment_id="test_exp_01")
        self.assertTrue(resumed_recorder.is_completed("ex_1"))
        self.assertFalse(resumed_recorder.is_completed("ex_2"))

        all_records = resumed_recorder.load_all_predictions()
        self.assertEqual(len(all_records), 1)
        self.assertEqual(all_records[0].example_id, "ex_1")
        self.assertTrue(all_records[0].is_correct)


class TestMetricRegistryAndAURC(unittest.TestCase):

    def test_metrics_calculation(self):
        records = [
            PredictionRecord(
                example_id="1", dataset="syn", split="test", question="Q1",
                gold_answer="A", predicted_answer="A", normalized_answer="A",
                is_correct=True, abstained=False, latency_seconds=1.0, rounds=1,
                heuristic_evidence_score=0.90,
            ),
            PredictionRecord(
                example_id="2", dataset="syn", split="test", question="Q2",
                gold_answer="B", predicted_answer="C", normalized_answer="C",
                is_correct=False, abstained=False, latency_seconds=2.0, rounds=2,
                heuristic_evidence_score=0.70,
            ),
            PredictionRecord(
                example_id="3", dataset="syn", split="test", question="Q3",
                gold_answer="C", predicted_answer=None, normalized_answer=None,
                is_correct=False, abstained=True, latency_seconds=0.5, rounds=1,
                heuristic_evidence_score=0.20,
            ),
            PredictionRecord(
                example_id="4", dataset="syn", split="test", question="Q4",
                gold_answer="D", predicted_answer="D", normalized_answer="D",
                is_correct=True, abstained=False, latency_seconds=1.5, rounds=1,
                heuristic_evidence_score=0.85,
            ),
        ]

        metrics = MetricRegistry.compute_all(records)

        self.assertEqual(metrics.total_examples, 4)
        self.assertEqual(metrics.evaluated_count, 3)
        self.assertEqual(metrics.abstained_count, 1)
        self.assertEqual(metrics.correct_count, 2)
        self.assertEqual(metrics.incorrect_count, 2)

        # Raw accuracy: 2 / 4 = 0.50
        self.assertAlmostEqual(metrics.accuracy, 0.50, places=2)
        # Selective accuracy: 2 / 3 = 0.6667
        self.assertAlmostEqual(metrics.selective_accuracy, 0.6667, places=2)
        # Abstention rate: 1 / 4 = 0.25
        self.assertAlmostEqual(metrics.abstention_rate, 0.25, places=2)
        # Coverage: 3 / 4 = 0.75
        self.assertAlmostEqual(metrics.coverage, 0.75, places=2)

        self.assertIsNotNone(metrics.aurc)
        self.assertAlmostEqual(metrics.mean_retrieval_rounds, 1.25, places=2)


class TestEvaluationRunnerEndToEnd(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_runner_execution_and_resume_recovery(self):
        config = ExperimentConfig(
            experiment_id="test_runner_e2e",
            benchmark_name="synthetic",
            split="test",
            limit=3,
            output_dir=self.test_dir,
            llm_backend="mock",
            resume=True,
        )

        # Mock MedRAG Pipeline to return deterministic predictions without neural models
        mock_pipeline = MagicMock()

        def fake_answer(prompt: str) -> PipelineResult:
            doc = Document(
                doc_id="mock_doc_1",
                title="Mock Doc",
                text="Evidence passage text",
                source="Mock",
                year=2024,
            )
            rdoc = RetrievedDocument(document=doc, score=0.95, bm25_rank=1)
            # Produce supported answer matching option A
            ev_state = EvidenceState(
                claim_evidences=[
                    ClaimEvidence("Mock claim statement", support_score=0.95, clinical_risk=0.2)
                ],
                round_number=1,
            )
            return PipelineResult(
                question=prompt,
                final_answer="The correct answer is Option A. Evidence supports this.",
                action="STOP",
                evidence_state=ev_state,
                rounds=1,
                retrieved_docs=[rdoc],
                history=[{"round": 1, "query": "Mock query", "new_docs": 1, "total_docs": 1, "reason": "Verified"}],
                latency_seconds=0.10,
            )

        mock_pipeline.answer = MagicMock(side_effect=fake_answer)

        runner = EvaluationRunner(
            config=config,
            pipeline=mock_pipeline,
        )

        # Run evaluation
        metrics = runner.run()

        self.assertEqual(metrics.total_examples, 3)
        self.assertGreaterEqual(metrics.accuracy, 0.0)

        # Check generated directory artifacts
        run_path = os.path.join(self.test_dir, "test_runner_e2e")
        self.assertTrue(os.path.exists(os.path.join(run_path, "config.yaml")))
        self.assertTrue(os.path.exists(os.path.join(run_path, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(run_path, "predictions.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(run_path, "retrieval_traces.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(run_path, "metrics.json")))

        # Check that 3 calls were made
        self.assertEqual(mock_pipeline.answer.call_count, 3)

        # Now test RESUME: Running again with same config should skip all 3 examples!
        mock_pipeline.answer.reset_mock()
        runner2 = EvaluationRunner(
            config=config,
            pipeline=mock_pipeline,
        )
        metrics2 = runner2.run()
        # No new calls to answer() because all 3 were already completed
        self.assertEqual(mock_pipeline.answer.call_count, 0)
        self.assertEqual(metrics2.total_examples, 3)


if __name__ == "__main__":
    unittest.main()
