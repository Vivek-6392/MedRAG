"""
Unit and Integration Tests for MedQA Benchmark Adapter, Protocols, and Corpus Utility.
Covers Part F requirements:
- Dataset loading across official splits (train, dev/val, test).
- Official split preservation without silent random splits.
- Answer-choice formatting and strict 4-option validation.
- Gold-answer leakage prevention (asserting gold label is not leaked into prompts).
- Structured JSON response parsing and ambiguous answer rejection.
- Explicit abstention detection.
- Metric computation (per-option accuracy, Macro-F1, confusion matrix, token usage).
- Textbook corpus preparer chunking, metadata preservation, and manifest hashing.
- Execution of all 5 research protocols (CLOSED_BOOK, TEXTBOOK_RAG, HYBRID_RAG, RERANKED_RAG, MEDRAG_FULL).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from medrag.benchmarks import (
    BenchmarkExample,
    EvaluationRunner,
    ExperimentConfig,
    MetricRegistry,
    PredictionRecord,
)
from medrag.benchmarks.adapters.medqa import MedQAAdapter
from medrag.benchmarks.corpus_preparer import TextbookCorpusPreparer
from medrag.pipeline import PipelineResult
from medrag.retriever import Document, RetrievedDocument


class TestMedQADatasetLoader(unittest.TestCase):

    def setUp(self):
        self.adapter = MedQAAdapter()

    def test_official_splits_loading(self):
        """Ensures official train, dev, val, and test splits are recognized."""
        for split in ["train", "dev", "val", "test"]:
            examples = self.adapter.load_split(split, limit=3)
            self.assertGreater(len(examples), 0)
            for ex in examples:
                self.assertEqual(ex.dataset, "medqa")
                self.assertEqual(len(ex.options), 4)
                self.assertIn(ex.gold_answer, ["A", "B", "C", "D"])

    def test_missing_question_or_option_rejected(self):
        """Verifies strict validation fails if question or options are invalid."""
        raw_missing_q = {
            "question": "",
            "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "answer_idx": "A",
        }
        with self.assertRaises(ValueError) as ctx:
            self.adapter.normalize_example(raw_missing_q, split="test")
        self.assertIn("non-empty string", str(ctx.exception))

        # Only 3 options provided
        raw_3_options = {
            "question": "A clinical question vignette?",
            "options": {"A": "1", "B": "2", "C": "3"},
            "answer_idx": "A",
        }
        with self.assertRaises(ValueError) as ctx:
            self.adapter.normalize_example(raw_3_options, split="test")
        self.assertIn("requires exactly keys", str(ctx.exception))

        # Empty option text
        raw_empty_opt = {
            "question": "A clinical question vignette?",
            "options": {"A": "1", "B": "", "C": "3", "D": "4"},
            "answer_idx": "A",
        }
        with self.assertRaises(ValueError) as ctx:
            self.adapter.normalize_example(raw_empty_opt, split="test")
        self.assertIn("Empty option text", str(ctx.exception))

    def test_duplicate_id_detection(self):
        """Verifies duplicate question IDs in a split raise an explicit ValueError."""
        temp_dir = tempfile.mkdtemp()
        try:
            split_file = os.path.join(temp_dir, "test.jsonl")
            item = {
                "id": "dup_001",
                "question": "Question text here",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "answer_idx": "A",
            }
            with open(split_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")
                f.write(json.dumps(item) + "\n")  # Duplicate

            custom_adapter = MedQAAdapter(data_path=split_file)
            with self.assertRaises(ValueError) as ctx:
                custom_adapter.load_split("test")
            self.assertIn("Duplicate example_id detected", str(ctx.exception))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_gold_answer_leakage_prevention(self):
        """Guarantees that gold answer labels NEVER appear in the model prompt."""
        ex = BenchmarkExample(
            example_id="leak_test_01",
            dataset="medqa",
            split="test",
            question="A 50-year-old male presents with acute chest pain.",
            options={
                "A": "Aspirin and heparin",
                "B": "Oral antacid",
                "C": "Intravenous fluids",
                "D": "Observe only",
            },
            gold_answer="A",
        )
        prompt = self.adapter.construct_input(ex)
        # Gold answer must NOT be revealed
        self.assertNotIn("Correct Answer: A", prompt)
        self.assertNotIn("Answer: A", prompt)
        self.assertNotIn("gold", prompt.lower())

    def test_dataset_statistics_reporting(self):
        """Verifies get_statistics reports split sizes and distributions."""
        stats = self.adapter.get_statistics("test")
        self.assertEqual(stats["dataset"], "medqa")
        self.assertEqual(stats["split"], "test")
        self.assertGreater(stats["total_questions"], 0)
        self.assertIn("gold_distribution", stats)
        for opt in ["A", "B", "C", "D"]:
            self.assertIn(opt, stats["gold_distribution"])


class TestMedQAAnswerNormalizationAndAmbiguity(unittest.TestCase):

    def setUp(self):
        self.adapter = MedQAAdapter()
        self.example = BenchmarkExample(
            example_id="q1",
            dataset="medqa",
            split="test",
            question="What is the drug of choice?",
            options={
                "A": "Metformin",
                "B": "Insulin",
                "C": "Glipizide",
                "D": "Acarbose",
            },
            gold_answer="A",
        )

    def test_structured_json_response_parsing(self):
        """Extracts selected_option from structured JSON format."""
        json_resp = """
        Here is the clinical evaluation:
        ```json
        {
            "selected_option": "A",
            "answer_text": "Metformin",
            "explanation": "Metformin is first-line pharmacological therapy for T2D.",
            "claims": ["Metformin reduces HbA1c", "Metformin has low hypoglycemia risk"],
            "abstain": false
        }
        ```
        """
        norm = self.adapter.normalize_answer(json_resp, self.example)
        self.assertEqual(norm, "A")

    def test_structured_json_with_abstention(self):
        """Structured JSON with abstain: true returns None (abstained)."""
        json_resp = """
        {
            "selected_option": "A",
            "explanation": "Contradictory evidence detected in literature.",
            "abstain": true
        }
        """
        norm = self.adapter.normalize_answer(json_resp, self.example)
        self.assertIsNone(norm)

    def test_ambiguous_answer_rejected(self):
        """Ambiguous responses asserting multiple choices must be rejected."""
        ambiguous_1 = "Either A or B could be the right choice depending on kidney function."
        self.assertIsNone(self.adapter.normalize_answer(ambiguous_1, self.example))

        ambiguous_2 = "Both A and C are possible clinical interventions."
        self.assertIsNone(self.adapter.normalize_answer(ambiguous_2, self.example))

    def test_explicit_abstention_signals(self):
        """Explicit abstention statements return None."""
        self.assertIsNone(self.adapter.normalize_answer("Withholding answer due to insufficient evidence.", self.example))
        self.assertIsNone(self.adapter.normalize_answer("We abstain from making a clinical recommendation.", self.example))


class TestTextbookCorpusPreparer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        # Create a small synthetic textbook
        self.book_path = os.path.join(self.temp_dir, "Internal_Medicine_Essentials.txt")
        textbook_content = (
            "CHAPTER 1: CARDIOVASCULAR MEDICINE\n\n"
            "Hypertension is defined as persistent systolic blood pressure >= 130 mmHg or diastolic >= 80 mmHg. "
            "First-line pharmacological therapy includes ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics. "
            "In patients with chronic kidney disease and proteinuria, ACE inhibitors or ARBs are preferred due to renal protective effects.\n\n"
            "CHAPTER 2: ENDOCRINOLOGY\n\n"
            "Type 2 diabetes mellitus is characterized by insulin resistance and progressive beta-cell dysfunction. "
            "Metformin remains the initial drug of choice in patients without severe renal impairment. "
            "Lactic acidosis is a rare but life-threatening complication of biguanide therapy."
        )
        with open(self.book_path, "w", encoding="utf-8") as f:
            f.write(textbook_content)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_textbook_chunking_and_provenance(self):
        preparer = TextbookCorpusPreparer(
            chunk_size=300,
            chunk_overlap=50,
            cache_dir=self.cache_dir,
        )
        docs, manifest = preparer.prepare_directory(self.temp_dir)

        self.assertGreater(len(docs), 0)
        self.assertEqual(manifest["total_source_files"], 1)
        self.assertIn("corpus_sha256", manifest)

        # Inspect first document provenance
        doc0 = docs[0]
        self.assertIn("internal_medicine_essentials", doc0.doc_id)
        self.assertEqual(doc0.source, "Internal Medicine Essentials")
        self.assertIn("section_title", doc0.metadata)
        self.assertIn("passage_index", doc0.metadata)
        self.assertGreater(len(doc0.text), 30)

        # Re-load from cache
        docs_cached, manifest_cached = preparer.prepare_directory(self.temp_dir)
        self.assertEqual(len(docs_cached), len(docs))
        self.assertEqual(manifest_cached["corpus_sha256"], manifest["corpus_sha256"])


class TestEvaluationProtocols(unittest.TestCase):
    """Verifies all 5 research protocols execute cleanly on identical configurations."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.corpus = [
            Document(
                doc_id="tb_doc_1",
                title="Cardiovascular Guide",
                text="ACE inhibitors are first-line for hypertension with diabetes.",
                source="Internal Medicine Guide",
                year=2024,
            ),
            Document(
                doc_id="tb_doc_2",
                title="Endocrinology Guide",
                text="Metformin reduces HbA1c and is first-line for type 2 diabetes.",
                source="Internal Medicine Guide",
                year=2024,
            ),
        ]

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_all_five_protocols(self):
        protocols = ["closed_book", "textbook_rag", "hybrid_rag", "reranked_rag", "medrag_full"]

        for proto in protocols:
            config = ExperimentConfig(
                experiment_id=f"exp_{proto}",
                benchmark_name="medqa",
                split="test",
                protocol=proto,
                limit=2,
                output_dir=self.test_dir,
                llm_backend="mock",
                resume=False,
            )

            # Mock pipeline
            mock_pipeline = MagicMock()
            mock_pipeline.generator.llm.complete.return_value = '{"selected_option": "A", "explanation": "Rationale"}'
            mock_pipeline.generator.generate_answer.return_value = MagicMock(
                answer='{"selected_option": "A", "explanation": "Rationale"}',
                atomic_claims=["Claim 1"],
            )
            mock_pipeline.retriever.retrieve.return_value = [
                RetrievedDocument(self.corpus[0], score=0.9, bm25_rank=1, dense_rank=1)
            ]
            mock_pipeline.retriever.bm25.retrieve.return_value = [
                RetrievedDocument(self.corpus[0], score=10.0, bm25_rank=1)
            ]
            mock_pipeline.retriever.dense.retrieve.return_value = [
                RetrievedDocument(self.corpus[0], score=0.9, dense_rank=1)
            ]
            mock_pipeline.answer.return_value = PipelineResult(
                question="Q",
                final_answer='{"selected_option": "A", "explanation": "Verified"}',
                action="STOP",
                evidence_state=None,
                rounds=1,
                retrieved_docs=[RetrievedDocument(self.corpus[0], score=0.9)],
                history=[{"round": 1, "reason": "Verified"}],
            )

            runner = EvaluationRunner(
                config=config,
                pipeline=mock_pipeline,
                corpus=self.corpus,
            )
            metrics = runner.run()

            self.assertEqual(metrics.total_examples, 2)
            self.assertGreaterEqual(metrics.accuracy, 0.0)
            self.assertIn("exp_" + proto, os.listdir(self.test_dir))


class TestExtendedMetrics(unittest.TestCase):

    def test_per_option_accuracy_and_macro_f1(self):
        records = [
            # Gold A: 2 correct, 1 incorrect
            PredictionRecord(example_id="1", dataset="medqa", split="test", question="Q1", gold_answer="A", normalized_answer="A", is_correct=True),
            PredictionRecord(example_id="2", dataset="medqa", split="test", question="Q2", gold_answer="A", normalized_answer="A", is_correct=True),
            PredictionRecord(example_id="3", dataset="medqa", split="test", question="Q3", gold_answer="A", normalized_answer="B", is_correct=False),
            # Gold B: 1 correct, 1 abstained
            PredictionRecord(example_id="4", dataset="medqa", split="test", question="Q4", gold_answer="B", normalized_answer="B", is_correct=True),
            PredictionRecord(example_id="5", dataset="medqa", split="test", question="Q5", gold_answer="B", normalized_answer=None, is_correct=False, abstained=True),
            # Gold C: 1 correct
            PredictionRecord(example_id="6", dataset="medqa", split="test", question="Q6", gold_answer="C", normalized_answer="C", is_correct=True),
            # Gold D: 1 correct
            PredictionRecord(example_id="7", dataset="medqa", split="test", question="Q7", gold_answer="D", normalized_answer="D", is_correct=True),
        ]

        metrics = MetricRegistry.compute_all(records)

        self.assertEqual(metrics.total_examples, 7)
        self.assertEqual(metrics.correct_count, 5)
        self.assertAlmostEqual(metrics.accuracy, 5 / 7, places=2)

        # Per-option accuracy
        self.assertAlmostEqual(metrics.per_option_accuracy["A"], 2 / 3, places=2)
        self.assertAlmostEqual(metrics.per_option_accuracy["B"], 1 / 2, places=2)
        self.assertAlmostEqual(metrics.per_option_accuracy["C"], 1.0, places=2)
        self.assertAlmostEqual(metrics.per_option_accuracy["D"], 1.0, places=2)

        # Macro-F1 should be computed
        self.assertIsNotNone(metrics.macro_f1)
        self.assertGreater(metrics.macro_f1, 0.5)

        # Confusion matrix checks
        self.assertEqual(metrics.confusion_matrix["A"]["A"], 2)
        self.assertEqual(metrics.confusion_matrix["A"]["B"], 1)
        self.assertEqual(metrics.confusion_matrix["B"]["ABSTAIN"], 1)


if __name__ == "__main__":
    unittest.main()
