"""
Comprehensive Test Suite for MedRAG Baselines, Ablations, Fairness,
and Statistical Significance Engine.
Tests:
1. Canonical Baseline Configurations (8 systems)
2. Leave-One-Out Ablations (9 components)
3. Fairness validation (PART C constraints)
4. McNemar's test (exact binomial & Edwards chi-squared)
5. Paired empirical bootstrap resampling (Accuracy, Selective Acc, Coverage)
6. Multi-configuration StudyRunner end-to-end execution and table exports
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import tempfile
from unittest.mock import MagicMock
import pytest

from medrag.evidence_state import ClaimEvidence, EvidenceState
from medrag.pipeline import PipelineResult
from medrag.retriever import Document, RetrievedDocument

from medrag.benchmarks import (
    ExperimentConfig,
    PredictionRecord,
    StudyRunner,
    StudyResult,
    create_baseline_config,
    create_ablation_config,
    get_baseline_suite,
    get_ablation_suite,
    validate_fairness,
    compute_mcnemar_test,
    paired_bootstrap_ci,
    compare_systems,
)


# ==============================================================================
# 1. Baseline Configurations Verification (PART A)
# ==============================================================================

def test_canonical_baseline_configurations():
    base = ExperimentConfig(
        experiment_id="test_suite",
        benchmark_name="synthetic",
        split="test",
        limit=5,
        llm_backend="mock",
        llm_model="mock-gpt4",
    )

    # 1. Closed Book
    cb = create_baseline_config("closed_book", base)
    assert cb.protocol == "closed_book"
    assert cb.max_search_rounds == 1
    assert cb.enable_hybrid is False
    assert cb.use_reranker is False
    assert cb.enable_nli_verification is False

    # 2. Single-pass Dense RAG
    dense = create_baseline_config("single_pass_dense_rag", base)
    assert dense.enable_hybrid is False
    assert dense.bm25_weight == 0.0
    assert dense.dense_weight == 1.0
    assert dense.use_reranker is False
    assert dense.enable_nli_verification is False

    # 3. Single-pass BM25 RAG
    bm25 = create_baseline_config("single_pass_bm25_rag", base)
    assert bm25.enable_hybrid is False
    assert bm25.bm25_weight == 1.0
    assert bm25.dense_weight == 0.0
    assert bm25.use_reranker is False
    assert bm25.enable_nli_verification is False

    # 4. Hybrid RAG
    hybrid = create_baseline_config("hybrid_rag", base)
    assert hybrid.enable_hybrid is True
    assert hybrid.bm25_weight == 0.4
    assert hybrid.dense_weight == 0.6
    assert hybrid.use_reranker is False

    # 5. Hybrid Reranked RAG
    reranked = create_baseline_config("hybrid_reranked_rag", base)
    assert reranked.enable_hybrid is True
    assert reranked.use_reranker is True
    assert reranked.enable_reranking is True
    assert reranked.enable_nli_verification is False

    # 6. Claim Verified RAG
    claim_ver = create_baseline_config("claim_verified_rag", base)
    assert claim_ver.enable_hybrid is True
    assert claim_ver.use_reranker is True
    assert claim_ver.enable_claim_decomposition is True
    assert claim_ver.enable_nli_verification is True
    assert claim_ver.enable_iterative is False  # Single pass

    # 7. Iterative RAG
    iter_rag = create_baseline_config("iterative_rag", base)
    assert iter_rag.enable_iterative is True
    assert iter_rag.max_search_rounds >= 2
    assert iter_rag.enable_risk_policy is False  # No risk policy

    # 8. Full MedRAG (Proposed system)
    full = create_baseline_config("full_medrag", base)
    assert full.enable_hybrid is True
    assert full.enable_reranking is True
    assert full.enable_claim_decomposition is True
    assert full.enable_nli_verification is True
    assert full.enable_evidence_state is True
    assert full.enable_iterative is True
    assert full.enable_risk_policy is True
    assert full.enable_abstention is True
    assert full.enable_source_quality_weighting is True


def test_baseline_suite_generator():
    suite = get_baseline_suite("medqa", "test", limit=10, backend="mock")
    assert len(suite) == 8
    protocols = [c.protocol for c in suite]
    assert "closed_book" in protocols
    assert "single_pass_dense_rag" in protocols
    assert "single_pass_bm25_rag" in protocols
    assert "hybrid_rag" in protocols
    assert "hybrid_reranked_rag" in protocols
    assert "claim_verified_rag" in protocols
    assert "iterative_rag" in protocols
    assert "full_medrag" in protocols


# ==============================================================================
# 2. Leave-One-Out Ablations Verification (PART B)
# ==============================================================================

def test_canonical_ablation_configurations():
    base = ExperimentConfig(
        experiment_id="ablation_test",
        benchmark_name="synthetic",
        split="test",
        limit=5,
        llm_backend="mock",
        llm_model="mock-gpt4",
    )

    full = create_baseline_config("full_medrag", base)

    # 1. Without hybrid
    ab1 = create_ablation_config("ablation_no_hybrid", base)
    assert ab1.enable_hybrid is False
    assert ab1.enable_reranking is full.enable_reranking
    assert ab1.enable_nli_verification is full.enable_nli_verification

    # 2. Without reranking
    ab2 = create_ablation_config("ablation_no_reranking", base)
    assert ab2.enable_reranking is False
    assert ab2.use_reranker is False
    assert ab2.enable_hybrid is full.enable_hybrid

    # 3. Without claim decomposition
    ab3 = create_ablation_config("ablation_no_claim_decomposition", base)
    assert ab3.enable_claim_decomposition is False
    assert ab3.enable_nli_verification is full.enable_nli_verification

    # 4. Without NLI verification
    ab4 = create_ablation_config("ablation_no_nli_verification", base)
    assert ab4.enable_nli_verification is False
    assert ab4.enable_claim_decomposition is full.enable_claim_decomposition

    # 5. Without evidence state
    ab5 = create_ablation_config("ablation_no_evidence_state", base)
    assert ab5.enable_evidence_state is False

    # 6. Without iterative
    ab6 = create_ablation_config("ablation_no_iterative", base)
    assert ab6.enable_iterative is False
    assert ab6.max_search_rounds == 1

    # 7. Without risk policy
    ab7 = create_ablation_config("ablation_no_risk_policy", base)
    assert ab7.enable_risk_policy is False

    # 8. Without abstention
    ab8 = create_ablation_config("ablation_no_abstention", base)
    assert ab8.enable_abstention is False

    # 9. Without source quality
    ab9 = create_ablation_config("ablation_no_source_quality", base)
    assert ab9.enable_source_quality_weighting is False


def test_ablation_suite_generator():
    suite = get_ablation_suite("medqa", "test", limit=5, backend="mock")
    # Reference full_medrag + 9 ablations = 10 configurations
    assert len(suite) == 10
    protocols = [c.protocol for c in suite]
    assert "full_medrag" in protocols
    assert "ablation_no_hybrid" in protocols
    assert "ablation_no_reranking" in protocols
    assert "ablation_no_abstention" in protocols


# ==============================================================================
# 3. Fairness Validation (PART C)
# ==============================================================================

def test_fairness_validation():
    base = ExperimentConfig(
        experiment_id="fairness_base",
        benchmark_name="medqa",
        split="test",
        limit=10,
        llm_backend="mock",
        llm_model="mock-gpt4",
        llm_temperature=0.0,
    )

    c1 = create_baseline_config("closed_book", base)
    c2 = create_baseline_config("hybrid_rag", base)
    c3 = create_baseline_config("full_medrag", base)

    # Identical underlying benchmark, split, model, temp, and limit -> Fair
    is_fair, violations = validate_fairness([c1, c2, c3])
    assert is_fair is True
    assert len(violations) == 0

    # Unfair case 1: Mismatched benchmark
    c_unfair_bm = c1.model_copy()
    c_unfair_bm.benchmark_name = "pubmedqa"
    is_fair_bm, viols_bm = validate_fairness([c1, c_unfair_bm])
    assert is_fair_bm is False
    assert any("benchmark" in v for v in viols_bm)

    # Unfair case 2: Mismatched model
    c_unfair_model = c1.model_copy()
    c_unfair_model.llm_model = "gpt-4o"
    is_fair_mod, viols_mod = validate_fairness([c1, c_unfair_model])
    assert is_fair_mod is False
    assert any("LLM" in v for v in viols_mod)

    # Unfair case 3: Mismatched temperature
    c_unfair_temp = c1.model_copy()
    c_unfair_temp.llm_temperature = 0.7
    is_fair_t, viols_t = validate_fairness([c1, c_unfair_temp])
    assert is_fair_t is False
    assert any("temperature" in v for v in viols_t)


# ==============================================================================
# 4. Statistical Significance Tests (PART E)
# ==============================================================================

def test_mcnemar_exact_binomial_small_samples():
    # 20 items:
    # A correct on 15, B correct on 5
    # Discordant: A correct & B wrong = 11; A wrong & B correct = 1 -> discordant = 12 (< 25)
    correct_a = [True] * 15 + [False] * 5
    correct_b = [True] * 4 + [False] * 11 + [True] * 1 + [False] * 4

    res = compute_mcnemar_test(correct_a, correct_b)
    assert res.method == "exact_binomial"
    assert res.table["a_only"] == 11
    assert res.table["b_only"] == 1
    # Significant difference (11 vs 1)
    assert res.p_value < 0.05
    assert res.is_significant_05 is True


def test_mcnemar_continuity_corrected_large_samples():
    # 100 items:
    # A only = 30, B only = 10 -> discordant = 40 (>= 25)
    correct_a = [True] * 30 + [False] * 10 + [True] * 30 + [False] * 30
    correct_b = [False] * 30 + [True] * 10 + [True] * 30 + [False] * 30

    res = compute_mcnemar_test(correct_a, correct_b)
    assert res.method == "edwards_continuity_corrected_chi2"
    assert res.table["a_only"] == 30
    assert res.table["b_only"] == 10
    # chi2 = (|30 - 10| - 1)^2 / 40 = 19^2 / 40 = 361 / 40 = 9.025
    assert res.statistic == pytest.approx(9.025, abs=0.01)
    assert res.p_value < 0.01
    assert res.is_significant_01 is True


def test_mcnemar_identical_systems():
    correct_a = [True, False, True, False, True]
    correct_b = [True, False, True, False, True]
    res = compute_mcnemar_test(correct_a, correct_b)
    assert res.table["a_only"] == 0
    assert res.table["b_only"] == 0
    assert res.p_value == 1.0
    assert res.is_significant_05 is False


def test_paired_bootstrap_confidence_interval():
    # Construct 50 paired records where System A has 80% accuracy and System B has 40%
    pairs = []
    for i in range(50):
        # A correct on first 40, B correct on first 20
        a_cor = i < 40
        b_cor = i < 20
        rec_a = PredictionRecord(
            example_id=f"q_{i}", dataset="syn", split="test", question="Q",
            gold_answer="A", predicted_answer="A" if a_cor else "B",
            is_correct=a_cor, abstained=False,
        )
        rec_b = PredictionRecord(
            example_id=f"q_{i}", dataset="syn", split="test", question="Q",
            gold_answer="A", predicted_answer="A" if b_cor else "B",
            is_correct=b_cor, abstained=False,
        )
        pairs.append((rec_a, rec_b))

    def _acc(recs):
        return sum(1 for r in recs if r.is_correct) / len(recs)

    res = paired_bootstrap_ci(pairs, _acc, metric_name="Accuracy", num_bootstraps=500, random_seed=42)
    assert res.metric_name == "Accuracy"
    # Observed difference: 80% - 40% = +40%
    assert res.observed_diff == pytest.approx(0.40, abs=0.01)
    # 95% CI bounds must be strictly positive
    assert res.ci_lower > 0.15
    assert res.ci_upper > res.ci_lower
    assert res.p_value < 0.01
    assert res.is_significant is True


def test_compare_systems_report():
    recs_a = [
        PredictionRecord(example_id="1", dataset="s", split="t", question="Q1", gold_answer="A", is_correct=True, abstained=False),
        PredictionRecord(example_id="2", dataset="s", split="t", question="Q2", gold_answer="B", is_correct=True, abstained=False),
        PredictionRecord(example_id="3", dataset="s", split="t", question="Q3", gold_answer="C", is_correct=False, abstained=False),
    ]
    recs_b = [
        PredictionRecord(example_id="1", dataset="s", split="t", question="Q1", gold_answer="A", is_correct=False, abstained=False),
        PredictionRecord(example_id="2", dataset="s", split="t", question="Q2", gold_answer="B", is_correct=True, abstained=False),
        PredictionRecord(example_id="3", dataset="s", split="t", question="Q3", gold_answer="C", is_correct=False, abstained=True),
    ]

    report = compare_systems(recs_a, recs_b, "FullMedRAG", "DenseRAG", num_bootstraps=200)
    assert report.total_aligned_examples == 3
    assert report.mcnemar is not None
    assert report.accuracy_bootstrap is not None
    assert report.selective_acc_bootstrap is not None

    rep_dict = report.to_dict()
    assert "FullMedRAG vs DenseRAG" in rep_dict["comparison"]
    assert "limitations_disclaimer" in rep_dict

    rep_md = report.to_markdown()
    assert "Statistical Significance" in rep_md
    assert "McNemar" in rep_md


# ==============================================================================
# 5. Multi-Configuration StudyRunner Integration (PART D)
# ==============================================================================

def test_study_runner_smoke_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = ExperimentConfig(
            experiment_id="study_smoke",
            benchmark_name="synthetic",
            split="test",
            limit=2,
            output_dir=tmpdir,
            llm_backend="mock",
            llm_model="mock-gpt4",
        )

        # Run 3 configurations: closed_book, hybrid_rag, full_medrag
        configs = [
            create_baseline_config("closed_book", base),
            create_baseline_config("hybrid_rag", base),
            create_baseline_config("full_medrag", base),
        ]

        def mock_pipeline_factory(cfg):
            mock_pipeline = MagicMock()
            doc = Document(
                doc_id="mock_doc_1",
                title="Mock Evidence",
                text="Mock document content",
                source="Mock",
                year=2024,
            )
            rdoc = RetrievedDocument(document=doc, score=0.90, bm25_rank=1)
            ev_state = EvidenceState(
                claim_evidences=[
                    ClaimEvidence("Evidence supports option A", support_score=0.95, clinical_risk=0.1)
                ],
                round_number=1,
            )

            # LLM generator mock for closed-book and single-pass
            mock_pipeline.generator.llm.complete.return_value = "The correct answer is Option A. Evidence confirms this."
            mock_pipeline.generator.generate.return_value = "The correct answer is Option A."
            mock_pipeline.generator.generate_with_citations.return_value = ("The correct answer is Option A. [1]", [rdoc])
            ans_obj = MagicMock()
            ans_obj.answer = "The correct answer is Option A."
            ans_obj.citations = [rdoc]
            mock_pipeline.generator.generate_answer.return_value = ans_obj

            # Retriever mock
            mock_pipeline.retriever.retrieve.return_value = [rdoc]
            mock_pipeline.retriever.dense_retriever.retrieve.return_value = [rdoc]
            mock_pipeline.retriever.bm25_retriever.retrieve.return_value = [rdoc]

            def fake_answer(prompt: str) -> PipelineResult:
                return PipelineResult(
                    question=prompt,
                    final_answer="The correct answer is Option A. Evidence confirms this.",
                    action="STOP",
                    evidence_state=ev_state,
                    rounds=1,
                    retrieved_docs=[rdoc],
                    history=[{"round": 1, "query": prompt, "new_docs": 1, "total_docs": 1, "reason": "Verified"}],
                    latency_seconds=0.05,
                )
            mock_pipeline.answer = MagicMock(side_effect=fake_answer)
            return mock_pipeline

        study = StudyRunner(
            study_id="smoke_baselines_study",
            configs=configs,
            output_dir=tmpdir,
            pipeline_factory=mock_pipeline_factory,
        )
        res: StudyResult = study.run()

        assert res.study_id == "smoke_baselines_study"
        assert len(res.metrics) == 3

        # Check artifact files written to study output dir
        out_dir = res.output_dir
        assert os.path.exists(os.path.join(out_dir, "comparison_table.md"))
        assert os.path.exists(os.path.join(out_dir, "comparison_table.json"))
        assert os.path.exists(os.path.join(out_dir, "per_example_comparison.csv"))
        assert os.path.exists(os.path.join(out_dir, "per_example_comparison.jsonl"))
        assert os.path.exists(os.path.join(out_dir, "ablation_table.md"))
        assert os.path.exists(os.path.join(out_dir, "statistical_report.json"))

        # Verify per-example CSV contents
        csv_path = os.path.join(out_dir, "per_example_comparison.csv")
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            assert "example_id" in header
            assert "closed_book_pred" in header
            assert "full_medrag_pred" in header
