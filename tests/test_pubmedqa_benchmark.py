"""
Comprehensive Test Suite for PubMedQA (PQA-L Expert-Labeled) Benchmark Adapter.
Verifies:
1. Strict PQA-L validation & rejection of PQA-U/PQA-A.
2. Label normalization ('yes', 'no', 'maybe') and option mapping.
3. Structured JSON answer generation and controlled recovery on ambiguity/conflicts.
4. PUBMEDQA_PROVIDED_CONTEXT and PUBMEDQA_RETRIEVAL protocol execution without leakage.
5. Claim-level evidence verification (SUPPORTED, CONTRADICTED, CONFLICTED, INSUFFICIENT_EVIDENCE).
6. Retrieval metrics (Recall@1/5/10, MRR, nDCG@k) and classification metrics (Macro-F1, per-class P/R, selective risk).
7. Official PubMedQA prediction export and evaluation compatibility.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import tempfile
import pytest

from medrag.benchmarks.registry import BenchmarkRegistry
from medrag.benchmarks.schema import BenchmarkExample, PredictionRecord
from medrag.benchmarks.adapters.pubmedqa import PubMedQAAdapter
from medrag.benchmarks.metrics import (
    MetricRegistry,
    compute_retrieval_metrics,
)
from medrag.benchmarks.runner import EvaluationRunner
from medrag.config_schema import ExperimentConfig
from medrag.evidence_state import ClaimEvidence, EvidencePassage


# ==============================================================================
# 1. Registration & Adapter Contract
# ==============================================================================

def test_pubmedqa_adapter_registration():
    adapter = BenchmarkRegistry.get("pubmedqa")
    assert isinstance(adapter, PubMedQAAdapter)
    assert adapter.dataset_id == "pubmedqa"
    assert "test" in adapter.supported_splits
    assert "dev" in adapter.supported_splits


# ==============================================================================
# 2. Part A: Dataset Integration & Strict Validation
# ==============================================================================

def test_pubmedqa_pmid_and_label_validation():
    adapter = PubMedQAAdapter()

    # Valid PQA-L record
    valid_raw = {
        "pmid": "21645374",
        "QUESTION": "Does preoperative statin therapy decrease adverse outcomes in cardiac surgery?",
        "CONTEXTS": ["Statins possess anti-inflammatory and plaque-stabilizing properties."],
        "final_decision": "yes",
        "YEAR": 2011,
    }
    ex = adapter.normalize_example(valid_raw, split="test")
    assert ex.example_id == "21645374"
    assert ex.gold_answer == "yes"
    assert len(ex.context) == 1
    assert ex.options == {"A": "yes", "B": "no", "C": "maybe"}

    # Missing PMID
    bad_pmid = dict(valid_raw)
    bad_pmid["pmid"] = ""
    with pytest.raises(ValueError, match="Missing PMID"):
        adapter.normalize_example(bad_pmid, split="test")

    # Missing Question
    bad_q = dict(valid_raw)
    bad_q["QUESTION"] = ""
    with pytest.raises(ValueError, match="Missing or invalid question"):
        adapter.normalize_example(bad_q, split="test")

    # Missing Context (must never be silently replaced)
    bad_ctx = dict(valid_raw)
    bad_ctx["CONTEXTS"] = []
    with pytest.raises(ValueError, match="Missing reference context"):
        adapter.normalize_example(bad_ctx, split="test")

    # Invalid final_decision (must never be silently replaced)
    bad_gold = dict(valid_raw)
    bad_gold["final_decision"] = "certainly"
    with pytest.raises(ValueError, match="Invalid or missing final_decision"):
        adapter.normalize_example(bad_gold, split="test")


def test_pubmedqa_reject_unlabeled_and_artificial():
    adapter = PubMedQAAdapter()

    # PQA-U / artificial item
    artificial_item = {
        "pmid": "99999999",
        "QUESTION": "Does drug X cure condition Y?",
        "CONTEXTS": ["Synthetic abstract sentence."],
        "final_decision": "yes",
        "subset": "pqa_artificial",
    }
    with pytest.raises(ValueError, match="Unlabeled or artificial PubMedQA subset"):
        adapter.normalize_example(artificial_item, split="test")

    pqau_item = {
        "pmid": "88888888",
        "QUESTION": "Is biomarker Z elevated in sepsis?",
        "CONTEXTS": ["Clinical abstract."],
        "final_decision": "yes",
        "subset": "pqau",
    }
    with pytest.raises(ValueError, match="Unlabeled or artificial PubMedQA subset"):
        adapter.normalize_example(pqau_item, split="test")


def test_pubmedqa_fallback_split_loading():
    adapter = PubMedQAAdapter(data_path="non_existent_dir_12345")
    examples = adapter.load_split("test", limit=3)
    assert len(examples) == 3
    for ex in examples:
        assert ex.dataset == "pubmedqa"
        assert ex.gold_answer in ("yes", "no", "maybe")
        assert len(ex.context) >= 1
        assert len(ex.question) > 10


# ==============================================================================
# 3. Part B: Protocol Input Construction & Leakage Prevention
# ==============================================================================

def test_pubmedqa_input_construction_protocols():
    adapter = PubMedQAAdapter()
    example = BenchmarkExample(
        example_id="21645374",
        dataset="pubmedqa",
        split="test",
        question="Does preoperative statin therapy decrease adverse outcomes in cardiac surgery?",
        options={"A": "yes", "B": "no", "C": "maybe"},
        context=["Statins possess anti-inflammatory properties.", "Trial showed reduced atrial fibrillation."],
        gold_answer="yes",
    )

    # Protocol 1: Provided Context
    prompt_provided = adapter.construct_input(example, include_context=True)
    assert "Reference Scientific Abstract:" in prompt_provided
    assert "Statins possess anti-inflammatory properties." in prompt_provided
    assert example.question in prompt_provided
    # Ensure gold answer is NOT leaked
    assert "final_decision" not in prompt_provided
    assert "Gold Answer: yes" not in prompt_provided

    # Protocol 2: Retrieval (Question Only)
    prompt_retrieval = adapter.construct_input(example, include_context=False)
    assert "Reference Scientific Abstract:" not in prompt_retrieval
    assert "Statins possess anti-inflammatory properties." not in prompt_retrieval
    assert example.question in prompt_retrieval
    assert "final_decision" not in prompt_retrieval


# ==============================================================================
# 4. Part D: Answer Generation, Structured JSON, & Controlled Recovery
# ==============================================================================

def test_pubmedqa_structured_json_parsing():
    adapter = PubMedQAAdapter()
    dummy_ex = BenchmarkExample(
        example_id="101",
        dataset="pubmedqa",
        split="test",
        question="Is aspirin effective?",
        gold_answer="yes",
    )

    # Valid direct JSON
    raw_json = json.dumps({
        "prediction": "yes",
        "explanation": "Multiple RCTs demonstrate risk reduction.",
        "claims": ["Aspirin reduces cardiovascular events."],
        "citations": ["101"],
        "abstain": False,
    })
    assert adapter.normalize_answer(raw_json, dummy_ex) == "yes"

    # Markdown fenced JSON
    markdown_json = f"Here is my analysis:\n```json\n{raw_json}\n```\nHope this helps."
    assert adapter.normalize_answer(markdown_json, dummy_ex) == "yes"

    # Option letter inside JSON
    opt_json = json.dumps({"prediction": "B", "abstain": False})
    assert adapter.normalize_answer(opt_json, dummy_ex) == "no"

    # Explicit abstention inside JSON
    abstain_json = json.dumps({
        "prediction": "yes",
        "explanation": "Conflicting literature.",
        "abstain": True,
    })
    assert adapter.normalize_answer(abstain_json, dummy_ex) is None


def test_pubmedqa_controlled_recovery_and_conflicts():
    adapter = PubMedQAAdapter()
    dummy_ex = BenchmarkExample(
        example_id="102",
        dataset="pubmedqa",
        split="test",
        question="Does treatment X work?",
        gold_answer="no",
    )

    # Unambiguous text conclusion
    assert adapter.normalize_answer("Conclusion: no. The clinical trials showed no benefit.", dummy_ex) == "no"
    assert adapter.normalize_answer("Answer: maybe. Further trials are necessary.", dummy_ex) == "maybe"
    assert adapter.normalize_answer("Option B", dummy_ex) == "no"

    # Explicit text abstention
    assert adapter.normalize_answer("There is insufficient evidence to conclude efficacy; abstaining.", dummy_ex) is None
    assert adapter.normalize_answer("Withholding judgment due to contradictory findings.", dummy_ex) is None

    # Conflicting signals in text (must safely abstain)
    conflicting_response = (
        "Prediction: yes. However, based on the second study, conclusion: no."
    )
    assert adapter.normalize_answer(conflicting_response, dummy_ex) is None

    # Ambiguous text containing differing decisions
    ambiguous_response = "Some say yes, others say no, while guidelines say maybe."
    assert adapter.normalize_answer(ambiguous_response, dummy_ex) is None

    # Completely unparseable noise
    assert adapter.normalize_answer("The clinical weather is cloudy today.", dummy_ex) is None
    assert adapter.normalize_answer("", dummy_ex) is None


# ==============================================================================
# 5. Part C: Evidence Verification & 4-Way Claim Status
# ==============================================================================

def test_pubmedqa_evidence_verification_statuses():
    # 1. SUPPORTED: high entailment, low contradiction
    sup_passage = EvidencePassage(
        doc_id="pmid_100_0",
        doc_title="Trial A",
        source="PubMed",
        year=2023,
        quality_score=0.95,
        text="Drug reduces mortality significantly.",
        entailment=0.92,
        contradiction=0.04,
        neutral=0.04,
    )
    claim_sup = ClaimEvidence(
        claim="Drug reduces mortality.",
        support_score=0.92,
        conflict_score=0.04,
        supporting_passages=[sup_passage],
    )
    assert claim_sup.verification_status == "SUPPORTED"

    # 2. CONTRADICTED: high contradiction, low entailment
    con_passage = EvidencePassage(
        doc_id="pmid_200_0",
        doc_title="Trial B",
        source="PubMed",
        year=2024,
        quality_score=0.90,
        text="Drug does not reduce mortality and increases adverse events.",
        entailment=0.05,
        contradiction=0.88,
        neutral=0.07,
    )
    claim_con = ClaimEvidence(
        claim="Drug reduces mortality.",
        support_score=0.05,
        conflict_score=0.88,
        contradicting_passages=[con_passage],
    )
    assert claim_con.verification_status == "CONTRADICTED"

    # 3. CONFLICTED: both high entailment and high contradiction across passages
    claim_conflicted = ClaimEvidence(
        claim="Drug reduces mortality.",
        support_score=0.85,
        conflict_score=0.82,
        supporting_passages=[sup_passage],
        contradicting_passages=[con_passage],
    )
    assert claim_conflicted.verification_status == "CONFLICTED"

    # 4. INSUFFICIENT_EVIDENCE: low entailment, low contradiction
    claim_insufficient = ClaimEvidence(
        claim="Drug reduces mortality.",
        support_score=0.10,
        conflict_score=0.05,
    )
    assert claim_insufficient.verification_status == "INSUFFICIENT_EVIDENCE"


# ==============================================================================
# 6. Part E: Evaluation Metrics (Retrieval & Classification)
# ==============================================================================

def test_pubmedqa_retrieval_metrics():
    # 3 records:
    # r1: relevant PMID "21645374" retrieved at rank 1
    # r2: relevant PMID "28342119" retrieved at rank 3
    # r3: relevant PMID "19401723" not in top 10
    records = [
        PredictionRecord(
            example_id="21645374",
            dataset="pubmedqa",
            split="test",
            question="Q1",
            gold_answer="yes",
            retrieved_doc_ids=["pmid_21645374_0", "pmid_99999_0", "pmid_88888_0"],
            metadata={"protocol": "pubmedqa_retrieval", "reference_pmid": "21645374"},
        ),
        PredictionRecord(
            example_id="28342119",
            dataset="pubmedqa",
            split="test",
            question="Q2",
            gold_answer="no",
            retrieved_doc_ids=["pmid_11111_0", "pmid_22222_0", "pmid_28342119_1"],
            metadata={"protocol": "pubmedqa_retrieval", "reference_pmid": "28342119"},
        ),
        PredictionRecord(
            example_id="19401723",
            dataset="pubmedqa",
            split="test",
            question="Q3",
            gold_answer="maybe",
            retrieved_doc_ids=["pmid_33333_0", "pmid_44444_0"],
            metadata={"protocol": "pubmedqa_retrieval", "reference_pmid": "19401723"},
        ),
    ]

    recall_at_k, mrr, ndcg_at_k = compute_retrieval_metrics(records)
    assert recall_at_k is not None
    assert mrr is not None
    assert ndcg_at_k is not None

    # Hits at 1: only r1 -> 1/3 = 0.3333
    assert recall_at_k["Recall@1"] == pytest.approx(0.3333, abs=0.01)
    # Hits at 5: r1 (rank 1) and r2 (rank 3) -> 2/3 = 0.6667
    assert recall_at_k["Recall@5"] == pytest.approx(0.6667, abs=0.01)
    assert recall_at_k["Recall@10"] == pytest.approx(0.6667, abs=0.01)

    # MRR: r1=1.0, r2=1/3=0.3333, r3=0.0 -> mean = (1.0 + 0.3333 + 0) / 3 = 0.4444
    assert mrr == pytest.approx(0.4444, abs=0.01)

    # nDCG@5: r1: 1.0; r2: 1/log2(4)=0.5; r3: 0 -> mean = 1.5 / 3 = 0.5000
    assert ndcg_at_k["nDCG@5"] == pytest.approx(0.5000, abs=0.01)


def test_pubmedqa_classification_metrics_and_selective_risk():
    records = [
        PredictionRecord(
            example_id="1", dataset="pubmedqa", split="test", question="Q1",
            gold_answer="yes", predicted_answer="yes", normalized_answer="yes",
            is_correct=True, abstained=False,
        ),
        PredictionRecord(
            example_id="2", dataset="pubmedqa", split="test", question="Q2",
            gold_answer="no", predicted_answer="no", normalized_answer="no",
            is_correct=True, abstained=False,
        ),
        PredictionRecord(
            example_id="3", dataset="pubmedqa", split="test", question="Q3",
            gold_answer="maybe", predicted_answer="yes", normalized_answer="yes",
            is_correct=False, abstained=False,
        ),
        PredictionRecord(
            example_id="4", dataset="pubmedqa", split="test", question="Q4",
            gold_answer="maybe", predicted_answer=None, normalized_answer=None,
            is_correct=False, abstained=True,
        ),
    ]

    metrics = MetricRegistry.compute_all(records)
    # Total = 4, Evaluated = 3, Correct = 2, Abstained = 1
    assert metrics.total_examples == 4
    assert metrics.evaluated_count == 3
    assert metrics.abstained_count == 1
    assert metrics.accuracy == pytest.approx(2 / 4, abs=0.01)
    assert metrics.selective_accuracy == pytest.approx(2 / 3, abs=0.01)
    assert metrics.selective_risk == pytest.approx(1.0 - (2 / 3), abs=0.01)
    assert metrics.abstention_rate == 0.25
    assert metrics.coverage == 0.75

    # Check per-class precision and recall
    assert "yes" in metrics.per_class_precision
    assert "no" in metrics.per_class_precision
    assert "maybe" in metrics.per_class_precision
    assert metrics.macro_f1 is not None


# ==============================================================================
# 7. Part F: Official Format Export & Compatibility
# ==============================================================================

def test_pubmedqa_official_export_and_evaluation():
    adapter = PubMedQAAdapter()
    records = [
        PredictionRecord(
            example_id="21645374", dataset="pubmedqa", split="test", question="Q1",
            gold_answer="yes", normalized_answer="yes", is_correct=True,
        ),
        PredictionRecord(
            example_id="28342119", dataset="pubmedqa", split="test", question="Q2",
            gold_answer="no", normalized_answer="no", is_correct=True,
        ),
        PredictionRecord(
            example_id="19401723", dataset="pubmedqa", split="test", question="Q3",
            gold_answer="maybe", normalized_answer=None, abstained=True, is_correct=False,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = os.path.join(tmpdir, "official_pubmedqa_predictions.json")
        adapter.export_official_format(records, export_file)

        assert os.path.exists(export_file)
        with open(export_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["21645374"] == "yes"
        assert data["28342119"] == "no"
        # Abstained record conservatively mapped to 'maybe' in official submission
        assert data["19401723"] == "maybe"

        # Test official evaluation script
        gold_dict = {
            "21645374": {"final_decision": "yes"},
            "28342119": {"final_decision": "no"},
            "19401723": {"final_decision": "maybe"},
        }
        res = adapter.evaluate_official_predictions(data, gold_dict)
        assert res["accuracy"] == 1.0
        assert res["macro_f1"] == 1.0


# ==============================================================================
# 8. Full Pipeline Runner Integration (Smoke Test for Both Protocols)
# ==============================================================================

def test_pubmedqa_runner_smoke_provided_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ExperimentConfig(
            benchmark_name="pubmedqa",
            split="test",
            protocol="pubmedqa_provided_context",
            limit=2,
            output_dir=tmpdir,
            experiment_id="pqa_prov_ctx_smoke",
            llm_backend="mock",
            llm_model="mock-gpt4",
        )

        runner = EvaluationRunner(config=cfg)
        metrics = runner.run()

        run_dir = runner.recorder.run_dir
        assert metrics.total_examples == 2
        assert os.path.exists(os.path.join(run_dir, "metrics.json"))
        assert os.path.exists(os.path.join(run_dir, "manifest.json"))
        assert os.path.exists(os.path.join(run_dir, "official_pubmedqa_predictions.json"))

        preds = runner.recorder.load_all_predictions()
        assert len(preds) == 2
        for p in preds:
            assert p.metadata.get("protocol") == "pubmedqa_provided_context"
            assert "claim_verification_details" in p.metadata


def test_pubmedqa_runner_smoke_retrieval():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ExperimentConfig(
            benchmark_name="pubmedqa",
            split="test",
            protocol="pubmedqa_retrieval",
            limit=2,
            output_dir=tmpdir,
            experiment_id="pqa_retrieval_smoke",
            llm_backend="mock",
            llm_model="mock-gpt4",
        )

        runner = EvaluationRunner(config=cfg)
        metrics = runner.run()

        run_dir = runner.recorder.run_dir
        assert metrics.total_examples == 2
        assert os.path.exists(os.path.join(run_dir, "metrics.json"))
        assert os.path.exists(os.path.join(run_dir, "official_pubmedqa_predictions.json"))

        preds = runner.recorder.load_all_predictions()
        assert len(preds) == 2
        for p in preds:
            assert p.metadata.get("protocol") == "pubmedqa_retrieval"
            assert p.metadata.get("reference_pmid") == p.example_id
