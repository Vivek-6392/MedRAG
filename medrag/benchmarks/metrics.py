"""
Scientific Metrics Engine and MetricRegistry for Medical QA Evaluation.
Computes standard and selective classification metrics, abstention rates,
per-option accuracy, Macro-F1, confusion matrix, token consumption,
Area Under the Risk-Coverage Curve (AURC), and retrieval efficiency statistics.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from .schema import PredictionRecord


class EvaluationMetrics(BaseModel):
    """Structured summary of benchmark evaluation metrics."""
    total_examples: int = 0
    evaluated_count: int = 0
    abstained_count: int = 0
    correct_count: int = 0
    incorrect_count: int = 0

    # Primary research accuracy & abstention metrics
    accuracy: float = Field(0.0, description="Overall accuracy counting abstentions as incorrect (correct / total)")
    selective_accuracy: float = Field(0.0, description="Accuracy on answered examples (correct / evaluated)")
    selective_risk: float = Field(0.0, description="Selective risk (1 - selective_accuracy)")
    abstention_rate: float = Field(0.0, description="Fraction of questions where system abstained")
    coverage: float = Field(0.0, description="Fraction of questions answered (1 - abstention_rate)")

    # Multi-choice & categorical classification metrics (MedQA & PubMedQA)
    per_option_accuracy: Dict[str, float] = Field(default_factory=dict, description="Accuracy per gold answer option/class")
    per_class_precision: Dict[str, float] = Field(default_factory=dict, description="Precision per class (yes/no/maybe or A/B/C/D)")
    per_class_recall: Dict[str, float] = Field(default_factory=dict, description="Recall per class (yes/no/maybe or A/B/C/D)")
    per_class_specificity: Dict[str, float] = Field(default_factory=dict, description="Specificity per class (TN / (TN + FP))")
    per_class_f1: Dict[str, float] = Field(default_factory=dict, description="F1-score per class")
    per_option_performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Detailed per-option breakdown: TP, FP, TN, FN, precision, recall, specificity, f1, support")
    macro_f1: Optional[float] = Field(None, description="Macro-averaged F1 score across options/classes")
    confusion_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Confusion matrix: gold -> {predicted: count}")

    # Selective classification risk-coverage
    aurc: Optional[float] = Field(None, description="Area Under the Risk-Coverage Curve (lower is better)")

    # Efficiency, latency, and tokens
    mean_latency_seconds: float = 0.0
    p95_latency_seconds: float = 0.0
    mean_retrieval_rounds: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    mean_total_tokens: float = 0.0
    mean_heuristic_score: Optional[float] = None

    # Retrieval metrics (PubMedQA retrieval mode)
    retrieval_recall_at_k: Optional[Dict[str, float]] = Field(None, description="Retrieval Recall@k against reference evidence")
    retrieval_mrr: Optional[float] = Field(None, description="Mean Reciprocal Rank for reference evidence retrieval")
    retrieval_ndcg_at_k: Optional[Dict[str, float]] = Field(None, description="Normalized Discounted Cumulative Gain at k")

    # Custom or dataset-specific metrics
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)


def compute_aurc(records: List[PredictionRecord]) -> Optional[float]:
    """
    Computes Area Under the Risk-Coverage Curve (AURC).
    Risk = Error Rate on answered queries; Coverage = Proportion of answered queries.
    Rank-orders examples by heuristic evidence score or confidence, sweeps coverage from 1/N to 1.0,
    and integrates risk over coverage.
    """
    if not records:
        return None

    valid_records = [
        r for r in records
        if r.heuristic_evidence_score is not None and not r.abstained
    ]
    if len(valid_records) < 2:
        return None

    valid_records.sort(key=lambda r: r.heuristic_evidence_score or 0.0, reverse=True)
    n = len(records)
    cumulative_errors = 0
    risk_points = []
    coverage_points = []

    for idx, r in enumerate(valid_records, 1):
        if not r.is_correct:
            cumulative_errors += 1
        coverage = idx / n
        risk = cumulative_errors / idx
        coverage_points.append(coverage)
        risk_points.append(risk)

    # Trapezoidal integration
    aurc = 0.0
    for i in range(1, len(coverage_points)):
        dx = coverage_points[i] - coverage_points[i - 1]
        y_mid = (risk_points[i] + risk_points[i - 1]) / 2.0
        aurc += dx * y_mid

    return round(float(aurc), 4)


def compute_multi_choice_metrics(
    records: List[PredictionRecord],
    canonical_options: Optional[Tuple[str, ...]] = None,
) -> Tuple[
    Dict[str, float],
    float,
    Dict[str, Dict[str, int]],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, Dict[str, Any]],
]:
    """
    Computes per-option accuracy, macro-F1, confusion matrix, per-class precision,
    per-class recall, per-class specificity, per-class F1, and detailed per-option performance.
    Handles multiple-choice (A/B/C/D for MedQA) and categorical (yes/no/maybe for PubMedQA) cleanly.
    """
    gold_set_lower = set(r.gold_answer.strip().lower() for r in records if r.gold_answer)
    is_pubmedqa = len(gold_set_lower) > 0 and gold_set_lower.issubset({"yes", "no", "maybe"})

    if is_pubmedqa:
        options = ["yes", "no", "maybe"]
        def case_fn(s: str) -> str:
            return s.strip().lower()
    else:
        def case_fn(s: str) -> str:
            return s.strip().upper()
        if canonical_options:
            options = list(canonical_options)
        else:
            gold_upper = set(case_fn(r.gold_answer) for r in records if r.gold_answer)
            if gold_upper.issubset({"A", "B", "C", "D"}):
                options = ["A", "B", "C", "D"]
            else:
                options = sorted(list(gold_upper)) if gold_upper else ["A", "B", "C", "D"]

    # Initialize confusion matrix
    cols = options + ["ABSTAIN", "OTHER"]
    matrix: Dict[str, Dict[str, int]] = {g: {c: 0 for c in cols} for g in options}

    per_opt_correct: Dict[str, int] = {opt: 0 for opt in options}
    per_opt_total: Dict[str, int] = {opt: 0 for opt in options}

    tp: Dict[str, int] = {opt: 0 for opt in options}
    fp: Dict[str, int] = {opt: 0 for opt in options}
    fn: Dict[str, int] = {opt: 0 for opt in options}

    for r in records:
        gold = case_fn(r.gold_answer) if r.gold_answer else "OTHER"
        if gold not in matrix:
            matrix[gold] = {c: 0 for c in cols}
            options.append(gold)
            per_opt_correct[gold] = 0
            per_opt_total[gold] = 0
            tp[gold] = 0
            fp[gold] = 0
            fn[gold] = 0

        per_opt_total[gold] += 1

        if r.abstained:
            pred_col = "ABSTAIN"
            fn[gold] += 1
        elif r.normalized_answer:
            pred = case_fn(r.normalized_answer)
            if pred in options:
                pred_col = pred
                if pred == gold:
                    tp[gold] += 1
                    per_opt_correct[gold] += 1
                else:
                    fp[pred] += 1
                    fn[gold] += 1
            else:
                pred_col = "OTHER"
                fn[gold] += 1
        else:
            pred_col = "OTHER"
            fn[gold] += 1

        matrix[gold][pred_col] = matrix[gold].get(pred_col, 0) + 1

    # Per-option accuracy
    per_opt_acc: Dict[str, float] = {}
    for opt in options:
        tot = per_opt_total[opt]
        per_opt_acc[opt] = round(per_opt_correct[opt] / tot, 4) if tot > 0 else 0.0

    # Macro F1, Precision, Recall, Specificity, and Detailed Breakdown
    f1_scores = []
    per_class_precision: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}
    per_class_specificity: Dict[str, float] = {}
    per_class_f1: Dict[str, float] = {}
    per_option_performance: Dict[str, Dict[str, Any]] = {}

    total_samples = len(records)
    for opt in options:
        t_pos = tp[opt]
        f_pos = fp[opt]
        f_neg = fn[opt]
        t_neg = max(0, total_samples - (t_pos + f_pos + f_neg))

        p = t_pos / (t_pos + f_pos) if (t_pos + f_pos) > 0 else 0.0
        r_rec = t_pos / (t_pos + f_neg) if (t_pos + f_neg) > 0 else 0.0
        spec = t_neg / (t_neg + f_pos) if (t_neg + f_pos) > 0 else 0.0
        f1 = (2 * p * r_rec) / (p + r_rec) if (p + r_rec) > 0 else 0.0

        if per_opt_total[opt] > 0:
            f1_scores.append(f1)

        per_class_precision[opt] = round(p, 4)
        per_class_recall[opt] = round(r_rec, 4)
        per_class_specificity[opt] = round(spec, 4)
        per_class_f1[opt] = round(f1, 4)

        per_option_performance[opt] = {
            "tp": t_pos,
            "fp": f_pos,
            "tn": t_neg,
            "fn": f_neg,
            "accuracy": per_opt_acc.get(opt, 0.0),
            "precision": round(p, 4),
            "recall": round(r_rec, 4),
            "specificity": round(spec, 4),
            "f1": round(f1, 4),
            "support": per_opt_total[opt],
        }

    macro_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0

    return (
        per_opt_acc,
        macro_f1,
        matrix,
        per_class_precision,
        per_class_recall,
        per_class_specificity,
        per_class_f1,
        per_option_performance,
    )


def compute_retrieval_metrics(
    records: List[PredictionRecord],
) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[Dict[str, float]]]:
    """
    Computes retrieval Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR),
    and nDCG@5, nDCG@10 against reference evidence (PMID, gold citations, or graded relevance).

    GUARDRAILS:
    - For PubMedQA retrieval: Evaluates against reference PMID from benchmark abstract.
    - For MedQA: Only calculates retrieval recall and ranking metrics when independent
      passage-level relevance annotations exist in metadata ('gold_citation_ids', 'passage_relevance',
      or 'passage_relevance_grades'). Gold retrieval labels are NEVER inferred from answer correctness.
    - When multiple relevance grades exist ('passage_relevance_grades'), computes true graded nDCG@k.
    """
    relevant_records: List[Tuple[PredictionRecord, List[str], Optional[Dict[str, float]]]] = []

    for r in records:
        refs: List[str] = []
        grades: Optional[Dict[str, float]] = None

        if r.metadata.get("passage_relevance_grades"):
            raw_grades = r.metadata["passage_relevance_grades"]
            if isinstance(raw_grades, dict):
                grades = {str(k).strip().lower(): float(v) for k, v in raw_grades.items()}
                refs.extend(list(grades.keys()))
        elif r.metadata.get("reference_pmid"):
            val = r.metadata["reference_pmid"]
            refs.extend([val] if isinstance(val, str) else [str(x) for x in val])
        elif r.metadata.get("gold_citation_ids"):
            val = r.metadata["gold_citation_ids"]
            refs.extend([val] if isinstance(val, str) else [str(x) for x in val])
        elif r.metadata.get("passage_relevance"):
            val = r.metadata["passage_relevance"]
            if isinstance(val, dict):
                refs.extend(list(val.keys()))
            else:
                refs.extend([str(x) for x in val])
        elif r.dataset == "pubmedqa" and r.metadata.get("protocol") in ("pubmedqa_retrieval", "retrieval"):
            refs.append(r.example_id)

        clean_refs = [x.strip().lower() for x in refs if x and str(x).strip()]
        if clean_refs:
            relevant_records.append((r, clean_refs, grades))

    if not relevant_records:
        return None, None, None

    def _matches_any_ref(doc_id: str, targets: List[str]) -> bool:
        norm_did = doc_id.strip().lower()
        base_did = norm_did.replace("pmid_", "").split("_")[0]
        for t in targets:
            norm_t = t.strip().lower()
            base_t = norm_t.replace("pmid_", "").split("_")[0]
            if norm_did == norm_t or base_did == base_t or norm_t in norm_did or base_t in base_did:
                return True
        return False

    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    reciprocal_ranks: List[float] = []
    ndcg_5_list: List[float] = []
    ndcg_10_list: List[float] = []

    for r, targets, grades in relevant_records:
        ret_ids = r.retrieved_doc_ids

        # Recall@1, 5, 10
        h1 = any(_matches_any_ref(did, targets) for did in ret_ids[:1])
        h5 = any(_matches_any_ref(did, targets) for did in ret_ids[:5])
        h10 = any(_matches_any_ref(did, targets) for did in ret_ids[:10])

        if h1:
            hits_1 += 1
        if h5:
            hits_5 += 1
        if h10:
            hits_10 += 1

        # MRR
        rr = 0.0
        for rank, did in enumerate(ret_ids, 1):
            if _matches_any_ref(did, targets):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # nDCG@k (Supports both binary and graded relevance)
        for k, ndcg_list in [(5, ndcg_5_list), (10, ndcg_10_list)]:
            dcg = 0.0
            for rank, did in enumerate(ret_ids[:k], 1):
                norm_did = did.strip().lower()
                rel = 0.0
                if grades:
                    rel = grades.get(norm_did, 0.0)
                    if rel == 0.0:
                        for g_k, g_v in grades.items():
                            if g_k in norm_did or norm_did in g_k:
                                rel = g_v
                                break
                elif _matches_any_ref(did, targets):
                    rel = 1.0

                if rel > 0.0:
                    dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1)

            # Compute Ideal DCG (IDCG@k)
            if grades:
                ideal_rels = sorted(list(grades.values()), reverse=True)[:k]
            else:
                ideal_rels = [1.0] * min(k, len(targets))

            idcg = sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
            ndcg_list.append(min(1.0, dcg / idcg) if idcg > 0.0 else 0.0)

    n = len(relevant_records)
    recall_at_k = {
        "Recall@1": round(hits_1 / n, 4),
        "Recall@5": round(hits_5 / n, 4),
        "Recall@10": round(hits_10 / n, 4),
    }
    mrr = round(sum(reciprocal_ranks) / n, 4) if n > 0 else 0.0
    ndcg_at_k = {
        "nDCG@5": round(sum(ndcg_5_list) / n, 4) if n > 0 else 0.0,
        "nDCG@10": round(sum(ndcg_10_list) / n, 4) if n > 0 else 0.0,
    }

    return recall_at_k, mrr, ndcg_at_k


class MetricRegistry:
    """
    Registry for evaluation metrics.
    Computes core benchmark metrics and allows registering custom metric callbacks.
    """
    _custom_metrics: Dict[str, Callable[[List[PredictionRecord]], Any]] = {}

    @classmethod
    def register_metric(cls, name: str) -> Callable:
        """Decorator to register a custom metric function."""
        def decorator(fn: Callable[[List[PredictionRecord]], Any]) -> Callable:
            cls._custom_metrics[name] = fn
            return fn
        return decorator

    @classmethod
    def compute_all(cls, records: List[PredictionRecord]) -> EvaluationMetrics:
        """
        Computes all standard benchmark metrics across prediction records.
        """
        total = len(records)
        if total == 0:
            return EvaluationMetrics()

        abstained = sum(1 for r in records if r.abstained)
        evaluated = total - abstained
        correct = sum(1 for r in records if r.is_correct is True)
        incorrect = sum(1 for r in records if r.is_correct is False)

        acc = correct / total if total > 0 else 0.0
        sel_acc = correct / evaluated if evaluated > 0 else 0.0
        sel_risk = round(1.0 - sel_acc, 4) if evaluated > 0 else 0.0
        abst_rate = abstained / total if total > 0 else 0.0
        coverage = evaluated / total if total > 0 else 0.0

        latencies = sorted(r.latency_seconds for r in records)
        mean_lat = sum(latencies) / total if total > 0 else 0.0
        p95_idx = min(int(math.ceil(0.95 * total)) - 1, total - 1)
        p95_lat = latencies[p95_idx] if latencies else 0.0

        mean_rounds = sum(r.rounds for r in records) / total if total > 0 else 0.0

        # Token statistics
        prompt_tokens = [r.token_usage.get("prompt_tokens", 0) for r in records if r.token_usage]
        comp_tokens = [r.token_usage.get("completion_tokens", 0) for r in records if r.token_usage]
        tot_tokens = [r.token_usage.get("total_tokens", 0) for r in records if r.token_usage]

        mean_pt = round(sum(prompt_tokens) / len(prompt_tokens), 1) if prompt_tokens else 0.0
        mean_ct = round(sum(comp_tokens) / len(comp_tokens), 1) if comp_tokens else 0.0
        mean_tt = round(sum(tot_tokens) / len(tot_tokens), 1) if tot_tokens else 0.0

        scores = [r.heuristic_evidence_score for r in records if r.heuristic_evidence_score is not None]
        mean_score = sum(scores) / len(scores) if scores else None

        aurc = compute_aurc(records)

        # Multi-choice classification breakdown
        (
            per_opt_acc,
            macro_f1,
            matrix,
            per_class_prec,
            per_class_rec,
            per_class_spec,
            per_class_f1,
            per_opt_perf,
        ) = compute_multi_choice_metrics(records)

        # Retrieval metrics (Recall@k, MRR, nDCG@k)
        recall_at_k, mrr, ndcg_at_k = compute_retrieval_metrics(records)

        # Run custom metrics
        custom_results: Dict[str, Any] = {}
        for name, fn in cls._custom_metrics.items():
            try:
                custom_results[name] = fn(records)
            except Exception as e:
                custom_results[name] = f"Error: {e}"

        # MedQA Retrieval guardrail documentation
        has_medqa = any(r.dataset == "medqa" for r in records)
        if has_medqa and recall_at_k is None:
            custom_results["medqa_retrieval_evaluation_note"] = (
                "Retrieval recall/MRR/nDCG omitted for MedQA: independent passage-level relevance "
                "annotations were not provided in dataset metadata. Gold retrieval labels are never "
                "inferred from answer correctness."
            )

        return EvaluationMetrics(
            total_examples=total,
            evaluated_count=evaluated,
            abstained_count=abstained,
            correct_count=correct,
            incorrect_count=incorrect,
            accuracy=round(acc, 4),
            selective_accuracy=round(sel_acc, 4),
            selective_risk=sel_risk,
            abstention_rate=round(abst_rate, 4),
            coverage=round(coverage, 4),
            per_option_accuracy=per_opt_acc,
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            per_class_specificity=per_class_spec,
            per_class_f1=per_class_f1,
            per_option_performance=per_opt_perf,
            macro_f1=macro_f1,
            confusion_matrix=matrix,
            aurc=aurc,
            mean_latency_seconds=round(mean_lat, 2),
            p95_latency_seconds=round(p95_lat, 2),
            mean_retrieval_rounds=round(mean_rounds, 2),
            mean_prompt_tokens=mean_pt,
            mean_completion_tokens=mean_ct,
            mean_total_tokens=mean_tt,
            mean_heuristic_score=round(mean_score, 4) if mean_score is not None else None,
            retrieval_recall_at_k=recall_at_k,
            retrieval_mrr=mrr,
            retrieval_ndcg_at_k=ndcg_at_k,
            custom_metrics=custom_results,
        )
