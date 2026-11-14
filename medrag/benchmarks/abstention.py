"""
Research-Grade Abstention, Selective Answering, and Risk-Coverage Engine.
Computes:
- Empirical Coverage and Selective Accuracy
- Selective Risk (1 - Selective Accuracy)
- Full Risk-Coverage Curves
- Area Under the Risk-Coverage Curve (AURC)
- Abstention Precision, Recall, and F1 against defined unanswerable/insufficient-evidence gold labels
- Rigorous Validation-Data Threshold Tuning (finds optimal operating points without test contamination)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from .schema import PredictionRecord


class RiskCoveragePoint(BaseModel):
    """Single operating point on the empirical risk-coverage curve."""
    threshold: float
    coverage: float
    selective_accuracy: float
    selective_risk: float
    answered_count: int
    error_count: int


class AbstentionClassificationMetrics(BaseModel):
    """Classification performance of abstention decisions against gold unanswerable labels."""
    tp_abstain: int = 0
    fp_abstain: int = 0
    fn_abstain: int = 0
    tn_abstain: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0


class AbstentionEvaluationReport(BaseModel):
    """Comprehensive evaluation of selective answering and risk-coverage."""
    total_examples: int = 0
    answered_count: int = 0
    abstained_count: int = 0
    coverage: float = 0.0
    selective_accuracy: float = 0.0
    selective_risk: float = 0.0
    raw_accuracy: float = 0.0
    aurc: Optional[float] = None
    curve_points: List[RiskCoveragePoint] = Field(default_factory=list)
    unanswerable_metrics: Optional[AbstentionClassificationMetrics] = None
    validation_tuning_applied: bool = False
    tuned_threshold: Optional[float] = None


class ThresholdTuningResult(BaseModel):
    """Result of tuning confidence/evidence thresholds on validation data."""
    target_risk: float
    min_coverage: float
    optimal_threshold: float
    val_coverage: float
    val_selective_risk: float
    val_selective_accuracy: float
    points_evaluated: int
    threshold_grid: List[Dict[str, float]] = Field(default_factory=list)
    methodology_note: str = (
        "Threshold was selected strictly on validation data to satisfy selective_risk <= target_risk. "
        "Must be held frozen when evaluating test split."
    )


def compute_risk_coverage_curve(
    records: List[PredictionRecord],
    score_key: str = "heuristic_evidence_score",
    num_threshold_bins: int = 50,
) -> Tuple[List[RiskCoveragePoint], Optional[float]]:
    """
    Computes empirical risk-coverage curve across varying confidence thresholds.

    Sorts records by predicted confidence/evidence score in descending order.
    Sweeps threshold from minimum to maximum, computing coverage and selective risk at each point.
    Computes AURC via trapezoidal integration over coverage in [0, 1].
    """
    if not records:
        return [], None

    # Filter records that have a valid score
    scored_records: List[Tuple[float, bool]] = []
    for r in records:
        val = getattr(r, score_key, None)
        if val is None and isinstance(r.metadata, dict):
            val = r.metadata.get("confidence") or r.metadata.get("support_score")
        if val is not None:
            scored_records.append((float(val), bool(r.is_correct)))
        else:
            scored_records.append((0.50, bool(r.is_correct)))

    if not scored_records:
        return [], None

    scored_records.sort(key=lambda x: x[0], reverse=True)
    n_total = len(scored_records)

    # Determine threshold sweep range
    min_score = min(s[0] for s in scored_records)
    max_score = max(s[0] for s in scored_records)
    if math.isclose(min_score, max_score):
        thresholds = [min_score]
    else:
        step = (max_score - min_score) / max(1, num_threshold_bins)
        thresholds = [min_score + i * step for i in range(num_threshold_bins + 1)]
        thresholds.sort(reverse=True)

    points: List[RiskCoveragePoint] = []
    seen_coverages = set()

    for th in thresholds:
        # Items answered at or above this threshold
        answered = [item for item in scored_records if item[0] >= th]
        ans_count = len(answered)
        if ans_count == 0:
            continue

        coverage = round(ans_count / n_total, 4)
        if coverage in seen_coverages:
            continue
        seen_coverages.add(coverage)

        correct_count = sum(1 for item in answered if item[1] is True)
        error_count = ans_count - correct_count
        sel_acc = round(correct_count / ans_count, 4)
        sel_risk = round(error_count / ans_count, 4)

        points.append(RiskCoveragePoint(
            threshold=round(th, 4),
            coverage=coverage,
            selective_accuracy=sel_acc,
            selective_risk=sel_risk,
            answered_count=ans_count,
            error_count=error_count,
        ))

    # Sort curve points by coverage ascending
    points.sort(key=lambda p: p.coverage)

    # Compute AURC (Trapezoidal integration over coverage)
    aurc = 0.0
    if len(points) >= 2:
        for i in range(1, len(points)):
            dx = points[i].coverage - points[i - 1].coverage
            y_mid = (points[i].selective_risk + points[i - 1].selective_risk) / 2.0
            aurc += dx * y_mid

    return points, round(aurc, 4) if points else None


def evaluate_abstention_classification(
    records: List[PredictionRecord],
    unanswerable_labels: Optional[List[str]] = None,
) -> Optional[AbstentionClassificationMetrics]:
    """
    Evaluates system's decision to ABSTAIN as a binary classification task against
    gold labels representing unanswerable questions (e.g. PubMedQA 'maybe', or explicit 'unanswerable').
    """
    if unanswerable_labels is None:
        unanswerable_labels = ["maybe", "unanswerable", "abstain", "insufficient_evidence", "none"]

    clean_unans = {lbl.strip().lower() for lbl in unanswerable_labels}

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    valid_count = 0

    for r in records:
        gold = (r.gold_answer or "").strip().lower()
        is_gold_unanswerable = gold in clean_unans or bool(r.metadata.get("is_unanswerable"))
        is_pred_abstain = bool(r.abstained)

        valid_count += 1
        if is_gold_unanswerable and is_pred_abstain:
            tp += 1
        elif not is_gold_unanswerable and is_pred_abstain:
            fp += 1
        elif is_gold_unanswerable and not is_pred_abstain:
            fn += 1
        else:
            tn += 1

    if valid_count == 0 or (tp + fn == 0 and fp + tn == 0):
        return None

    prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round((2 * prec * rec) / (prec + rec), 4) if (prec + rec) > 0 else 0.0
    acc = round((tp + tn) / valid_count, 4) if valid_count > 0 else 0.0

    return AbstentionClassificationMetrics(
        tp_abstain=tp,
        fp_abstain=fp,
        fn_abstain=fn,
        tn_abstain=tn,
        precision=prec,
        recall=rec,
        f1=f1,
        accuracy=acc,
    )


def tune_abstention_thresholds(
    val_records: List[PredictionRecord],
    target_risk: float = 0.05,
    min_coverage: float = 0.50,
    score_key: str = "heuristic_evidence_score",
) -> ThresholdTuningResult:
    """
    Calibrates confidence / evidence thresholds strictly on validation records.

    Finds the minimum threshold tau that satisfies:
      selective_risk(tau) <= target_risk
    while maximizing coverage >= min_coverage.
    """
    points, _ = compute_risk_coverage_curve(val_records, score_key=score_key, num_threshold_bins=100)
    if not points:
        return ThresholdTuningResult(
            target_risk=target_risk,
            min_coverage=min_coverage,
            optimal_threshold=0.65,
            val_coverage=0.0,
            val_selective_risk=0.0,
            val_selective_accuracy=0.0,
            points_evaluated=0,
        )

    grid: List[Dict[str, float]] = []
    admissible_points: List[RiskCoveragePoint] = []

    for p in points:
        grid.append({
            "threshold": p.threshold,
            "coverage": p.coverage,
            "selective_risk": p.selective_risk,
            "selective_accuracy": p.selective_accuracy,
        })
        if p.selective_risk <= target_risk and p.coverage >= min_coverage:
            admissible_points.append(p)

    if admissible_points:
        # Choose the point that maximizes coverage among risk-admissible candidates
        best = max(admissible_points, key=lambda p: (p.coverage, -p.selective_risk))
    else:
        # Fallback: Minimum risk point that meets min_coverage, or best available risk
        coverage_candidates = [p for p in points if p.coverage >= min_coverage]
        if coverage_candidates:
            best = min(coverage_candidates, key=lambda p: p.selective_risk)
        else:
            best = min(points, key=lambda p: p.selective_risk)

    return ThresholdTuningResult(
        target_risk=target_risk,
        min_coverage=min_coverage,
        optimal_threshold=best.threshold,
        val_coverage=best.coverage,
        val_selective_risk=best.selective_risk,
        val_selective_accuracy=best.selective_accuracy,
        points_evaluated=len(points),
        threshold_grid=grid[:20],  # Sample for serialization
    )


def evaluate_abstention(
    records: List[PredictionRecord],
    tuned_threshold: Optional[float] = None,
    unanswerable_labels: Optional[List[str]] = None,
) -> AbstentionEvaluationReport:
    """
    Full Abstention and Selective Answering evaluation for a prediction run.
    """
    total = len(records)
    if total == 0:
        return AbstentionEvaluationReport()

    abstained = sum(1 for r in records if r.abstained)
    answered = total - abstained
    coverage = round(answered / total, 4) if total > 0 else 0.0

    correct = sum(1 for r in records if r.is_correct is True)
    sel_acc = round(correct / answered, 4) if answered > 0 else 0.0
    sel_risk = round(1.0 - sel_acc, 4) if answered > 0 else 0.0
    raw_acc = round(correct / total, 4) if total > 0 else 0.0

    curve_points, aurc = compute_risk_coverage_curve(records)
    unans_metrics = evaluate_abstention_classification(records, unanswerable_labels=unanswerable_labels)

    return AbstentionEvaluationReport(
        total_examples=total,
        answered_count=answered,
        abstained_count=abstained,
        coverage=coverage,
        selective_accuracy=sel_acc,
        selective_risk=sel_risk,
        raw_accuracy=raw_acc,
        aurc=aurc,
        curve_points=curve_points,
        unanswerable_metrics=unans_metrics,
        validation_tuning_applied=tuned_threshold is not None,
        tuned_threshold=tuned_threshold,
    )
