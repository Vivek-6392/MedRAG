"""
Research-Grade Claim-Level Evaluation and Faithfulness Engine.
Provides standardized claim-evidence annotation schema, manual subset evaluation,
and faithfulness metrics:
- Claim-level Precision, Recall, and Macro-F1 across 4 canonical labels:
    * SUPPORTED
    * CONTRADICTED
    * NOT_ENOUGH_INFORMATION
    * PARTIALLY_SUPPORTED
- Unsupported Claim Rate (fraction of generated claims lacking adequate grounding)
- False-Support Rate (claims predicted as supported but contradicted/unsupported by gold)
- Citation Precision (proportion of cited passages that genuinely entail the claim)
- Citation Completeness (proportion of necessary gold reference evidence retrieved/cited)
"""
from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field

from .schema import PredictionRecord


class ClaimEvidenceLabel(str, Enum):
    """Canonical ground-truth labels for atomic claim verification."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"
    PARTIALLY_SUPPORTED = "partially_supported"

    @classmethod
    def from_str(cls, val: str) -> ClaimEvidenceLabel:
        v = str(val).strip().lower().replace("-", "_").replace(" ", "_")
        if v in ("supported", "support", "entailed", "entailment"):
            return cls.SUPPORTED
        if v in ("contradicted", "contradiction", "refuted", "conflict"):
            return cls.CONTRADICTED
        if v in ("not_enough_information", "nei", "missing", "neutral", "insufficient"):
            return cls.NOT_ENOUGH_INFORMATION
        if v in ("partially_supported", "partial", "partially"):
            return cls.PARTIALLY_SUPPORTED
        raise ValueError(f"Unrecognized claim evidence label: '{val}'. Expected one of {list(cls)}.")


class ClaimAnnotation(BaseModel):
    """
    Standardized Claim-Evidence Ground-Truth Annotation Record.
    """
    example_id: str = Field(..., description="Benchmark example identifier")
    claim_id: str = Field(..., description="Unique atomic claim identifier within the example")
    claim: str = Field(..., description="Atomic claim statement in natural language")
    evidence_id: str = Field(..., description="Target reference evidence passage or document ID")
    label: ClaimEvidenceLabel = Field(..., description="Expert verification label")
    annotator_id: str = Field("expert_annotator", description="Identifier of the clinical/expert annotator")
    adjudication_status: str = Field(
        "adjudicated",
        description="Status: 'single_annotator', 'consensus', 'adjudicated', or 'gold'"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional annotation metadata")


class ClaimEvaluationMetrics(BaseModel):
    """
    Evaluation Metrics for Atomic Claim Faithfulness and Attribution.
    """
    total_annotated_claims: int = 0
    total_generated_claims: int = 0
    annotated_subset_evaluated: int = 0

    # Macro & overall claim classification metrics
    precision: float = Field(0.0, description="Macro-averaged precision across claim labels")
    recall: float = Field(0.0, description="Macro-averaged recall across claim labels")
    macro_f1: float = Field(0.0, description="Macro-averaged F1 score across claim labels")

    # Per-label breakdown
    per_label_precision: Dict[str, float] = Field(default_factory=dict)
    per_label_recall: Dict[str, float] = Field(default_factory=dict)
    per_label_f1: Dict[str, float] = Field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    # Core faithfulness & hallucination failure rates
    unsupported_claim_rate: float = Field(
        0.0,
        description="Proportion of generated claims lacking support (contradicted or NEI)"
    )
    false_support_rate: float = Field(
        0.0,
        description="Proportion of claims predicted as supported whose gold label is contradicted or NEI"
    )

    # Citation quality metrics
    citation_precision: float = Field(
        0.0,
        description="Fraction of citations made by the model that genuinely support the claim"
    )
    citation_completeness: float = Field(
        0.0,
        description="Fraction of necessary gold evidence passages that were successfully cited"
    )

    # Guardrail and provenance notice
    annotation_subset_warning: str = Field(
        "Measured strictly on a manually annotated subset. These metrics must not be extrapolated "
        "to the entire unannotated benchmark."
    )


def load_claim_annotations(path: str) -> List[ClaimAnnotation]:
    """
    Loads claim annotations from JSON or JSONL format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Claim annotation file not found: {path}")

    records: List[ClaimAnnotation] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []

        if content.startswith("["):
            items = json.loads(content)
            for it in items:
                records.append(ClaimAnnotation(**it))
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    records.append(ClaimAnnotation(**json.loads(line)))

    return records


def save_claim_annotations(annotations: List[ClaimAnnotation], path: str) -> None:
    """
    Saves claim annotations to JSONL.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann.model_dump(), default=str) + "\n")


def map_pipeline_status_to_label(status_str: str) -> ClaimEvidenceLabel:
    """Maps internal pipeline ClaimEvidence status to canonical ClaimEvidenceLabel."""
    s = str(status_str).strip().lower()
    if s in ("supported", "ok"):
        return ClaimEvidenceLabel.SUPPORTED
    if s in ("conflicted", "conflict", "contradicted"):
        return ClaimEvidenceLabel.CONTRADICTED
    if s in ("missing", "uncertain", "nei", "not_enough_information"):
        return ClaimEvidenceLabel.NOT_ENOUGH_INFORMATION
    if s in ("partially_supported", "partial"):
        return ClaimEvidenceLabel.PARTIALLY_SUPPORTED
    return ClaimEvidenceLabel.NOT_ENOUGH_INFORMATION


def evaluate_claim_faithfulness(
    predictions: List[PredictionRecord],
    annotations: List[ClaimAnnotation],
    subset_only: bool = True,
) -> ClaimEvaluationMetrics:
    """
    Evaluates claim-level verification and attribution against gold annotations.

    Strictly matches by (example_id, claim_id) or fuzzy claim text overlap where claim_id is unset.
    Computes precision, recall, macro-F1, unsupported-claim rate, false-support rate,
    and citation precision & completeness.
    """
    labels = [
        ClaimEvidenceLabel.SUPPORTED.value,
        ClaimEvidenceLabel.CONTRADICTED.value,
        ClaimEvidenceLabel.NOT_ENOUGH_INFORMATION.value,
        ClaimEvidenceLabel.PARTIALLY_SUPPORTED.value,
    ]

    # Index annotations by (example_id, claim_id) and example_id
    gold_by_key: Dict[Tuple[str, str], ClaimAnnotation] = {}
    gold_by_ex: Dict[str, List[ClaimAnnotation]] = {}
    for ann in annotations:
        gold_by_key[(ann.example_id, ann.claim_id)] = ann
        gold_by_ex.setdefault(ann.example_id, []).append(ann)

    # Initialize confusion matrix
    matrix: Dict[str, Dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    tp: Dict[str, int] = {lbl: 0 for lbl in labels}
    fp: Dict[str, int] = {lbl: 0 for lbl in labels}
    fn: Dict[str, int] = {lbl: 0 for lbl in labels}

    evaluated_count = 0
    total_generated = 0
    unsupported_claims = 0
    false_support_count = 0
    total_predicted_supported = 0

    cited_passages_total = 0
    cited_passages_correct = 0
    gold_passages_needed: Set[Tuple[str, str]] = set()
    gold_passages_cited: Set[Tuple[str, str]] = set()

    for pred in predictions:
        ex_id = pred.example_id
        gold_anns = gold_by_ex.get(ex_id, [])
        if subset_only and not gold_anns:
            continue

        ev_state = getattr(pred, "evidence_state", None) or pred.metadata.get("evidence_state")
        if isinstance(ev_state, dict):
            raw_claims = ev_state.get("claims", [])
        elif pred.metadata.get("claim_verification_details"):
            raw_claims = pred.metadata.get("claim_verification_details")
        else:
            raw_claims = [
                {"claim": c, "status": "supported" if pred.supporting_evidence_ids else "missing"}
                for c in pred.generated_claims
            ]
        total_generated += len(raw_claims)

        for c_idx, c_dict in enumerate(raw_claims, 1):
            claim_text = c_dict.get("claim", "").strip()
            c_id = c_dict.get("claim_id") or f"c{c_idx}"
            pred_status = c_dict.get("status", "missing")
            pred_label = map_pipeline_status_to_label(pred_status).value

            # Match with gold annotation
            gold_ann = gold_by_key.get((ex_id, c_id))
            if gold_ann is None:
                # Fallback: Best textual match in example
                for candidate in gold_anns:
                    if candidate.claim.strip().lower() in claim_text.lower() or claim_text.lower() in candidate.claim.strip().lower():
                        gold_ann = candidate
                        break

            if gold_ann is None:
                continue

            evaluated_count += 1
            gold_label = gold_ann.label.value

            matrix[gold_label][pred_label] += 1

            if pred_label == gold_label:
                tp[gold_label] += 1
            else:
                fp[pred_label] += 1
                fn[gold_label] += 1

            if gold_label in (ClaimEvidenceLabel.CONTRADICTED.value, ClaimEvidenceLabel.NOT_ENOUGH_INFORMATION.value):
                unsupported_claims += 1

            if pred_label == ClaimEvidenceLabel.SUPPORTED.value:
                total_predicted_supported += 1
                if gold_label != ClaimEvidenceLabel.SUPPORTED.value:
                    false_support_count += 1

            # Citation evaluation
            gold_ev_id = str(gold_ann.evidence_id).strip().lower()
            if gold_ev_id and gold_ev_id != "none":
                gold_passages_needed.add((ex_id, gold_ev_id))

            # Citations reported for this claim or prediction
            model_cited_docs = [
                str(d).strip().lower()
                for d in c_dict.get("cited_doc_ids", []) or pred.retrieved_doc_ids[:3]
            ]
            for c_doc in model_cited_docs:
                cited_passages_total += 1
                if c_doc == gold_ev_id or gold_ev_id in c_doc or c_doc in gold_ev_id:
                    cited_passages_correct += 1
                    gold_passages_cited.add((ex_id, gold_ev_id))

    # Compute macro precision, recall, and F1
    per_label_p: Dict[str, float] = {}
    per_label_r: Dict[str, float] = {}
    per_label_f1: Dict[str, float] = {}
    macro_p_list: List[float] = []
    macro_r_list: List[float] = []
    macro_f_list: List[float] = []

    for lbl in labels:
        p = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0.0
        r = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        per_label_p[lbl] = round(p, 4)
        per_label_r[lbl] = round(r, 4)
        per_label_f1[lbl] = round(f1, 4)

        if (tp[lbl] + fn[lbl]) > 0:
            macro_p_list.append(p)
            macro_r_list.append(r)
            macro_f_list.append(f1)

    macro_precision = round(sum(macro_p_list) / len(macro_p_list), 4) if macro_p_list else 0.0
    macro_recall = round(sum(macro_r_list) / len(macro_r_list), 4) if macro_r_list else 0.0
    macro_f1 = round(sum(macro_f_list) / len(macro_f_list), 4) if macro_f_list else 0.0

    unsupported_rate = round(unsupported_claims / evaluated_count, 4) if evaluated_count > 0 else 0.0
    false_support_rate = round(false_support_count / total_predicted_supported, 4) if total_predicted_supported > 0 else 0.0

    cit_precision = round(cited_passages_correct / cited_passages_total, 4) if cited_passages_total > 0 else 0.0
    cit_completeness = round(len(gold_passages_cited) / len(gold_passages_needed), 4) if gold_passages_needed else 0.0

    return ClaimEvaluationMetrics(
        total_annotated_claims=len(annotations),
        total_generated_claims=total_generated,
        annotated_subset_evaluated=evaluated_count,
        precision=macro_precision,
        recall=macro_recall,
        macro_f1=macro_f1,
        per_label_precision=per_label_p,
        per_label_recall=per_label_r,
        per_label_f1=per_label_f1,
        confusion_matrix=matrix,
        unsupported_claim_rate=unsupported_rate,
        false_support_rate=false_support_rate,
        citation_precision=cit_precision,
        citation_completeness=cit_completeness,
    )
