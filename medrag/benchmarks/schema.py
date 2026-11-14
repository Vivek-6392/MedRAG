"""
Normalized Benchmark Schemas for Medical QA Evaluation.
Defines schemas for benchmark examples, prediction records, and retrieval traces.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkExample(BaseModel):
    """
    Normalized representation of a single benchmark evaluation item.
    Supports multiple-choice, boolean, and open-ended medical QA tasks.
    """
    example_id: str = Field(..., description="Unique example identifier within the dataset")
    dataset: str = Field(..., description="Dataset identifier, e.g. 'medqa', 'pubmedqa', 'synthetic'")
    split: str = Field(..., description="Dataset split: 'train', 'dev', 'val', or 'test'")
    question: str = Field(..., description="Clinical question or patient vignette")
    options: Dict[str, str] = Field(default_factory=dict, description="Multiple choice options mapping label -> text")
    context: List[str] = Field(default_factory=list, description="Ground-truth passages, abstracts, or vignette notes")
    gold_answer: str = Field(..., description="Canonical ground-truth answer (option letter or label)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary dataset-specific metadata")

    def formatted_input(self, include_context: bool = False) -> str:
        """
        Formats the clinical question and options into a standard evaluation prompt.
        """
        lines = []
        if include_context and self.context:
            lines.append("Context:")
            for idx, ctx in enumerate(self.context, 1):
                lines.append(f"[{idx}] {ctx.strip()}")
            lines.append("")

        lines.append(f"Question: {self.question.strip()}")

        if self.options:
            lines.append("\nOptions:")
            # Sort keys naturally: A, B, C, D...
            for key in sorted(self.options.keys()):
                lines.append(f"{key}. {self.options[key].strip()}")
            lines.append("\nProvide the correct option letter and rationale.")

        return "\n".join(lines).strip()


class RetrievalTrace(BaseModel):
    """
    Records round-by-round retrieval provenance for an evaluated question.
    """
    example_id: str
    round_number: int
    search_query: str
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    retrieval_scores: Dict[str, float] = Field(default_factory=dict)
    new_docs_count: int = 0
    total_docs_count: int = 0


class PredictionRecord(BaseModel):
    """
    Structured record of a single benchmark prediction and its full verification provenance.
    """
    example_id: str
    dataset: str
    split: str
    question: str
    gold_answer: str
    predicted_answer: Optional[str] = None
    raw_model_response: str = ""
    normalized_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    abstained: bool = False
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    retrieval_scores: Dict[str, float] = Field(default_factory=dict)
    generated_claims: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    evidence_state: Optional[Dict[str, Any]] = None
    rounds: int = 1
    decision_action: str = "STOP"
    decision_reason: str = ""
    heuristic_evidence_score: Optional[float] = None
    latency_seconds: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
