"""
MedRag — Adaptive Evidence-Aware Retrieval and Verification Framework
Public API exports.
"""
from .pipeline import MedRagPipeline, PipelineResult
from .retriever import Document, HybridRetriever, RetrievedDocument
from .generator import LLMClient, GeneratedAnswer
from .evidence_state import EvidenceState, ClaimEvidence, ClaimStatus
from .controller import Action, ControllerDecision
from .query_understanding import ClinicalQuery

__all__ = [
    "MedRagPipeline",
    "PipelineResult",
    "Document",
    "HybridRetriever",
    "RetrievedDocument",
    "LLMClient",
    "GeneratedAnswer",
    "EvidenceState",
    "ClaimEvidence",
    "ClaimStatus",
    "Action",
    "ControllerDecision",
    "ClinicalQuery",
]
