"""
Pydantic Schemas for Configuration, Experiment Tracking, and Freeze Verification.
Enforces configuration immutability and provenance during academic benchmarks.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    """
    Configuration specification for reproducible research evaluations.
    """
    schema_version: str = Field("1.0.0", description="Semantic version of configuration schema")
    experiment_id: str = Field(..., description="Unique experiment identifier, e.g. 'medqa_hybrid_baseline'")
    description: str = Field("", description="Detailed hypothesis or ablation description")
    benchmark_name: str = Field(..., description="Target benchmark name (e.g., 'synthetic', 'medqa', 'pubmedqa')")
    split: str = Field("test", description="Dataset split ('val' for tuning, 'test' for final evaluation)")
    protocol: str = Field("rag_adaptive", description="Evaluation protocol: 'zero_shot', 'rag_standard', or 'rag_adaptive'")
    limit: Optional[int] = Field(None, ge=1, description="Evaluation limit for smoke testing")
    resume: bool = Field(True, description="Whether to resume from existing predictions if interrupted")
    corpus_path: Optional[str] = Field(None, description="Optional path to custom knowledge corpus")
    output_dir: str = Field("runs", description="Root directory for experiment outputs")
    
    # LLM settings
    llm_backend: str = Field("groq", description="LLM backend: 'groq', 'openai', 'ollama', or 'mock'")
    llm_model: str = Field("openai/gpt-oss-120b", description="Model identifier")
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(1024, gt=0)

    # Retrieval settings
    top_k_retrieve: int = Field(20, gt=0)
    top_k_rerank: int = Field(5, gt=0)
    bm25_weight: float = Field(0.4, ge=0.0, le=1.0)
    dense_weight: float = Field(0.6, ge=0.0, le=1.0)
    dense_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    use_reranker: bool = Field(True)
    reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Claim Verification settings
    nli_model: str = Field("cross-encoder/nli-MiniLM2-L6-H768")
    nli_batch_size: int = Field(16, gt=0)
    support_threshold: float = Field(0.70, ge=0.0, le=1.0)
    conflict_threshold: float = Field(0.35, ge=0.0, le=1.0)
    missing_threshold: float = Field(0.35, ge=0.0, le=1.0)
    clinical_confidence_threshold: float = Field(0.65, ge=0.0, le=1.0)

    # Controller / Stopping settings
    max_search_rounds: int = Field(3, ge=1, le=10)
    high_risk_threshold: float = Field(0.75, ge=0.0, le=1.0)
    enable_abstention: bool = Field(True, description="Whether to allow selective abstention on risk/gaps")

    # Component toggles for Baselines and Ablations
    enable_hybrid: bool = Field(True, description="Whether hybrid (BM25+Dense RRF) retrieval is enabled")
    enable_reranking: bool = Field(True, description="Whether cross-encoder neural reranking is enabled")
    enable_claim_decomposition: bool = Field(True, description="Whether answers are decomposed into atomic claims")
    enable_nli_verification: bool = Field(True, description="Whether sentence-level NLI verification is performed")
    enable_evidence_state: bool = Field(True, description="Whether evidence state and conflict tracking are active")
    enable_iterative: bool = Field(True, description="Whether iterative multi-round search is enabled")
    enable_risk_policy: bool = Field(True, description="Whether risk-aware adaptive stopping policy is active")
    enable_source_quality_weighting: bool = Field(True, description="Whether source quality and recency decay are applied")

    # Seeds and determinism
    random_seed: int = Field(42, description="Random seed for deterministic runs")

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ExperimentConfig":
        if self.missing_threshold > self.support_threshold:
            raise ValueError("missing_threshold must not exceed support_threshold.")
        if self.top_k_rerank > self.top_k_retrieve:
            raise ValueError("top_k_rerank cannot exceed top_k_retrieve.")
        return self

    def config_hash(self) -> str:
        """Computes a SHA256 signature of the configuration to verify frozen state."""
        raw = self.model_dump_json(exclude={"description"})
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_medrag_config(self) -> Any:
        """Converts to runtime MedRagConfig dataclass."""
        from config import MedRagConfig
        return MedRagConfig(
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            llm_temperature=self.llm_temperature,
            llm_max_tokens=self.llm_max_tokens,
            top_k_retrieve=self.top_k_retrieve,
            top_k_rerank=self.top_k_rerank,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            dense_model=self.dense_model,
            use_reranker=self.use_reranker and self.enable_reranking,
            reranker_model=self.reranker_model,
            nli_model=self.nli_model,
            nli_batch_size=self.nli_batch_size,
            support_threshold=self.support_threshold,
            conflict_threshold=self.conflict_threshold,
            missing_threshold=self.missing_threshold,
            clinical_confidence_threshold=self.clinical_confidence_threshold,
            max_search_rounds=self.max_search_rounds if self.enable_iterative else 1,
            high_risk_threshold=self.high_risk_threshold,
            enable_hybrid=self.enable_hybrid,
            enable_reranking=self.enable_reranking,
            enable_claim_decomposition=self.enable_claim_decomposition,
            enable_nli_verification=self.enable_nli_verification,
            enable_evidence_state=self.enable_evidence_state,
            enable_iterative=self.enable_iterative,
            enable_risk_policy=self.enable_risk_policy,
            enable_abstention=self.enable_abstention,
            enable_source_quality_weighting=self.enable_source_quality_weighting,
        )

    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        """Loads and validates ExperimentConfig from a YAML or JSON file."""
        import os
        import yaml
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                raw_dict = yaml.safe_load(f)
            else:
                raw_dict = json.load(f)
        return cls.model_validate(raw_dict)

    def to_file(self, path: str) -> None:
        """Serializes ExperimentConfig to a YAML or JSON file."""
        import os
        import yaml
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        dump_dict = self.model_dump()
        with open(path, "w", encoding="utf-8") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                yaml.dump(dump_dict, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(dump_dict, f, indent=2)

