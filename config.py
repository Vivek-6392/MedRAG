"""
MedRag Configuration
Adaptive Evidence-Aware Retrieval and Verification Framework
"""
import os
import math
from dataclasses import dataclass
from typing import Optional

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass
class MedRagConfig:
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_backend: str = "groq"        # "openai" | "groq" | "ollama" | "mock"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_retrieve: int = 20           # candidates from BM25 + Dense each
    top_k_rerank: int = 5              # final docs passed to generator/verifier
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "chromadb"     # "chromadb" | "faiss" | "numpy"
    collection_name: str = "medrag_corpus"
    chroma_persist_dir: Optional[str] = "./data/chroma_db"
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── NLI Verifier ─────────────────────────────────────────────────────────
    nli_model: str = "ygivenx/modernbert-base-mednli"
    nli_batch_size: int = 16

    # ── Relevance gate ────────────────────────────────────────────────────────
    # If set, abstain when the best round-1 reranker score is below this value
    # (out-of-corpus questions). None disables. Calibrate on in/out-of-corpus questions.
    relevance_threshold: Optional[float] = None

    # ── Evidence thresholds ───────────────────────────────────────────────────
    support_threshold: float = 0.70    # min entailment for a claim to be "supported"
    conflict_threshold: float = 0.35   # max contradiction allowed
    missing_threshold: float = 0.35    # threshold below which evidence is considered missing
    clinical_confidence_threshold: float = 0.65

    # ── Controller ────────────────────────────────────────────────────────────
    max_search_rounds: int = 3
    high_risk_threshold: float = 0.75  # unresolved claims at/above this risk cannot be partial
    high_risk_penalty: float = 0.15    # retained for compatibility; risk policy is action-based

    # ── Component Toggles for Baselines and Ablations ────────────────────────
    enable_hybrid: bool = True
    enable_reranking: bool = True
    enable_claim_decomposition: bool = True
    enable_nli_verification: bool = True
    enable_evidence_state: bool = True
    enable_iterative: bool = True
    enable_risk_policy: bool = True
    enable_abstention: bool = True
    enable_source_quality_weighting: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.top_k_retrieve < 1 or self.top_k_rerank < 1:
            raise ValueError("Retrieval limits top_k_retrieve and top_k_rerank must be positive.")
        if self.nli_batch_size < 1:
            raise ValueError("nli_batch_size must be positive.")
        if self.max_search_rounds < 1:
            raise ValueError("max_search_rounds must be at least 1.")

        thresholds = {
            "support_threshold": self.support_threshold,
            "conflict_threshold": self.conflict_threshold,
            "missing_threshold": self.missing_threshold,
            "clinical_confidence_threshold": self.clinical_confidence_threshold,
            "high_risk_threshold": self.high_risk_threshold,
        }
        for name, value in thresholds.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value between 0 and 1.")
        if self.missing_threshold > self.support_threshold:
            raise ValueError("missing_threshold must not exceed support_threshold.")

    @classmethod
    def from_env(cls) -> "MedRagConfig":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        cfg = cls()

        from medrag.provider import ProviderResolver
        try:
            resolved = ProviderResolver.resolve()
            cfg.llm_backend = resolved.backend
            cfg.llm_model = resolved.model
            cfg.llm_api_key = resolved.api_key
            cfg.llm_base_url = resolved.base_url
        except Exception:
            # Fallback for minimal test environments without keys
            backend = os.environ.get("LLM_BACKEND", "").strip().lower()
            if backend:
                cfg.llm_backend = backend
            if cfg.llm_backend == "groq":
                cfg.llm_model = os.environ.get("GROQ_MODEL") or os.environ.get("LLM_MODEL") or DEFAULT_GROQ_MODEL
                cfg.llm_api_key = os.environ.get("GROQ_API_KEY")
                cfg.llm_base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            elif cfg.llm_backend == "openai":
                cfg.llm_model = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
                cfg.llm_api_key = os.environ.get("OPENAI_API_KEY")
                cfg.llm_base_url = os.environ.get("OPENAI_BASE_URL")

        if os.environ.get("LLM_MODEL"):
            cfg.llm_model = os.environ["LLM_MODEL"].strip()
        elif os.environ.get("MODEL"):
            cfg.llm_model = os.environ["MODEL"].strip()

        if os.environ.get("LLM_BASE_URL"):
            cfg.llm_base_url = os.environ["LLM_BASE_URL"].strip()
        if os.environ.get("LLM_API_KEY"):
            cfg.llm_api_key = os.environ["LLM_API_KEY"].strip()

        if os.environ.get("NLI_MODEL"):
            cfg.nli_model = os.environ["NLI_MODEL"].strip()

        return cfg

    def resolve_for_backend(self, backend: str, model: Optional[str] = None) -> "MedRagConfig":
        """Re-resolves provider credentials and endpoints for a specific backend."""
        from medrag.provider import ProviderResolver
        resolved = ProviderResolver.resolve(
            requested_provider=backend,
            model=model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )
        self.llm_backend = resolved.backend
        self.llm_model = resolved.model
        self.llm_api_key = resolved.api_key
        self.llm_base_url = resolved.base_url
        return self
