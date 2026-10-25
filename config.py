"""
MedRag Configuration
Adaptive Evidence-Aware Retrieval and Verification Framework
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MedRagConfig:
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_backend: str = "openai"        # "openai" | "groq" | "ollama" | "mock"
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
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── NLI Verifier ─────────────────────────────────────────────────────────
    nli_model: str = "cross-encoder/nli-MiniLM2-L6-H768"
    nli_batch_size: int = 16

    # ── Evidence thresholds ───────────────────────────────────────────────────
    support_threshold: float = 0.70    # min entailment for a claim to be "supported"
    conflict_threshold: float = 0.35   # max contradiction allowed

    # ── Controller ────────────────────────────────────────────────────────────
    max_search_rounds: int = 3
    high_risk_threshold: float = 0.75  # clinical risk above which threshold tightens
    high_risk_penalty: float = 0.15    # extra support needed for high-risk claims

    @classmethod
    def from_env(cls) -> "MedRagConfig":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        cfg = cls()
        backend = os.environ.get("LLM_BACKEND", "").strip().lower()
        if backend:
            cfg.llm_backend = backend

        if os.environ.get("OPENAI_API_KEY"):
            cfg.llm_api_key = os.environ["OPENAI_API_KEY"]
            if not backend:
                cfg.llm_backend = "openai"
            if os.environ.get("OPENAI_MODEL"):
                cfg.llm_model = os.environ["OPENAI_MODEL"]
        elif os.environ.get("GROQ_API_KEY"):
            cfg.llm_api_key = os.environ["GROQ_API_KEY"]
            if not backend:
                cfg.llm_backend = "groq"
            cfg.llm_base_url = "https://api.groq.com/openai/v1"
            cfg.llm_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

        if os.environ.get("LLM_MODEL"):
            cfg.llm_model = os.environ["LLM_MODEL"]
        if os.environ.get("LLM_BASE_URL"):
            cfg.llm_base_url = os.environ["LLM_BASE_URL"]

        return cfg
