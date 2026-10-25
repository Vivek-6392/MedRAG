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

        # 1. Determine Backend
        backend = os.environ.get("LLM_BACKEND", "").strip().lower()
        if backend:
            cfg.llm_backend = backend
        elif os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            cfg.llm_backend = "groq"
        elif os.environ.get("OPENAI_API_KEY"):
            cfg.llm_backend = "openai"

        # 2. Configure Backend Credentials & Model from .env
        if cfg.llm_backend == "groq":
            cfg.llm_api_key = os.environ.get("GROQ_API_KEY")
            cfg.llm_base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            cfg.llm_model = (
                os.environ.get("GROQ_MODEL")
                or os.environ.get("LLM_MODEL")
                or os.environ.get("MODEL")
            )
        elif cfg.llm_backend == "openai":
            cfg.llm_api_key = os.environ.get("OPENAI_API_KEY")
            cfg.llm_base_url = os.environ.get("OPENAI_BASE_URL")
            cfg.llm_model = (
                os.environ.get("OPENAI_MODEL")
                or os.environ.get("LLM_MODEL")
                or os.environ.get("MODEL")
            )
        elif cfg.llm_backend == "ollama":
            cfg.llm_base_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
            cfg.llm_model = (
                os.environ.get("OLLAMA_MODEL")
                or os.environ.get("LLM_MODEL")
                or os.environ.get("MODEL")
            )
        else:
            cfg.llm_model = os.environ.get("LLM_MODEL") or os.environ.get("MODEL")

        # 3. Global Overrides from .env
        if os.environ.get("LLM_MODEL"):
            cfg.llm_model = os.environ["LLM_MODEL"].strip()
        elif os.environ.get("MODEL"):
            cfg.llm_model = os.environ["MODEL"].strip()

        if os.environ.get("LLM_BASE_URL"):
            cfg.llm_base_url = os.environ["LLM_BASE_URL"].strip()
        if os.environ.get("LLM_API_KEY"):
            cfg.llm_api_key = os.environ["LLM_API_KEY"].strip()

        return cfg
