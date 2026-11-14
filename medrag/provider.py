"""
Centralized LLM Provider Resolver.
Manages provider resolution, credential lookup, model defaults, URL endpoints,
and configuration validation across CLI, Streamlit, and experiment runners.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .logger import get_logger

logger = get_logger("medrag.provider")

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class ProviderType(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"
    MOCK = "mock"


@dataclass
class ResolvedProvider:
    """Fully resolved and validated LLM provider configuration."""
    backend: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1024
    is_auto_selected: bool = False
    selection_reason: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_client_kwargs(self) -> Dict[str, Any]:
        """Convert into kwargs for LLMClient initialization."""
        return {
            "backend": self.backend,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class ProviderResolver:
    """
    Centralized resolver that maps requested backends and environment variables
    to validated provider configurations.
    """

    SUPPORTED_PROVIDERS = {
        ProviderType.OPENAI.value,
        ProviderType.GROQ.value,
        ProviderType.OLLAMA.value,
        ProviderType.MOCK.value,
    }

    @classmethod
    def resolve(
        cls,
        requested_provider: Optional[str] = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        env: Optional[Dict[str, str]] = None,
    ) -> ResolvedProvider:
        """
        Resolves provider, credentials, endpoints, and models with explicit provenance.
        """
        environ = os.environ if env is None else env
        target = (requested_provider or "auto").strip().lower()

        # ── 1. Genuine Automatic Selection ────────────────────────────────────
        is_auto = target in ("auto", "", "none")
        selection_reason = ""

        if is_auto:
            env_backend = environ.get("LLM_BACKEND", "").strip().lower()
            if env_backend and env_backend in cls.SUPPORTED_PROVIDERS:
                target = env_backend
                selection_reason = f"Selected '{target}' via explicit LLM_BACKEND environment variable"
            elif environ.get("GROQ_API_KEY"):
                target = ProviderType.GROQ.value
                selection_reason = "Auto-selected 'groq' (found GROQ_API_KEY in environment)"
            elif environ.get("OPENAI_API_KEY"):
                target = ProviderType.OPENAI.value
                selection_reason = "Auto-selected 'openai' (found OPENAI_API_KEY in environment)"
            elif environ.get("OLLAMA_BASE_URL") or environ.get("OLLAMA_MODEL"):
                target = ProviderType.OLLAMA.value
                selection_reason = "Auto-selected 'ollama' (found OLLAMA configuration in environment)"
            else:
                target = ProviderType.MOCK.value
                selection_reason = (
                    "Auto-selected 'mock' backend (no API credentials found in environment). "
                    "Set GROQ_API_KEY or OPENAI_API_KEY for live models."
                )
        else:
            selection_reason = f"Explicitly requested provider '{target}'"

        if target not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{target}'. Supported providers: {sorted(cls.SUPPORTED_PROVIDERS)}"
            )

        # ── 2. Credential & Endpoint Resolution ──────────────────────────────
        resolved_api_key = api_key
        resolved_base_url = base_url
        resolved_model = model

        if target == ProviderType.GROQ.value:
            resolved_api_key = resolved_api_key or environ.get("GROQ_API_KEY") or environ.get("LLM_API_KEY")
            resolved_base_url = (
                resolved_base_url
                or environ.get("GROQ_BASE_URL")
                or environ.get("LLM_BASE_URL")
                or DEFAULT_GROQ_BASE_URL
            )
            resolved_model = (
                resolved_model
                or environ.get("GROQ_MODEL")
                or environ.get("LLM_MODEL")
                or DEFAULT_GROQ_MODEL
            )

        elif target == ProviderType.OPENAI.value:
            resolved_api_key = resolved_api_key or environ.get("OPENAI_API_KEY") or environ.get("LLM_API_KEY")
            resolved_base_url = (
                resolved_base_url
                or environ.get("OPENAI_BASE_URL")
                or environ.get("LLM_BASE_URL")
            )
            resolved_model = (
                resolved_model
                or environ.get("OPENAI_MODEL")
                or environ.get("LLM_MODEL")
                or DEFAULT_OPENAI_MODEL
            )

        elif target == ProviderType.OLLAMA.value:
            resolved_base_url = (
                resolved_base_url
                or environ.get("OLLAMA_BASE_URL")
                or environ.get("LLM_BASE_URL")
                or DEFAULT_OLLAMA_BASE_URL
            )
            resolved_model = (
                resolved_model
                or environ.get("OLLAMA_MODEL")
                or environ.get("LLM_MODEL")
                or DEFAULT_OLLAMA_MODEL
            )

        elif target == ProviderType.MOCK.value:
            resolved_model = resolved_model or "mock-clinical-model"

        # Global model override check if not already explicitly supplied
        if not model and environ.get("MODEL"):
            resolved_model = environ["MODEL"].strip()

        # ── 3. Validation ────────────────────────────────────────────────────
        cls._validate_provider_config(target, resolved_model, resolved_api_key)

        logger.info(
            f"Provider resolved: backend='{target}', model='{resolved_model}', "
            f"base_url='{resolved_base_url or 'default'}'. Reason: {selection_reason}"
        )

        return ResolvedProvider(
            backend=target,
            model=resolved_model or "default",
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            is_auto_selected=is_auto,
            selection_reason=selection_reason,
        )

    @classmethod
    def _validate_provider_config(
        cls,
        backend: str,
        model: Optional[str],
        api_key: Optional[str],
    ) -> None:
        if backend == ProviderType.GROQ.value:
            if not api_key:
                raise ValueError(
                    "Groq provider selected but GROQ_API_KEY is missing. "
                    "Please set GROQ_API_KEY in your .env or pass --api-key."
                )
        elif backend == ProviderType.OPENAI.value:
            if not api_key:
                raise ValueError(
                    "OpenAI provider selected but OPENAI_API_KEY is missing. "
                    "Please set OPENAI_API_KEY in your .env or pass --api-key."
                )
        elif backend == ProviderType.OLLAMA.value:
            if not model:
                raise ValueError("Ollama provider requires a non-empty model identifier.")
