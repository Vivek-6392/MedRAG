"""
Experiment Manifest and Provenance Tracking.
Captures system environment, model versions, git commit, dataset splits,
and prompt hashes to guarantee end-to-end scientific reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from ..config_schema import ExperimentConfig


def get_git_commit_sha() -> str:
    """Retrieves the current git commit SHA, or 'unknown' if not in a repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_software_versions() -> Dict[str, str]:
    """Captures installed versions of core ML and evaluation libraries without importing heavyweight modules."""
    import importlib.metadata
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    packages = ["torch", "transformers", "pydantic", "sentence-transformers", "rank-bm25", "pyyaml"]
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "not_installed"
    return versions


def get_hardware_info() -> Dict[str, Any]:
    """Captures hardware and accelerator execution details for reproducibility."""
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": False,
        "gpu_device_name": "none",
        "gpu_count": 0,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def compute_prompts_hash() -> str:
    """Computes a SHA256 digest of core answer generation and verification prompt templates."""
    from ..generator import AnswerGenerator
    from ..query_understanding import ClinicalQueryUnderstanding

    prompts = [
        getattr(AnswerGenerator, "_SYSTEM_PROMPT", ""),
        getattr(AnswerGenerator, "_USER_PROMPT", ""),
        getattr(AnswerGenerator, "_CLAIMS_PROMPT", ""),
        getattr(AnswerGenerator, "_GAP_QUERY_PROMPT", ""),
        getattr(ClinicalQueryUnderstanding, "_EXTRACTION_PROMPT", ""),
    ]
    raw = "||".join(prompts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class ExperimentManifest(BaseModel):
    """
    Immutable provenance record saved with every evaluation run.
    Guarantees that an independent researcher can reproduce exact conditions.
    """
    experiment_id: str
    status: str = Field(default="running", description="Status: 'running', 'completed', 'failed', 'resumed'")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    # Versioning & Environment
    git_commit: str = Field(default_factory=get_git_commit_sha)
    software_versions: Dict[str, str] = Field(default_factory=get_software_versions)
    hardware_info: Dict[str, Any] = Field(default_factory=get_hardware_info)

    # Dataset & Split
    dataset_id: str
    dataset_version: str
    split: str
    num_examples: int = 0
    limit: Optional[int] = None
    corpus_version: str = "v1"
    corpus_hash: Optional[str] = None

    # Models & Components
    config_hash: str
    protocol: str = "rag_adaptive"
    llm_backend: str
    llm_model: str
    llm_model_revision: str = "default"
    dense_model: str
    dense_model_revision: str = "main"
    reranker_model: str
    reranker_model_revision: str = "main"
    nli_model: str
    nli_model_revision: str = "main"
    random_seed: int = 42

    # Chunking & Retrieval Parameters
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
    bm25_weight: float = 0.4
    dense_weight: float = 0.6

    # Prompts & Controller
    prompt_templates_hash: str = Field(default_factory=compute_prompts_hash)
    llm_temperature: float = 0.0
    support_threshold: float
    conflict_threshold: float
    high_risk_threshold: float
    max_search_rounds: int

    # Summary metrics placeholder
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_config_and_adapter(
        cls,
        config: ExperimentConfig,
        adapter_version: str,
        num_examples: int = 0,
        notes: str = "",
    ) -> "ExperimentManifest":
        """Factory constructor instantiating manifest from config and adapter metadata."""
        return cls(
            experiment_id=config.experiment_id,
            dataset_id=config.benchmark_name,
            dataset_version=adapter_version,
            split=config.split,
            num_examples=num_examples,
            limit=config.limit,
            config_hash=config.config_hash(),
            protocol=config.protocol,
            llm_backend=config.llm_backend,
            llm_model=config.llm_model,
            dense_model=config.dense_model,
            reranker_model=config.reranker_model,
            nli_model=config.nli_model,
            random_seed=config.random_seed,
            top_k_retrieve=config.top_k_retrieve,
            top_k_rerank=config.top_k_rerank,
            bm25_weight=config.bm25_weight,
            dense_weight=config.dense_weight,
            llm_temperature=config.llm_temperature,
            support_threshold=config.support_threshold,
            conflict_threshold=config.conflict_threshold,
            high_risk_threshold=config.high_risk_threshold,
            max_search_rounds=config.max_search_rounds,
            notes=notes,
        )

    def mark_completed(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Marks the manifest as completed and attaches final evaluation metrics."""
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if metrics:
            self.summary_metrics = metrics

    def mark_failed(self, error: str) -> None:
        """Marks the manifest as failed with the error explanation."""
        self.status = "failed"
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.notes = f"{self.notes} [FAILED: {error}]".strip()

    def save(self, filepath: str) -> None:
        """Saves manifest to JSON, creating parent directories if necessary."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str) -> "ExperimentManifest":
        """Loads manifest from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
