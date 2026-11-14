"""
Pytest configuration for MedRAG test suite.
Ensures deterministic, offline-capable, and safe test execution.
"""
import os
from pathlib import Path

# Set offline flags if Hugging Face cache is populated to avoid outbound network calls during tests
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
if hf_cache.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Disable tokenizers parallelism to prevent deadlocks in multi-threaded test runners
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
