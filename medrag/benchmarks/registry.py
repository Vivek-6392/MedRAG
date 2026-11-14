"""
Benchmark Registry for Medical QA Datasets.
Provides adapter registration, discovery, and instantiation.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Type
from .adapter import DatasetAdapter


class BenchmarkRegistry:
    """
    Central registry for medical benchmark dataset adapters.
    """
    _adapters: Dict[str, Type[DatasetAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: Type[DatasetAdapter]) -> None:
        """Registers a DatasetAdapter class under a canonical name."""
        key = name.strip().lower()
        cls._adapters[key] = adapter_cls

    @classmethod
    def register_adapter(cls, name: str) -> Callable[[Type[DatasetAdapter]], Type[DatasetAdapter]]:
        """Decorator to register a DatasetAdapter class."""
        def decorator(adapter_cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
            cls.register(name, adapter_cls)
            return adapter_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> DatasetAdapter:
        """Instantiates and returns the requested dataset adapter."""
        key = name.strip().lower()
        if key not in cls._adapters:
            available = ", ".join(sorted(cls._adapters.keys()))
            raise KeyError(
                f"Unknown benchmark '{name}'. Available benchmarks: [{available}]."
            )
        return cls._adapters[key](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Returns the list of registered benchmark identifiers."""
        return sorted(list(cls._adapters.keys()))

    @classmethod
    def clear(cls) -> None:
        """Clears registered adapters (useful for test isolation)."""
        cls._adapters.clear()
