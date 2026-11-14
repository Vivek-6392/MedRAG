"""
Common Dataset Adapter Interface for Medical QA Benchmarks.
Provides the abstract base class and contract required for all dataset adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..retriever import Document
from .schema import BenchmarkExample


class DatasetAdapter(ABC):
    """
    Abstract contract for medical QA dataset adapters.
    Encapsulates dataset-specific ingestion, normalization, prompt construction,
    answer normalization, and evaluation logic without modifying core MedRAG.
    """

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Canonical identifier for the benchmark (e.g. 'medqa', 'pubmedqa', 'synthetic')."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Dataset version or commit tag (e.g. '1.0.0', 'usmle-4opt-v1')."""
        pass

    @property
    @abstractmethod
    def supported_splits(self) -> List[str]:
        """List of official splits supported by this benchmark (e.g. ['train', 'dev', 'test'])."""
        pass

    @abstractmethod
    def load_split(self, split: str, limit: Optional[int] = None) -> List[BenchmarkExample]:
        """
        Loads and normalizes raw dataset examples for the requested split.
        """
        pass

    @abstractmethod
    def normalize_example(self, raw_item: Dict[str, Any], split: str) -> BenchmarkExample:
        """
        Converts a raw dataset item into the standardized BenchmarkExample schema.
        """
        pass

    @abstractmethod
    def extract_gold_answer(self, raw_item: Dict[str, Any]) -> str:
        """
        Extracts the canonical ground-truth answer from a raw item.
        """
        pass

    def construct_input(self, example: BenchmarkExample, include_context: bool = False) -> str:
        """
        Constructs the clinical query or prompt string to be processed by MedRAGPipeline.
        Default implementation formats question, context (if requested), and multiple-choice options.
        """
        return example.formatted_input(include_context=include_context)

    @abstractmethod
    def normalize_answer(self, raw_response: str, example: BenchmarkExample) -> Optional[str]:
        """
        Extracts and normalizes the prediction from raw LLM output into canonical form.
        Returns None if the response represents an abstention or cannot be parsed.
        """
        pass

    def evaluate_prediction(
        self,
        predicted: Optional[str],
        gold: str,
        example: BenchmarkExample,
    ) -> bool:
        """
        Determines whether the predicted answer matches the gold answer.
        Default implementation performs case-insensitive stripped exact matching.
        """
        if predicted is None or not str(predicted).strip():
            return False
        return str(predicted).strip().upper() == str(gold).strip().upper()

    def get_default_corpus(self, examples: Optional[List[BenchmarkExample]] = None) -> Optional[List[Document]]:
        """
        Optionally extracts or provides an indexed Document corpus from dataset context passages.
        Returns None if an external corpus must be provided.
        """
        if not examples:
            return None
        
        docs: List[Document] = []
        seen = set()
        for ex in examples:
            for idx, text in enumerate(ex.context):
                doc_id = f"{ex.example_id}_ctx_{idx}"
                if doc_id not in seen and text.strip():
                    seen.add(doc_id)
                    docs.append(Document(
                        doc_id=doc_id,
                        title=f"Context for {ex.example_id}",
                        text=text.strip(),
                        source=ex.dataset,
                        year=2024,
                        quality_score=0.90,
                    ))
        return docs if docs else None
