"""
Dataset Adapters Package.
Automatically registers MedQA, PubMedQA, and Synthetic adapters into BenchmarkRegistry.
"""
from .synthetic import SyntheticAdapter
from .medqa import MedQAAdapter
from .pubmedqa import PubMedQAAdapter

__all__ = ["SyntheticAdapter", "MedQAAdapter", "PubMedQAAdapter"]
