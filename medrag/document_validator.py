"""
Document Validation Engine.
Provides consistent, strict validation rules for documents entering through CLI,
Streamlit, or programmatic dataset adapters.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from .retriever import Document


DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
URL_PATTERN = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)

VALID_DOC_TYPES = {
    "guideline",
    "rct",
    "meta_analysis",
    "systematic_review",
    "observational",
    "review",
    "case_report",
    "abstract",
    "textbook",
    "other",
}


class ValidatedDocumentInput(BaseModel):
    """Pydantic schema for strict document ingress validation."""
    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., min_length=15, description="Full passage or abstract text")
    title: str = Field(default="Untitled Document")
    source: str = Field(default="Unknown Source")
    year: int = Field(default=2024)
    quality_score: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    url: Optional[str] = None
    doi: Optional[str] = None
    doc_type: str = Field(default="other")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("Document id cannot be empty.")
        return s

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: Any) -> str:
        s = str(v).strip()
        if len(s) < 15:
            raise ValueError(f"Document text too short ({len(s)} chars); minimum is 15.")
        return s

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Any) -> int:
        try:
            yr = int(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid publication year: {v}")
        max_year = datetime.now().year + 1
        if not (1800 <= yr <= max_year):
            raise ValueError(f"Publication year {yr} must be between 1800 and {max_year}.")
        return yr

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip():
            s = v.strip()
            if not URL_PATTERN.match(s):
                raise ValueError(f"Invalid URL format: '{s}'")
            return s
        return None

    @field_validator("doi")
    @classmethod
    def validate_doi(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip():
            s = v.strip()
            # Strip standard prefixes like https://doi.org/ or doi:
            cleaned = re.sub(r"^(https?://(?:dx\.)?doi\.org/|doi:)", "", s, flags=re.I).strip()
            if not DOI_PATTERN.match(cleaned):
                raise ValueError(f"Invalid DOI format: '{s}'")
            return cleaned
        return None

    @field_validator("doc_type")
    @classmethod
    def validate_doc_type(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if s not in VALID_DOC_TYPES:
            return "other"
        return s


def validate_document(raw: Dict[str, Any]) -> Document:
    """
    Validates a raw document dictionary and converts it into a runtime Document instance.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Document must be a dictionary, got {type(raw).__name__}")

    # Map legacy aliases
    doc_id = raw.get("id") or raw.get("doc_id") or raw.get("pmid")
    text = raw.get("text") or raw.get("abstract") or raw.get("content") or ""
    title = raw.get("title") or "Untitled Document"
    source = raw.get("source") or raw.get("journal") or "Unknown Source"
    year = raw.get("year") or raw.get("pub_year") or 2024
    quality = raw.get("quality_score")
    url = raw.get("url") or raw.get("link")
    doi = raw.get("doi")
    doc_type = raw.get("doc_type") or raw.get("type") or "other"
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    validated = ValidatedDocumentInput(
        id=str(doc_id) if doc_id is not None else "",
        text=str(text),
        title=str(title).strip(),
        source=str(source).strip(),
        year=year,
        quality_score=quality,
        url=url,
        doi=doi,
        doc_type=doc_type,
        metadata=metadata,
    )

    is_default_q = validated.quality_score is None
    final_quality = validated.quality_score if validated.quality_score is not None else 0.50

    combined_meta = dict(validated.metadata)
    if validated.url:
        combined_meta["url"] = validated.url
    if validated.doi:
        combined_meta["doi"] = validated.doi
    combined_meta["doc_type"] = validated.doc_type
    combined_meta["is_default_quality"] = is_default_q

    return Document(
        doc_id=validated.id,
        title=validated.title,
        text=validated.text,
        source=validated.source,
        year=validated.year,
        quality_score=final_quality,
        metadata=combined_meta,
    )


def validate_corpus(items: List[Any]) -> List[Document]:
    """
    Validates an entire collection of documents.
    Raises ValueError with item index on failure.
    """
    if not isinstance(items, list):
        raise TypeError(f"Corpus items must be a list, got {type(items).__name__}")

    documents: List[Document] = []
    seen_ids = set()

    for i, item in enumerate(items):
        try:
            doc = validate_document(item)
            if doc.doc_id in seen_ids:
                raise ValueError(f"Duplicate document id '{doc.doc_id}' detected at index {i}.")
            seen_ids.add(doc.doc_id)
            documents.append(doc)
        except Exception as e:
            raise ValueError(f"Corpus validation failed at document index {i}: {e}") from e

    return documents
