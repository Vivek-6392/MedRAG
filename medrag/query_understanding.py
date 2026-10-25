"""
Phase 1 — Clinical Query Understanding
Extracts: entities • intent • clinical context • risk level
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import LLMClient


@dataclass
class ClinicalQuery:
    original_question: str
    entities: List[str] = field(default_factory=list)   # diseases, drugs, symptoms
    intent: str = "general"                              # see INTENT_KEYWORDS
    context: str = ""                                    # patient-specific context
    clinical_risk: float = 0.5                          # 0–1 (1 = highest risk)
    search_query: str = ""                               # cleaned retrieval query


class ClinicalQueryUnderstanding:
    """
    Parses a medical question to produce a structured ClinicalQuery.
    Falls back to heuristic parsing when no LLM client is available.
    """

    INTENT_KEYWORDS: dict[str, list[str]] = {
        "diagnosis": [
            "diagnose", "diagnosis", "what is", "identify", "cause",
            "symptom", "present", "differential",
        ],
        "treatment": [
            "treat", "treatment", "therapy", "management", "drug",
            "medication", "prescribe", "regimen",
        ],
        "prognosis": [
            "prognosis", "survival", "outcome", "mortality", "life expectancy",
            "recurrence",
        ],
        "dosage": [
            "dose", "dosage", "how much", " mg", "frequency",
            "administration", "titrat",
        ],
        "safety": [
            "safe", "safety", "contraindication", "side effect", "adverse",
            "risk", "pregnancy", "breastfeed", "allerg",
        ],
        "mechanism": [
            "mechanism", "how does", "why does", "pathway", "action", "pharmacokinetic",
        ],
    }

    HIGH_RISK_TERMS: list[str] = [
        "pregnancy", "pregnant", "pediatric", "child", "infant", "neonatal",
        "renal failure", "hepatic failure", "overdose", "emergency",
        "anaphylaxis", "contraindicated", "fatal", "icu", "critically ill",
        "sepsis", "cardiac arrest", "allergy",
    ]

    def __init__(self, llm_client: Optional["LLMClient"] = None):
        self.llm = llm_client

    # ── public API ──────────────────────────────────────────────────────────

    def parse(self, question: str) -> ClinicalQuery:
        if self.llm and self.llm.backend != "mock":
            try:
                return self._llm_parse(question)
            except Exception:
                pass
        return self._heuristic_parse(question)

    # ── private ─────────────────────────────────────────────────────────────

    def _llm_parse(self, question: str) -> ClinicalQuery:
        prompt = (
            "You are a clinical NLP expert. Analyse the following medical question and "
            "return a JSON object with these exact keys:\n"
            "  entities       : list of medical entities (diseases, drugs, symptoms, procedures)\n"
            '  intent         : one of [diagnosis, treatment, prognosis, dosage, safety, mechanism, general]\n'
            "  context        : any patient-specific context mentioned (empty string if none)\n"
            "  clinical_risk  : float 0–1 (1 = highest, e.g. paediatric dosing, pregnancy, emergency)\n"
            "  search_query   : a concise retrieval query (≤12 words)\n\n"
            f"Question: {question}\n\n"
            "Respond with ONLY valid JSON — no markdown, no extra text."
        )
        raw = self.llm.complete(prompt, max_tokens=300)
        data = json.loads(raw)
        return ClinicalQuery(
            original_question=question,
            entities=data.get("entities", []),
            intent=data.get("intent", "general"),
            context=data.get("context", ""),
            clinical_risk=float(data.get("clinical_risk", 0.5)),
            search_query=data.get("search_query", question),
        )

    def _heuristic_parse(self, question: str) -> ClinicalQuery:
        q = question.lower()

        # Intent
        intent = "general"
        for name, kws in self.INTENT_KEYWORDS.items():
            if any(kw in q for kw in kws):
                intent = name
                break

        # Risk
        hits = sum(1 for t in self.HIGH_RISK_TERMS if t in q)
        risk = min(0.3 + hits * 0.15, 1.0)
        if intent in ("dosage", "safety"):
            risk = max(risk, 0.6)

        # Entities — capitalised noun phrases
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", question)

        return ClinicalQuery(
            original_question=question,
            entities=entities[:10],
            intent=intent,
            context="",
            clinical_risk=round(risk, 2),
            search_query=question,
        )
