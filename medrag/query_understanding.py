"""
Phase 1 — Clinical Query Understanding
Extracts: entities • intent • clinical context • risk level
"""
from __future__ import annotations

import json
import math
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
    QUESTION_STARTERS: set[str] = {
        "are", "can", "could", "does", "do", "has", "have", "how", "is",
        "may", "might", "should", "what", "when", "where", "which", "who",
        "why", "will", "would",
    }

    HIGH_RISK_TERMS: list[str] = [
        "pregnancy", "pregnant", "pediatric", "child", "infant", "neonatal",
        "renal failure", "renal impairment", "chronic kidney disease", "hepatic failure",
        "overdose", "emergency", "elderly", "older adult", "anticoagulant",
        "anticoagulation", "stroke prevention",
        "anaphylaxis", "contraindicated", "fatal", "icu", "critically ill",
        "sepsis", "cardiac arrest", "allergy",
    ]
    CRITICAL_RISK_TERMS: list[str] = [
        "overdose", "emergency", "anaphylaxis", "fatal", "icu",
        "critically ill", "sepsis", "cardiac arrest",
    ]

    def __init__(self, llm_client: Optional["LLMClient"] = None):
        self.llm = llm_client

    # ── public API ──────────────────────────────────────────────────────────

    def parse(self, question: str) -> ClinicalQuery:
        if self.llm and self.llm.backend != "mock":
            try:
                return self._llm_parse(question)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                print(f"[Warning] Invalid clinical query response; using heuristic parsing: {exc}")
            except Exception as exc:
                print(f"[Warning] Clinical query LLM failed; using heuristic parsing: {exc}")
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
        if self.llm.call_failed:
            raise RuntimeError(f"Clinical query LLM failure: {self.llm.last_error}")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Clinical query response must be a JSON object.")

        entities = data.get("entities")
        intent = data.get("intent")
        context = data.get("context")
        risk = data.get("clinical_risk")
        search_query = data.get("search_query")
        valid_intents = {*self.INTENT_KEYWORDS, "general"}
        if (
            not isinstance(entities, list)
            or not all(isinstance(entity, str) for entity in entities)
            or intent not in valid_intents
            or not isinstance(context, str)
            or not isinstance(risk, (int, float))
            or isinstance(risk, bool)
            or not math.isfinite(risk)
            or not 0.0 <= risk <= 1.0
            or not isinstance(search_query, str)
            or not search_query.strip()
        ):
            raise ValueError("Clinical query response does not match the required schema.")

        return ClinicalQuery(
            original_question=question,
            entities=entities[:10],
            intent=intent,
            context=context,
            clinical_risk=float(risk),
            search_query=search_query.strip(),
        )

    def _heuristic_parse(self, question: str) -> ClinicalQuery:
        from .risk_policy import ClinicalRiskPolicy

        q = question.lower()

        # Intent
        intent = "general"
        for name, kws in self.INTENT_KEYWORDS.items():
            if any(kw in q for kw in kws):
                intent = name
                break

        # Risk via ClinicalRiskPolicy
        assessment = ClinicalRiskPolicy.assess_query(question, intent=intent)
        risk = assessment.score

        # Entities — capitalised noun phrases
        entities = [
            entity
            for entity in re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", question)
            if entity.casefold() not in self.QUESTION_STARTERS
        ]

        return ClinicalQuery(
            original_question=question,
            entities=entities[:10],
            intent=intent,
            context="",
            clinical_risk=round(risk, 2),
            search_query=question,
        )
