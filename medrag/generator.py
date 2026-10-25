"""
Phase 3 — Answer Generator + Atomic Claim Decomposer
LLM client supporting OpenAI / Groq / Ollama / Mock backends.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── LLM Client ───────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified client for OpenAI-compatible LLM APIs.
    Backends:
      "openai"  — OpenAI API
      "groq"    — Groq API (set base_url to https://api.groq.com/openai/v1)
      "ollama"  — Local Ollama (set base_url to http://localhost:11434/v1)
      "mock"    — Rule-based stub for testing without an API key
    """

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None and self.backend != "mock":
            try:
                from openai import OpenAI
                kwargs: dict = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                print("[Warning] openai package not installed — falling back to mock mode.")
                self.backend = "mock"
        return self._client

    def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if self.backend == "mock":
            return self._mock_complete(prompt)
        client = self._get_client()
        if client is None:
            return self._mock_complete(prompt)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return response.choices[0].message.content or ""

    # ── mock backend ─────────────────────────────────────────────────────────

    @staticmethod
    def _mock_complete(prompt: str) -> str:
        p = prompt.lower()

        if "json array" in p or "atomic" in p or "decompose" in p:
            return json.dumps([
                "The treatment is evidence-based and guideline-recommended.",
                "The dosage falls within standard therapeutic range.",
                "No absolute contraindications are mentioned in the evidence.",
                "Monitoring of key parameters is required during therapy.",
            ])

        if "search query" in p or "gap" in p or "missing" in p:
            return "evidence-based treatment efficacy safety contraindications"

        if "entities" in p and "intent" in p:
            question = ""
            if "question:" in p:
                question = prompt.split("Question:")[-1].strip()[:80]
            return json.dumps({
                "entities": ["Condition", "Treatment"],
                "intent": "treatment",
                "context": "",
                "clinical_risk": 0.5,
                "search_query": question or "medical treatment query",
            })

        # Default: generate a template answer
        return (
            "Based on the available clinical evidence, this condition should be managed "
            "according to current evidence-based guidelines. Treatment selection must "
            "account for individual patient factors, comorbidities, and risk-benefit "
            "considerations. Regular monitoring and follow-up are essential. "
            "Contraindications and drug interactions should be assessed before initiating therapy."
        )


# ── Generated Answer ──────────────────────────────────────────────────────────

@dataclass
class GeneratedAnswer:
    answer: str
    rationale: str
    atomic_claims: List[str] = field(default_factory=list)
    confidence: float = 0.5


# ── Answer Generator ─────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Phase 3: Generate answer from question + retrieved evidence,
    then decompose the answer into atomic, verifiable claims.
    """

    _ANSWER_PROMPT = """\
You are an expert medical consultant. Answer the following question using the provided evidence.

Question: {question}

Retrieved Evidence:
{evidence}

Instructions:
- Base your answer on the evidence above.
- Be precise and cite evidence numbers where relevant (e.g. [1], [2]).
- Acknowledge uncertainty or gaps in the evidence if present.
- Do NOT invent facts not supported by the evidence.

Answer:"""

    _CLAIMS_PROMPT = """\
Extract atomic, independently verifiable medical claims from the following medical answer.

Answer: {answer}

Rules:
- Each claim must be a single, self-contained factual statement.
- No conjunctions combining multiple facts.
- Specific and verifiable against medical literature.
- Maximum 20 words per claim.

Return ONLY a JSON array of strings, no markdown, no explanation.
Example: ["Metformin reduces HbA1c by 1-2%.", "Metformin is contraindicated in eGFR < 30."]"""

    _GAP_QUERY_PROMPT = """\
You are a medical information specialist. Generate a targeted PubMed-style search query \
to find evidence for the following unresolved or conflicted medical claims.

Original Question: {question}

Unresolved claims:
{weak_claims}

Return ONE concise search query (≤10 words). No explanation, just the query."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    # ── public API ───────────────────────────────────────────────────────────

    def generate_answer(
        self,
        question: str,
        retrieved_docs: list,
        max_evidence_chars: int = 3500,
    ) -> GeneratedAnswer:
        evidence_text = self._format_evidence(retrieved_docs, max_evidence_chars)
        prompt = self._ANSWER_PROMPT.format(question=question, evidence=evidence_text)
        answer = self.llm.complete(prompt)
        claims = self.decompose_claims(answer)
        return GeneratedAnswer(
            answer=answer,
            rationale=answer,
            atomic_claims=claims,
        )

    def decompose_claims(self, answer: str) -> List[str]:
        prompt = self._CLAIMS_PROMPT.format(answer=answer)
        raw = self.llm.complete(prompt, max_tokens=512)
        # Try to extract a JSON array from the response
        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [str(c).strip() for c in parsed if c]
        except (json.JSONDecodeError, AttributeError):
            pass
        # Fallback: split by sentence endings
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        return [s.strip() for s in sentences if len(s.strip()) > 20][:8]

    def generate_gap_query(self, original_question: str, weak_claims: list) -> str:
        if not weak_claims:
            return original_question
        claims_text = "\n".join(f"- {c.claim}" for c in weak_claims[:3])
        prompt = self._GAP_QUERY_PROMPT.format(
            question=original_question, weak_claims=claims_text
        query = self.llm.complete(prompt, max_tokens=60).strip().strip('"').strip("'")
        return query if query else original_question

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format_evidence(retrieved_docs: list, max_chars: int) -> str:
        parts: List[str] = []
        running = 0
        for i, rdoc in enumerate(retrieved_docs[:8]):
            doc = rdoc.document
            snippet = doc.text[:500]
            entry = f"[{i+1}] {doc.title} ({doc.year}, {doc.source})\n{snippet}"
            if running + len(entry) > max_chars:
                break
            parts.append(entry)
            running += len(entry)
        return "\n\n".join(parts) if parts else "No evidence retrieved."
