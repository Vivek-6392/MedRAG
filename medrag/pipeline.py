"""
Main Pipeline — Adaptive Evidence-Aware Retrieval and Verification Framework
Orchestrates Clinical Query Understanding, Hybrid Retrieval, Answer Generation,
Claim Verification, Evidence State Management, and the STOP / PARTIAL / CONFLICTED / SEARCH / ABSTAIN Controller.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .query_understanding import ClinicalQueryUnderstanding, ClinicalQuery
from .retriever import HybridRetriever, RetrievedDocument, Document
from .generator import AnswerGenerator, GeneratedAnswer, LLMClient
from .verifier import NLIVerifier
from .evidence_state import EvidenceState
from .controller import EvidenceController, Action, ControllerDecision


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    question: str
    final_answer: str
    action: str                                         # final STOP | PARTIAL | CONFLICTED | ABSTAIN
    evidence_state: Optional[EvidenceState]
    rounds: int
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)
    clinical_query: Optional[ClinicalQuery] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0

    @property
    def trajectory(self) -> List[Dict[str, Any]]:
        """Alias for history (round-by-round trajectory)."""
        return self.history

    def print_summary(self) -> None:
        sep = "=" * 72
        print(f"\n{sep}")
        print("  MEDRAG RESULT")
        print(sep)
        print(f"  Question : {self.question}")
        print(f"  Decision : {self.action}")
        print(f"  Rounds   : {self.rounds}")
        print(f"  Latency  : {self.latency_seconds:.1f}s")
        if self.evidence_state:
            s = self.evidence_state.summary_dict()
            print(f"\n  Evidence State:")
            print(f"    Claims     : {s['total_claims']} total | "
                  f"{s['supported']} supported | "
                  f"{s['conflicted']} conflicted | "
                  f"{s['missing']} missing | "
                  f"{s['uncertain']} uncertain")
            print(f"    Mean support  : {s['mean_support']:.3f}")
            print(f"    Max conflict  : {s['max_conflict']:.3f}")
            print(f"    Confidence    : {s.get('mean_confidence', 0.0):.3f}")
            print(f"    Clinical risk : {s['max_clinical_risk']:.3f}")
            print()
            for ce in self.evidence_state.claim_evidences:
                status_icon = {
                    "supported": "OK",
                    "conflicted": "XX",
                    "missing": "??",
                    "uncertain": "~~",
                }.get(ce.status.value, "  ")
                print(f"    [{status_icon}] {ce.claim[:65]}")
                print(f"         sup={ce.support_score:.2f}  "
                      f"con={ce.conflict_score:.2f}  "
                      f"conf={ce.clinical_confidence:.2f}  "
                      f"src={ce.best_source[:35]!r}  "
                      f"yr={ce.recency_year}")
        print(f"\n  Answer:\n{self.final_answer}")
        print(sep)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "action": self.action if isinstance(self.action, str) else self.action.value,
            "rounds": self.rounds,
            "latency_seconds": self.latency_seconds,
            "evidence_state": self.evidence_state.detailed_dict() if self.evidence_state else None,
            "history": self.history,
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MedRagPipeline:
    """
    Adaptive Evidence-Aware Retrieval and Verification Framework

    6-Phase Architecture
    ────────────────────
    Phase 1  Clinical Query Understanding   (entity / intent / risk extraction)
    Phase 2  Evidence-Aware Hybrid Retrieval (BM25 + Dense + Cross-Encoder Reranker)
    Phase 3  Answer Generation + Claim Decomposer (LLM)
    Phase 4  Claim-Level Verification       (Sentence-Level NLI)
    Phase 5  Evidence State Management      (support · conflict · recency · quality · risk)
    Phase 6  Next-Action Controller         (STOP / PARTIAL / CONFLICTED / SEARCH / ABSTAIN)
    """

    def __init__(
        self,
        documents: List[Document],
        llm_client: LLMClient,
        config=None,
    ) -> None:
        if config is None:
            try:
                from config import MedRagConfig
                config = MedRagConfig.from_env()
            except ImportError:
                config = _DefaultConfig()
        self.config = config
        validate_config = getattr(config, "validate", None)
        if callable(validate_config):
            validate_config()

        print("\n[MedRag] Initialising pipeline components...")

        # Phase 1
        self.query_understander = ClinicalQueryUnderstanding(llm_client)

        # Phase 2
        self.retriever = HybridRetriever(
            dense_model=config.dense_model,
            reranker_model=config.reranker_model,
            bm25_weight=config.bm25_weight,
            dense_weight=config.dense_weight,
            top_k_retrieve=config.top_k_retrieve,
            top_k_rerank=config.top_k_rerank,
            use_reranker=config.use_reranker,
        )
        self.retriever.index(documents)

        # Phase 3
        self.generator = AnswerGenerator(llm_client)

        # Phase 4
        self.verifier = NLIVerifier(
            model_name=config.nli_model,
            batch_size=config.nli_batch_size,
        )

        # Phase 6
        self.controller = EvidenceController(
            support_threshold=config.support_threshold,
            conflict_threshold=config.conflict_threshold,
            clinical_confidence_threshold=getattr(config, "clinical_confidence_threshold", 0.65),
            max_search_rounds=config.max_search_rounds,
            high_risk_threshold=getattr(config, "high_risk_threshold", 0.75),
            high_risk_penalty=getattr(config, "high_risk_penalty", 0.15),
        )

        print("[MedRag] Pipeline ready.\n")

    # ── main entry point ─────────────────────────────────────────────────────

    def answer(self, question: str) -> PipelineResult:
        """Run the adaptive evidence loop for one medical question."""
        t0 = time.time()
        history: List[Dict[str, Any]] = []

        # ── Phase 1: Clinical Query Understanding ────────────────────────────
        print(f"[Phase 1] Clinical Query Understanding")
        cq: ClinicalQuery = self.query_understander.parse(question)
        print(f"  intent={cq.intent}  risk={cq.clinical_risk:.2f}  entities={cq.entities[:4]}")

        search_query = cq.search_query or question
        all_retrieved: List[RetrievedDocument] = []
        seen_doc_ids: set[str] = set()

        max_rounds = int(self.config.max_search_rounds)
        if max_rounds < 1:
            raise ValueError("max_search_rounds must be at least 1.")
        round_num = 1
        rounds_run = 0
        stable_claims: List[str] = []
        initial_generated_answer: Optional[GeneratedAnswer] = None
        final_state: Optional[EvidenceState] = None
        decision: Optional[ControllerDecision] = None

        while round_num <= max_rounds:
            rounds_run = round_num
            label = f"Round {round_num}"

            # ── Phase 2: Evidence-Aware Retrieval ────────────────────────────
            print(f"\n[{label}] Evidence-Aware Retrieval  query={search_query!r:.70}")
            new_hits = self.retriever.retrieve(search_query)

            # Deduplicate and accumulate evidence from all rounds
            new_added = 0
            for hit in new_hits:
                if hit.document.doc_id not in seen_doc_ids:
                    all_retrieved.append(hit)
                    seen_doc_ids.add(hit.document.doc_id)
                    new_added += 1

            print(f"  +{new_added} new docs retrieved  (total unique={len(all_retrieved)})")

            if not all_retrieved:
                print("  [Warning] No documents retrieved.")
                decision = ControllerDecision(
                    action=Action.ABSTAIN,
                    reason="No relevant clinical documents were retrieved from the corpus.",
                    confidence=0.0,
                )
                history.append({
                    "round": round_num,
                    "query": search_query,
                    "new_docs": 0,
                    "total_docs": 0,
                    "n_claims": 0,
                    "action": decision.action.value,
                    "reason": decision.reason,
                })
                break

            # ── Phase 3: Answer Generation + Claim Extraction (Round 1) ──────
            # Claims remain stable across rounds so the pipeline verifies the same facts
            if round_num == 1:
                print(f"[{label}] Generating answer + decomposing claims...")
                context_docs = all_retrieved[: self.config.top_k_rerank]
                initial_generated_answer = self.generator.generate_answer(
                    question, context_docs
                )

                if initial_generated_answer.is_fallback:
                    print(f"  [Error] LLM backend failure: {initial_generated_answer.error}")
                    decision = ControllerDecision(
                        action=Action.ABSTAIN,
                        reason=f"LLM API failure ({self.generator.llm.backend}): {initial_generated_answer.error}. Withholding ungrounded response.",
                        confidence=0.0,
                    )
                    history.append({
                        "round": round_num,
                        "query": search_query,
                        "new_docs": new_added,
                        "total_docs": len(all_retrieved),
                        "n_claims": 0,
                        "action": decision.action.value,
                        "reason": decision.reason,
                    })
                    break

                stable_claims = initial_generated_answer.atomic_claims
                print(f"  {len(stable_claims)} atomic claims extracted")

                if not stable_claims:
                    print("  [Warning] No atomic claims extracted.")
                    decision = ControllerDecision(
                        action=Action.ABSTAIN,
                        reason="No atomic clinical claims could be extracted for verification.",
                        confidence=0.0,
                    )
                    history.append({
                        "round": round_num,
                        "query": search_query,
                        "new_docs": new_added,
                        "total_docs": len(all_retrieved),
                        "n_claims": 0,
                        "action": decision.action.value,
                        "reason": decision.reason,
                    })
                    break

            # ── Phase 4+5: Claim Verification against ALL Accumulated Evidence ──
            # Fix: Verifies against all accumulated documents, not truncated top_k_rerank
            print(f"[{label}] Verifying claims against {len(all_retrieved)} accumulated documents...")
            state: EvidenceState = self.verifier.verify_claims(
                claims=stable_claims,
                retrieved_docs=all_retrieved,
                clinical_risk=cq.clinical_risk,
                support_threshold=self.config.support_threshold,
                conflict_threshold=self.config.conflict_threshold,
                missing_threshold=getattr(self.config, "missing_threshold", 0.35),
                round_number=round_num,
                clinical_confidence_threshold=getattr(
                    self.config, "clinical_confidence_threshold", 0.65
                ),
            )
            final_state = state

            s = state.summary_dict()
            print(
                f"  support={s['mean_support']:.2f}  "
                f"conflict={s['max_conflict']:.2f}  "
                f"supported={s['supported']}  "
                f"conflicted={s['conflicted']}  "
                f"missing={s['missing']}  "
                f"uncertain={s['uncertain']}"
            )

            # ── Phase 6: Next-Action Decision ────────────────────────────────
            decision = self.controller.decide(
                state=state,
                current_round=round_num,
                max_rounds=max_rounds,
            )
            print(f"  -> {decision.action}  {decision.reason}")

            history.append({
                "round": round_num,
                "query": search_query,
                "new_docs": new_added,
                "total_docs": len(all_retrieved),
                "n_claims": len(stable_claims),
                "evidence": s,
                "action": decision.action.value if hasattr(decision.action, "value") else str(decision.action),
                "reason": decision.reason,
            })

            # Check if terminating
            if decision.action in (Action.STOP, Action.VERIFIED, Action.PARTIAL, Action.CONFLICTED, Action.ABSTAIN):
                break

            # Action == SEARCH: Generate targeted gap query for next round
            gap_query = self.generator.generate_gap_query(question, state.weakest_claims)
            print(f"  Gap query: {gap_query!r}")
            if not gap_query or not gap_query.strip():
                decision = ControllerDecision(
                    action=Action.ABSTAIN,
                    reason=(
                        "Unresolved claims did not yield a specific search query; "
                        "stopping rather than repeating a generic medical search."
                    ),
                    confidence=state.mean_confidence,
                )
                history[-1].update(
                    action=decision.action.value,
                    reason=decision.reason,
                )
                break
            search_query = gap_query
            round_num += 1

        # Synthesize final answer based on verified evidence state
        final_answer = self._synthesize_final_answer(
            question=question,
            state=final_state,
            decision=decision,
            retrieved_docs=all_retrieved,
        )

        return PipelineResult(
            question=question,
            final_answer=final_answer,
            action=decision.action if decision else Action.ABSTAIN,
            evidence_state=final_state,
            rounds=rounds_run,
            retrieved_docs=all_retrieved,
            clinical_query=cq,
            history=history,
            latency_seconds=time.time() - t0,
        )

    # ── synthesis helper ─────────────────────────────────────────────────────

    def _synthesize_final_answer(
        self,
        question: str,
        state: Optional[EvidenceState],
        decision: Optional[ControllerDecision],
        retrieved_docs: List[RetrievedDocument],
    ) -> str:
        if not decision or not state:
            reason = decision.reason if decision else "No controller decision was produced."
            return (
                "**ABSTAIN — no verified clinical statement produced.**\n\n"
                f"Reason: {reason}\n\n"
                "Evidence status is relative to the supplied corpus."
            )

        act = decision.action

        if act in (Action.STOP, Action.VERIFIED):
            lines = [
                "**Evidence-supported findings:**",
                f"**Question:** {question}",
                f"All generated claims met the configured support criteria across "
                f"{len(retrieved_docs)} retrieved document(s).",
            ]
            for i, c in enumerate(state.claim_evidences):
                lines.append(f"- **Claim {i + 1}:** {c.claim}")
                if c.supporting_passages:
                    for passage in [
                        p for p in c.supporting_passages if not p.illustrative
                    ][:3]:
                        lines.append(
                            f"  - [{passage.doc_id}] {passage.doc_title} — "
                            f"{passage.source}, {passage.year} "
                            f"(source quality {passage.quality_score:.2f}; "
                            f"clinical confidence {c.clinical_confidence:.0%})"
                        )
                        lines.append(f"    > {passage.text[:350]}")
            lines.append(
                "\nClinical confidence is heuristic. Evidence status is relative to "
                "the supplied corpus and does not "
                "establish clinical truth or replace professional judgment."
            )
            return "\n".join(lines)

        if act == Action.PARTIAL:
            lines = [
                "⚠️ **Qualified Partial Clinical Statement**\n",
                "Some claims are verified by retrieved literature while others remain unconfirmed. Clinical recommendations are qualified accordingly.\n",
                "### Verified Clinical Findings:",
            ]
            for c in state.supported_claims:
                lines.append(f"- ✅ **{c.claim}** *(Support: {c.support_score:.0%})*")
                for passage in [
                    p for p in c.supporting_passages if not p.illustrative
                ][:3]:
                    lines.append(
                        f"  - [{passage.doc_id}] {passage.doc_title} — "
                        f"{passage.source}, {passage.year} "
                        f"(quality {passage.quality_score:.2f})"
                    )
                    lines.append(f"    > {passage.text[:350]}")

            lines.append("\n### Unresolved / Lacking Evidence:")
            for c in state.weakest_claims:
                status_label = "Missing in corpus" if c.missing else "Inconclusive / Uncertain"
                lines.append(f"- ❓ **{c.claim}** *({status_label})*")

            lines.append(
                "\n*Evidence status is relative to the supplied corpus. Treatment or "
                "dosage decisions should not be based on unconfirmed claims.*"
            )
            return "\n".join(lines)

        if act == Action.CONFLICTED:
            lines = [
                "⚔️ **Contradictory Clinical Evidence Detected**\n",
                "The retrieved corpus contains material disagreements regarding this clinical query. Conflicting evidence is summarized below:\n",
            ]
            for c in state.conflicted_claims:
                lines.append(f"#### Disputed Assertion: {c.claim}")
                for passage in c.supporting_passages[:3]:
                    sample_label = " · illustrative sample" if passage.illustrative else ""
                    lines.append(
                        f"- **Supporting Evidence:** [{passage.doc_id}] "
                        f"*{passage.doc_title} ({passage.year}, {passage.source})* "
                        f"[Entailment: {passage.entailment:.0%}; "
                        f"quality: {passage.quality_score:.2f}{sample_label}]"
                    )
                    lines.append(f"  > *\"{passage.text[:300]}\"*")
                for passage in c.contradicting_passages[:3]:
                    sample_label = " · illustrative sample" if passage.illustrative else ""
                    lines.append(
                        f"- **Contradicting Evidence:** [{passage.doc_id}] "
                        f"*{passage.doc_title} ({passage.year}, {passage.source})* "
                        f"[Contradiction: {passage.contradiction:.0%}; "
                        f"quality: {passage.quality_score:.2f}{sample_label}]"
                    )
                    lines.append(f"  > *\"{passage.text[:300]}\"*")

            lines.append(
                "\n*No clinical conclusion is selected because the supplied corpus "
                "contains contradictory passages.*"
            )
            return "\n".join(lines)

        # ABSTAIN
        reason = decision.reason if decision else "Insufficient evidence."
        lines = [
            "🛑 **MedRAG Abstained from Clinical Recommendation**\n",
            f"**Clinical Decision:** ABSTAIN\n",
            f"**Reason:** {reason}\n\n",
            "The retrieved evidence does not meet the necessary clinical certainty threshold to ensure patient safety.",
        ]
        lines.append(
            "\nEvidence status is relative to the supplied corpus; no unsupported "
            "clinical recommendation is provided."
        )
        return "\n".join(lines)


# ── Minimal fallback config (used when config.py is not importable) ───────────

class _DefaultConfig:
    dense_model = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model = "cross-encoder/nli-MiniLM2-L6-H768"
    bm25_weight = 0.4
    dense_weight = 0.6
    top_k_retrieve = 20
    top_k_rerank = 5
    use_reranker = True
    support_threshold = 0.70
    conflict_threshold = 0.35
    missing_threshold = 0.35
    clinical_confidence_threshold = 0.65
    max_search_rounds = 3
    nli_batch_size = 16
    high_risk_threshold = 0.75
    high_risk_penalty = 0.15
