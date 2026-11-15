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
            print("\n  Evidence State:")
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
        use_reranker = bool(config.use_reranker and getattr(config, "enable_reranking", True))
        vector_backend = getattr(config, "vector_store", "chromadb")
        collection_name = getattr(config, "collection_name", "medrag_corpus")
        persist_dir = getattr(config, "chroma_persist_dir", "./data/chroma_db")
        self.retriever = HybridRetriever(
            dense_model=config.dense_model,
            reranker_model=config.reranker_model,
            bm25_weight=config.bm25_weight,
            dense_weight=config.dense_weight,
            top_k_retrieve=config.top_k_retrieve,
            top_k_rerank=config.top_k_rerank,
            use_reranker=use_reranker,
            vector_backend=vector_backend,
            collection_name=collection_name,
            persist_dir=persist_dir,
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
        print("[Phase 1] Clinical Query Understanding")
        cq: ClinicalQuery = self.query_understander.parse(question)
        print(f"  intent={cq.intent}  risk={cq.clinical_risk:.2f}  entities={cq.entities[:4]}")

        search_query = cq.search_query or question
        all_retrieved: List[RetrievedDocument] = []
        seen_doc_ids: set[str] = set()

        max_rounds = 1 if not getattr(self.config, "enable_iterative", True) else int(self.config.max_search_rounds)
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
            if not getattr(self.config, "enable_hybrid", True):
                dense_retriever = getattr(self.retriever, "dense", None)
                if self.config.dense_weight > 0 and dense_retriever is not None and getattr(dense_retriever, "_embeddings", None) is not None:
                    new_hits = dense_retriever.retrieve(search_query, top_k=self.config.top_k_retrieve)
                else:
                    new_hits = self.retriever.bm25.retrieve(search_query, top_k=self.config.top_k_retrieve)
            else:
                new_hits = self.retriever.retrieve(search_query)

            # Deduplicate and accumulate evidence from all rounds
            new_added = 0
            for hit in new_hits:
                if hit.document.doc_id not in seen_doc_ids:
                    all_retrieved.append(hit)
                    seen_doc_ids.add(hit.document.doc_id)
                    new_added += 1

            print(f"  +{new_added} new docs retrieved  (total unique={len(all_retrieved)})")

            # Relevance gate (round 1): the retriever always returns top-k, even for
            # out-of-corpus questions. Abstain if nothing is plausibly relevant.
            rel_threshold = getattr(self.config, "relevance_threshold", None)
            if round_num == 1 and rel_threshold is not None:
                rerank_scores = [h.rerank_score for h in new_hits if h.rerank_score is not None]
                if rerank_scores:
                    best_rel = max(rerank_scores)
                    print(f"  best rerank score={best_rel:.3f} (gate={rel_threshold})")
                    if best_rel < rel_threshold:
                        decision = ControllerDecision(
                            action=Action.ABSTAIN,
                            reason=(
                                f"Retrieved documents are not relevant to the question "
                                f"(best rerank score {best_rel:.3f} < {rel_threshold}); "
                                "the question appears to be outside the indexed corpus."
                            ),
                            confidence=0.0,
                        )
                        history.append({
                            "round": round_num, "query": search_query,
                            "new_docs": new_added, "total_docs": len(all_retrieved),
                            "n_claims": 0, "action": decision.action.value,
                            "reason": decision.reason,
                        })
                        break

            if not all_retrieved:
                print("  [Warning] No documents retrieved.")
                if not getattr(self.config, "enable_abstention", True):
                    raw_ans = self.generator.llm.complete(question)
                    decision = ControllerDecision(
                        action=Action.STOP,
                        reason="Zero retrieval but abstention disabled by ablation; emitted parametric generation.",
                        confidence=0.5,
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
                    return PipelineResult(
                        question=question,
                        final_answer=raw_ans,
                        action="STOP",
                        evidence_state=None,
                        rounds=rounds_run,
                        retrieved_docs=[],
                        clinical_query=cq,
                        history=history,
                        latency_seconds=time.time() - t0,
                    )
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

                if not getattr(self.config, "enable_claim_decomposition", True):
                    raw_clean = initial_generated_answer.answer.strip()
                    first_stmt = raw_clean.split(".")[0].strip() if "." in raw_clean else raw_clean
                    stable_claims = [first_stmt if len(first_stmt) > 15 else raw_clean[:200]]
                    print("  [Ablation] Claim decomposition disabled; using single holistic statement")
                else:
                    stable_claims = initial_generated_answer.atomic_claims
                    print(f"  {len(stable_claims)} atomic claims extracted")

                if not stable_claims:
                    print("  [Warning] No atomic claims extracted.")
                    if not getattr(self.config, "enable_abstention", True):
                        decision = ControllerDecision(
                            action=Action.STOP,
                            reason="No claims extracted but abstention disabled by ablation; emitted unverified answer.",
                            confidence=0.5,
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
            if not getattr(self.config, "enable_nli_verification", True):
                print("  [Ablation] NLI verification disabled; approximating support from retrieval ranking...")
                from .evidence_state import EvidenceState, ClaimEvidence, EvidencePassage
                top_score = all_retrieved[0].score if all_retrieved else 0.5
                norm_score = min(1.0, max(0.0, top_score if top_score <= 1.0 else top_score / 10.0))
                claim_evs = []
                for c in stable_claims:
                    claim_evs.append(ClaimEvidence(
                        claim=c,
                        support_score=norm_score,
                        conflict_score=0.0,
                        supporting_passages=[
                            EvidencePassage(
                                doc_id=d.document.doc_id,
                                doc_title=d.document.title,
                                source=d.document.source,
                                year=d.document.year,
                                quality_score=1.0 if not getattr(self.config, "enable_source_quality_weighting", True) else d.document.quality_score,
                                text=d.document.text[:300],
                                entailment=norm_score,
                                contradiction=0.0,
                                neutral=1.0 - norm_score,
                            )
                            for d in all_retrieved[:3]
                        ],
                    ))
                state = EvidenceState(claim_evidences=claim_evs, round_number=round_num)
            else:
                state: EvidenceState = self.verifier.verify_claims(
                    claims=stable_claims,
                    retrieved_docs=all_retrieved,
                    clinical_risk=cq.clinical_risk if getattr(self.config, "enable_risk_policy", True) else 0.0,
                    support_threshold=self.config.support_threshold,
                    conflict_threshold=self.config.conflict_threshold,
                    missing_threshold=getattr(self.config, "missing_threshold", 0.35),
                    round_number=round_num,
                    clinical_confidence_threshold=getattr(
                        self.config, "clinical_confidence_threshold", 0.65
                    ),
                )

            # Apply ablation: without source quality weighting
            if not getattr(self.config, "enable_source_quality_weighting", True):
                for ce in state.claim_evidences:
                    ce.quality_weight = 0.0
                    ce.annual_recency_decay = 0.0
                    for p in ce.supporting_passages + getattr(ce, "contradicting_passages", []):
                        p.quality_score = 1.0
                        p.year = 2024

            # Apply ablation: without evidence state tracking
            if not getattr(self.config, "enable_evidence_state", True):
                for ce in state.claim_evidences:
                    ce.conflict_penalty_weight = 0.0
                    ce.quality_weight = 0.0
                    ce.annual_recency_decay = 0.0

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

            # Apply ablation: without abstention (force answering)
            if not getattr(self.config, "enable_abstention", True):
                if decision.action == Action.ABSTAIN:
                    decision = ControllerDecision(
                        action=Action.STOP,
                        reason="Abstention disabled by ablation study; forced to emit best candidate answer.",
                        confidence=state.mean_confidence if state else 0.5,
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
            candidate_answer=initial_generated_answer.answer if initial_generated_answer else "",
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
        candidate_answer: str = "",
    ) -> str:
        # If abstention is disabled by ablation, return candidate answer if system would otherwise abstain
        if not getattr(self.config, "enable_abstention", True):
            if not decision or not state or decision.action in (Action.ABSTAIN, Action.PARTIAL, Action.CONFLICTED):
                if candidate_answer:
                    return candidate_answer

        if not decision or not state:
            reason = decision.reason if decision else "No controller decision was produced."
            return (
                "**ABSTAIN — no verified clinical statement produced.**\n\n"
                f"Reason: {reason}\n\n"
                "Evidence status is relative to the supplied corpus."
            )

        act = decision.action

        if act in (Action.STOP, Action.VERIFIED):
            lines = []
            if candidate_answer:
                lines.append(f"{candidate_answer}\n")
            lines.extend([
                "**Evidence-supported findings:**",
                f"**Question:** {question}",
                f"All generated claims met the configured support criteria across "
                f"{len(retrieved_docs)} retrieved document(s).",
            ])
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
            "**Clinical Decision:** ABSTAIN\n",
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
    nli_model = "ygivenx/modernbert-base-mednli"
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
