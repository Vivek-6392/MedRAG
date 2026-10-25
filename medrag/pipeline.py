"""
Main Pipeline — Adaptive Evidence-Aware Retrieval and Verification Framework
Orchestrates all 6 phases and the SEARCH / STOP / ABSTAIN control loop.
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
    action: str                                         # final STOP | ABSTAIN | SEARCH
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
                      f"src={ce.best_source[:40]!r}  "
                      f"yr={ce.recency_year}")
        print(f"\n  Answer:\n{self.final_answer}")
        print(sep)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "action": self.action,
            "rounds": self.rounds,
            "latency_seconds": self.latency_seconds,
            "evidence_state": self.evidence_state.detailed_dict() if self.evidence_state else None,
            "history": self.history,
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MedRagPipeline:
    """
    Adaptive Evidence-Aware Retrieval and Verification Framework

    3-Phase Hybrid Architecture
    ───────────────────────────
    Phase 1  Clinical Query Understanding   (entity / intent / risk extraction)
    Phase 2  Evidence-Aware Retrieval        (BM25 + Dense + Reranker)
    Phase 3  Generate Answer + Atomic Claims (LLM)
    Phase 4  Claim-Level Verification        (NLI)
    Phase 5  Evidence State                  (support · conflict · recency · quality · risk)
    Phase 6  Next-Action Decision            (SEARCH / STOP / ABSTAIN)
    """

    def __init__(
        self,
        documents: List[Document],
        llm_client: LLMClient,
        config=None,
    ) -> None:
        # Resolve config
        if config is None:
            try:
                from config import MedRagConfig
                config = MedRagConfig.from_env()
            except ImportError:
                config = _DefaultConfig()
        self.config = config

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
        print(f"  intent={cq.intent}  risk={cq.clinical_risk:.2f}  "
              f"entities={cq.entities[:4]}")

        search_query = cq.search_query or question
        final_answer = ""
        final_state: Optional[EvidenceState] = None
        all_retrieved: List[RetrievedDocument] = []
        decision: Optional[ControllerDecision] = None
        round_num = 0

        while True:
            label = f"Round {round_num + 1}"

            # ── Phase 2: Evidence-Aware Retrieval ────────────────────────────
            print(f"\n[{label}] Evidence-Aware Retrieval  query={search_query!r:.70}")
            new_hits = self.retriever.retrieve(search_query)

            # Merge & deduplicate across rounds
            seen: set[str] = {r.document.doc_id for r in all_retrieved}
            for hit in new_hits:
                if hit.document.doc_id not in seen:
                    all_retrieved.append(hit)
                    seen.add(hit.document.doc_id)

            print(f"  +{len(new_hits)} docs retrieved  "
                  f"(total unique={len(all_retrieved)})")

            context_docs = all_retrieved[: self.config.top_k_rerank]

            # ── Phase 3: Generate Answer + Atomic Claims ─────────────────────
            print(f"[{label}] Generating answer + decomposing claims...")
            gen: GeneratedAnswer = self.generator.generate_answer(
                question, context_docs
            )
            final_answer = gen.answer
            print(f"  {len(gen.atomic_claims)} atomic claims extracted")

            # ── Phase 4+5: Claim Verification → Evidence State ───────────────
            print(f"[{label}] Verifying claims with NLI...")
            state: EvidenceState = self.verifier.verify_claims(
                claims=gen.atomic_claims,
                retrieved_docs=context_docs,
                clinical_risk=cq.clinical_risk,
            )
            state.round_number = round_num
            final_state = state

            s = state.summary_dict()
            print(
                f"  support={s['mean_support']:.2f}  "
                f"conflict={s['max_conflict']:.2f}  "
                f"missing={s['missing']}  conflicted={s['conflicted']}"
            )

            # ── Phase 6: Next-Action Decision ────────────────────────────────
            decision = self.controller.decide(state)
            print(f"  -> {decision.action}  {decision.reason}")

            history.append({
                "round": round_num,
                "query": search_query,
                "new_docs": len(new_hits),
                "total_docs": len(all_retrieved),
                "n_claims": len(gen.atomic_claims),
                "evidence": s,
                "action": decision.action,
                "reason": decision.reason,
            })

            if decision.action == Action.STOP:
                break

            if decision.action == Action.ABSTAIN:
                final_answer = (
                    f"[ABSTAIN — insufficient evidence]\n\n"
                    f"After {round_num + 1} retrieval round(s), the evidence remains "
                    f"inadequate to make a reliable clinical statement.\n\n"
                    f"Reason: {decision.reason}\n\n"
                    f"Partial answer (low confidence):\n{final_answer}"
                )
                break

            # Action == SEARCH → generate targeted gap query
            gap_query = self.generator.generate_gap_query(
                question, state.weakest_claims
            )
            print(f"  Gap query: {gap_query!r}")
            search_query = gap_query if gap_query and gap_query.strip() else (cq.search_query or question)
            round_num += 1

        return PipelineResult(
            question=question,
            final_answer=final_answer,
            action=decision.action if decision else "UNKNOWN",
            evidence_state=final_state,
            rounds=round_num + 1,
            retrieved_docs=all_retrieved,
            clinical_query=cq,
            history=history,
            latency_seconds=time.time() - t0,
        )


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
    max_search_rounds = 3
    nli_batch_size = 16
    high_risk_threshold = 0.75
    high_risk_penalty = 0.15
