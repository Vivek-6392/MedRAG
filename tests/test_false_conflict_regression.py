"""Regression tests: out-of-corpus questions must ABSTAIN, not report a 'conflict'."""
from types import SimpleNamespace

from medrag.evidence_state import ClaimEvidence, ClaimStatus, EvidencePassage
from medrag.generator import AnswerGenerator
from medrag.pipeline import MedRagPipeline
from medrag.query_understanding import ClinicalQuery
from medrag.retriever import Document, RetrievedDocument
from medrag.verifier import NLIVerifier


class _StubLLM:
    backend = "stub"
    call_failed = False
    last_error = None

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt, max_tokens=None):
        return self.reply


def _passage(ent=0.0, con=0.0, doc_id="d1"):
    return EvidencePassage(
        doc_id=doc_id, doc_title="T", source="S", year=2024, quality_score=0.9,
        text="mitochondria are organelles", entailment=ent, contradiction=con,
        neutral=max(0.0, 1 - ent - con),
    )


# ── 1. meta-claims are not verifiable medical claims ─────────────────────────
def test_meta_claims_are_dropped_json_path():
    llm = _StubLLM(
        '["The provided evidence does not address the role of mitochondria in lace plants.", '
        '"Based on the supplied evidence, we cannot determine whether mitochondria play a role."]'
    )
    assert AnswerGenerator(llm).decompose_claims("anything") == []


def test_meta_claims_are_dropped_fallback_path():
    llm = _StubLLM("not json")
    answer = "The provided evidence does not address this topic at all."
    assert AnswerGenerator(llm).decompose_claims(answer) == []


def test_real_claims_survive_filter():
    llm = _StubLLM('["Metformin is contraindicated in severe renal impairment (eGFR < 30)."]')
    assert len(AnswerGenerator(llm).decompose_claims("x")) == 1


# ── 2. contradiction without support is not CONFLICTED ───────────────────────
def test_contradiction_without_support_is_not_conflicted():
    ce = ClaimEvidence(claim="c", support_score=0.1, conflict_score=0.95,
                       contradicting_passages=[_passage(con=0.95)])
    assert ce.status != ClaimStatus.CONFLICTED
    assert ce.status == ClaimStatus.UNCERTAIN


def test_support_plus_contradiction_is_conflicted():
    ce = ClaimEvidence(claim="c", support_score=0.9, conflict_score=0.9,
                       supporting_passages=[_passage(ent=0.9)],
                       contradicting_passages=[_passage(con=0.9, doc_id="d2")])
    assert ce.status == ClaimStatus.CONFLICTED


# ── 3. one document = one contradiction vote, margin required ────────────────
def _verify(scores):
    v = NLIVerifier()
    v._score_pairs = lambda pairs: [scores] * len(pairs)  # no model download needed
    doc = Document(doc_id="d1", title="Katzung", text="Mitochondria are organelles in cells. "
                   "Cytochrome P450 enzymes metabolise many drugs.")
    state = v.verify_claims(["Mitochondria are organelles."], [RetrievedDocument(doc, 1.0)])
    return state.claim_evidences[0]


def test_same_document_counted_once():
    ce = _verify((0.02, 0.95, 0.03))  # (ent, con, neu)
    assert len(ce.contradicting_passages) == 1


def test_borderline_contradiction_needs_margin():
    ce = _verify((0.30, 0.40, 0.30))  # con is argmax but margin is only 0.10
    assert ce.contradicting_passages == []


# ── 4. relevance gate abstains on out-of-corpus questions ────────────────────
def _bare_pipeline(threshold, rerank_score):
    doc = Document(doc_id="katzung_1", title="Katzung", text="Pharmacokinetics text " * 5)
    hit = RetrievedDocument(document=doc, score=0.016, rerank_score=rerank_score)
    p = object.__new__(MedRagPipeline)
    p.config = SimpleNamespace(
        enable_iterative=True, max_search_rounds=3, enable_hybrid=True,
        enable_abstention=True, relevance_threshold=threshold, top_k_rerank=5,
    )
    p.query_understander = SimpleNamespace(
        parse=lambda q: ClinicalQuery(original_question=q, search_query=q, clinical_risk=0.3))
    p.retriever = SimpleNamespace(retrieve=lambda q: [hit])
    return p


def test_relevance_gate_abstains_out_of_corpus():
    res = _bare_pipeline(threshold=0.5, rerank_score=0.01).answer(
        "Role of mitochondria in lace plant programmed cell death?")
    assert str(res.action.value if hasattr(res.action, "value") else res.action) == "ABSTAIN"
    assert "not relevant" in res.history[-1]["reason"]
    assert "Contradictory" not in res.final_answer
