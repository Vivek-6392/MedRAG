"""
Unit and Integration Tests for MedRAG Framework
Covers:
- Supported evidence verification
- Unsupported / Missing evidence and no false attribution
- Independent evaluation of contradiction and support (co-existing)
- Controller decisions: STOP, PARTIAL, CONFLICTED, SEARCH, ABSTAIN
- Clinical risk policy bounds
- Claim stability and multi-round evidence accumulation
- Empty evidence, empty claims, and LLM failure handling
- Representative clinical queries (Metformin, Aspirin, DOACs)
"""
import unittest
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from medrag.evidence_state import EvidencePassage, ClaimEvidence, EvidenceState, ClaimStatus
from medrag.controller import EvidenceController, Action, ControllerDecision
from medrag.retriever import Document, RetrievedDocument, BM25Retriever, HybridRetriever
from medrag.generator import LLMClient, AnswerGenerator, GeneratedAnswer
from medrag.verifier import NLIVerifier
from medrag.pipeline import MedRagPipeline, PipelineResult
from config import MedRagConfig


def _supporting_passage(
    doc_id="source",
    quality=0.95,
    year=2025,
    entailment=0.9,
    illustrative=False,
):
    return EvidencePassage(
        doc_id=doc_id,
        doc_title="Supporting source",
        source="Test source",
        year=year,
        quality_score=quality,
        text="A supporting evidence passage.",
        entailment=entailment,
        contradiction=0.0,
        neutral=1.0 - entailment,
        illustrative=illustrative,
    )


class TestEvidenceStateAndPassages(unittest.TestCase):
    """Tests for EvidencePassage, ClaimEvidence, thresholds, and confidence."""

    def test_passage_structure(self):
        p = EvidencePassage(
            doc_id="doc1",
            doc_title="Metformin in CKD",
            source="Lancet",
            year=2024,
            quality_score=0.92,
            text="Metformin is contraindicated when eGFR is under 30.",
            entailment=0.95,
            contradiction=0.02,
            neutral=0.03,
        )
        d = p.to_dict()
        self.assertEqual(d["doc_id"], "doc1")
        self.assertEqual(d["source"], "Lancet")
        self.assertAlmostEqual(d["entailment"], 0.95)

    def test_independent_support_and_contradiction(self):
        """A supporting passage must not hide contradictory evidence elsewhere."""
        sup_p = EvidencePassage(
            doc_id="1", doc_title="Doc A", source="NEJM", year=2024, quality_score=0.9,
            text="Aspirin reduces cardiovascular events in all adults.",
            entailment=0.92, contradiction=0.03, neutral=0.05
        )
        con_p = EvidencePassage(
            doc_id="2", doc_title="Doc B", source="USPSTF", year=2023, quality_score=0.9,
            text="Aspirin is not recommended for primary prevention in adults aged 60+.",
            entailment=0.05, contradiction=0.88, neutral=0.07
        )

        ce = ClaimEvidence(
            claim="Aspirin is universally safe for primary prevention in elderly patients.",
            support_score=0.92,
            conflict_score=0.88,
            best_source="Doc A",
            supporting_passages=[sup_p],
            contradicting_passages=[con_p],
            support_threshold=0.70,
            conflict_threshold=0.35,
        )

        # Both scores must be retained
        self.assertAlmostEqual(ce.support_score, 0.92)
        self.assertAlmostEqual(ce.conflict_score, 0.88)
        self.assertEqual(len(ce.supporting_passages), 1)
        self.assertEqual(len(ce.contradicting_passages), 1)

        # Contradiction must take precedence in status
        self.assertEqual(ce.status, ClaimStatus.CONFLICTED)
        self.assertTrue(ce.has_conflict)
        self.assertTrue(ce.has_support)

    def test_unsupported_claim_has_no_arbitrary_source(self):
        """When no support exists, best_source must NOT be set to an arbitrary document."""
        ce = ClaimEvidence(
            claim="Unicorn horns cure diabetes mellitus.",
            support_score=0.05,
            conflict_score=0.02,
            best_source="",
            missing=True,
            support_threshold=0.70,
            missing_threshold=0.35,
        )
        self.assertEqual(ce.status, ClaimStatus.MISSING)
        self.assertTrue(ce.missing)
        self.assertEqual(ce.best_source, "")

    def test_clinical_confidence_uses_quality_recency_and_conflict(self):
        """Clinical confidence must temper entailment by quality and penalize conflict."""
        high_quality = ClaimEvidence(
            claim="Metformin reduces HbA1c.",
            support_score=0.95,
            conflict_score=0.0,
            source_quality=0.95,
            recency_year=2025,
            supporting_passages=[_supporting_passage(quality=0.95, year=2025, entailment=0.95)],
        )
        low_quality = ClaimEvidence(
            claim="Metformin reduces HbA1c.",
            support_score=0.95,
            conflict_score=0.0,
            source_quality=0.50,
            recency_year=2015,
            supporting_passages=[_supporting_passage(quality=0.50, year=2015, entailment=0.95)],
        )
        conflicted = ClaimEvidence(
            claim="Metformin reduces HbA1c.",
            support_score=0.95,
            conflict_score=0.80,
            source_quality=0.95,
            recency_year=2025,
            supporting_passages=[_supporting_passage(quality=0.95, year=2025, entailment=0.95)],
        )

        self.assertGreater(high_quality.clinical_confidence, low_quality.clinical_confidence)
        self.assertGreater(high_quality.clinical_confidence, conflicted.clinical_confidence)

    def test_low_source_quality_prevents_verified_status(self):
        ce = ClaimEvidence(
            claim="A high NLI score is not sufficient by itself.",
            support_score=0.90,
            source_quality=0.40,
            recency_year=2024,
            support_threshold=0.70,
            clinical_confidence_threshold=0.65,
            supporting_passages=[_supporting_passage(quality=0.40, entailment=0.90)],
        )
        self.assertEqual(ce.status, ClaimStatus.UNCERTAIN)

    def test_illustrative_corpus_cannot_become_verified(self):
        passage = EvidencePassage(
            doc_id="sample", doc_title="Sample", source="Example", year=2025,
            quality_score=1.0, text="Illustrative evidence.", entailment=0.99,
            contradiction=0.0, neutral=0.01, illustrative=True,
        )
        ce = ClaimEvidence(
            claim="Example claim.",
            support_score=0.99,
            source_quality=1.0,
            supporting_passages=[passage],
        )
        self.assertEqual(ce.clinical_confidence, 0.0)
        self.assertEqual(ce.status, ClaimStatus.UNCERTAIN)

    def test_all_passages_are_serialized(self):
        passages = [
            EvidencePassage(
                doc_id=str(i), doc_title=f"Document {i}", source="Test source",
                year=2024, quality_score=0.8, text=f"Passage {i}",
                entailment=0.1, contradiction=0.1, neutral=0.8,
            )
            for i in range(4)
        ]
        ce = ClaimEvidence(claim="Claim", neutral_passages=passages)
        self.assertEqual(len(ce.to_dict()["neutral_passages"]), 4)


class TestVerifierPassageClassification(unittest.TestCase):
    def test_support_and_contradiction_from_same_document_are_preserved(self):
        doc = Document(
            doc_id="aspirin-guideline",
            title="Aspirin primary prevention",
            text=(
                "Aspirin reduces cardiovascular events in selected adults. "
                "Aspirin is not recommended for primary prevention in older adults."
            ),
            source="Clinical guideline",
            year=2024,
            quality_score=0.95,
        )
        verifier = NLIVerifier()
        verifier._score_pairs = MagicMock(return_value=[
            (0.05, 0.10, 0.85),
            (0.92, 0.02, 0.06),
            (0.04, 0.91, 0.05),
        ])

        state = verifier.verify_claims(
            ["Aspirin is beneficial for primary prevention in older adults."],
            [RetrievedDocument(doc, score=0.99)],
            support_threshold=0.70,
            conflict_threshold=0.35,
            missing_threshold=0.35,
        )

        evidence = state.claim_evidences[0]
        self.assertEqual(evidence.status, ClaimStatus.CONFLICTED)
        self.assertAlmostEqual(evidence.support_score, 0.92)
        self.assertAlmostEqual(evidence.conflict_score, 0.91)
        self.assertEqual(len(evidence.supporting_passages), 1)
        self.assertEqual(len(evidence.contradicting_passages), 1)
        self.assertEqual(len(evidence.neutral_passages), 1)
        self.assertEqual(evidence.supporting_passages[0].doc_id, "aspirin-guideline")

    def test_configured_support_threshold_and_no_false_source(self):
        doc = Document(
            doc_id="unrelated",
            title="Metformin report",
            text="The report contains no evidence for the requested claim.",
            source="Sample corpus",
            year=2024,
            quality_score=0.9,
        )
        verifier = NLIVerifier()
        verifier._score_pairs = MagicMock(return_value=[
            (0.80, 0.05, 0.15),
            (0.80, 0.05, 0.15),
        ])

        state = verifier.verify_claims(
            ["Aspirin prevents stroke in all elderly patients."],
            [RetrievedDocument(doc, score=0.999)],
            support_threshold=0.90,
            conflict_threshold=0.85,
            missing_threshold=0.35,
        )

        evidence = state.claim_evidences[0]
        self.assertEqual(evidence.status, ClaimStatus.UNCERTAIN)
        self.assertAlmostEqual(evidence.support_score, 0.80)
        self.assertEqual(evidence.supporting_passages, [])
        self.assertEqual(evidence.best_source, "")

    def test_empty_claims_and_empty_evidence_are_safe(self):
        verifier = NLIVerifier()
        empty_claims = verifier.verify_claims(["  "], [])
        self.assertEqual(empty_claims.claim_evidences, [])

        no_docs = verifier.verify_claims(["A clinical claim."], [])
        self.assertEqual(no_docs.claim_evidences[0].status, ClaimStatus.MISSING)


class TestConfigurationValidation(unittest.TestCase):
    def test_groq_model_defaults_to_current_migration_target(self):
        with patch.dict(os.environ, {"LLM_BACKEND": "groq"}, clear=True):
            with patch("dotenv.load_dotenv"):
                config = MedRagConfig.from_env()
        self.assertEqual(config.llm_model, "openai/gpt-oss-120b")

    def test_groq_model_environment_override_is_preserved(self):
        with patch.dict(
            os.environ,
            {"LLM_BACKEND": "groq", "GROQ_MODEL": "account-specific-model"},
            clear=True,
        ):
            with patch("dotenv.load_dotenv"):
                config = MedRagConfig.from_env()
        self.assertEqual(config.llm_model, "account-specific-model")

    def test_invalid_retrieval_and_round_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            MedRagConfig(top_k_retrieve=0)
        with self.assertRaises(ValueError):
            MedRagConfig(max_search_rounds=0)

    def test_representative_high_risk_queries_get_high_risk_tier(self):
        from medrag.query_understanding import ClinicalQueryUnderstanding

        understander = ClinicalQueryUnderstanding(LLMClient(backend="mock"))
        questions = [
            "Is metformin safe for patients with renal impairment?",
            "Should aspirin be used for primary prevention in elderly patients?",
            "What anticoagulant is preferred for stroke prevention in non-valvular atrial fibrillation?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertGreaterEqual(understander.parse(question).clinical_risk, 0.75)

    def test_controller_uses_configured_threshold_without_risk_shifts(self):
        controller = EvidenceController(
            support_threshold=0.80,
            conflict_threshold=0.30,
            clinical_confidence_threshold=0.60,
            max_search_rounds=2,
        )
        claim = ClaimEvidence(
            "Claim has some evidence but misses the configured threshold.",
            support_score=0.75,
            source_quality=1.0,
            clinical_risk=0.1,
            support_threshold=0.80,
            conflict_threshold=0.30,
            clinical_confidence_threshold=0.60,
            supporting_passages=[_supporting_passage(entailment=0.75)],
        )
        state = EvidenceState(
            claim_evidences=[claim],
            support_threshold=0.80,
            conflict_threshold=0.30,
            clinical_confidence_threshold=0.60,
        )
        self.assertEqual(controller.decide(state, current_round=1).action, Action.SEARCH)

    def test_builtin_corpus_is_marked_illustrative(self):
        from main import _build_sample_corpus

        self.assertTrue(all(doc.metadata.get("illustrative") for doc in _build_sample_corpus()))


class TestEvidenceController(unittest.TestCase):
    """Tests for the 5 controller actions: STOP, PARTIAL, CONFLICTED, SEARCH, ABSTAIN."""

    def setUp(self):
        self.controller = EvidenceController(
            support_threshold=0.70,
            conflict_threshold=0.35,
            max_search_rounds=3,
            high_risk_threshold=0.75,
        )

    def test_all_supported_triggers_stop(self):
        c1 = ClaimEvidence("Claim 1", support_score=0.85, conflict_score=0.05, source_quality=0.95, supporting_passages=[_supporting_passage(entailment=0.85)], support_threshold=0.70, conflict_threshold=0.35)
        c2 = ClaimEvidence("Claim 2", support_score=0.90, conflict_score=0.02, source_quality=0.95, supporting_passages=[_supporting_passage(entailment=0.90)], support_threshold=0.70, conflict_threshold=0.35)
        state = EvidenceState(claim_evidences=[c1, c2], round_number=1)

        decision = self.controller.decide(state, current_round=1, max_rounds=3)
        self.assertEqual(decision.action, Action.STOP)

    def test_gaps_with_budget_triggers_search(self):
        c1 = ClaimEvidence("Claim 1", support_score=0.85, conflict_score=0.05, source_quality=0.95, supporting_passages=[_supporting_passage(entailment=0.85)], support_threshold=0.70, conflict_threshold=0.35)
        c2 = ClaimEvidence("Claim 2", support_score=0.20, conflict_score=0.05, missing=True, support_threshold=0.70, conflict_threshold=0.35)
        state = EvidenceState(claim_evidences=[c1, c2], round_number=1)

        decision = self.controller.decide(state, current_round=1, max_rounds=3)
        self.assertEqual(decision.action, Action.SEARCH)
        self.assertIn("Claim 2", decision.gap_focus)

    def test_conflicted_literature_triggers_conflicted(self):
        c1 = ClaimEvidence("Claim 1", support_score=0.85, conflict_score=0.75, support_threshold=0.70, conflict_threshold=0.35)
        state = EvidenceState(claim_evidences=[c1], round_number=3)

        decision = self.controller.decide(state, current_round=3, max_rounds=3)
        self.assertEqual(decision.action, Action.CONFLICTED)

    def test_partial_support_when_budget_exhausted(self):
        c1 = ClaimEvidence("Claim 1", support_score=0.85, conflict_score=0.05, source_quality=0.95, supporting_passages=[_supporting_passage(entailment=0.85)], support_threshold=0.70, conflict_threshold=0.35)
        c2 = ClaimEvidence("Claim 2", support_score=0.20, conflict_score=0.05, missing=True, clinical_risk=0.5, support_threshold=0.70, conflict_threshold=0.35)
        state = EvidenceState(claim_evidences=[c1, c2], round_number=3)

        decision = self.controller.decide(state, current_round=3, max_rounds=3)
        self.assertEqual(decision.action, Action.PARTIAL)

    def test_high_risk_prevents_partial_answer_and_abstains(self):
        """Under high clinical risk, missing evidence must force ABSTAIN rather than PARTIAL."""
        c1 = ClaimEvidence("Claim 1", support_score=0.85, conflict_score=0.05, source_quality=0.95, supporting_passages=[_supporting_passage(entailment=0.85)], clinical_risk=0.85, support_threshold=0.70, conflict_threshold=0.35)
        c2 = ClaimEvidence("Claim 2", support_score=0.20, conflict_score=0.05, missing=True, clinical_risk=0.85, support_threshold=0.70, conflict_threshold=0.35)
        state = EvidenceState(claim_evidences=[c1, c2], round_number=3)

        decision = self.controller.decide(state, current_round=3, max_rounds=3)
        self.assertEqual(decision.action, Action.ABSTAIN)

    def test_empty_claims_abstains(self):
        state = EvidenceState(claim_evidences=[], round_number=1)
        decision = self.controller.decide(state, current_round=1, max_rounds=3)
        self.assertEqual(decision.action, Action.ABSTAIN)


class TestGeneratorAndErrorHandling(unittest.TestCase):
    """Tests for evidence-grounded generation, LLM failures, and specific gap queries."""

    def test_mock_generator_extracts_evidence_and_decomposes(self):
        llm = LLMClient(backend="mock")
        gen = AnswerGenerator(llm)

        doc = Document(
            doc_id="1",
            title="Metformin in CKD",
            text="Metformin is contraindicated when eGFR is under 30. Regular monitoring of renal function is mandatory.",
            source="Lancet",
            year=2023,
            quality_score=0.9,
        )
        rdoc = RetrievedDocument(document=doc, score=0.9)

        res = gen.generate_answer(
            question="Is metformin safe for renal impairment?",
            retrieved_docs=[rdoc],
        )

        self.assertFalse(res.is_fallback)
        self.assertTrue(len(res.atomic_claims) > 0)
        # Verify extracted claims reflect the actual evidence
        any_metformin = any("metformin" in c.lower() for c in res.atomic_claims)
        self.assertTrue(any_metformin)

    def test_targeted_gap_query_is_specific_not_generic(self):
        llm = LLMClient(backend="mock")
        gen = AnswerGenerator(llm)

        weak_claim = ClaimEvidence(
            claim="Metformin is contraindicated in severe renal impairment (eGFR < 30 mL/min/1.73 m2).",
            support_score=0.1,
            conflict_score=0.0,
            missing=True,
        )

        query = gen.generate_gap_query(
            original_question="Is metformin safe for patients with renal failure?",
            weak_claims=[weak_claim],
        )

        # Must not be the old generic string
        self.assertNotEqual(query, "evidence-based treatment efficacy safety contraindications")
        # Must contain specific clinical terms
        self.assertTrue(any(term in query.lower() for term in ["metformin", "renal", "egfr", "impairment"]))

    def test_llm_api_failure_is_flagged(self):
        """API failures must not silently produce ungrounded fake outputs."""
        llm = LLMClient(backend="openai", api_key="invalid-key")
        # Force a failure during complete
        with patch.object(llm, "_get_client", return_value=MagicMock(chat=MagicMock(completions=MagicMock(create=MagicMock(side_effect=RuntimeError("API Connection Timeout")))))):
            resp = llm.complete("Test medical prompt")
            self.assertTrue(llm.call_failed)
            self.assertIn("API Connection Timeout", llm.last_error)

        with patch.object(llm, "_get_client", side_effect=ImportError("missing openai package")):
            llm.complete("Test medical prompt")
            self.assertTrue(llm.call_failed)
            self.assertEqual(llm.backend, "openai")

    def test_groq_gpt_oss_uses_reasoning_token_budget(self):
        llm = LLMClient(
            backend="groq",
            model="openai/gpt-oss-120b",
            api_key="test-key",
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="A grounded response."),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(completion_tokens=100),
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch.object(llm, "_get_client", return_value=client):
            output = llm.complete("Test prompt", max_tokens=60)

        self.assertEqual(output, "A grounded response.")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["max_completion_tokens"], 4096)
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertNotIn("max_tokens", request)

    def test_empty_llm_content_reports_completion_diagnostics(self):
        llm = LLMClient(backend="groq", model="openai/gpt-oss-120b")
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="length",
            )],
            usage=SimpleNamespace(completion_tokens=4096),
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch.object(llm, "_get_client", return_value=client):
            output = llm.complete("Test prompt")

        self.assertEqual(output, "")
        self.assertTrue(llm.call_failed)
        self.assertIn("finish_reason='length'", llm.last_error)
        self.assertIn("completion_tokens=4096", llm.last_error)

    def test_malformed_query_json_uses_question_specific_heuristic(self):
        from medrag.query_understanding import ClinicalQueryUnderstanding

        llm = LLMClient(backend="openai")
        llm.complete = MagicMock(return_value='{"entities": "not a list"}')
        parsed = ClinicalQueryUnderstanding(llm).parse(
            "Is metformin safe for patients with renal impairment?"
        )
        self.assertEqual(
            parsed.search_query,
            "Is metformin safe for patients with renal impairment?",
        )
        self.assertNotEqual(parsed.search_query, "medical query")

    def test_decomposition_failure_does_not_return_answer_claims_as_success(self):
        llm = LLMClient(backend="openai")
        calls = 0

        def fail_decomposition(prompt, max_tokens=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                llm.call_failed = False
                return "Metformin is safe for every patient."
            llm.call_failed = True
            llm.last_error = "timeout"
            return ""

        llm.complete = fail_decomposition
        generator = AnswerGenerator(llm)
        result = generator.generate_answer(
            "Is metformin safe?",
            [RetrievedDocument(
                Document("1", "Metformin", "A sample passage.", "Sample", 2024, 0.9),
                score=0.8,
            )],
        )
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.atomic_claims, [])
        self.assertEqual(result.answer, "")

    def test_generic_gap_query_is_replaced_with_claim_specific_terms(self):
        llm = LLMClient(backend="mock")
        generator = AnswerGenerator(llm)
        weak_claim = ClaimEvidence(
            claim="Metformin dosing in renal impairment at eGFR below 30.",
            missing=True,
        )
        with patch.object(llm, "complete", return_value="general medical research"):
            query = generator.generate_gap_query("What should be done?", [weak_claim])
        self.assertIn("metformin", query.lower())
        self.assertIn("renal", query.lower())


class TestPipelineAccumulationAndStability(unittest.TestCase):
    """Tests for multi-round evidence accumulation and claim stability."""

    def test_claims_remain_stable_and_accumulated_docs_verified(self):
        doc1 = Document(
            doc_id="d1", title="Metformin First-Line",
            text="Metformin reduces HbA1c by 1-2%.", source="NEJM", year=2024, quality_score=0.95
        )
        doc2 = Document(
            doc_id="d2", title="Metformin CKD",
            text="Metformin is contraindicated when eGFR is below 30.", source="Lancet", year=2023, quality_score=0.93
        )

        llm = LLMClient(backend="mock")
        cfg = MedRagConfig()
        cfg.max_search_rounds = 2
        cfg.top_k_rerank = 1  # Intentionally small to test truncation fix!

        with patch.object(HybridRetriever, "index"):
            pipeline = MedRagPipeline(documents=[doc1, doc2], llm_client=llm, config=cfg)

        # Mock retriever: Round 1 returns only d1, Round 2 returns d2
        mock_r1 = [RetrievedDocument(doc1, score=0.9)]
        mock_r2 = [RetrievedDocument(doc2, score=0.85)]

        call_count = 0
        def mocked_retrieve(query):
            nonlocal call_count
            call_count += 1
            return mock_r1 if call_count == 1 else mock_r2

        pipeline.retriever.retrieve = mocked_retrieve

        # Mock verifier to simulate finding evidence in round 2
        def mocked_verify(claims, retrieved_docs, **kwargs):
            doc_ids = {r.document.doc_id for r in retrieved_docs}
            evidences = []
            for c in claims:
                # If d2 is present, claim 2 is supported!
                if "contraindicated" in c.lower() and "d2" in doc_ids:
                    passage = EvidencePassage(
                        doc_id="d2", doc_title="Metformin CKD", source="Lancet",
                        year=2023, quality_score=0.95,
                        text="Metformin is contraindicated below eGFR 30.",
                        entailment=0.9, contradiction=0.0, neutral=0.1,
                    )
                    evidences.append(ClaimEvidence(
                        c, support_score=0.9, conflict_score=0.0, source_quality=0.95,
                        best_source="Metformin CKD", supporting_passages=[passage],
                    ))
                elif "hba1c" in c.lower() and "d1" in doc_ids:
                    passage = EvidencePassage(
                        doc_id="d1", doc_title="Metformin First-Line", source="NEJM",
                        year=2024, quality_score=0.95,
                        text="Metformin reduces HbA1c by 1-2%.",
                        entailment=0.9, contradiction=0.0, neutral=0.1,
                    )
                    evidences.append(ClaimEvidence(
                        c, support_score=0.9, conflict_score=0.0, source_quality=0.95,
                        best_source="Metformin First-Line", supporting_passages=[passage],
                    ))
                else:
                    evidences.append(ClaimEvidence(c, support_score=0.1, conflict_score=0.0, missing=True))
            return EvidenceState(claim_evidences=evidences, round_number=kwargs.get("round_number", 1))

        pipeline.verifier.verify_claims = mocked_verify

        # Force initial answer with 2 claims
        pipeline.generator.generate_answer = MagicMock(return_value=GeneratedAnswer(
            answer="Metformin reduces HbA1c. Metformin is contraindicated in severe renal impairment.",
            rationale="Test rationale",
            atomic_claims=["Metformin reduces HbA1c.", "Metformin is contraindicated in severe renal impairment."],
            confidence=0.8,
        ))

        res = pipeline.answer("Is metformin safe?")

        # Check that d2 was accumulated and verified
        self.assertEqual(len(res.retrieved_docs), 2)
        doc_ids_retrieved = {r.document.doc_id for r in res.retrieved_docs}
        self.assertIn("d1", doc_ids_retrieved)
        self.assertIn("d2", doc_ids_retrieved)

        # In round 2 with d2 present, all claims should be supported!
        self.assertEqual(res.action, Action.STOP)
        self.assertEqual(res.rounds, 2)
        self.assertEqual([step["n_claims"] for step in res.history], [2, 2])
        self.assertEqual(res.history[1]["new_docs"], 1)
        self.assertIn("[d1]", res.final_answer)
        self.assertIn("[d2]", res.final_answer)

    def test_empty_corpus_abstains_and_counts_the_retrieval_round(self):
        cfg = MedRagConfig(use_reranker=False)
        pipeline = MedRagPipeline(documents=[], llm_client=LLMClient("mock"), config=cfg)
        result = pipeline.answer("Is this treatment safe?")

        self.assertEqual(result.action, Action.ABSTAIN)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.history[0]["new_docs"], 0)
        self.assertIn("ABSTAIN", result.final_answer)


class TestRepresentativeClinicalQueries(unittest.TestCase):
    """End-to-end tests on representative clinical scenarios."""

    def test_metformin_renal_safety_abstain_or_partial_when_gaps(self):
        from main import _build_sample_corpus
        corpus = _build_sample_corpus()

        llm = LLMClient(backend="mock")
        cfg = MedRagConfig()
        cfg.max_search_rounds = 2
        cfg.use_reranker = False

        pipeline = MedRagPipeline(documents=corpus, llm_client=llm, config=cfg)
        res = pipeline.answer("Is metformin safe for patients with renal impairment?")

        # The pipeline must complete safely and return valid evidence state
        self.assertIsNotNone(res.evidence_state)
        self.assertNotEqual(res.action, Action.STOP)
        self.assertIn(res.action, [Action.PARTIAL, Action.CONFLICTED, Action.ABSTAIN])
        self.assertTrue(len(res.retrieved_docs) > 0)
        self.assertFalse(hasattr(res.final_answer, "exception"))

    def test_aspirin_primary_prevention_elderly(self):
        from main import _build_sample_corpus
        corpus = _build_sample_corpus()

        llm = LLMClient(backend="mock")
        cfg = MedRagConfig()
        cfg.max_search_rounds = 2
        cfg.use_reranker = False

        pipeline = MedRagPipeline(documents=corpus, llm_client=llm, config=cfg)
        res = pipeline.answer("Should aspirin be used for primary prevention of cardiovascular events in elderly patients?")

        self.assertIsNotNone(res.evidence_state)
        self.assertNotEqual(res.action, Action.STOP)
        self.assertIn(res.action, [Action.PARTIAL, Action.CONFLICTED, Action.ABSTAIN])
        self.assertTrue(len(res.retrieved_docs) > 0)

    def test_doac_atrial_fibrillation(self):
        from main import _build_sample_corpus
        corpus = _build_sample_corpus()

        llm = LLMClient(backend="mock")
        cfg = MedRagConfig()
        cfg.max_search_rounds = 2
        cfg.use_reranker = False

        pipeline = MedRagPipeline(documents=corpus, llm_client=llm, config=cfg)
        res = pipeline.answer("What anticoagulant is preferred for stroke prevention in non-valvular atrial fibrillation?")

        self.assertIsNotNone(res.evidence_state)
        self.assertNotEqual(res.action, Action.STOP)
        self.assertIn(res.action, [Action.PARTIAL, Action.CONFLICTED, Action.ABSTAIN])
        self.assertTrue(len(res.retrieved_docs) > 0)


if __name__ == "__main__":
    unittest.main()
