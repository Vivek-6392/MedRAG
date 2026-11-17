"""
Evaluation Runner for Medical QA Benchmarks.
Coordinates dataset loading, pipeline execution, resume/recovery,
prediction recording, and metric calculation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..config_schema import ExperimentConfig
from ..document_validator import validate_corpus
from ..generator import LLMClient
from ..logger import get_logger
from ..pipeline import MedRagPipeline, PipelineResult
from ..retriever import Document
from .adapter import DatasetAdapter
from .manifest import ExperimentManifest
from .metrics import EvaluationMetrics, MetricRegistry
from .recorder import PredictionRecorder
from .registry import BenchmarkRegistry
from .schema import BenchmarkExample, PredictionRecord, RetrievalTrace
from .statistical_report import generate_comprehensive_statistical_report
from .calibration import evaluate_calibration


class EvaluationRunner:
    """
    Unified experiment execution engine for MedRAG benchmarks.
    Supports dataset selection, limits, resuming, thresholding, and provenance logging.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        pipeline: Optional[MedRagPipeline] = None,
        adapter: Optional[DatasetAdapter] = None,
        corpus: Optional[List[Document]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or get_logger(f"medrag.eval.{config.experiment_id}")
        self.recorder = PredictionRecorder(config.output_dir, config.experiment_id)

        # Attach file handler to write logs/run.log alongside predictions
        _log_path = os.path.join(config.output_dir, config.experiment_id, "logs", "run.log")
        os.makedirs(os.path.dirname(_log_path), exist_ok=True)
        _fh = logging.FileHandler(_log_path, encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        self.logger.addHandler(_fh)

        # 1. Resolve Dataset Adapter
        self.adapter: DatasetAdapter = adapter or BenchmarkRegistry.get(config.benchmark_name)

        # 2. Load Split Examples
        self.logger.info(
            f"Loading dataset '{self.adapter.dataset_id}' split='{config.split}' "
            f"(limit={config.limit})"
        )
        self.examples: List[BenchmarkExample] = self.adapter.load_split(
            split=config.split,
            limit=config.limit,
        )
        self.logger.info(f"Loaded {len(self.examples)} benchmark examples.")

        # 3. Setup or Validate Knowledge Corpus
        protocol_clean = (config.protocol or "medrag_full").strip().lower()
        if protocol_clean in ("closed_book", "closed-book", "pubmedqa_provided_context", "provided_context"):
            # Protocols with zero external retrieval do not require indexing external corpus
            self.corpus = []
        elif corpus is not None:
            self.corpus = corpus
        elif config.corpus_path:
            with open(config.corpus_path, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)
            self.corpus = validate_corpus(raw_docs)
        elif config.benchmark_name == "medqa" and os.environ.get("MEDQA_CORPUS_DIR") and os.path.exists(os.environ["MEDQA_CORPUS_DIR"]):
            from .corpus_preparer import TextbookCorpusPreparer
            preparer = TextbookCorpusPreparer()
            docs, _ = preparer.prepare_directory(os.environ["MEDQA_CORPUS_DIR"])
            self.corpus = docs
        else:
            default_docs = self.adapter.get_default_corpus(self.examples)
            if default_docs:
                self.corpus = default_docs
            else:
                # Minimal fallback document if neither corpus nor context passages exist
                self.corpus = [
                    Document(
                        doc_id="fallback_corpus_doc",
                        title="General Medical Knowledge",
                        text="Standard clinical reference document for medical query evaluation.",
                        source="Reference Literature",
                        year=2024,
                        quality_score=0.85,
                    )
                ]

        # 4. Save Experiment Configuration
        self.recorder.save_config(config)

        # 5. Initialize or Load Experiment Manifest
        self.manifest = ExperimentManifest.from_config_and_adapter(
            config=config,
            adapter_version=self.adapter.version,
            num_examples=len(self.examples),
        )
        self.recorder.save_manifest(self.manifest)

        # 6. Pipeline Initialization
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.logger.info(
                f"Initializing MedRAG pipeline: backend={config.llm_backend}, "
                f"model={config.llm_model}, dense={config.dense_model}"
            )
            llm = LLMClient(
                backend=config.llm_backend,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
            medrag_cfg = config.to_medrag_config()
            if hasattr(llm, "api_key") and llm.api_key:
                medrag_cfg.llm_api_key = llm.api_key
            if hasattr(llm, "base_url") and llm.base_url:
                medrag_cfg.llm_base_url = llm.base_url
            if protocol_clean in ("closed_book", "closed-book"):
                medrag_cfg.use_reranker = False
            self.pipeline = MedRagPipeline(
                documents=self.corpus,
                llm_client=llm,
                config=medrag_cfg,
            )

    def run(self) -> EvaluationMetrics:
        """
        Executes the evaluation loop across all loaded examples.
        Handles resume/skip of completed examples, streaming record writes,
        and final metric computation.
        """
        self.logger.info(
            f"Starting evaluation run '{self.config.experiment_id}' "
            f"({len(self.examples)} total items)"
        )
        t_start = time.time()
        protocol_clean = (self.config.protocol or "medrag_full").strip().lower()

        interrupted = False
        try:
            for idx, example in enumerate(self.examples, 1):
                if self.config.resume and self.recorder.is_completed(example.example_id):
                    self.logger.info(
                        f"[{idx}/{len(self.examples)}] Resuming: Skipping already completed '{example.example_id}'"
                    )
                    continue

                self.logger.info(
                    f"[{idx}/{len(self.examples)}] Evaluating '{example.example_id}' "
                    f"(gold={example.gold_answer})"
                )

                is_provided_context = protocol_clean in ("pubmedqa_provided_context", "provided_context")
                try:
                    prompt = self.adapter.construct_input(example, include_context=is_provided_context)
                except TypeError:
                    prompt = self.adapter.construct_input(example)
                t0 = time.time()

                try:
                    result: PipelineResult = self._execute_protocol(prompt, example)
                    latency = round(time.time() - t0, 3)
                except Exception as e:
                    self.logger.error(f"Error evaluating '{example.example_id}': {e}", exc_info=True)
                    # Record failed/abstained execution safely
                    rec = PredictionRecord(
                        example_id=example.example_id,
                        dataset=example.dataset,
                        split=example.split,
                        question=example.question,
                        gold_answer=example.gold_answer,
                        predicted_answer=None,
                        raw_model_response="",
                        normalized_answer=None,
                        is_correct=False,
                        abstained=True,
                        decision_action="ABSTAIN",
                        decision_reason=f"Pipeline execution failure: {str(e)}",
                        latency_seconds=round(time.time() - t0, 3),
                        metadata={"error": str(e), "protocol": protocol_clean},
                    )
                    self.recorder.record_prediction(rec)
                    continue

                raw_response = result.final_answer
                norm_answer = self.adapter.normalize_answer(raw_response, example)

                # Determine decision action string
                action_str = result.action.value if hasattr(result.action, "value") else str(result.action)

                # Check abstention: either controller signaled ABSTAIN or response could not be parsed
                abstained = (action_str == "ABSTAIN" or norm_answer is None)
                if abstained:
                    is_correct = False
                else:
                    is_correct = self.adapter.evaluate_prediction(norm_answer, example.gold_answer, example)

                # Extract retrieved doc IDs and scores
                retrieved_ids: List[str] = []
                retrieval_scores: Dict[str, float] = {}
                for rdoc in result.retrieved_docs:
                    did = rdoc.document.doc_id
                    retrieved_ids.append(did)
                    retrieval_scores[did] = round(rdoc.score, 4)

                # Extract claim-level verification state
                generated_claims: List[str] = []
                supporting_ids: List[str] = []
                contradicting_ids: List[str] = []
                claim_verification_details: List[Dict[str, Any]] = []
                heuristic_score: Optional[float] = None

                if result.evidence_state:
                    heuristic_score = round(result.evidence_state.mean_confidence, 4)
                    for c_idx, ce in enumerate(result.evidence_state.claim_evidences, 1):
                        generated_claims.append(ce.claim)
                        status_name = getattr(ce, "verification_status", "INSUFFICIENT_EVIDENCE")
                        sup_list = [p.to_dict() for p in ce.supporting_passages]
                        con_list = [p.to_dict() for p in getattr(ce, "contradicting_passages", getattr(ce, "refuting_passages", []))]
                        claim_verification_details.append({
                            "claim_id": f"claim_{c_idx}",
                            "claim_text": ce.claim,
                            "status": status_name,
                            "nli_entailment": round(ce.support_score, 4),
                            "nli_contradiction": round(ce.conflict_score, 4),
                            "supporting_passages": sup_list,
                            "contradicting_passages": con_list,
                        })
                        for p in ce.supporting_passages:
                            if not p.illustrative and p.doc_id not in supporting_ids:
                                supporting_ids.append(p.doc_id)
                        for p in getattr(ce, "contradicting_passages", getattr(ce, "refuting_passages", [])):
                            if not getattr(p, "illustrative", False) and getattr(p, "doc_id", None) not in contradicting_ids:
                                contradicting_ids.append(p.doc_id)

                # Extract last decision reason
                decision_reason = ""
                if result.history:
                    decision_reason = result.history[-1].get("reason", "")

                # Approximate token usage
                prompt_tokens = len(prompt.split())
                completion_tokens = len(raw_response.split())
                token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

                rec_metadata: Dict[str, Any] = {
                    "config_hash": self.config.config_hash(),
                    "llm_backend": self.config.llm_backend,
                    "llm_model": self.config.llm_model,
                    "protocol": protocol_clean,
                    "claim_verification_details": claim_verification_details,
                }
                if protocol_clean in ("pubmedqa_retrieval", "retrieval"):
                    rec_metadata["reference_pmid"] = example.example_id
                    norm_ex_id = str(example.example_id).strip().lower()
                    ret_match = any(
                        norm_ex_id in did.lower() or did.lower().replace("pmid_", "").split("_")[0] == norm_ex_id
                        for did in retrieved_ids
                    )
                    rec_metadata["reference_retrieved"] = ret_match

                # Create and record prediction record
                rec = PredictionRecord(
                    example_id=example.example_id,
                    dataset=example.dataset,
                    split=example.split,
                    question=example.question,
                    gold_answer=example.gold_answer,
                    predicted_answer=norm_answer,
                    raw_model_response=raw_response,
                    normalized_answer=norm_answer,
                    is_correct=is_correct,
                    abstained=abstained,
                    retrieved_doc_ids=retrieved_ids,
                    retrieval_scores=retrieval_scores,
                    generated_claims=generated_claims,
                    supporting_evidence_ids=supporting_ids,
                    contradicting_evidence_ids=contradicting_ids,
                    rounds=result.rounds,
                    decision_action=action_str,
                    decision_reason=decision_reason,
                    heuristic_evidence_score=heuristic_score,
                    latency_seconds=latency,
                    token_usage=token_usage,
                    metadata=rec_metadata,
                )
                self.recorder.record_prediction(rec)

                # Record round-by-round retrieval traces
                traces: List[RetrievalTrace] = []
                for h in result.history:
                    traces.append(RetrievalTrace(
                        example_id=example.example_id,
                        round_number=h.get("round", 1),
                        search_query=h.get("query", ""),
                        new_docs_count=h.get("new_docs", 0),
                        total_docs_count=h.get("total_docs", 0),
                    ))
                self.recorder.record_retrieval_traces(traces)

                # Gentle pacing for cloud providers with per-minute token limits (e.g. Groq 8k TPM)
                if getattr(self.config, "llm_backend", "").lower() == "groq":
                    time.sleep(2.0)

        except KeyboardInterrupt:
            interrupted = True
            evaluated_so_far = self.recorder.load_all_predictions()
            partial_metrics = MetricRegistry.compute_all(evaluated_so_far)
            partial_dict = partial_metrics.model_dump()
            # Mark manifest as interrupted (partial) with current metrics
            self.manifest.status = "interrupted"
            self.manifest.notes = (
                f"Run interrupted by user after {len(evaluated_so_far)}/{len(self.examples)} examples. "
                f"Partial accuracy: {partial_metrics.accuracy:.3f}"
            )
            from datetime import datetime, timezone
            self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
            self.manifest.summary_metrics = partial_dict
            self.recorder.save_manifest(self.manifest)
            self.recorder.save_metrics(partial_dict)
            self.logger.warning(
                f"Evaluation interrupted after {len(evaluated_so_far)}/{len(self.examples)} examples. "
                f"Partial results saved. Resume with --config to continue."
            )
            print(
                f"\n[Interrupted] Saved partial results ({len(evaluated_so_far)}/{len(self.examples)} examples). "
                f"Re-run with same experiment ID to resume.\n"
            )
            return partial_metrics

        # 7. Aggregate all evaluated records
        all_records = self.recorder.load_all_predictions()
        metrics: EvaluationMetrics = MetricRegistry.compute_all(all_records)

        # 8. Save Metrics & Mark Manifest Complete
        metrics_dict = metrics.model_dump()
        self.recorder.save_metrics(metrics_dict)
        self.manifest.mark_completed(metrics_dict)
        self.recorder.save_manifest(self.manifest)

        # 8b. Export Statistical and Calibration Reports
        try:
            stat_rep = generate_comprehensive_statistical_report(all_records, experiment_id=self.config.experiment_id)
            with open(os.path.join(self.recorder.run_dir, "statistical_report.md"), "w", encoding="utf-8") as f:
                f.write(stat_rep.to_markdown())

            calib_rep = evaluate_calibration(all_records)
            with open(os.path.join(self.recorder.run_dir, "calibration_metrics.json"), "w", encoding="utf-8") as f:
                f.write(calib_rep.model_dump_json(indent=2))
        except Exception as e:
            self.logger.warning(f"Could not generate statistical or calibration report: {e}")

        # 9. Export official predictions format if supported by adapter
        if hasattr(self.adapter, "export_official_format"):
            try:
                # Save to run_dir (with all experiment artifacts) and base output_dir for convenience
                official_export_path = os.path.join(
                    self.recorder.run_dir,
                    f"official_{self.adapter.dataset_id}_predictions.json",
                )
                self.adapter.export_official_format(all_records, official_export_path)
                if os.path.abspath(self.config.output_dir) != os.path.abspath(self.recorder.run_dir):
                    base_export_path = os.path.join(
                        self.config.output_dir,
                        f"official_{self.adapter.dataset_id}_predictions.json",
                    )
                    self.adapter.export_official_format(all_records, base_export_path)
            except Exception as e:
                self.logger.warning(f"Failed to export official predictions: {e}")

        total_time = round(time.time() - t_start, 2)
        self.logger.info(
            f"Experiment '{self.config.experiment_id}' completed in {total_time}s. "
            f"Total: {metrics.total_examples}, Accuracy: {metrics.accuracy:.3f}, "
            f"Selective Accuracy: {metrics.selective_accuracy:.3f}, "
            f"Abstention Rate: {metrics.abstention_rate:.3f}"
        )

        return metrics

    def _execute_protocol(self, prompt: str, example: BenchmarkExample) -> PipelineResult:
        """
        Executes evaluation according to the selected research protocol:
        1. CLOSED_BOOK: Question and choices given to LLM without retrieval.
        2. TEXTBOOK_RAG: Question and choices given, relevant textbook passages retrieved (dense).
        3. HYBRID_RAG: Sparse (BM25) and dense retrieval fused with RRF.
        4. RERANKED_RAG: Hybrid retrieval followed by cross-encoder reranking.
        5. MEDRAG_FULL: Complete iterative MedRAG pipeline with claim extraction,
           sentence-level NLI verification, and adaptive stopping/abstention decisions.
        """
        protocol = (self.config.protocol or "medrag_full").strip().lower()

        # Protocol 1: CLOSED_BOOK (No retrieval)
        if protocol in ("closed_book", "closed-book", "closed_book_llm"):
            t0 = time.time()
            raw_response = self.pipeline.generator.llm.complete(
                prompt,
                max_tokens=self.config.llm_max_tokens,
            )
            return PipelineResult(
                question=prompt,
                final_answer=raw_response,
                action="STOP",
                evidence_state=None,
                rounds=0,
                retrieved_docs=[],
                history=[{
                    "round": 0,
                    "protocol": "closed_book",
                    "new_docs": 0,
                    "total_docs": 0,
                    "query": "",
                    "reason": "Closed-book direct generation (zero retrieval)",
                }],
                latency_seconds=round(time.time() - t0, 3),
            )

        # Protocol 2: SINGLE_PASS_DENSE_RAG (Single-round dense retrieval)
        elif protocol in ("single_pass_dense_rag", "dense_rag", "textbook_rag"):
            t0 = time.time()
            dense_retriever = getattr(self.pipeline.retriever, "dense", None)
            if dense_retriever is not None and getattr(dense_retriever, "_embeddings", None) is not None:
                docs = dense_retriever.retrieve(example.question, top_k=self.config.top_k_retrieve)
            else:
                docs = self.pipeline.retriever.retrieve(example.question)
            context_docs = docs[: self.config.top_k_rerank]
            gen_ans = self.pipeline.generator.generate_answer(prompt, context_docs)
            return PipelineResult(
                question=prompt,
                final_answer=gen_ans.answer,
                action="STOP",
                evidence_state=None,
                rounds=1,
                retrieved_docs=context_docs,
                history=[{
                    "round": 1,
                    "protocol": "single_pass_dense_rag",
                    "new_docs": len(context_docs),
                    "total_docs": len(context_docs),
                    "query": example.question,
                    "reason": "Single-round textbook dense retrieval",
                }],
                latency_seconds=round(time.time() - t0, 3),
            )

        # Protocol 3: SINGLE_PASS_BM25_RAG (Single-round sparse lexical retrieval)
        elif protocol in ("single_pass_bm25_rag", "bm25_rag"):
            t0 = time.time()
            hits = self.pipeline.retriever.bm25.retrieve(example.question, top_k=self.config.top_k_rerank)
            gen_ans = self.pipeline.generator.generate_answer(prompt, hits)
            return PipelineResult(
                question=prompt,
                final_answer=gen_ans.answer,
                action="STOP",
                evidence_state=None,
                rounds=1,
                retrieved_docs=hits,
                history=[{
                    "round": 1,
                    "protocol": "single_pass_bm25_rag",
                    "new_docs": len(hits),
                    "total_docs": len(hits),
                    "query": example.question,
                    "reason": "Single-round sparse BM25 retrieval",
                }],
                latency_seconds=round(time.time() - t0, 3),
            )

        # Protocol 4: HYBRID_RAG (BM25 + Dense with RRF, no reranking)
        elif protocol in ("hybrid_rag", "rrf_rag"):
            t0 = time.time()
            from ..retriever import reciprocal_rank_fusion
            bm25_hits = self.pipeline.retriever.bm25.retrieve(example.question, top_k=self.config.top_k_retrieve)
            dense_hits = []
            if getattr(self.pipeline.retriever, "dense", None) is not None:
                dense_hits = self.pipeline.retriever.dense.retrieve(example.question, top_k=self.config.top_k_retrieve)
            fused = reciprocal_rank_fusion(
                bm25_hits,
                dense_hits,
                bm25_weight=self.config.bm25_weight,
                dense_weight=self.config.dense_weight,
            )
            context_docs = fused[: self.config.top_k_rerank]
            gen_ans = self.pipeline.generator.generate_answer(prompt, context_docs)
            return PipelineResult(
                question=prompt,
                final_answer=gen_ans.answer,
                action="STOP",
                evidence_state=None,
                rounds=1,
                retrieved_docs=context_docs,
                history=[{
                    "round": 1,
                    "protocol": "hybrid_rag",
                    "new_docs": len(context_docs),
                    "total_docs": len(context_docs),
                    "query": example.question,
                    "reason": "Hybrid BM25 + Dense RRF retrieval",
                }],
                latency_seconds=round(time.time() - t0, 3),
            )

        # Protocol 5: HYBRID_RERANKED_RAG (Hybrid RRF + Cross-Encoder Reranker)
        elif protocol in ("hybrid_reranked_rag", "reranked_rag", "rerank_rag"):
            t0 = time.time()
            hits = self.pipeline.retriever.retrieve(example.question)
            context_docs = hits[: self.config.top_k_rerank]
            gen_ans = self.pipeline.generator.generate_answer(prompt, context_docs)
            return PipelineResult(
                question=prompt,
                final_answer=gen_ans.answer,
                action="STOP",
                evidence_state=None,
                rounds=1,
                retrieved_docs=context_docs,
                history=[{
                    "round": 1,
                    "protocol": "hybrid_reranked_rag",
                    "new_docs": len(context_docs),
                    "total_docs": len(context_docs),
                    "query": example.question,
                    "reason": "Hybrid retrieval with cross-encoder reranking",
                }],
                latency_seconds=round(time.time() - t0, 3),
            )

        # Protocol 6: CLAIM_VERIFIED_RAG (Single-round hybrid + rerank + claim extraction + NLI)
        elif protocol in ("claim_verified_rag", "claim_verified"):
            t0 = time.time()
            hits = self.pipeline.retriever.retrieve(example.question)
            context_docs = hits[: self.config.top_k_rerank]
            gen_ans = self.pipeline.generator.generate_answer(prompt, context_docs)
            evidence_state = None
            if getattr(self.pipeline, "verifier", None):
                claims = getattr(gen_ans, "atomic_claims", []) or [gen_ans.answer[:200]]
                evidence_state = self.pipeline.verifier.verify_claims(
                    claims=claims,
                    retrieved_docs=context_docs,
                    clinical_risk=0.5,
                    support_threshold=self.config.support_threshold,
                    conflict_threshold=self.config.conflict_threshold,
                    round_number=1,
                )
            return PipelineResult(
                question=prompt,
                final_answer=gen_ans.answer,
                action="STOP",
                evidence_state=evidence_state,
                rounds=1,
                retrieved_docs=context_docs,
                history=[{
                    "round": 1,
                    "protocol": "claim_verified_rag",
                    "new_docs": len(context_docs),
                    "total_docs": len(context_docs),
                    "query": example.question,
                    "reason": "Single-round hybrid retrieval with claim extraction and NLI verification",
                }],
                latency_seconds=round(time.time() - t0, 3),
            )

        # Protocol: PUBMEDQA_PROVIDED_CONTEXT (Benchmark-supplied abstract evidence)
        elif protocol in ("pubmedqa_provided_context", "provided_context"):
            t0 = time.time()
            ctx_docs = []
            for c_idx, passage in enumerate(example.context):
                ctx_docs.append(Document(
                    doc_id=f"pmid_{example.example_id}_{c_idx}",
                    title=f"PubMed Abstract {example.example_id}",
                    text=passage.strip(),
                    source="PubMed Central",
                    year=int(example.metadata.get("year") or 2024),
                    quality_score=0.90,
                ))

            # Generate structured answer using provided context
            gen_ans = self.pipeline.generator.generate_answer(prompt, ctx_docs)

            # Perform claim verification against provided abstract context if verifier available
            evidence_state = None
            if getattr(self.pipeline, "claim_extractor", None) and getattr(self.pipeline, "verifier", None):
                try:
                    claims = self.pipeline.claim_extractor.extract_claims(gen_ans.answer)
                    if claims:
                        evidence_state = self.pipeline.verifier.verify(claims, ctx_docs)
                except Exception as e:
                    self.logger.warning(f"Claim verification failed on provided context: {e}")

            from ..retriever import RetrievedDocument
            ret_docs = [RetrievedDocument(document=d, score=1.0) for d in ctx_docs]

            return PipelineResult(
                question=prompt,
                final_answer=gen_ans.answer,
                action="STOP",
                evidence_state=evidence_state,
                rounds=0,
                retrieved_docs=ret_docs,
                history=[{
                    "round": 0,
                    "protocol": "pubmedqa_provided_context",
                    "new_docs": len(ctx_docs),
                    "total_docs": len(ctx_docs),
                    "query": "",
                    "reason": "Evaluated using benchmark-supplied reference abstract (zero external retrieval)",
                }],
                latency_seconds=round(time.time() - t0, 3),
            )

        # Protocol: PUBMEDQA_RETRIEVAL (External biomedical corpus retrieval)
        elif protocol in ("pubmedqa_retrieval", "retrieval"):
            t0 = time.time()
            # Retrieve from background corpus using question ONLY
            ret_hits = self.pipeline.retriever.retrieve(example.question)
            context_docs = ret_hits[: self.config.top_k_rerank]
            gen_ans = self.pipeline.generator.generate_answer(prompt, context_docs)

            evidence_state = None
            if getattr(self.pipeline, "claim_extractor", None) and getattr(self.pipeline, "verifier", None):
                try:
                    claims = self.pipeline.claim_extractor.extract_claims(gen_ans.answer)
                    if claims:
                        evidence_state = self.pipeline.verifier.verify(claims, context_docs)
                except Exception as e:
                    self.logger.warning(f"Claim verification failed on retrieval: {e}")

            return PipelineResult(
                question=prompt,
                final_answer=gen_ans.answer,
                action="STOP",
                evidence_state=evidence_state,
                rounds=1,
                retrieved_docs=context_docs,
                history=[{
                    "round": 1,
                    "protocol": "pubmedqa_retrieval",
                    "new_docs": len(context_docs),
                    "total_docs": len(context_docs),
                    "query": example.question,
                    "reason": "External retrieval across documented biomedical corpus",
                }],
                latency_seconds=round(time.time() - t0, 3),
            )

        # Protocol 5: MEDRAG_FULL / RAG_ADAPTIVE (Full iterative claim verification & stopping)
        else:
            return self.pipeline.answer(prompt)
