# MedRAG Research Implementation Roadmap

**Date:** November 2026  
**Status:** Step-by-Step Engineering & Experimental Transformation Plan

---

## 1. Overview and Phasing Strategy

To achieve the research objective of transforming MedRAG into an independently reproducible, publication-grade medical RAG research artifact without breaking working code or introducing ungrounded claims, work is organized into **five progressive phases**:

1. **Phase A:** Core Bug Fixes, Metadata Integrity & Pipeline Hardening (Prioritizing Correctness).
2. **Phase B:** Benchmark Adapters & Split Management (MedQA & PubMedQA).
3. **Phase C:** Statistical Evaluation Engine & Research Metrics (AURC, Attribution, Confidence Intervals).
4. **Phase D:** Baselines & Ablation Studies Suite.
5. **Phase E:** Experiment Orchestration, Configuration Freezing & Publication Tables.

---

## 2. Detailed Modification Specifications

### Phase A: Core Bug Fixes & Pipeline Hardening (Correctness First)

#### Modification A.1: Fix Dynamic Provider Switching & Key Resolution
- **Why it is needed:** When switching providers in Streamlit (`app.py`) or CLI (`main.py`), credentials and base URLs are not dynamically updated from the environment, causing authentication errors when switching from Groq to OpenAI or Ollama. In `main.py`, default `-b auto` causes the mock fallback block to be unreachable.
- **Affected Existing Files:**
  - [`app.py` Lines 306-320](file:///c:/Users/vivek/OneDrive/Desktop/MedRag/app.py#L306-L320)
  - [`main.py` Lines 308-335](file:///c:/Users/vivek/OneDrive/Desktop/MedRag/main.py#L308-L335)
  - [`config.py` Lines 72-128](file:///c:/Users/vivek/OneDrive/Desktop/MedRag/config.py#L72-L128)
- **New Module Needed:** No.
- **Expected Research Benefit:** Ensures headless experiment scripts and Streamlit researchers can seamlessly switch between providers (OpenAI, Groq, local Ollama, offline Mock) without runtime crashes or credential leaks.
- **Required Tests:**
  - Unit tests verifying provider-switching updates API keys and base URLs accordingly.
  - Test verifying `main.py` CLI in `--backend auto` falls back to `mock` when no environment keys exist.

#### Modification A.2: Retain Complete Ranking & Score Metadata in RRF
- **Why it is needed:** In `medrag/retriever.py`, documents present in both BM25 and Dense sets have their `dense_rank` overwritten with `None`, and raw retrieval scores are discarded.
- **Affected Existing Files:**
  - [`medrag/retriever.py` Lines 158-188](file:///c:/Users/vivek/OneDrive/Desktop/MedRag/medrag/retriever.py#L158-L188)
- **New Module Needed:** No.
- **Expected Research Benefit:** Allows fine-grained retrieval analysis (e.g. comparing MRR of BM25 vs. Dense vs. RRF, and analyzing rank correlation).
- **Required Tests:**
  - Unit test verifying that a document appearing in both BM25 and Dense sets preserves both `bm25_rank` and `dense_rank`.
  - Test verifying that raw component scores are preserved in `RetrievedDocument` metadata.

#### Modification A.3: Align Citation Indexing with Permanent Provenance IDs
- **Why it is needed:** Generator prompt formats evidence with temporary integer indices `[1], [2], ...`, while `_synthesize_final_answer` formats findings using permanent document IDs `[passage.doc_id]`. If `doc_id` is a PMID string (e.g. `pmid_34211`), citations generated in the text do not match the final synthesized provenance keys.
- **Affected Existing Files:**
  - [`medrag/generator.py` Lines 235-250, 403-414](file:///c:/Users/vivek/OneDrive/Desktop/MedRag/medrag/generator.py#L403-L414)
  - [`medrag/pipeline.py` Lines 380-420](file:///c:/Users/vivek/OneDrive/Desktop/MedRag/medrag/pipeline.py#L380-L420)
- **New Module Needed:** No.
- **Expected Research Benefit:** Resolves citation mismatches and enables automated citation precision and recall measurement against ground-truth corpora.
- **Required Tests:**
  - Unit test asserting that document citations in the generated answer match the actual document IDs in the evidence state.

#### Modification A.4: Unify Temporal Decay and Recency Calibration
- **Why it is needed:** `ClaimEvidence` has two competing formulas (`clinical_confidence` vs `effective_support`) with different decay rates and floors, and hardcodes `datetime.now().year`, creating calendar-dependent drift.
- **Affected Existing Files:**
  - [`medrag/evidence_state.py` Lines 102-133](file:///c:/Users/vivek/OneDrive/Desktop/MedRag/medrag/evidence_state.py#L102-L133)
- **New Module Needed:** No.
- **Expected Research Benefit:** Guarantees deterministic, reproducible confidence scoring independent of the execution year by allowing a configurable reference year (e.g. publication year of the benchmark).
- **Required Tests:**
  - Unit test verifying identical confidence output across simulated calendar years given a fixed reference year.

---

### Phase B: Benchmark Adapters & Split Isolation (MedQA & PubMedQA)

#### Modification B.1: Deterministic Benchmark Ingestion & Checksum Verification
- **Why it is needed:** Research Requirement 12 strictly forbids checking copyrighted benchmark corpora into Git. Datasets must be downloaded deterministically with SHA-256 integrity verification.
- **Affected Existing Files:** None.
- **New Module Needed:** Yes: `medrag_benchmarks/download.py`.
- **Expected Research Benefit:** Researchers can execute a single command (`python -m medrag_benchmarks.download --all`) to prepare authentic evaluation data without repository bloat or licensing violations.
- **Required Tests:**
  - Checksum validation tests for downloaded files.
  - Mock download tests simulating network error handling and cache validation.

#### Modification B.2: Split-Aware Dataset Adapters for MedQA and PubMedQA
- **Why it is needed:** Research protocols mandate keeping training, validation, and test splits strictly separated. Hyperparameters and prompts must only be tuned on validation splits.
- **Affected Existing Files:** None.
- **New Module Needed:** Yes:
  - `medrag_benchmarks/schemas.py`: Standardized Pydantic schemas.
  - `medrag_benchmarks/loaders/medqa_loader.py`: Official USMLE 4-option split manager.
  - `medrag_benchmarks/loaders/pubmedqa_loader.py`: PQA-L expert subset manager.
  - `medrag_benchmarks/adapters/medqa_adapter.py`: Multiple-choice prediction parser.
  - `medrag_benchmarks/adapters/pubmedqa_adapter.py`: Yes/No/Maybe decision extractor.
- **Expected Research Benefit:** Prevents data leakage between validation tuning and final test reporting. Ensures exact comparability with existing published literature.
- **Required Tests:**
  - Test verifying zero sample overlap between train, dev, and test splits.
  - Adapter parser tests on synthetic multiple-choice questions and yes/no/maybe responses.

---

### Phase C: Statistical Evaluation Engine & Research Metrics

#### Modification C.1: Multi-Metric Evaluation Module
- **Why it is needed:** Benchmarking clinical RAG requires measuring accuracy, selective answering, and evidence attribution. Accuracy alone does not capture clinical reliability.
- **Affected Existing Files:** None.
- **New Module Needed:** Yes:
  - `medrag_eval/metrics/classification.py`: Accuracy, Balanced Accuracy, Macro-F1.
  - `medrag_eval/metrics/selective_answering.py`: AURC, Risk@Coverage, Abstention rate.
  - `medrag_eval/metrics/attribution.py`: Citation precision, recall, and hallucination rate.
  - `medrag_eval/metrics/retrieval.py`: Recall@k, MRR@k, NDCG@k.
- **Expected Research Benefit:** Quantifies the trade-off between coverage and accuracy, showing that selective abstention protects against high-risk hallucinations.
- **Required Tests:**
  - Unit tests for edge cases in AURC (monotonicity, 100% coverage, 0% coverage).
  - Unit tests verifying Macro-F1 with imbalanced multi-class confusion matrices.
  - Citation attribution precision/recall tests against known ground-truth citations.

#### Modification C.2: Confidence Intervals and Hypothesis Testing
- **Why it is needed:** Publication standards require confidence intervals and statistical significance testing rather than point estimates alone.
- **Affected Existing Files:** None.
- **New Module Needed:** Yes: `medrag_eval/stats/bootstrap.py` and `medrag_eval/stats/significance_tests.py`.
- **Expected Research Benefit:** Enables reporting results with 95% confidence intervals (e.g., $74.2\% \pm 2.1\%$) and confirming whether MedRAG's gains over baselines are statistically significant ($p < 0.05$).
- **Required Tests:**
  - Bootstrap CI tests verifying bounded outputs ($0.0 \le \text{lower} \le \text{upper} \le 1.0$).
  - McNemar's test validation against known contingency tables.

---

### Phase D: Baseline Systems & Ablation Matrix

#### Modification D.1: Standardized Comparison Baselines
- **Why it is needed:** Validating research claims requires running all baselines on identical evaluation examples with the same base models.
- **Affected Existing Files:** None.
- **New Module Needed:** Yes:
  - `medrag_baselines/direct_generation.py`: Zero-shot LLM without retrieval.
  - `medrag_baselines/standard_bm25_rag.py`: Single-round BM25 baseline.
  - `medrag_baselines/standard_dense_rag.py`: Single-round Dense baseline.
  - `medrag_baselines/hybrid_standard_rag.py`: Single-round hybrid retrieval without verification/controller.
- **Expected Research Benefit:** Demonstrates empirical progression: Direct Gen $\rightarrow$ Naive RAG $\rightarrow$ Hybrid RAG $\rightarrow$ MedRAG.
- **Required Tests:**
  - Execution tests for each baseline ensuring uniform output schema (`EvaluationRecord`).

#### Modification D.2: Automated Ablation Suite
- **Why it is needed:** An ablation study is essential to isolate the impact of each core component:
  1. Full MedRAG.
  2. No NLI Claim Verification (LLM self-assessment).
  3. No Iterative Controller (fixed 1 round).
  4. No Risk-Aware Policy (uniform thresholds).
- **Affected Existing Files:** None.
- **New Module Needed:** Yes: `scripts/run_ablations.py`.
- **Expected Research Benefit:** Validates the scientific hypothesis that claim-level NLI verification and risk-aware stopping are the primary drivers of reduced medical error rates.
- **Required Tests:**
  - Automated ablation smoke tests running over a synthetic validation subset.

---

### Phase E: Experiment Orchestrator, Freezing & Publication Tables

#### Modification E.1: Experiment Runner with Configuration Freezing
- **Why it is needed:** Every experiment must be reproducible by locking configuration parameters with SHA-256 hashes and saving raw per-sample prediction logs (`predictions.jsonl`).
- **Affected Existing Files:** None.
- **New Module Needed:** Yes: `scripts/run_benchmark.py` and `medrag_utils/artifact_saver.py`.
- **Expected Research Benefit:** Independent researchers can verify and audit every prediction and claim decision without having to re-run expensive LLM inference.
- **Required Tests:**
  - Test verifying configuration hash matches artifact metadata.
  - Test verifying JSONL prediction logs contain all required audit fields.

#### Modification E.2: Publication Table & LaTeX Generator
- **Why it is needed:** Automatically compiles evaluation outputs into publication-ready LaTeX tables (`results/tables/benchmark_results.tex`), avoiding manual transcription errors.
- **Affected Existing Files:** None.
- **New Module Needed:** Yes: `scripts/generate_paper_tables.py`.
- **Expected Research Benefit:** Direct inclusion into academic LaTeX manuscripts with zero transcription errors.
- **Required Tests:**
  - Test verifying valid LaTeX table compilation from mock evaluation metrics JSON.

---

## 3. Implementation Order & Safety Checklist

| Step | Task | Priority | Dependencies | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Fix provider switching & API key resolution | Critical | Existing code | Ready to start |
| **2** | Fix RRF metadata preservation (`dense_rank`, raw scores) | High | Existing code | Ready to start |
| **3** | Align citation indices with permanent `doc_id` | High | Existing code | Ready to start |
| **4** | Unify temporal decay and recency calibration | Medium | Existing code | Ready to start |
| **5** | Implement benchmark download & adapters (MedQA, PubMedQA) | High | Steps 1-4 | Blocked by 1-4 |
| **6** | Implement evaluation metrics (AURC, CIs, attribution) | High | Step 5 | Blocked by 5 |
| **7** | Implement baselines & ablation engine | Medium | Steps 5-6 | Blocked by 5-6 |
| **8** | Implement experiment runner & publication table generator | Medium | Step 7 | Blocked by 7 |

> [!IMPORTANT]
> In accordance with user instructions, no implementation modifications have been made during this turn. All changes will proceed sequentially starting with Phase A upon authorization.
