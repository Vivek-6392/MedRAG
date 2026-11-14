# Changelog

All notable changes to the MedRAG research codebase will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to reproducible scientific software practices.

## [Unreleased] - Phase A: Core Correctness & Provenance Infrastructure

### Added
- **Centralized Provider Resolution (`medrag/provider.py`)**:
  - `ProviderResolver`: Centralized provider registry that deterministically maps and validates backend providers (`openai`, `groq`, `ollama`, `mock`).
  - Strict credential validation: Explicitly checks for provider-specific API keys (`OPENAI_API_KEY`, `GROQ_API_KEY`) and raises clear `ValueError` with remediation instructions rather than silently falling back or failing with raw HTTP errors.
  - Transparent auto-selection: Implemented verifiable auto-selection prioritizing OpenAI, then Groq, then local Ollama, falling back to Mock with an explicit warning and recorded selection reason (`resolution_reason`).
  - Provider switching in Streamlit (`app.py`): Dynamic sidebar provider selection now properly reconfigures model endpoints and API keys via `cfg.resolve_for_backend(...)`.

- **Clinical Risk Policy Engine (`medrag/risk_policy.py`)**:
  - Explicit risk tiers: `RiskTier.LOW` (0.00-0.34), `RiskTier.MODERATE` (0.35-0.69), `RiskTier.HIGH` (0.70-0.89), `RiskTier.CRITICAL` (0.90-1.00).
  - Word-boundary regex matching: Fixed substring false positives (e.g., query containing "adrenal gland" was incorrectly matching substring "renal").
  - Adaptive decision thresholds: `compute_effective_thresholds` tightens support thresholds (up to +0.10) and conflict tolerance (-0.10) for high-risk clinical queries.
  - Conservative abstention: Strictly prohibits partial answer generation (`allow_partial=False`) for high- and critical-risk queries when any claims remain unverified.
  - Safe sanitization: Catches and bounds malformed, non-numeric, or NaN risk inputs to safe defaults.

- **Unified Document Validation Engine (`medrag/document_validator.py`)**:
  - Pydantic schema `ValidatedDocumentInput` enforcing data integrity across CLI (`main.py`) and Web UI (`app.py`) ingress points.
  - Validates document IDs (non-empty, uniqueness checking across corpus sets).
  - Validates text content (minimum length 15 characters, non-whitespace).
  - Validates publication year (1800 to current year + 1).
  - Strict DOI syntax validation using regex `^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$`.
  - Strict URL syntax validation for HTTP/HTTPS protocols.
  - Safe default imputation for missing `quality_score` (defaults to 0.50 with `is_default_quality=True` flag).

- **Research Infrastructure & Safety Foundations**:
  - `medrag/logger.py`: Implemented structured JSON/stream logger with automated regex redaction for API keys (`sk-...`, `gsk_...`) and Patient Protected Health Information (SSNs, emails, MRNs).
  - `medrag/config_schema.py`: Pydantic research configuration with immutable hashing (`sha256`), configuration freezing, and strict parameter bounds.
  - Regression test suites added under `tests/`:
    - `test_provider_resolver.py`: 8 unit tests covering auto-resolution, credential validation, and provider mapping.
    - `test_retrieval_rrf.py`: 5 unit tests covering BM25-only, dense-only, dual-hit rank merging, deterministic tie-breaking, and empty sets.
    - `test_evidence_scoring.py`: 5 unit tests covering source reliability, recency decay, contradiction separation, and safe metadata handling.
    - `test_risk_policy.py`: 5 unit tests covering word boundaries, critical risk, malformed inputs, threshold adaptation, and controller abstention.
    - `test_document_validator.py`: 7 unit tests covering schema validation, length, year bounds, DOI/URL formatting, and duplicate ID detection.
    - `test_research_foundation.py`: 6 unit tests covering log redaction, PHI scrubbing, and configuration freezing.

### Changed
- **Reciprocal Rank Fusion Overhaul (`medrag/retriever.py`)**:
  - Fixed provenance loss: Merged documents now preserve both `bm25_rank` and `dense_rank` instead of dropping dense ranks for dual hits.
  - Preserved raw component scores: `bm25_score` and `dense_score` are recorded directly on `RetrievedDocument`.
  - Deterministic tie-breaking: RRF rankings break ties deterministically using `(-rrf_score, doc_id)`.
  - Safe empty handling: Empty candidate lists now return safely without raising exceptions or computing invalid ranks.

- **Evidence Scoring & Calibration Calibration (`medrag/evidence_state.py`, `medrag/verifier.py`)**:
  - Renamed `clinical_confidence` property to `heuristic_confidence` to clearly distinguish heuristic scores from calibrated clinical probabilities, while maintaining a backward-compatible alias.
  - Separated evidence support (`entailment`) from source reliability (`source_reliability`).
  - Removed undocumented arbitrary score adjustments and hardcoded floors.
  - Added explicit, configurable parameters: `reference_year` (prevents calendar drift during multi-year evaluations), `quality_weight` (default 0.6), `annual_recency_decay` (default 0.02), and `conflict_penalty_weight` (default 0.5).
  - Handled missing source quality metadata safely: unrated passages receive baseline quality 0.50 with explicit provenance tracking.

## [Unreleased] - Phase B: Reusable Medical Benchmark Framework

### Added
- **Reusable Benchmark Architecture (`medrag/benchmarks/`)**:
  - `DatasetAdapter` (`medrag/benchmarks/adapter.py`): Common abstract contract for dataset adapters (split loading, normalization, input prompt construction, answer normalization, gold extraction, evaluation).
  - `BenchmarkExample` & `PredictionRecord` (`medrag/benchmarks/schema.py`): Normalized benchmark schemas supporting multiple-choice, boolean, and open-ended clinical QA tasks.
  - `BenchmarkRegistry` (`medrag/benchmarks/registry.py`): Dynamic registry supporting adapter discovery, instantiation, and custom registration via `@register_adapter`.
  - `EvaluationRunner` (`medrag/benchmarks/runner.py`): Unified benchmark orchestration supporting dataset and split selection, limits, dynamic corpus setup, pipeline execution, and resume recovery.
  - `PredictionRecorder` (`medrag/benchmarks/recorder.py`): Streaming JSONL persistence for predictions and retrieval traces, resume detection, and artifact storage under `runs/<experiment_id>/`.
  - `MetricRegistry` (`medrag/benchmarks/metrics.py`): Scientific evaluation metrics computing Raw Accuracy, Selective Accuracy, Abstention Rate, Coverage, Area Under the Risk-Coverage Curve (AURC), and retrieval/latency efficiency.
  - `ExperimentManifest` (`medrag/benchmarks/manifest.py`): Immutable provenance tracking recording git commit SHA, library versions, prompt template hashes, and random seeds.
- **Concrete Dataset Adapters (`medrag/benchmarks/adapters/`)**:
  - `SyntheticAdapter`: Curated multi-choice clinical vignettes with paired literature passages for reproducible offline testing and smoke runs.
  - `MedQAAdapter`: USMLE Step 1/2CK/3 4-option subset adapter with option extraction and local/fallback loading.
  - `PubMedQAAdapter`: PQA-L expert-labeled subset adapter with yes/no/maybe normalization and context abstract extraction.
- **Evaluation Command-Line Interface (`eval.py`)**:
  - Standalone CLI for launching, resuming, and evaluating benchmark runs (`--smoke-test`, `--dataset`, `--split`, `--limit`, `--backend`, `--max-rounds`, `--resume`, `--list-benchmarks`).
- **Comprehensive Documentation & Regression Suite**:
  - `docs/BENCHMARK_GUIDE.md`: Step-by-step instructions for running smoke tests, validation split tuning, test set freezing, and artifact inspection.
  - `tests/test_benchmark_framework.py`: 9 unit and integration tests covering adapter contracts, normalization, registry lookup, streaming recorder, resume recovery, AURC computation, and runner execution.

## [Unreleased] - Phase C: MedQA Benchmark Adapter & Textbook Corpus Infrastructure

### Added
- **MedQA (USMLE 4-Option) Benchmark Adapter (`medrag/benchmarks/adapters/medqa.py`)**:
  - Robust ingestion of official `train`, `dev`, and `test` splits of the English USMLE 4-option subset.
  - Strict validation: requires exactly 4 options (`A`, `B`, `C`, `D`), non-empty vignette text, and canonical gold answer.
  - Duplicate detection: identifies and logs duplicate question IDs or deterministic question hashes.
  - Leakage prevention: guarantees that gold answer labels and choices never leak into prompt text.
  - Structured JSON output format with controlled fallback for unparseable or ambiguous model completions.
  - Medical textbook corpus preparer (`TextbookCorpusPreparer`): chunking (800 chars, 150 overlap), provenance tagging, and SHA-256 corpus digest generation.
- **Experimental Protocol & Testing**:
  - `docs/experiments/MEDQA.md`: Comprehensive experimental protocol covering the five comparison baselines, textbook chunking, split discipline, and clinical safety disclaimers.
  - `tests/test_medqa_benchmark.py`: 12 unit and integration tests covering split loading, leakage prevention, structured JSON parsing, textbook chunking, and protocol execution.

## [Unreleased] - Phase D: PubMedQA Benchmark Adapter & Dual Protocol Infrastructure

### Added
- **PubMedQA (PQA-L Expert-Labeled Subset) Adapter (`medrag/benchmarks/adapters/pubmedqa.py`)**:
  - Strict PQA-L validation: validates non-empty PMID, question, reference abstract contexts, and gold decision (`yes`, `no`, `maybe`).
  - Strict subset separation: explicitly rejects unlabeled PQA-U and artificially generated PQA-A items to prevent benchmark corruption.
  - Dual research evaluation protocols:
    - `PUBMEDQA_PROVIDED_CONTEXT`: Evaluates answer reasoning and claim verification given benchmark-supplied abstract evidence (zero external retrieval).
    - `PUBMEDQA_RETRIEVAL`: Evaluates external retrieval across a biomedical corpus; questions only provided with zero leakage of abstract or gold labels; tracks reference PMID retrieval.
  - Structured JSON answer generation schema with claims, citations, and explicit abstention support.
  - Controlled recovery: parses markdown code fences, handles option mappings (`A`->`yes`, `B`->`no`, `C`->`maybe`), detects conflicting signals (`yes` vs `no` conclusions), and safely abstains.
  - Official evaluation compatibility: exports predictions in `{ "PMID": "yes|no|maybe" }` format to `official_pubmedqa_predictions.json` and provides exact Python implementation of official evaluation script (`evaluate_official_predictions`).
- **Claim-Level Evidence Verification Engine (`medrag/evidence_state.py`)**:
  - Added explicit 4-way claim verification taxonomy (`SUPPORTED`, `CONTRADICTED`, `CONFLICTED`, `INSUFFICIENT_EVIDENCE`).
  - Provenance tracking with PMID and passage indexing.
  - Epistemic discipline: explicit documentation that NLI entailment scores represent linguistic compatibility, never calibrated clinical probabilities.
- **Scientific Retrieval & Selective Classification Metrics (`medrag/benchmarks/metrics.py`)**:
  - Retrieval metrics: `Recall@1`, `Recall@5`, `Recall@10`, Mean Reciprocal Rank (`MRR`), and `nDCG@5`, `nDCG@10`.
  - Selective classification metrics: `selective_risk` ($1 - \text{Selective Accuracy}$), per-class precision, per-class recall, and Macro-F1 across `yes`, `no`, `maybe`.
- **Experimental Protocol Documentation & Regression Suite**:
  - `docs/experiments/PUBMEDQA.md`: Comprehensive protocol documentation specifying PQA-L dataset properties, dual evaluation protocols, claim verification taxonomy, metric definitions, CLI workflows, and clinical ethics disclaimers.
  - `tests/test_pubmedqa_benchmark.py`: 13 comprehensive unit and integration tests covering all requirements.

## [Unreleased] - Phase E: Standardized Baselines & Ablation Framework

### Added
- **Standardized Baseline Systems (`medrag/benchmarks/experiments.py`)**:
  - `BASELINE 1: CLOSED_BOOK_LLM`: Direct zero-shot generation without external retrieval.
  - `BASELINE 2: SINGLE_PASS_DENSE_RAG`: Single-round dense embedding retrieval (MiniLM) + generation.
  - `BASELINE 3: SINGLE_PASS_BM25_RAG`: Single-round BM25 lexical keyword retrieval + generation.
  - `BASELINE 4: HYBRID_RAG`: BM25 + dense retrieval fused via Reciprocal Rank Fusion (RRF).
  - `BASELINE 5: HYBRID_RERANKED_RAG`: Hybrid retrieval + cross-encoder neural reranking.
  - `BASELINE 6: CLAIM_VERIFIED_RAG`: Single-round hybrid retrieval + claim extraction + NLI verification.
  - `BASELINE 7: ITERATIVE_RAG`: Multi-round iterative search and claim verification with standard threshold stopping (no risk policy).
  - `PROPOSED SYSTEM: FULL_MEDRAG`: Full architecture with iterative hybrid retrieval, reranking, claim decomposition, NLI verification, evidence-state tracking, source-quality weighting, risk-aware adaptive stopping, and selective abstention.
- **Leave-One-Out Ablation Framework (`medrag/benchmarks/experiments.py`)**:
  - Configurations for all 9 core dimensions: `ablation_no_hybrid`, `ablation_no_reranking`, `ablation_no_claim_decomposition`, `ablation_no_nli_verification`, `ablation_no_evidence_state`, `ablation_no_iterative`, `ablation_no_risk_policy`, `ablation_no_abstention`, and `ablation_no_source_quality`.
  - Component toggles added to `MedRagConfig` and `ExperimentConfig` ensuring strictly one component is disabled at a time while holding all other parameters identical.
  - `validate_fairness`: Automated verification of Part C fairness constraints (identical benchmark, split, LLM model, temperature, retrieval limits, and seeds).
- **Statistical Significance Engine (`medrag/benchmarks/significance.py`)**:
  - `compute_mcnemar_test`: Paired binary correctness testing supporting both Edwards continuity-corrected $\chi^2$ test ($b+c \ge 25$) and exact two-tailed binomial test ($b+c < 25$).
  - `paired_bootstrap_ci`: Paired empirical bootstrap resampling ($B=1000$) computing 95% confidence intervals and empirical two-tailed $p$-values for continuous metrics (Accuracy, Selective Accuracy, Coverage).
  - `compare_systems`: High-level paired comparison report generator with Markdown and JSON exports and statistical limitations disclaimer.
- **Multi-Configuration Study Execution (`medrag/benchmarks/study.py`)**:
  - `StudyRunner`: Executes complete baseline and ablation suites in one command (`python eval.py --study baselines` or `--study ablations`).
  - Automated output compilation: `comparison_table.md`, `ablation_table.md`, `per_example_comparison.csv` (side-by-side example predictions), and `statistical_report.json`.
- **Experimental Protocols & Test Suite**:
  - `docs/experiments/BASELINES.md`: Formal specification of the 7 baselines and proposed system, activation matrix, fairness protocols, and CLI usage.
  - `docs/experiments/ABLATIONS.md`: Detailed leave-one-out hypotheses, statistical testing criteria, and scientific reporting standards.
  - `tests/test_baselines_and_ablations.py`: 11 unit and integration tests covering presets, leave-one-out toggles, fairness validation, McNemar's exact/corrected test, paired bootstrap resampling, and multi-configuration study runner execution.

