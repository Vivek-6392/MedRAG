# MedRAG Research Architecture & Modular Design

**Date:** November 2026  
**Status:** Architectural Specification for Publication-Grade Medical RAG System

---

## 1. Architectural Philosophy & Principles

To produce an academic research artifact that external researchers can independently inspect, clone, configure, reproduce, and extend, MedRAG adheres to six core principles:

1. **Strict Separation of Concerns:** Core inference, benchmark ingestion, evaluation metrics, and baseline comparisons must reside in decoupled modules.
2. **Immutable Experiment Provenance:** Every evaluation run binds a frozen, validated configuration (`ExperimentConfig`) hashed via SHA-256 to raw output logs and predictions.
3. **Strict Split Isolation:** Benchmark training, validation, and test splits are physically and programmatically quarantined. Prompt tuning, threshold discovery, and hyperparameter optimization occur solely on validation splits. Test splits are evaluated only with frozen parameters.
4. **Statistical Rigor:** All reported benchmarks must include sample sizes, point estimates, confidence intervals (95% bootstrap / Wilson score), and statistical significance tests ($p$-values via McNemar's or paired permutation tests).
5. **Safe Attribution & Abstention:** Medical statements must never be asserted without explicit, claim-level passage provenance. When clinical risk exceeds confidence or contradictions exist, the system safely withholds recommendations.
6. **No Copyrighted Data in Version Control:** Benchmark corpora (e.g. USMLE questions, PubMed abstracts) are downloaded programmatically via deterministic scripts with checksum validation.

---

## 2. Proposed System Directory Structure

The repository architecture is organized into ten distinct functional subsystems:

```
MedRag/
│
├── .github/
│   └── workflows/
│       └── ci.yml                   # CI pipeline (Python 3.10-3.12, tests, lint)
│
├── configs/                         # Subsystem 6: Experiment Configurations
│   ├── base_config.json             # Default hyperparameter template
│   ├── benchmarks/                  # Benchmark-specific configurations
│   │   ├── medqa_usmle.json
│   │   └── pubmedqa_pqal.json
│   └── ablations/                   # Pre-defined ablation configurations
│       ├── ablation_no_controller.json
│       ├── ablation_no_nli_verifier.json
│       ├── ablation_no_reranker.json
│       └── ablation_no_risk_policy.json
│
├── data/                            # Local data cache (git-ignored)
│   ├── benchmarks/                  # Downloaded benchmark files
│   │   ├── medqa/
│   │   └── pubmedqa/
│   └── embeddings/                  # Cached document embeddings
│
├── docs/                            # Subsystem 10: Publication Documentation
│   ├── CODE_AUDIT.md                # Source-code audit report
│   ├── RESEARCH_ARCHITECTURE.md     # Architectural specification (this document)
│   ├── IMPLEMENTATION_ROADMAP.md    # Phased transformation plan
│   ├── METHODOLOGY.md               # Mathematical formulations & formal definitions
│   └── REPRODUCIBILITY_GUIDE.md     # Step-by-step reproduction guide
│
├── medrag/                          # Subsystem 1: Core MedRAG Pipeline
│   ├── __init__.py                  # Public API exports
│   ├── config.py                    # Backward-compatible runtime configuration
│   ├── config_schema.py             # Pydantic v2 schemas with SHA-256 freezing
│   ├── controller.py                # Phase 6: Action Controller (STOP, PARTIAL, etc.)
│   ├── evidence_state.py            # Phase 5: Passage & Claim Evidence State
│   ├── generator.py                 # Phase 3: Grounded Answer & Claim Extraction
│   ├── logger.py                    # Privacy-preserving logging with PHI/Key redaction
│   ├── pipeline.py                  # End-to-end 6-phase coordinator
│   ├── query_understanding.py       # Phase 1: Clinical Query Understanding
│   ├── retriever.py                 # Phase 2: Hybrid BM25 + Dense + RRF + Reranker
│   └── verifier.py                  # Phase 4: Sentence-level Cross-Encoder NLI Verifier
│
├── medrag_benchmarks/               # Subsystems 2 & 3: Adapters & Loaders
│   ├── __init__.py
│   ├── download.py                  # Automated downloader with SHA-256 verification
│   ├── schemas.py                   # Standardized Pydantic items (MedQAItem, PubMedQAItem)
│   ├── adapters/                    # Subsystem 2: Task-Specific Adapters
│   │   ├── base_adapter.py          # Abstract task adapter interface
│   │   ├── medqa_adapter.py         # USMLE 4-option multiple-choice adapter
│   │   └── pubmedqa_adapter.py      # Yes/No/Maybe decision & reasoning adapter
│   └── loaders/                     # Subsystem 3: Split-Aware Dataset Loaders
│       ├── base_loader.py           # Split enumeration and caching
│       ├── medqa_loader.py          # Train / Val / Test split manager for MedQA
│       └── pubmedqa_loader.py       # PQA-L expert subset loader
│
├── medrag_baselines/                # Subsystem 5: Baseline Systems
│   ├── __init__.py
│   ├── base_baseline.py             # Reusable baseline runner interface
│   ├── direct_generation.py         # Zero-shot / Chain-of-Thought (No Retrieval)
│   ├── standard_bm25_rag.py         # Single-round lexical BM25 RAG
│   ├── standard_dense_rag.py        # Single-round dense semantic RAG
│   ├── hybrid_standard_rag.py       # Hybrid retrieval without verification/controller
│   └── iterative_no_verify_rag.py   # Iterative retrieval without claim NLI
│
├── medrag_eval/                     # Subsystems 4 & 8: Metrics & Statistical Analysis
│   ├── __init__.py
│   ├── evaluator.py                 # Benchmark evaluation orchestrator
│   ├── metrics/                     # Subsystem 4: Evaluation Metrics
│   │   ├── classification.py        # Accuracy, Balanced Accuracy, Macro/Micro-F1
│   │   ├── selective_answering.py   # AURC, Risk@Coverage, Coverage@Risk
│   │   ├── retrieval.py             # Recall@k, MRR@k, NDCG@k, Precision@k
│   │   └── attribution.py           # Citation Precision, Recall, Hallucination Rate
│   └── stats/                       # Subsystem 8: Statistical Analysis
│       ├── bootstrap.py             # Percentile & BCa confidence intervals
│       ├── confidence_intervals.py  # Wilson score and Clopper-Pearson intervals
│       └── significance_tests.py    # McNemar's test, paired permutation tests
│
├── medrag_utils/                    # Subsystem 9: Reproducibility Utilities
│   ├── __init__.py
│   ├── reproducibility.py           # Deterministic seeding (Python, NumPy, PyTorch)
│   ├── artifact_saver.py            # Serializes raw predictions, configs, metrics
│   └── provenance.py                # Environment fingerprinting (OS, CPU, GPU, packages)
│
├── results/                         # Subsystem 7: Result Storage (git-ignored raw runs)
│   ├── runs/                        # Structured JSONL prediction logs per run
│   │   └── <experiment_id>_<timestamp>/
│   │       ├── config.json          # Frozen config with SHA-256 hash
│   │       ├── environment.json     # Hardware and dependency versions
│   │       ├── predictions.jsonl    # Per-sample predictions, claims, decisions
│   │       └── summary_metrics.json # Computed aggregate metrics with 95% CIs
│   └── tables/                      # Publication-ready LaTeX and Markdown tables
│
├── scripts/                         # Executable entry points
│   ├── run_benchmark.py             # Primary CLI runner for benchmarks
│   ├── run_ablations.py             # Automated ablation matrix executor
│   ├── run_baselines.py             # Executes all baseline comparisons
│   └── generate_paper_tables.py     # Compiles LaTeX tables from results/
│
├── tests/                           # Unit & Integration Test Suite
│   ├── conftest.py                  # Pytest configuration & offline model fixtures
│   ├── test_medrag.py               # Existing 35 core pipeline tests
│   ├── test_research_foundation.py  # Logging, redaction, config schema tests
│   ├── test_benchmarks.py           # Synthetic MedQA/PubMedQA adapter tests
│   ├── test_metrics.py              # Metric calculation tests (AURC, CIs, F1)
│   ├── test_baselines.py            # Baseline compatibility tests
│   └── test_reproducibility.py      # Seed determinism & artifact saving tests
│
├── app.py                           # Preserved Streamlit Web Application
├── main.py                          # Preserved Interactive CLI
├── pytest.ini                       # Test runner configuration
└── requirements.txt                 # Project dependencies
```

---

## 3. Subsystem Detailed Specifications

### 3.1. Benchmark Dataset Loaders & Split Management (`medrag_benchmarks/`)
- **MedQA Loader (`medqa_loader.py`):**
  - Dataset: English USMLE four-option subset.
  - Formats: JSONL with `question`, `options: {"A": ..., "B": ..., "C": ..., "D": ...}`, `answer_idx`, `meta_info`.
  - Splits: Official MedQA USMLE splits (`train`: 10,178, `dev`: 1,272, `test`: 1,273).
  - Validation-Only Rule: All prompt templates, few-shot exemplars, and confidence thresholds are tuned on `dev`. Evaluation on `test` is executed once with frozen configurations.
- **PubMedQA Loader (`pubmedqa_loader.py`):**
  - Dataset: Expert-labeled subset (`PQA-L`, 1,000 instances).
  - Formats: JSON with `QUESTION`, `CONTEXTS`, `LABELS`, `LONG_ANSWER`, `final_decision` (`yes` / `no` / `maybe`).
  - Splits: Official 10-fold cross-validation or standardized train/dev/test splits (500 train, 500 test).

### 3.2. Evaluation Metrics Engine (`medrag_eval/metrics/`)
- **Classification Performance:**
  - Standard Accuracy: $\text{Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$.
  - Macro-F1 across classes to account for class imbalance (crucial for PubMedQA `maybe` minority class).
- **Selective Answering & Risk-Coverage (AURC):**
  - Coverage: fraction of questions answered with a non-abstaining terminal action (`STOP`, `PARTIAL`).
  - Selective Accuracy: accuracy on answered subset.
  - Risk: error rate on answered subset.
  - Area Under the Risk-Coverage Curve (AURC): Area under empirical curve plotting selective risk against coverage $\in [0, 1]$. Lower AURC indicates superior calibration and risk awareness.
- **Attribution & Provenance Fidelity:**
  - Citation Precision: Fraction of cited passages that entail the corresponding generated claim.
  - Citation Recall: Fraction of claims requiring external evidence that possess at least one valid supporting citation.
  - Citation Hallucination Rate: Frequency of fabricated or non-existent document references.

### 3.3. Statistical Analysis Module (`medrag_eval/stats/`)
- **Confidence Intervals:**
  - Non-parametric empirical bootstrap with 2,000 resamples for metrics where analytical variance is complex (AURC, macro-F1, selective accuracy).
  - Wilson score interval for binomial proportions (accuracy, abstention rate).
- **Hypothesis Testing:**
  - Paired comparison between MedRAG and baselines on identical evaluation examples using McNemar's test for binary classification outcomes and paired permutation tests for continuous metric scores.

### 3.4. Baseline Implementations (`medrag_baselines/`)
1. **Direct Generation (Zero-shot LLM):** Prompting the LLM directly without retrieval or verification.
2. **Standard BM25 RAG:** Top-$k$ BM25 retrieval + generator prompt; single-round, no verifier, no controller.
3. **Standard Dense RAG:** Top-$k$ Dense retrieval + generator prompt; single-round.
4. **Hybrid Standard RAG:** BM25 + Dense + Cross-Encoder reranker; single-round.
5. **Iterative Search without Claim Verification:** Multi-round iterative retrieval driven by LLM self-reflection rather than sentence-level NLI entailment.

---

## 4. Interfaces and Data Contracts

### 4.1. Benchmark Item Contract
```python
class BenchmarkQuestion(BaseModel):
    id: str
    benchmark: str  # "medqa" | "pubmedqa"
    split: str      # "train" | "dev" | "test"
    question: str
    context: Optional[str] = None
    options: Optional[Dict[str, str]] = None  # e.g. {"A": ..., "B": ..., "C": ..., "D": ...}
    ground_truth: str                         # "A" or "yes"
```

### 4.2. Raw Prediction Record Contract
```python
class EvaluationRecord(BaseModel):
    question_id: str
    benchmark: str
    split: str
    config_hash: str
    predicted_label: Optional[str]
    ground_truth: str
    is_correct: Optional[bool]
    action: str  # "STOP", "PARTIAL", "CONFLICTED", "ABSTAIN"
    rounds_executed: int
    confidence: float
    total_claims: int
    supported_claims: int
    conflicted_claims: int
    missing_claims: int
    latency_seconds: float
```
