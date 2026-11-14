# MedRAG Benchmark Evaluation Guide

This guide describes how to run reproducible, claim-level medical Question Answering benchmark evaluations using MedRAG's unified evaluation framework.

---

## 1. Framework Architecture

The framework decouples benchmark dataset formats from MedRAG's core iterative retrieval and verification engine:

```
                           BenchmarkRegistry
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
           SyntheticAdapter  MedQAAdapter  PubMedQAAdapter
                  │               │               │
                  └───────────────┬───────────────┘
                                  ▼
                         BenchmarkExample
                                  │
                                  ▼
                          EvaluationRunner
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
       MedRagPipeline                        PredictionRecorder
(Retrieve ↔ Verify ↔ Stop)                   (Streaming JSONL)
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                           PredictionRecord
                                  │
                                  ▼
                            MetricRegistry
               (Accuracy, Selective Acc, Abstention, AURC)
```

### Core Components
- **`DatasetAdapter`** (`medrag/benchmarks/adapter.py`): Common interface providing split loading, example normalization into `BenchmarkExample`, input prompt construction, answer normalization, and evaluation.
- **`BenchmarkRegistry`** (`medrag/benchmarks/registry.py`): Central registry for discovering and instantiating benchmark adapters.
- **`EvaluationRunner`** (`medrag/benchmarks/runner.py`): Coordinates data loading, pipeline execution, resume checks, streaming prediction recording, and metric calculations.
- **`PredictionRecorder`** (`medrag/benchmarks/recorder.py`): Atomic streaming writer for predictions and retrieval traces, supporting resume and crash recovery.
- **`MetricRegistry`** (`medrag/benchmarks/metrics.py`): Computes scientific metrics including Raw Accuracy, Selective Accuracy, Abstention Rate, Coverage, and Area Under the Risk-Coverage Curve (AURC).
- **`ExperimentConfig`** (`medrag/config_schema.py`): Pydantic configuration schema supporting parameter freezing and SHA-256 configuration hashing.
- **`ExperimentManifest`** (`medrag/benchmarks/manifest.py`): Immutable provenance manifest capturing git commit SHA, library versions, prompt template hashes, and random seeds.

---

## 2. Quick Smoke Testing (Before Full Experiments)

Always run a small smoke test before launching long-running benchmark evaluations. The smoke test verifies end-to-end integration across embedding models, BM25 indexing, answer generation, claim decomposition, NLI cross-encoder verification, and artifact writing without requiring external API credits.

### Automated Smoke Test Command

```bash
python eval.py --smoke-test
```

This runs 3 synthetic clinical questions through the full pipeline using the local offline `mock` LLM backend and cached neural models (`sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/nli-MiniLM2-L6-H768`).

### Expected Smoke Test Output

```text
======================================================================
  MedRAG Benchmark Evaluation Engine
  Experiment ID : smoke_test_20261114_192725
  Dataset       : synthetic (test)
  Limit         : 3
  Backend       : mock (mock-clinical-llm)
  Output Dir    : runs\smoke_test_20261114_192725
======================================================================

[MedRag] Initialising pipeline components...
  Indexing 3 documents with BM25 + Dense...
  Indexing complete.
[MedRag] Pipeline ready.

...
======================================================================
  EVALUATION RESULTS — smoke_test_20261114_192725
======================================================================
  Total Examples      : 3
  Evaluated (Answered): 3
  Abstained           : 0
  Raw Accuracy        : 100.00%
  Selective Accuracy  : 100.00%
  Abstention Rate     : 0.00%
  Coverage            : 100.00%
  AURC Risk-Coverage  : 0.0000
  Mean Latency        : 38.18s
  Mean Search Rounds  : 3.00
======================================================================
  Run artifacts saved to: runs\smoke_test_20261114_192725
```

---

## 3. Supported Benchmarks

List available benchmarks:
```bash
python eval.py --list-benchmarks
```

1. **`synthetic`** (Version `1.0.0-fixtures`):
   - 5 curated multi-choice clinical vignettes across endocrinology, cardiology, hematology, and infectious disease.
   - Includes paired ground-truth literature passages for fully self-contained offline testing.
2. **`medqa`** (Version `usmle-4opt-v1`):
   - USMLE Step 1, 2 CK, and 3 clinical multiple-choice questions (options A, B, C, D).
   - Supported splits: `dev`, `val`, `test`, `train`.
   - Ingests from `data/medqa/test.jsonl` or uses built-in representative USMLE fixtures if data files are not yet downloaded.
3. **`pubmedqa`** (Version `pqa-labeled-v1`):
   - Biomedical question answering over PubMed abstracts with categorical conclusions (`yes`, `no`, `maybe`).
   - Supported splits: `dev`, `val`, `test`, `train`.
   - Ingests from `data/pubmedqa/ori_pqal.json` or uses built-in representative fixtures.

---

## 4. Running Benchmark Evaluations

### A. Quick Evaluation with Limit (Smoke Run on Real Dataset)

```bash
# Evaluate 5 examples from MedQA test split with mock backend
python eval.py --dataset medqa --split test --limit 5 --backend mock --experiment-id medqa_smoke_01

# Evaluate 5 examples from PubMedQA test split with mock backend
python eval.py --dataset pubmedqa --split test --limit 5 --backend mock --experiment-id pubmedqa_smoke_01
```

### B. Full Benchmark Run with Groq LLM Backend

```bash
# Set your API key
export GROQ_API_KEY="gsk_..."

# Run full MedQA validation split for prompt/threshold tuning
python eval.py \
  --dataset medqa \
  --split val \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --experiment-id medqa_val_tuning_01 \
  --max-rounds 3 \
  --support-threshold 0.70 \
  --conflict-threshold 0.35 \
  --high-risk-threshold 0.75
```

### C. Final Test Evaluation (Frozen Configuration)

> [!IMPORTANT]
> Tune prompts, thresholds, and hyperparameters **only on the validation split (`--split val`)**. Freeze the configuration and record the SHA-256 configuration hash before running the final test set (`--split test`).

```bash
python eval.py \
  --dataset medqa \
  --split test \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --experiment-id medqa_test_final_frozen
```

---

## 5. Resume and Failure Recovery

If a benchmark evaluation is interrupted by network failure, power outage, or rate limits, the runner can resume automatically without losing progress or re-evaluating previously processed examples.

```bash
# Resume an interrupted experiment (default behavior)
python eval.py --dataset medqa --split test --experiment-id medqa_test_final_frozen

# Force re-evaluation from scratch (disables resume)
python eval.py --dataset medqa --split test --experiment-id medqa_test_final_frozen --no-resume
```

---

## 6. Experiment Output Directory Layout

Every experiment saves structured artifacts under `runs/<experiment_id>/`:

```
runs/
  <experiment_id>/
    ├── config.yaml              # Frozen experiment configuration (no API keys)
    ├── manifest.json            # Provenance: git commit, library versions, prompt hash
    ├── predictions.jsonl        # Claim-level prediction records (streaming line-by-line)
    ├── retrieval_traces.jsonl   # Round-by-round retrieval queries, doc counts, and scores
    ├── metrics.json             # Aggregate scientific metrics (Accuracy, AURC, latency)
    └── logs/
        └── run.log              # Sanitized execution log (API keys and PHI redacted)
```

### Example `predictions.jsonl` Record
```json
{
  "example_id": "syn_001",
  "dataset": "synthetic",
  "split": "test",
  "question": "Which of the following antidiabetic agents is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min/1.73 m2)?",
  "gold_answer": "A",
  "predicted_answer": "A",
  "raw_model_response": "...",
  "normalized_answer": "A",
  "is_correct": true,
  "abstained": false,
  "retrieved_doc_ids": ["doc_syn_001_0", "doc_syn_003_0"],
  "retrieval_scores": {"doc_syn_001_0": 0.0164, "doc_syn_003_0": 0.0161},
  "generated_claims": [
    "Metformin is an oral biguanide widely prescribed as first-line therapy for type 2 diabetes.",
    "Metformin is contraindicated in patients with advanced renal failure (eGFR < 30 mL/min/1.73 m2)."
  ],
  "supporting_evidence_ids": ["doc_syn_001_0"],
  "contradicting_evidence_ids": [],
  "rounds": 1,
  "decision_action": "STOP",
  "decision_reason": "All 2 claims meet configured evidence criteria.",
  "heuristic_evidence_score": 0.8540,
  "latency_seconds": 12.4,
  "token_usage": {"prompt_tokens": 43, "completion_tokens": 590, "total_tokens": 633},
  "metadata": {"config_hash": "afe09e352ce01e6d", "llm_backend": "mock"}
}
```

---

## 7. Metrics Definition

- **Raw Accuracy**: $\frac{\text{Correct}}{\text{Total}}$
- **Selective Accuracy**: $\frac{\text{Correct}}{\text{Evaluated}}$ (accuracy on questions the system decided to answer)
- **Abstention Rate**: $\frac{\text{Abstained}}{\text{Total}}$
- **Coverage**: $1 - \text{Abstention Rate}$
- **Area Under the Risk-Coverage Curve (AURC)**: Integrates selective prediction risk (error rate on answered questions) across varying coverage levels determined by confidence thresholds. Lower values indicate better risk-calibrated performance.
