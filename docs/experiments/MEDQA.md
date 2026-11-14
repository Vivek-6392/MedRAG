# MedQA (USMLE 4-Option Subset) Research Benchmark

**Dataset**: MedQA — English USMLE 4-Option Subset  
**Official Source**: [https://github.com/jind11/MedQA](https://github.com/jind11/MedQA)  
**Task**: Multiple-choice clinical question answering (USMLE Steps 1, 2 CK, and 3).  

---

## 1. Overview & Research Objectives

The MedQA benchmark evaluates clinical reasoning and factual knowledge retrieval across realistic medical licensing vignettes. Each question consists of a patient clinical vignette followed by four candidate answer choices (`A`, `B`, `C`, `D`), of which exactly one is correct.

The MedRAG evaluation framework compares five distinct scientific protocols under identical underlying model configurations:

1. **`CLOSED_BOOK`**: Direct zero-shot LLM inference using parametric knowledge without external retrieval.
2. **`TEXTBOOK_RAG`**: Single-round dense retrieval from medical textbooks followed by evidence-grounded answer generation.
3. **`HYBRID_RAG`**: Fused sparse (BM25) and dense retrieval via Reciprocal Rank Fusion (RRF).
4. **`RERANKED_RAG`**: Hybrid retrieval followed by cross-encoder neural reranking.
5. **`MEDRAG_FULL`**: Adaptive evidence-aware loop combining hybrid retrieval, claim decomposition, sentence-level NLI contradiction detection, evidence state tracking, and risk-aware stopping/abstention.

---

## 2. Dataset Specification & Official Splits

MedRAG adheres strictly to official benchmark splits to prevent data leakage and ensure comparability with published medical NLP literature:

| Split | Question Format | Options | Purpose |
| :--- | :--- | :--- | :--- |
| **`train`** | Clinical Vignette | 4 (`A`, `B`, `C`, `D`) | In-context exemplar selection / retrieval corpus calibration |
| **`dev` / `val`** | Clinical Vignette | 4 (`A`, `B`, `C`, `D`) | **Prompt engineering, hyperparameter tuning & threshold calibration** |
| **`test`** | Clinical Vignette | 4 (`A`, `B`, `C`, `D`) | **Final evaluation only (FROZEN CONFIGURATION)** |

> [!IMPORTANT]
> **Split Hygiene**: Hyperparameters (retrieval $k$, NLI support/conflict thresholds, and risk weights) must **only** be tuned on the validation split (`--split val`). The configuration must be frozen (`config_hash` recorded in `manifest.json`) prior to running evaluation on the test split (`--split test`).

---

## 3. Medical Textbook Corpus Preparation

MedQA provides an accompanying corpus of 18 English medical textbooks covering internal medicine, pathology, pharmacology, and pediatrics.

### Preprocessing & Chunking Protocol
The `TextbookCorpusPreparer` utility normalizes textbook documents and produces deterministic, sentence-aligned chunks:
- **Chunk Size**: 800 characters (~150 words).
- **Chunk Overlap**: 150 characters (~30 words).
- **Provenance**: Preserves `{book_title}_{chapter}_{section}_{chunk_id}` in document metadata.
- **Corpus Integrity**: Computes SHA-256 digest across all chunked documents and saves a manifest to verify reproducible retrieval across experiments.

```bash
# Chunk and index textbook corpus directory
python eval.py \
  --dataset medqa \
  --split val \
  --protocol hybrid_rag \
  --corpus-dir data/medqa/textbooks \
  --chunk-size 800 \
  --chunk-overlap 150
```

> [!CAUTION]
> **Data Contamination Safeguard**: Benchmark test questions and answer keys must never be added to the retrieval corpus. Source textbooks must not be committed to Git unless permissible under their respective publisher licenses.

---

## 4. Evaluation Protocols & CLI Commands

### Protocol 1: Closed-Book Direct Generation
Evaluates parametric clinical reasoning without external evidence:
```bash
python eval.py \
  --dataset medqa \
  --split test \
  --protocol closed_book \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --experiment-id medqa_closed_book_test
```

### Protocol 2: Textbook Dense RAG
Single-round dense retrieval using medical textbook embeddings:
```bash
python eval.py \
  --dataset medqa \
  --split test \
  --protocol textbook_rag \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --experiment-id medqa_textbook_rag_test
```

### Protocol 3: Hybrid Retrieval with RRF
Sparse BM25 combined with dense semantic retrieval:
```bash
python eval.py \
  --dataset medqa \
  --split test \
  --protocol hybrid_rag \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --experiment-id medqa_hybrid_rag_test
```

### Protocol 4: Hybrid RRF with Cross-Encoder Reranking
Top retrieved documents reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`:
```bash
python eval.py \
  --dataset medqa \
  --split test \
  --protocol reranked_rag \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --experiment-id medqa_reranked_rag_test
```

### Protocol 5: Full Adaptive MedRAG
Full iterative claim verification with selective abstention:
```bash
python eval.py \
  --dataset medqa \
  --split test \
  --protocol medrag_full \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --max-rounds 3 \
  --support-threshold 0.70 \
  --conflict-threshold 0.35 \
  --high-risk-threshold 0.75 \
  --experiment-id medqa_full_adaptive_test
```

---

## 5. Structured Answer Generation Schema

To eliminate parsing ambiguity, the model is prompted to emit a structured JSON response:

```json
{
  "selected_option": "B",
  "answer_text": "Leucovorin (folinic acid)",
  "explanation": "Leucovorin bypasses dihydrofolate reductase inhibition to rescue healthy cells during methotrexate toxicity.",
  "claims": [
    "Methotrexate competitively inhibits dihydrofolate reductase.",
    "Leucovorin provides reduced folate cofactors directly."
  ],
  "citations": ["[1]", "[3]"],
  "abstain": false
}
```

### Ambiguity Rejection
If the response asserts conflicting options (e.g. *"both A and B are plausible"*), or fails to select an unequivocal option, the adapter rejects the prediction and treats it as unresolved or abstained, rather than silently guessing.

---

## 6. Metrics & Reporting

Every experiment computes:
1. **Raw Accuracy**: $\frac{\text{Correct}}{\text{Total}}$ (counting abstentions as incorrect).
2. **Selective Accuracy**: $\frac{\text{Correct}}{\text{Evaluated}}$ (accuracy on questions the system chose to answer).
3. **Abstention Rate**: $\frac{\text{Abstained}}{\text{Total}}$.
4. **Coverage**: $1 - \text{Abstention Rate}$.
5. **Per-Option Accuracy**: Accuracy for gold answers `A`, `B`, `C`, and `D`.
6. **Macro-F1**: Unweighted harmonic mean of precision and recall across all four options.
7. **Confusion Matrix**: $4 \times 4$ contingency table tracking gold options against predicted options and abstentions.
8. **Area Under the Risk-Coverage Curve (AURC)**: Measures the trade-off between coverage and selective error rate.
9. **Efficiency**: Average latency per question, average search rounds, and average token consumption.

---

## 7. Artifact Persistence Layout

Results are saved under `runs/<experiment_id>/`:
- `config.yaml`: Frozen configuration parameters.
- `manifest.json`: Full environment provenance (commit SHA, prompt hash, library versions).
- `predictions.jsonl`: Line-by-line prediction records with retrieved doc IDs and claim-level evidence states.
- `retrieval_traces.jsonl`: Round-by-round queries and document counts.
- `metrics.json`: Aggregate evaluation statistics.
- `logs/run.log`: Sanitized run log.

---

## 8. Clinical Research Ethics Notice

> [!WARNING]
> **Clinical Safety Notice**: Benchmark accuracy on multiple-choice licensing examinations (USMLE) does not constitute proof of clinical safety or diagnostic efficacy in real-world clinical care. Benchmark evaluations must be interpreted as measurements of textual factual recall and semantic reasoning, not clinical competency.
