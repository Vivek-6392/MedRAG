# Dataset Provenance, Preparation & Licensing Guide

This document details the acquisition, directory structure, integrity verification, and chunking procedures for all benchmarks supported by MedRAG.

---

## 1. Supported Benchmarks Summary

| Benchmark | Subset / Task | Official Split | Official Source | License / Terms |
| :--- | :--- | :--- | :--- | :--- |
| **MedQA** | USMLE English (4-Option) | `train`, `dev`, `test` | [MedQA GitHub](https://github.com/jind11/MedQA) | MIT (Code/Adapter); Textbooks subject to original publisher rights |
| **PubMedQA** | PQA-L (Expert-Labeled) | `train`, `dev`, `test` | [PubMedQA Site](https://pubmedqa.github.io/) | MIT License |
| **Synthetic** | Pipeline Smoke Fixtures | `train`, `dev`, `test` | Internal repository fixtures | MIT License |

---

## 2. Directory Layout & Environment Variables

MedRAG resolves dataset and corpus paths from environment variables or standard project directories:

```text
MedRAG/
├── data/
│   ├── medqa/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   ├── test.jsonl
│   │   └── textbooks/          # Directory of .txt medical textbooks
│   │       ├── Anatomy.txt
│   │       ├── Pathology.txt
│   │       └── ...
│   └── pubmedqa/
│       ├── ori_pqal.json       # Expert-labeled PQA-L subset
│       ├── test_ground_truth.json
│       └── corpus/             # Optional background literature
```

### Environment Variables
Configure these in your `.env` file:
```bash
MEDQA_DATA_DIR=data/medqa
MEDQA_CORPUS_DIR=data/medqa/textbooks
PUBMEDQA_DATA_DIR=data/pubmedqa
PUBMED_CORPUS_DIR=data/pubmedqa/corpus
```

---

## 3. MedQA Preparation & Deterministic Chunking

### A. Download Questions
Download the English 4-option questions from the official [MedQA repository](https://github.com/jind11/MedQA):
Place `train.jsonl`, `dev.jsonl`, and `test.jsonl` into `data/medqa/`.

### B. Textbook Corpus Preparation
Medical textbooks provided by the MedQA authors must be stored as plain-text `.txt` files in `data/medqa/textbooks/`.

Run deterministic sentence-boundary chunking:
```bash
python eval.py --dataset medqa --corpus-dir data/medqa/textbooks --chunk-size 800 --chunk-overlap 150
```

This procedure:
1. Segments text at sentence boundaries into 800-character chunks with 150-character overlaps.
2. Embeds document provenance metadata (`doc_id`, `title`, `chunk_id`, `year`, `source`).
3. Computes a SHA-256 digest across the entire corpus for version tracking (`medqa-textbook-v1`).
4. Caches processed chunks into `data/medqa/cache/corpus_800_150.jsonl`.

---

## 4. PubMedQA Preparation

### A. Download PQA-L
Download the expert-labeled subset (`ori_pqal.json` or split files) from [PubMedQA](https://pubmedqa.github.io/).
Place the file into `data/pubmedqa/ori_pqal.json`.

### B. Evaluation Protocols
1. **`pubmedqa_provided_context`**: Uses the benchmark's bundled scientific abstract passages. No external corpus indexing is required.
2. **`pubmedqa_retrieval`**: Question-only retrieval across external biomedical abstracts. Indexes background literature passages.

---

## 5. Integrity Verification & Guardrails

- **Split Preservation**: Never mix official `dev` and `test` splits. Tune decision thresholds strictly on validation data.
- **Leakage Prevention**: MedRAG benchmark adapters verify that gold answer choices and labels are never passed in model prompt inputs.
- **Subset Isolation**: Artificial or machine-generated subsets (`PQA-A` and `PQA-U`) are filtered out by the adapter to prevent contaminating expert-labeled evaluations.
