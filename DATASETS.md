# Dataset Provenance, Automated Setup & Licensing Guide

This document details the acquisition, directory layout, automated setup scripts, integrity verification, and chunking procedures for all benchmarks supported by MedRAG.

---

## 1. Supported Benchmarks Summary

| Benchmark | Subset / Task | Official Split | Official Source | Code License | Underlying Content Terms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MedQA** | USMLE English (4-Option) | `train` (10,178), `dev` (1,272), `test` (1,273) | [jind11/MedQA](https://github.com/jind11/MedQA) | MIT License | Questions from USMLE board exams; Textbooks subject to medical publisher copyright |
| **PubMedQA** | PQA-L (Expert-Labeled) | `test` (500), `dev` (100), `train` (400) | [pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) | MIT License | Scientific abstracts governed by NLM Open Access / Terms of Service |
| **Synthetic** | Diagnostic Smoke Fixtures | `train`, `dev`, `test` | Repository fixtures | MIT License | Open-source synthetic clinical testbeds |

---

## 2. Default Storage: External to Git Repository

To prevent committing benchmark questions, answer keys, or copyrighted textbooks to Git, MedRAG stores datasets **outside the repository by default** under:

```text
Windows: %USERPROFILE%\.medrag\data\
Linux/macOS: ~/.medrag/data/
```

### Layout
```text
~/.medrag/data/
├── medqa/
│   ├── train.jsonl             # 10,178 USMLE English 4-option questions
│   ├── dev.jsonl               # 1,272 dev questions
│   ├── test.jsonl              # 1,273 test questions
│   └── textbooks/              # 18 official English reference medical textbooks (.txt)
│       ├── Anatomy_Gray.txt
│       ├── Pathology_Robbins.txt
│       ├── InternalMed_Harrison.txt
│       └── ... (85.3 MB total)
└── pubmedqa/
    ├── ori_pqal.json           # 1,000 expert-labeled QA pairs
    ├── test_ground_truth.json  # 500 official test labels
    ├── test.json               # 500 test questions + contexts + labels
    ├── dev.json                # 100 dev questions + contexts + labels
    └── train.json              # 400 train questions + contexts + labels
```

---

## 3. Windows PowerShell Step-by-Step Setup

Follow these exact commands in Windows PowerShell:

### Step 1: Set Up MedQA Benchmark

If you downloaded the official `data_clean.zip` from Google Drive (or have extracted it):

```powershell
# Point to your extracted data_clean directory
python scripts/download_medqa.py --source-dir "C:\Users\$env:USERNAME\Downloads\data_clean" --force
```

This command automatically:
1. Detects `phrases_no_exclude_{train,dev,test}.jsonl` under `questions/US/4_options/`.
2. Standardizes them as `train.jsonl`, `dev.jsonl`, and `test.jsonl` in `$HOME\.medrag\data\medqa\`.
3. Copies all 18 official English reference medical textbooks to `$HOME\.medrag\data\medqa\textbooks\`.
4. Validates question formatting, ensures exactly 4 choices (A, B, C, D), and reports split counts.

### Step 2: Set Up PubMedQA Benchmark

If you cloned or downloaded the official PubMedQA repository (`pubmedqa-master`):

```powershell
# Point to your downloaded pubmedqa repository
python scripts/download_pubmedqa.py --source-dir "C:\Users\$env:USERNAME\Downloads\pubmedqa-master" --force
```

Alternatively, to clone directly from the official repository:
```powershell
python scripts/download_pubmedqa.py --force
```

This command automatically:
1. Extracts `ori_pqal.json` (1,000 expert-labeled instances).
2. Extracts `test_ground_truth.json` (500 test labels).
3. Generates the official splits (`test.json`, `dev.json`, `train.json`) with questions and full context abstracts.
4. Validates record fields and reports label distributions (`yes`, `no`, `maybe`).

### Step 3: Configure Environment Variables

Add the resolved paths to your `.env` file (copied from `.env.example`):

```ini
MEDQA_DATA_DIR=C:\Users\<YourUsername>\.medrag\data\medqa
MEDQA_CORPUS_DIR=C:\Users\<YourUsername>\.medrag\data\medqa\textbooks
PUBMEDQA_DATA_DIR=C:\Users\<YourUsername>\.medrag\data\pubmedqa
PUBMED_CORPUS_DIR=C:\Users\<YourUsername>\.medrag\data\pubmedqa\corpus
```

Or export them in your active PowerShell session:

```powershell
$env:MEDQA_DATA_DIR = "$HOME\.medrag\data\medqa"
$env:MEDQA_CORPUS_DIR = "$HOME\.medrag\data\medqa\textbooks"
$env:PUBMEDQA_DATA_DIR = "$HOME\.medrag\data\pubmedqa"
```

### Step 4: Run Dataset Verification Command

Verify that both datasets are recognized, validated, and loadable by their respective benchmark adapters:

```powershell
python -m medrag.cli verify-datasets
```

Expected output:
```text
==============================================================================
MEDRAG BENCHMARK DATASET & CORPUS VERIFICATION AUDIT
==============================================================================

1. MedQA (USMLE English 4-option) -> [PASS]
   Data Directory:   C:\Users\<User>\.medrag\data\medqa
   Corpus Directory: C:\Users\<User>\.medrag\data\medqa\textbooks
   Splits:
      - TRAIN:  10178 questions | Path: ...\train.jsonl
      - DEV  :   1272 questions | Path: ...\dev.jsonl
      - TEST :   1273 questions | Path: ...\test.jsonl
   Textbooks: 18 files (85.29 MB) -> All 18 official textbooks present
   Adapter Integration: PASS

2. PubMedQA (PQA-L Expert-Labeled) -> [PASS]
   Data Directory: C:\Users\<User>\.medrag\data\pubmedqa
   Splits:
      - TEST :    500 instances | Path: ...\test.json
      - DEV  :    100 instances | Path: ...\dev.json
      - TRAIN:    400 instances | Path: ...\train.json
   Adapter Integration: PASS

------------------------------------------------------------------------------
[AUDIT RESULT] ALL BENCHMARK DATASETS VERIFIED SUCCESSFULLY
==============================================================================
```

---

## 4. Benchmark Adapter Execution

Run benchmark evaluations directly against the verified local datasets:

### MedQA USMLE Evaluation:
```powershell
python eval.py --dataset medqa --config configs/medqa_eval.yaml --run-dir runs/medqa_official_eval
```

### PubMedQA Provided-Context Evaluation:
```powershell
python eval.py --dataset pubmedqa --protocol pubmedqa_provided_context --config configs/pubmedqa_provided_context.yaml
```

### PubMedQA Open Corpus Retrieval Evaluation:
```powershell
python eval.py --dataset pubmedqa --protocol pubmedqa_retrieval --config configs/pubmedqa_retrieval.yaml
```

---

## 5. Licensing & Copyright Disclaimers

### A. MedQA
- **Dataset Code & Processing:** MIT License ([jind11/MedQA](https://github.com/jind11/MedQA)).
- **Exam Questions:** Collected from USMLE Step 1, 2, and 3 examinations.
- **Reference Textbooks:** Copyrighted medical textbooks (Harrison's Internal Medicine, Robbins Pathology, etc.).
- **Notice:** Textbooks and questions are provided strictly for non-commercial academic research and benchmarking under fair use. Do not host, publish, or commit textbook text files to GitHub.

### B. PubMedQA
- **Dataset Code & Annotations:** MIT License ([pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa)).
- **PubMed Abstracts:** National Library of Medicine (NLM) Open Access subset.
- **Terms:** Governed by NLM Terms and Conditions for text mining.

### C. Git Hygiene Notice
The repository's `.gitignore` explicitly excludes:
- `data/`
- `*.jsonl`
- `*.txt` in corpus folders
- `.medrag/`
- `.env`

Never force-add (`git add -f`) datasets or copyrighted textbooks to the git repository.
