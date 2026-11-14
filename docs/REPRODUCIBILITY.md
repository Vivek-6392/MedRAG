# MedRAG Academic Reproduction Playbook

This document provides exact, end-to-end instructions for independent researchers to reproduce the MedRAG software environment, benchmark evaluations, statistical significance tests, and publication figures without access to private machine credentials.

---

## 1. System Requirements

- **Operating System**: Linux (Ubuntu 20.04/22.04/24.04), macOS, or Windows 10/11
- **Python**: Version 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- **Memory**: Minimum 8 GB RAM (16 GB recommended for local dense embedding caches)
- **Disk Space**: ~2 GB for dependencies and model weights; ~10 GB if indexing full textbook corpora
- **GPU (Optional)**: CUDA-compatible GPU accelerates dense embeddings and cross-encoders, but CPU execution is fully supported.

---

## 2. Environment Setup & Clean Installation

### Option A: Using Standard Python Virtual Environment

#### On Linux / macOS:
```bash
# 1. Clone repository
git clone https://github.com/Vivek-6392/MedRAG.git
cd MedRAG

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-optional.txt
```

#### On Windows (PowerShell):
```powershell
# 1. Clone repository
git clone https://github.com/Vivek-6392/MedRAG.git
cd MedRAG

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-optional.txt
```

### Option B: Using Conda
```bash
conda create -n medrag python=3.11 -y
conda activate medrag
pip install -r requirements.txt -r requirements-dev.txt -r requirements-optional.txt
```

---

## 3. Immediate Verification (Zero Credentials Required)

MedRAG includes offline synthetic fixtures and a deterministic mock LLM backend to verify the complete software stack without paid API keys:

```bash
# 1. Run unit, integration, and benchmark tests (59 tests)
python -m pytest tests/ -v

# 2. Run CLI smoke test
python eval.py --smoke-test

# 3. List registered benchmark adapters
python eval.py --list-benchmarks
```

---

## 4. API Key Configuration

To run experiments with live frontier models:
1. Copy the template:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` to supply your provider keys:
   ```env
   LLM_BACKEND=groq
   GROQ_API_KEY=your_actual_groq_key
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

---

## 5. Reproducing Baseline Experiments

To run the study comparing all 8 canonical configurations:
1. `closed_book`
2. `single_pass_dense_rag`
3. `single_pass_bm25_rag`
4. `hybrid_rag`
5. `hybrid_reranked_rag`
6. `claim_verified_rag`
7. `iterative_rag`
8. `full_medrag`

Run using the explicit YAML configuration:
```bash
python eval.py --config configs/baseline_experiments.yaml --study baselines
```

Artifacts will be written to `runs/baselines_study/`:
- `comparison_table.md` (Markdown summary table)
- `comparison_table.json` (Structured JSON table)
- `per_example_comparison.csv` (Per-question predictions and correctness)
- `statistical_report.md` (Wilson CIs, McNemar's tests, paired bootstrap deltas)

---

## 6. Reproducing Component Ablations

To isolate the independent contributions of individual modules:
```bash
python eval.py --config configs/ablation_experiments.yaml --study ablations
```

Evaluates 8 leave-one-out ablations:
- `ablation_no_hybrid`
- `ablation_no_reranking`
- `ablation_no_claim_decomposition`
- `ablation_no_nli_verification`
- `ablation_no_evidence_state`
- `ablation_no_iterative`
- `ablation_no_risk_policy`
- `ablation_no_abstention`

---

## 7. Reproducing MedQA (USMLE 4-Option)

1. Place official MedQA question files into `data/medqa/` (`train.jsonl`, `dev.jsonl`, `test.jsonl`).
2. Place medical textbook files into `data/medqa/textbooks/*.txt`.
3. Execute evaluation:
   ```bash
   python eval.py --config configs/medqa_eval.yaml
   ```

---

## 8. Reproducing PubMedQA Evaluations

### Protocol 1: Provided Reference Abstract (`pubmedqa_provided_context`)
```bash
python eval.py --config configs/pubmedqa_provided_context.yaml
```

### Protocol 2: Question-Only External Retrieval (`pubmedqa_retrieval`)
```bash
python eval.py --config configs/pubmedqa_retrieval.yaml
```

---

## 9. Regenerating Publication Figures

To generate all 9 publication figures (PNG @ 300 DPI and vector PDF) from existing saved runs:
```bash
python scripts/generate_publication_figures.py --runs-dir runs --output-dir publication/figures
```

Output figures in `publication/figures/`:
1. `baseline_accuracy_comparison.png` / `.pdf`
2. `retrieval_recall_at_k.png` / `.pdf`
3. `claim_verification_performance.png` / `.pdf`
4. `confusion_matrix_*.png` / `.pdf`
5. `ablation_delta_accuracy.png` / `.pdf`
6. `risk_coverage_curve.png` / `.pdf`
7. `reliability_diagram.png` / `.pdf`
8. `latency_vs_accuracy.png` / `.pdf`
9. `retrieval_round_distribution.png` / `.pdf`

---

## 10. Hardware & Provenance Verification

Every run records an immutable `manifest.json` file containing:
- Exact Git commit SHA
- Python and library versions (`torch`, `transformers`, `sentence-transformers`, `pydantic`)
- Hardware platform, CPU cores, RAM, and GPU device name
- SHA-256 prompt template hashes and configuration digests
- Random seeds and threshold parameters
