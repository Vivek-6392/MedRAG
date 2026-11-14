# MedRAG Open-Source Academic Release Checklist

This document serves as the pre-release and reproduction verification checklist for MedRAG prior to public release and academic manuscript submission.

---

## 1. Codebase & Integrity Verification

- [x] **Supported Python Versions:** Tested and compatible with Python 3.10 through 3.13.
- [x] **Dependencies Segregated & Pinned:**
  - `requirements.txt`: Core runtime dependencies pinned to compatible versions.
  - `requirements-dev.txt`: Test harness (`pytest`), linters (`ruff`), and coverage tools.
  - `requirements-optional.txt`: UI framework (`streamlit`), GPU acceleration packages.
  - `pyproject.toml`: Modern packaging metadata and project specifications.
- [x] **Linting & Code Formatting:**
  - Ran `ruff check .` — Passed with 0 errors.
- [x] **Automated Test Suite:**
  - Ran `pytest tests/` — All 59 tests passing across all test modules.
- [x] **Environment Isolation:**
  - Tested clean installation inside fresh virtual environment (`venv`).

---

## 2. Security, Privacy & Secret Protection

- [x] **Zero Credentials Committed:**
  - Verified `.env` is listed in `.gitignore` and not tracked in git tree.
  - Scanned for accidental hardcoded API tokens (`sk-`, `ghp_`, `anthropic`). Zero detected.
  - Validated `.env.example` contains only template placeholders.
- [x] **Protected Health Information (PHI):**
  - Zero patient records or private health data present in repository.
  - All testbed cases utilize synthetic or publicly available benchmark items.

---

## 3. Data Governance & Copyright Compliance

- [x] **No Copyrighted Corpora Committed:**
  - Verified no copyrighted medical textbook PDFs or epubs are tracked in git.
  - Verified PubMed raw bulk dumps are excluded via `.gitignore`.
- [x] **Environment Variable Pathways:**
  - Evaluated dataset path resolution via `MEDQA_DATA_DIR`, `MEDQA_CORPUS_DIR`, `PUBMEDQA_DATA_DIR`, `PUBMED_CORPUS_DIR`.
- [x] **Dataset Documentation:**
  - Verified `DATASETS.md` documents dataset versions, download instructions, and SHA256 checksums.

---

## 4. Research Artifacts & Publication Materials

- [x] **Academic Manuscript:**
  - `paper/manuscript.md` complete with all 16 academic sections.
  - Mathematical formalization matches implemented code.
  - Zero fabricated results: clear distinction between smoke runs and pending cluster runs.
- [x] **Publication Tables (`paper/tables/`):**
  - Generated all 7 tables in both Markdown and JSON:
    1. `medqa_baselines.md` & `.json`
    2. `pubmedqa_baselines.md` & `.json`
    3. `retrieval_metrics.md` & `.json`
    4. `claim_verification.md` & `.json`
    5. `ablation_table.md` & `.json`
    6. `abstention_table.md` & `.json`
    7. `efficiency_table.md` & `.json`
  - Explicit sample sizes (N) and 95% Wilson / bootstrap confidence intervals.
- [x] **Publication Figures (`paper/figures/`):**
  - All 9 publication figures available in both vector PDF and 300 DPI PNG formats.
- [x] **Formal Error Analysis:**
  - `docs/research/ERROR_ANALYSIS.md` complete covering 10 clinical failure modes with case studies and anonymized IDs.

---

## 5. Repository Documentation & Presentation

- [x] **README.md:** Comprehensive overhaul covering all 15 required sections with Mermaid architecture diagrams.
- [x] **Licensing:** Permissive MIT License established in `LICENSE`.
- [x] **Citation Metadata:** `CITATION.cff` formatted with authors, date, and repository metadata.
- [x] **Contributing & Community Guidelines:** Established in `CONTRIBUTING.md` and `SECURITY.md`.
- [x] **Model Cards:** Documented in `MODEL_CARDS.md` for bi-encoder, cross-encoder, and NLI models.
- [x] **Reproducibility Guide:** Established in `docs/REPRODUCIBILITY.md`.

---

## 6. Execution Command Smoke Check

Verify that core commands execute cleanly:

```bash
# 1. Run unit test suite
pytest tests/

# 2. Check code quality
ruff check .

# 3. Validate dataset directory mapping
python eval.py --validate-data

# 4. Run baseline smoke evaluation
python eval.py --study baselines --config configs/baseline_experiments.yaml --run-dir runs/smoke_check_baseline

# 5. Run ablation smoke evaluation
python eval.py --study ablations --config configs/ablation_experiments.yaml --run-dir runs/smoke_check_ablation
```

*Release sign-off complete.*
