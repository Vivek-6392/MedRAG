# Contributing to MedRAG

Thank you for your interest in contributing to MedRAG. We welcome contributions that improve reproducibility, add validated benchmark adapters, or enhance evidence verification methodologies.

---

## 1. Development Principles

1. **Academic Reproducibility**: Every algorithm, evaluation metric, and ablation must be deterministic and fully reproducible.
2. **Clinical Safety & Truthfulness**: Never make claims of clinical readiness or safety based solely on multiple-choice accuracy or benchmark test results.
3. **No Proprietary Data in Commits**: Do not commit private patient data, copyrighted medical textbooks, or credentials.
4. **Resilient Testing**: All unit and integration tests must run without requiring active network access or paid external API keys. Use the `mock` LLM backend and local test fixtures for automated testing.

---

## 2. Setting Up a Development Environment

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- Git

### Linux / macOS
```bash
git clone https://github.com/Vivek-6392/MedRAG.git
cd MedRAG
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt -r requirements-optional.txt
```

### Windows (PowerShell)
```powershell
git clone https://github.com/Vivek-6392/MedRAG.git
cd MedRAG
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt -r requirements-optional.txt
```

---

## 3. Code Standards & Quality Checks

Before submitting a pull request, run the following verification checks:

1. **Linting Check**:
   ```bash
   ruff check medrag/ tests/ eval.py
   ```
2. **Configuration Validation**:
   ```bash
   python -c "import glob; from medrag.benchmarks import ExperimentConfig; [ExperimentConfig.from_file(f) for f in glob.glob('configs/*.yaml')]; print('Configs valid!')"
   ```
3. **Benchmark Adapter Registration**:
   ```bash
   python eval.py --list-benchmarks
   ```
4. **Test Suite**:
   ```bash
   python -m pytest tests/ -v
   ```

---

## 4. Submitting Pull Requests

1. Create a feature branch: `git checkout -b feature/your-contribution`.
2. Commit your changes with clear, descriptive commit messages.
3. Ensure all tests pass in local offline mode (`MEDRAG_LLM_BACKEND=mock`).
4. Submit your pull request to the `main` branch.
