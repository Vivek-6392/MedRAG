# Security and Privacy Policy

## 1. Scope & Research Objective
MedRAG is an academic research system designed to evaluate retrieval, natural language inference (NLI), claim-level evidence verification, and selective abstention.

> **CRITICAL DISCLAIMER**: MedRAG is NOT a medical device, is NOT HIPAA-compliant out of the box, and is NOT certified for clinical diagnosis or direct patient care. Real Protected Health Information (PHI) must NEVER be transmitted to external cloud LLM APIs without compliant Business Associate Agreements (BAA) and certified technical safeguards.

---

## 2. Security Safeguards

### A. API Key and Credential Management
- Provider resolution is centralized in `medrag.provider.ProviderResolver`.
- API keys are loaded strictly from environment variables (`.env`) or local session configurations.
- Credentials, bearer tokens, and key prefixes (`sk-`, `gsk_`) are filtered by `RedactionFilter` in `medrag.logger` and never persisted to experiment manifests, logs, or prediction artifacts.

### B. Prompt Injection Defense (Untrusted Evidence)
Retrieved documents from search indexes or external corpora represent untrusted data:
- Excerpts are isolated within `<untrusted_evidence>` XML boundaries in model prompts.
- Prompts instruct generative models to treat evidence passages strictly as reference data and explicitly ignore any directives, instructions, or format-override commands embedded inside retrieved text.

### C. Logging & Protected Health Information (PHI)
- `RedactionFilter` monitors logging channels and masks phone numbers, email addresses, Social Security Numbers (SSNs), Medical Record Numbers (MRNs), and bearer tokens.
- Query logging can be toggled or routed to local files using the `MEDRAG_LOG_LEVEL` environment variable.

### D. External Cloud LLM Data Transmission
When using cloud backends (`openai`, `groq`), query text and retrieved evidence passages are transmitted over TLS to third-party endpoints.
- Researchers conducting investigations with sensitive data must use local offline backends (e.g. `ollama`, local vLLM, or `mock`) to prevent data exfiltration.

### E. File Upload & Input Validation
- File upload components in the Streamlit application (`app.py`) restrict uploads to `.json` and `.jsonl` extensions.
- File size is capped at 50 MB to prevent denial-of-service memory exhaustion.
- Document inputs are validated against `medrag.document_validator.validate_corpus` with UTF-8 replacement handling.

---

## 3. Vulnerability Reporting
If you identify a security issue or credential exposure risk, please open a private security advisory on GitHub or contact the repository maintainers.
