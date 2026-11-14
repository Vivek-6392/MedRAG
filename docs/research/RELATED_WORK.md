# Related Work: Clinical Information Retrieval, Verification, and Selective RAG

This document surveys literature across medical QA, hybrid retrieval, self-corrective RAG, claim decomposition, biomedical NLI, iterative retrieval, and selective prediction, situating MedRAG within the field.

---

## 1. Domain-Specific Taxonomy of Related Work

### A. Medical Retrieval-Augmented Generation
Early medical RAG frameworks applied standard dense or sparse retrieval directly to clinical multiple-choice benchmarks:
- **MedQA / USMLE Benchmarking** (Jin et al., 2021): Introduced the United States Medical Licensing Examination (USMLE) dataset in English, Simplified Chinese, and Traditional Chinese, establishing textbook retrieval baselines.
- **PubMedQA** (Jin et al., 2019): Formulated biomedical QA over PubMed abstracts with yes/no/maybe decisions, separating reasoning from external search.
- **MedRAG Benchmark Studies** (Xiong et al., 2024): Evaluated diverse retrieval backends across MedQA, PubMedQA, and MMLU-Clinical, demonstrating that textbook and PubMed retrieval substantially improves raw LLM performance.
- *Limitation*: Existing medical RAG systems operate primarily in a single-pass, greedy generation mode without explicit claim-level verification or risk-conditioned abstention.

### B. Hybrid Retrieval & Reranking
- **Reciprocal Rank Fusion (RRF)** (Cormack, Clarke & Buettcher, 2009): Proved that ranking fusion based on reciprocal ranks ($1 / (k + r)$) outperforms complex score normalization techniques without hyperparameter tuning.
- **Dense-Sparse Hybrid Search** (Karpukhin et al., 2020; Gao et al., 2021): Demonstrated that dense bi-encoders (e.g. DPR, Contriever) and lexical BM25 capture complementary retrieval signals.
- **Neural Cross-Encoder Reranking** (Nogueira & Cho, 2019): Showed that cross-encoders jointly attending to query and candidate tokens achieve superior ranking accuracy over bi-encoders, motivating two-stage retrieval architectures.

### C. Self-Corrective and Verifying RAG Frameworks
- **Self-RAG** (Asai et al., 2023): Introduced special reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) trained into language models to control retrieval and evaluate passage relevance.
- **CRAG (Corrective RAG)** (Yan et al., 2024): Introduced a lightweight retrieval evaluator to classify retrieval outcomes into Correct, Ambiguous, and Incorrect, triggering web search fallback when document relevance is low.
- *Distinction from MedRAG*: Self-RAG and CRAG evaluate document-level relevance rather than fine-grained atomic claim entailment; they lack asymmetric contradiction penalties, source authority weighting, and clinical risk tiering.

### D. Claim-Level Factuality and Decomposition
- **FActScore** (Min et al., 2023): Operationalized factual precision by decomposing generated biographies into atomic facts and verifying each against Wikipedia using retrieval and prompting.
- **FacTool** (Chern et al., 2023): Extended claim decomposition across multiple domains (code, math, knowledge QA) using search engines.
- **SAFE (Search-Augmented Factuality Evaluator)** (Wei et al., 2024): Used multi-step search queries to verify individual atomic claims.
- *Distinction from MedRAG*: These methods are post-hoc evaluation metrics rather than runtime iterative controllers. They treat factuality as binary without modeling domain-specific clinical risk or temporal evidence decay.

### E. Biomedical Natural Language Inference
- **BioBERT** (Lee et al., 2020) and **PubMedBERT** (Gu et al., 2021): Demonstrated that pretraining language models from scratch or fine-tuning on PubMed Central abstracts significantly improves biomedical NLP performance.
- **MedNLI** (Romanov & Shivade, 2018): Developed the canonical clinical NLI dataset annotated by medical experts based on MIMIC-III clinical notes.
- *Role in MedRAG*: MedRAG uses cross-encoder sequence classification (`nli-MiniLM2-L6-H768`) trained on entailment, neutral, and contradiction, but departs from standard NLI by independently tracking contradiction and support rather than collapsing them to a single scalar.

### F. Iterative and Multi-Hop Retrieval
- **IRCoT (Interleaving Retrieval with Chain-of-Thought)** (Trivedi et al., 2022): Interleaved reasoning steps with retrieval queries for multi-hop open-domain QA.
- **ITER-RETGEN** (Shao et al., 2023): Iteratively interleaved retrieval and generation, demonstrating that generated text helps retrieve better passages and vice-versa.
- *Distinction from MedRAG*: IRCoT and ITER-RETGEN iterate on free-text model thoughts. MedRAG specifically directs gap queries at unverified, missing, or conflicted atomic claims identified by the NLI verifier.

### G. Selective Prediction, Calibration, and Abstention
- **Selective Classification** (Geifman & El-Yaniv, 2017): Established mathematical foundations for selective prediction, optimizing coverage under a bounded risk constraint $\mathcal{R}(f, g) \le \epsilon^*$.
- **Temperature Scaling & Calibration** (Guo et al., 2017): Showed that post-hoc temperature scaling effectively calibrates neural network logits without degrading classification accuracy.
- **Selective Answering in Medical QA** (Feng et al., 2024): Demonstrated that selective abstention prevents clinical errors in medical LLMs.
- *Role in MedRAG*: MedRAG formalizes a 5-way decision policy (`STOP`, `SEARCH`, `PARTIAL`, `CONFLICTED`, `ABSTAIN`) conditioned on clinical risk and empirical evidence thresholds, evaluating performance via risk-coverage curves and AURC.

---

## 2. Comprehensive Comparative Matrix

| System / Paper | Retrieval Paradigm | Verification Granularity | Contradiction Handling | Iteration Mechanism | Abstention Capability | Clinical Risk Conditioning | Primary Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Medical RAG** (Xiong et al., 2024) | Dense or BM25 (Single-pass) | None (Unverified generation) | None (Ignored) | None (Single pass) | None (Forced 100% coverage) | None | Benchmark Accuracy |
| **Self-RAG** (Asai et al., 2023) | Dense (Retrieval on-demand) | Document-level (Special tokens) | Implicit | None | Implicit via generation | None | Open Domain QA |
| **CRAG** (Yan et al., 2024) | Dense + Web fallback | Passage-level relevance | None | Single fallback step | None | None | General Web RAG |
| **FActScore** (Min et al., 2023) | Dense / Google Search | Atomic claim decomposition | Binary (Supported / Unsupported) | None (Post-hoc evaluation) | N/A (Metric only) | None | Factuality Metric |
| **IRCoT** (Trivedi et al., 2022) | Dense / BM25 | None | None | Reasoning Chain (CoT) | None | None | Multi-Hop Reasoning |
| **PubMedBERT NLI** (Gu et al., 2021) | None (Classifier only) | Sentence-pair | Symmetric NLI loss | None | None | None | NLI Classification |
| **Selective Medical QA** (Feng et al., 2024) | None (Closed-book LLM) | Output probability / Perplexity | None | None | Binary Abstain | Uncalibrated heuristic | Selective Answering |
| **Proposed MedRAG** | **Hybrid (BM25 + Dense RRF) + Cross-Encoder Rerank** | **Atomic Claim-Level (Multi-Granularity Premises)** | **Asymmetric Contradiction Isolation & Penalty** | **Claim-Specific Targeted Gap Queries ($T \le 3$)** | **5-Way Policy (`STOP`, `SEARCH`, `PARTIAL`, `CONFLICT`, `ABSTAIN`)** | **Dynamic Thresholds $\tau(\rho_x)$ & Bounded Risk Optimization** | **Evidence-Attributed Medical RAG** |

---

## 3. Critical Differentiation Summary

MedRAG does not claim novelty in the individual mathematical primitives:
- BM25 and Dense vector embeddings are standard IR techniques.
- Reciprocal Rank Fusion (RRF) and Cross-Encoder rerankers are established information retrieval tools.
- Atomic claim extraction and 3-class NLI are well-known NLP formulations.

**The genuine scientific contribution of MedRAG lies in the holistic, source-aware and risk-calibrated verification architecture**:
1. Formulating medical evidence verification as **asymmetric, independent tracking of support and contradiction**, ensuring that literature disagreements cannot be hidden by sheer document volume.
2. Directing iterative retrieval loops **specifically at unresolved and conflicted claim gaps** rather than unconstrained reasoning steps.
3. Integrating source authority grades and temporal recency decay into claim confidence scoring.
4. Conditioning next-action stopping and selective abstention decisions on **explicit clinical risk tiering**, enabling optimization on the selective risk-coverage Pareto frontier.
