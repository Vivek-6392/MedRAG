# Model Cards & Architecture Reference

This document describes the neural and generative models utilized in the MedRAG research pipeline.

---

## 1. Dense Retriever: `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture**: 6-layer Transformer (BERT-mini backbone)
- **Embedding Dimension**: 384
- **Similarity Metric**: Cosine similarity / Dot product
- **Role in MedRAG**: Encodes question queries and textbook/literature passages into dense semantic vectors for first-stage hybrid retrieval.
- **HuggingFace Identifier**: `sentence-transformers/all-MiniLM-L6-v2`
- **Revision**: `main`

---

## 2. Neural Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Architecture**: 6-layer Cross-Encoder
- **Role in MedRAG**: Takes top candidate pairs `(query, document_passage)` retrieved by sparse BM25 and dense retrieval, outputting direct relevance scores to filter down to the top-$k$ context passages.
- **HuggingFace Identifier**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Revision**: `main`

---

## 3. NLI Verifier: `cross-encoder/nli-MiniLM2-L6-H768`
- **Architecture**: Cross-Encoder sequence classification model fine-tuned on MNLI and SNLI
- **Role in MedRAG**: Performs sentence-level Natural Language Inference to evaluate extracted atomic clinical claims against retrieved evidence passages.
- **Output Classes**: Entailment, Neutral, Contradiction
- **Score Mapping**:
  - `entailment_score >= support_threshold` (default 0.70) $\rightarrow$ `SUPPORTED`
  - `contradiction_score >= conflict_threshold` (default 0.35) $\rightarrow$ `CONTRADICTED`
  - Otherwise $\rightarrow$ `NOT_ENOUGH_INFORMATION` / Missing Evidence
- **HuggingFace Identifier**: `cross-encoder/nli-MiniLM2-L6-H768`
- **Revision**: `main`

---

## 4. Generative LLMs
- **LLaMA-3.3-70B-Versatile**: 70B parameter open-weights model hosted via Groq Cloud API for low-latency reasoning and synthesis.
- **GPT-4o-mini**: OpenAI frontier lightweight model used for comparative closed-book and RAG baselines.
- **MockClinicalLLM**: Rule-based offline generator used for continuous integration, deterministic smoke testing, and airgapped test reproduction without external API keys.
