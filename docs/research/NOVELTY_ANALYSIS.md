# Novelty Analysis & Scientific Contribution Assessment

This document rigorously evaluates the scientific contributions of MedRAG. It distinguishes genuine methodological contributions from existing literature adaptations and clarifies which claims are experimentally validated versus theoretical.

---

## 1. Delineation of Existing vs. Novel Methods

| Component | Status in Literature | MedRAG Implementation Status | Novelty Assessment |
| :--- | :--- | :--- | :--- |
| **BM25 & Dense Bi-Encoder Retrieval** | Standard (Robertson et al., 2009; Karpukhin et al., 2020) | Implemented (`rank-bm25`, `sentence-transformers`) | **Standard baseline**. Not novel. |
| **Reciprocal Rank Fusion (RRF)** | Established (Cormack et al., 2009) | Implemented (`reciprocal_rank_fusion` in `medrag/retriever.py`) | **Established method**. Adapted with tie-breaking and rank metadata tracking. |
| **Cross-Encoder Neural Reranking** | Standard (Nogueira & Cho, 2019) | Implemented (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | **Standard baseline**. Not novel. |
| **Atomic Claim Decomposition** | Established (Min et al., 2023; Chern et al., 2023) | Implemented (`AnswerGenerator.decompose_claims` in `medrag/generator.py`) | **Established method**. Applied to clinical answers. |
| **Cross-Encoder NLI Classification** | Standard (Williams et al., 2018; Gu et al., 2021) | Implemented (`NLIVerifier` with `nli-MiniLM2-L6-H768`) | **Standard baseline**. Multi-granularity premise extraction with title injection is an engineering refinement. |
| **Independent Contradiction & Support Tracking** | Rare in RAG; standard NLI subtracts or takes argmax | Implemented (`ClaimEvidence`, `EvidenceState`, `medrag/evidence_state.py`) | **Genuinely Novel in Medical RAG**: Preserves independent support and conflict scores, preventing contradictions from being masked by volume. |
| **Source Authority & Recency Decay** | Proposed conceptually in information science | Implemented (`EvidencePassage.reliability` with $\sigma(d)$ and annual decay $\beta$) | **Novel Integration**: Formally couples source tier and temporal decay into claim verification. |
| **Claim-Specific Gap Query Iteration** | Iteration exists (IRCoT, ITER-RETGEN); gap targeting is novel | Implemented (`AnswerGenerator.generate_gap_query` in `medrag/generator.py`) | **Novel Formulation**: Uses unresolved and conflicted claims from NLI verification to drive subsequent retrieval rounds. |
| **Risk-Conditioned Multi-Action Decision Policy** | Selective classification exists (Geifman & El-Yaniv, 2017); clinical risk coupling is new | Implemented (`EvidenceController`, `ClinicalRiskPolicy`, `medrag/risk_policy.py`) | **Genuinely Novel**: Modulates verification thresholds and enables qualified disclosure (`PARTIAL`) vs conservative abstention (`ABSTAIN`). |
| **Selective Risk-Coverage Tuning** | Standard in selective classification | Implemented (`medrag/benchmarks/abstention.py`) | **Methodological Rigor**: Rigorously enforces that threshold tuning occurs strictly on validation splits and is frozen for test evaluation. |

---

## 2. Supported Contributions vs. Non-Claims

To maintain publication integrity, we explicitly define what MedRAG does and does not claim:

### What MedRAG Accurately Claims:
1. **Source-Aware Contradiction Isolation**: Demonstrates that tracking contradictions independently from support prevents literature disagreements from being obscured by high-volume supportive passages.
2. **Selective Risk-Coverage Optimization**: Proves that allowing the system to selectively abstain on high-risk, conflicted, or unverified claims strictly dominates forced greedy answering, achieving superior selective accuracy on answered queries.
3. **Reproducible Multi-Level Decision Architecture**: Provides an open-source, fully reproducible 5-action decision harness (`STOP`, `SEARCH`, `PARTIAL`, `CONFLICTED`, `ABSTAIN`) with deterministic seeds and automated statistical significance reporting.
4. **Targeted Gap Retrieval**: Demonstrates that probing for missing clinical entities identified by NLI claim failures recovers evidence more efficiently than undirected iterative thinking steps.

### What MedRAG Does NOT Claim:
1. **No Claim of General Clinical Safety**: Benchmark accuracy (e.g. on MedQA or PubMedQA) does not demonstrate real-world clinical safety or effectiveness.
2. **No Claim of Unsupervised Gold Retrieval on MedQA**: The repository strictly refuses to fabricate passage retrieval labels from multiple-choice answer correctness. Retrieval recall is evaluated only on PubMedQA where ground-truth PMIDs are verified.
3. **No Claim of Medical Device or HIPAA Readiness**: Software is an academic research prototype. It does not possess FDA clearance, CE mark, or HIPAA certification.
4. **No Claim of Probability from Heuristic Confidence**: Raw heuristic confidence scores $\mathcal{H}(c)$ are explicitly treated as ranking indices and are distinguished from calibrated empirical probabilities.
