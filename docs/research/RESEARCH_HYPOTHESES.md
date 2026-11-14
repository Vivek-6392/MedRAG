# Research Hypotheses & Experimental Validation Protocols

This document defines the formal research hypotheses of MedRAG, their theoretical justifications, operational metrics, null hypotheses, and statistical validation strategies.

---

## Hypothesis 1 (H1): Hybrid Retrieval Superiority

> **Statement**: Combining sparse lexical retrieval (BM25) with dense bi-encoder semantic retrieval via Reciprocal Rank Fusion (RRF) significantly improves medical evidence retrieval quality over sparse-only or dense-only retrieval.

### Theoretical Justification
Biomedical queries contain both exact terminology (drug names, gene symbols, specific numerical lab cutoffs like "eGFR < 30") where lexical BM25 excels, and conceptual descriptive phrasing ("cardiac remodeling prevention") where dense neural embeddings capture semantic synonyms. Neither paradigm alone suffices across diverse medical queries.

### Operational Setup & Comparison
- **System A**: `hybrid_rag` (BM25 + Dense RRF)
- **Baseline 1**: `single_pass_bm25_rag`
- **Baseline 2**: `single_pass_dense_rag`
- **Benchmark**: PubMedQA Retrieval (`pubmedqa_retrieval` protocol) where ground-truth reference abstracts (PMIDs) are documented.

### Metrics
- **Recall@1, Recall@5, Recall@10**
- **Mean Reciprocal Rank (MRR)**
- **nDCG@5**

### Null Hypothesis
$$H_0^{(1)}: \text{Recall@5}(\text{Hybrid}) \le \max\left(\text{Recall@5}(\text{BM25}), \; \text{Recall@5}(\text{Dense})\right)$$

### Validation Strategy
- Paired bootstrap resampling ($B = 1,000$ iterations) over aligned retrieval queries.
- Reject $H_0^{(1)}$ if the empirical 95% bootstrap confidence interval of $\Delta \text{Recall@5}$ is strictly positive ($CI_{\text{lower}} > 0$) with $p < 0.05$.

---

## Hypothesis 2 (H2): Claim-Level Verification and Hallucination Reduction

> **Statement**: Decomposing candidate medical answers into atomic claims and verifying each against retrieved premises via sentence-level NLI significantly reduces unsupported and confabulated clinical assertions compared to standard generation.

### Theoretical Justification
Standard generative LLMs suffer from "citation hallucination," where passages are cited that do not actually entail the assertion, or where plausible clinical claims are fabricated beyond the evidence. Sentence-level cross-encoder NLI provides direct token-interaction verification to detect unentailed statements.

### Operational Setup & Comparison
- **System A**: `claim_verified_rag`
- **Baseline**: `hybrid_reranked_rag` (unverified generation)
- **Evaluation**: Manually adjudicated claim subset evaluated via `evaluate_claim_faithfulness`.

### Metrics
- **Unsupported Claim Rate**: Proportion of generated claims labeled as `NOT_ENOUGH_INFORMATION` or unentailed.
- **Citation Precision**: Proportion of cited documents that genuinely entail the citing claim.
- **Citation Completeness**: Proportion of generated claims possessing at least one valid supporting citation.

### Null Hypothesis
$$H_0^{(2)}: \text{UnsupportedRate}(\text{Verified}) \ge \text{UnsupportedRate}(\text{Unverified})$$

### Validation Strategy
- Exact binomial test and McNemar's test on paired claim-level correctness across the annotated gold standard subset.
- Reject $H_0^{(2)}$ if the unsupported claim rate reduction is statistically significant ($p < 0.01$).

---

## Hypothesis 3 (H3): Contradiction-Sensitive Aggregation & Risk Prevention

> **Statement**: Explicitly isolating contradictory evidence and enforcing an asymmetric contradiction penalty prevents the system from generating unsafe clinical answers on queries with conflicting biomedical evidence.

### Theoretical Justification
In biomedical research, observational studies and randomized trials frequently report contradictory efficacy or safety findings. Standard retrieval systems aggregate evidence by raw passage count or average embedding similarity, causing a small number of critical contradiction passages (e.g. "increased mortality in subgroup X") to be drowned out by numerous supportive marketing excerpts.

### Operational Setup & Comparison
- **System A**: `full_medrag` (asymmetric conflict thresholding and conflict penalty)
- **Baseline**: `ablation_no_nli_verification` or `ablation_no_evidence_state` (naive average relevance scoring)
- **Benchmark**: Conflicted clinical query subset (queries exhibiting material literature controversy).

### Metrics
- **False-Support Rate**: Proportion of contested claims improperly labeled as `SUPPORTED`.
- **Selective Accuracy on Conflicted Queries**.
- **Contradiction Detection F1**.

### Null Hypothesis
$$H_0^{(3)}: \text{FalseSupportRate}(\text{ContradictionSensitive}) \ge \text{FalseSupportRate}(\text{NaiveAggregation})$$

### Validation Strategy
- Stratified bootstrap evaluation on the subset of queries with confirmed multi-source evidence conflict.
- Reject $H_0^{(3)}$ if $\Delta \text{FalseSupportRate} < -0.15$ with $p < 0.01$.

---

## Hypothesis 4 (H4): Iterative Retrieval & Evidence Completeness

> **Statement**: Iterative targeted retrieval directed specifically at unresolved claim gaps recovers missing evidence and resolves epistemic uncertainty more effectively than single-pass retrieval with an equivalent total passage budget.

### Theoretical Justification
First-round queries are formulated before the model knows what specific evidence is missing. By extracting unresolved claims $\mathcal{C}_{\text{weak}}$ after Round 1 and generating focused medical gap queries, the retriever probes the corpus for specific missing clinical endpoints, contraindications, or dosing limits.

### Operational Setup & Comparison
- **System A**: `full_medrag` ($T_{\text{max}} = 3$ iterative rounds)
- **Baseline**: `ablation_no_iterative` (single-pass retrieval with top-$k=15$ context documents to match total retrieved volume).
- **Benchmark**: MedQA USMLE test split and PubMedQA retrieval test split.

### Metrics
- **Evidence Completeness** (proportion of claims resolved from `MISSING` to `SUPPORTED`).
- **Mean Search Rounds to Convergence**.
- **Final Answer Accuracy & Selective Accuracy**.

### Null Hypothesis
$$H_0^{(4)}: \text{Accuracy}(\text{Iterative}) \le \text{Accuracy}(\text{SinglePass}_{\text{matched\_budget}})$$

### Validation Strategy
- Paired McNemar's test on test split predictions.
- Reject $H_0^{(4)}$ if McNemar's $p < 0.05$ with an improvement in evidence completeness $\ge 10\%$.

---

## Hypothesis 5 (H5): Risk-Aware Selective Abstention & The Risk-Coverage Frontier

> **Statement**: Conditioning abstention decisions on clinical risk tiering and evidence confidence strictly dominates unconditional greedy answering, achieving higher selective accuracy and lower selective risk across all operating coverage levels.

### Theoretical Justification
In medicine, forced 100% coverage guarantees that errors will occur on unanswerable, ambiguous, or contradictory questions. Selective prediction allows the system to trade coverage for precision by abstaining when confidence is low or patient risk is high.

### Operational Setup & Comparison
- **System A**: `full_medrag` with risk-conditioned policy $\tau(\rho_x)$
- **Baseline 1**: `closed_book` (100% forced coverage)
- **Baseline 2**: `ablation_no_abstention` (forced answering on all retrieved queries)
- **Baseline 3**: Naive uncalibrated LLM self-confidence abstention.

### Metrics
- **Area Under Risk-Coverage Curve (AURC)**: Lower is better.
- **Selective Accuracy at $\kappa = 80\%$ Coverage**.
- **Selective Risk at $\kappa = 80\%$ Coverage**.
- **Abstention Precision & Recall** on defined unanswerable queries.

### Null Hypothesis
$$H_0^{(5)}: \text{AURC}(\text{RiskAwareMedRAG}) \ge \text{AURC}(\text{UncalibratedBaseline})$$

### Validation Strategy
- Compute empirical risk-coverage curves across threshold sweep $\tau \in [0, 1]$.
- Paired bootstrap resampling ($B = 1,000$) of AURC difference:
  $$\Delta \text{AURC} = \text{AURC}(\text{MedRAG}) - \text{AURC}(\text{Baseline})$$
- Reject $H_0^{(5)}$ if $CI_{\text{upper}}(\Delta \text{AURC}) < 0$ with $p < 0.01$.
