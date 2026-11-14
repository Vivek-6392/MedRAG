# MedRAG Evaluation & Statistical Analysis Methodology

## 1. Overview and Core Philosophy

The MedRAG evaluation engine provides mathematically rigorous, reproducible, and transparent evidence regarding medical Question Answering (QA), biomedical evidence retrieval, claim-level faithfulness, and clinical abstention. 

### Methodological Guardrails
1. **No Conflation of Benchmark Accuracy with Clinical Safety:** Statistical significance ($p < 0.05$) or high benchmark accuracy (e.g., on USMLE-style questions) indicates empirical consistency on sampled multiple-choice problems. It **does not** prove patient safety, clinical efficacy, or the absence of hazardous medical hallucinations in real-world deployment.
2. **Separation of Heuristic Confidence from Calibrated Probability:** Raw Cross-Encoder NLI scores and reciprocal rank scores are heuristic ordinal indicators, **not** true Bayesian posterior probabilities. A confidence estimate is only designated as *calibrated* if explicit post-hoc calibration (e.g., temperature scaling, Platt scaling) was performed on a held-out validation split.
3. **Strict Attribution & Ground-Truth Provenance:** Retrieval metrics on MedQA are computed **only** when independent passage-level relevance annotations exist. Gold retrieval labels are **never** inferred from answer correctness.
4. **Claim-Level Grounding Scope:** Atomic claim verification metrics are reported strictly on manually annotated subsets and are never extrapolated to an entire unannotated benchmark.

---

## 2. PART A: Answer Quality Evaluation

### 2.1 MedQA (English USMLE 4-Option Subset)
For each multiple-choice option $c \in \{A, B, C, D\}$:
- **True Positives ($TP_c$):** Gold is $c$ and predicted is $c$.
- **False Positives ($FP_c$):** Gold is not $c$ and predicted is $c$.
- **False Negatives ($FN_c$):** Gold is $c$ and predicted is not $c$ (including abstentions).
- **True Negatives ($TN_c$):** Gold is not $c$ and predicted is not $c$.

For each choice, the engine computes:
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad \text{Specificity}_c = \frac{TN_c}{TN_c + FP_c}$$
$$\text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
$$\text{Macro-F1} = \frac{1}{4} \sum_{c \in \{A, B, C, D\}} \text{F1}_c$$

Full confusion matrices $M \in \mathbb{R}^{4 \times 5}$ track gold choices against predictions, explicitly including an `ABSTAIN` column.

### 2.2 PubMedQA (Biomedical Research QA)
Evaluates categorical reasoning over $c \in \{\text{yes}, \text{no}, \text{maybe}\}$:
- Per-class precision, recall, specificity, and F1.
- Macro-averaged F1 across all three classes.
- Full confusion matrix highlighting false-positive affirmations versus safe abstentions on inconclusive studies (`maybe`).

---

## 3. PART B: Retrieval Quality Evaluation

### 3.1 PubMedQA Retrieval Mode
Evaluates MedRAG's ability to search an external corpus and locate the benchmark's gold abstract without leaking gold labels or questions into the index:
- **Recall@k:** Proportion of questions where the gold PMID is retrieved within the top-$k$ candidates:
  $$\text{Recall}@k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left(\text{PMID}_q \in \text{Retrieved}_k(q)\right), \quad k \in \{1, 5, 10\}$$
- **Mean Reciprocal Rank (MRR):**
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$
- **Discounted Cumulative Gain (nDCG@k):** Supports binary targets and multi-grade relevance annotations ($rel_i \in \{0, 1, 2\}$):
  $$\text{DCG}@k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$$

### 3.2 MedQA Retrieval Guardrail
For MedQA, textbook retrieval recall and ranking metrics are **only** computed when independent passage-level relevance annotations exist (`metadata.gold_citation_ids` or `metadata.passage_relevance`). MedRAG explicitly refuses to synthesize artificial retrieval labels from answer correctness, preventing circular evaluation artifacts.

---

## 4. PART C: Claim-Level Faithfulness & Citation Attribution

### 4.1 Standardized Claim-Evidence Schema
To rigorously verify hallucination rates and sentence-level evidence attribution, annotations follow the schema:
```json
{
  "example_id": "medqa_test_042",
  "claim_id": "c1",
  "claim": "Metformin is contraindicated when eGFR is below 30 mL/min/1.73m2.",
  "evidence_id": "textbook_renal_ch4_p12",
  "label": "supported",
  "annotator_id": "clinical_fellow_1",
  "adjudication_status": "adjudicated"
}
```

Canonical Ground-Truth Labels:
1. `SUPPORTED`: The cited reference passage directly and fully entails the assertion.
2. `CONTRADICTED`: The cited passage explicitly refutes or disputes the assertion.
3. `NOT_ENOUGH_INFORMATION`: The reference does not contain sufficient factual evidence to confirm or deny.
4. `PARTIALLY_SUPPORTED`: The reference confirms key elements but lacks critical quantifiers, dosages, or constraints.

### 4.2 Faithfulness & Attribution Metrics
- **Claim-Level Precision, Recall, and Macro-F1:** Evaluated across the 4 canonical verification categories.
- **Unsupported Claim Rate:** Proportion of generated claims labeled `NOT_ENOUGH_INFORMATION` or `CONTRADICTED`:
  $$\text{Unsupported Rate} = \frac{N_{\text{NEI}} + N_{\text{CONTRADICTED}}}{N_{\text{total\_claims}}}$$
- **False-Support Rate:** Proportion of claims that MedRAG's verifier accepted as `SUPPORTED`, but whose ground-truth label was refuted or ungrounded:
  $$\text{False-Support Rate} = \frac{|\{c \mid \text{pred}(c) = \text{SUPPORTED} \land \text{gold}(c) \ne \text{SUPPORTED}\}|}{|\{c \mid \text{pred}(c) = \text{SUPPORTED}\}|}$$
- **Citation Precision:** Proportion of model-cited documents that genuinely entail the claimed medical fact.
- **Citation Completeness:** Proportion of essential gold reference documents that were successfully cited.

---

## 5. PART D: Abstention & Selective Classification

In high-stakes clinical decision support, the ability to withhold judgment on unconfirmed or contradictory queries is paramount.

### 5.1 Selective Answering Metrics
- **Coverage ($\Phi$):** Fraction of benchmark questions where the system issues a definitive answer:
  $$\Phi = \frac{|\mathcal{A}|}{|\mathcal{D}|}, \quad \mathcal{A} = \{q \in \mathcal{D} \mid \text{decision}(q) \ne \text{ABSTAIN}\}$$
- **Selective Accuracy ($\text{Acc}_{sel}$):** Accuracy conditioned strictly on answered queries:
  $$\text{Acc}_{sel} = \frac{1}{|\mathcal{A}|} \sum_{q \in \mathcal{A}} \mathbb{I}(\hat{y}_q = y_q^*)$$
- **Selective Risk ($\mathcal{R}_{sel}$):** Empirical error rate on answered queries:
  $$\mathcal{R}_{sel} = 1 - \text{Acc}_{sel} = \frac{1}{|\mathcal{A}|} \sum_{q \in \mathcal{A}} \mathbb{I}(\hat{y}_q \ne y_q^*)$$
- **Risk-Coverage Curve & AURC:** Sweeps verification thresholds to generate the empirical trade-off curve $(\Phi(\tau), \mathcal{R}_{sel}(\tau))$. The Area Under the Risk-Coverage Curve (AURC) is computed via trapezoidal numerical integration:
  $$\text{AURC} = \int_0^1 \mathcal{R}_{sel}(\phi) \, d\phi$$
  *Lower AURC indicates a superior trade-off where risk drops rapidly as abstention is exercised.*

### 5.2 Threshold Calibration on Validation Data
Thresholds for clinical confidence, support, and contradiction must **never** be tuned on the test split. The module provides `tune_abstention_thresholds(val_records, target_risk=0.05, min_coverage=0.50)`:
$$\tau^* = \arg\max_{\tau} \Phi_{val}(\tau) \quad \text{s.t.} \quad \mathcal{R}_{sel, val}(\tau) \le \text{target\_risk} \land \Phi_{val}(\tau) \ge \text{min\_coverage}$$
The optimal threshold $\tau^*$ is then held frozen when evaluating test performance.

---

## 6. PART E: Confidence Calibration & Reliability

When MedRAG outputs a numerical confidence score $\hat{p} \in [0, 1]$, its probabilistic calibration is assessed using:

### 6.1 Brier Score
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2, \quad y_i \in \{0, 1\}$$

### 6.2 Expected Calibration Error (ECE)
Partitions $[0, 1]$ into $M=10$ equal-width bins $B_1, \dots, B_M$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
where $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$ and $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$.

### 6.3 Post-hoc Temperature Scaling on Validation
To calibrate heuristic logits $z_i = \log(\hat{p}_i / (1 - \hat{p}_i))$, a temperature parameter $T > 0$ is optimized strictly on the validation split:
$$\min_T \frac{1}{|D_{val}|} \sum_{i \in D_{val}} \left(\sigma\left(\frac{z_i}{T}\right) - y_i\right)^2$$
Calibrated probabilities on the test split are computed as $\hat{p}_{cal} = \sigma(z_{test} / T)$.

---

## 7. PART F: Statistical Reporting & Significance Testing

For every reported metric, MedRAG outputs:
1. **Sample Size ($N$):** Number of evaluated examples.
2. **Point Estimate:** Empirical metric value.
3. **95% Confidence Intervals:**
   - **Binomial Proportions (Accuracy, Coverage, Selective Accuracy):** Computed via the **Wilson score interval with continuity correction**:
     $$CI_{95\%} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}, \quad z = 1.95996$$
     *Guarantees exact interval limits within $[0, 1]$ without normal approximation degradation.*
   - **Continuous and Macro Metrics (Macro-F1, AURC, Latency, ECE):** Computed via non-parametric empirical bootstrap resampling ($B=1,000$).
4. **Paired Statistical Tests vs. Baselines:**
   - **McNemar's Test:** Exact two-tailed binomial test for discordant pairs $b + c < 25$, Edwards continuity-corrected $\chi^2$ test with exact analytical $p$-value for $b + c \ge 25$.
   - **Paired Bootstrap:** 95% percentile CIs for $\Delta \text{Accuracy}$, $\Delta \text{Selective Accuracy}$, and empirical two-tailed $p$-values.

---

## 8. PART G: Visualizations

Publication-grade figures are rendered at 300 DPI (PNG) and as vector graphics (PDF) for LaTeX compilation in `publication/figures/`:

| Figure Name | File | Content & Academic Purpose |
| :--- | :--- | :--- |
| **Fig 1. Baseline Accuracy** | `baseline_accuracy_comparison.png/.pdf` | Dual-bar chart of Raw vs Selective Accuracy across 8 canonical systems. |
| **Fig 2. Retrieval Recall@k** | `retrieval_recall_at_k.png/.pdf` | Recall@1, 5, 10 curves across dense, BM25, hybrid, and reranked pipelines. |
| **Fig 3. Claim Verification** | `claim_verification_performance.png/.pdf` | Precision, Recall, and F1 across 4 canonical claim-evidence categories. |
| **Fig 4. Confusion Matrices** | `confusion_matrix_{name}.png/.pdf` | Annotated heatmaps of gold vs predicted choices including `ABSTAIN`. |
| **Fig 5. Component Ablations** | `ablation_delta_accuracy.png/.pdf` | Horizontal waterfall chart of $\Delta \text{Acc}$ with 95% bootstrap error bars. |
| **Fig 6. Risk-Coverage Curves** | `risk_coverage_curve.png/.pdf` | Empirical selective risk vs coverage trade-offs with AURC annotations. |
| **Fig 7. Reliability Diagram** | `reliability_diagram.png/.pdf` | Binned model confidence vs empirical accuracy with ECE and sample weights. |
| **Fig 8. Latency vs Accuracy** | `latency_vs_accuracy.png/.pdf` | Pareto efficiency frontier comparing response latency against clinical accuracy. |
| **Fig 9. Rounds Distribution** | `retrieval_round_distribution.png/.pdf` | Histogram of multi-round search stopping decisions across questions. |

All figures are synthesized dynamically from logged experimental outputs, eliminating manual entry or subjective chart adjustments.
