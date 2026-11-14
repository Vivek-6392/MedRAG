# Formal Problem Formulation: Evidence-Aware Medical RAG

## 1. Problem Overview and Clinical Motivation

Retrieval-Augmented Generation (RAG) systems in medicine face fundamental challenges that standard open-domain RAG frameworks cannot address:

1. **High-Stakes Asymmetry**: In clinical decision support, the penalty for an undetected contradiction, adverse interaction, or contraindication is far higher than the reward for generating a plausible answer.
2. **Conflicting Evidence in Biomedical Literature**: Unlike static factual knowledge bases, biomedical literature contains evolving clinical trials, conflicting observational studies, and superseded clinical guidelines.
3. **Epistemic Uncertainty and Hallucination**: Generative Large Language Models (LLMs) frequently confabulate plausibly sounding clinical justifications or cite non-existent evidence when unconstrained.
4. **Binary Forcing vs. Clinical Abstention**: Standard RAG pipelines force a generative model to synthesize an answer even when retrieved evidence is empty, contradictory, or insufficient. In medicine, selective abstention ("withholding a recommendation pending further workup") is often the safest clinical action.

MedRAG formalizes medical QA not as single-shot text generation, but as an **evidence-state tracking and risk-conditioned decision process**.

---

## 2. Mathematical Notation and Problem Space

Let:
- $\mathcal{X}$ denote the space of clinical queries.
- $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$ denote an external biomedical corpus of documents, where each document $d_j = \langle \text{id}_j, \text{title}_j, \text{text}_j, \text{source}_j, Y_j, \sigma_j \rangle$ is annotated with publication year $Y_j$ and source reliability grade $\sigma_j \in [0, 1]$.
- $\mathcal{Y}$ denote the space of candidate answers.
- $\mathcal{C}(y) = \{c_1, c_2, \dots, c_m\}$ denote a set of atomic, independently falsifiable clinical claims extracted from candidate answer $y$.
- $\mathcal{A} = \{\text{STOP}, \text{SEARCH}, \text{PARTIAL}, \text{CONFLICTED}, \text{ABSTAIN}\}$ denote the set of next-action control decisions.

Given query $x \in \mathcal{X}$, the system's objective is to produce a decision $a \in \mathcal{A}$ and, when $a \in \{\text{STOP}, \text{PARTIAL}\}$, an evidence-attributed answer $y^*$ such that:
1. Every retained claim $c \in \mathcal{C}(y^*)$ is strongly entailed by verified evidence passages in $\mathcal{D}$.
2. No claim $c \in \mathcal{C}(y^*)$ is materially contradicted by credible literature in $\mathcal{D}$.
3. When evidence is contradictory or insufficient under high clinical risk, the system abstains ($a = \text{ABSTAIN}$ or $a = \text{CONFLICTED}$), guaranteeing that selective risk remains below a pre-specified tolerance $\epsilon^*$.

---

## 3. Formal Evidence-State Model

For each atomic claim $c_i \in \mathcal{C}(y)$, we define an evidence state $\mathcal{S}(c_i)$ over the retrieved premise passages $\mathcal{P} = \{p_1, \dots, p_K\}$ extracted from the accumulated documents $\mathcal{D}_{\text{accum}}$:

$$\mathcal{S}(c_i) = \left\langle \mathcal{E}^+(c_i), \mathcal{E}^-(c_i), \mathcal{E}^{\emptyset}(c_i), s_{\text{sup}}(c_i), s_{\text{con}}(c_i), w_{\text{source}}(c_i), \text{status}(c_i) \right\rangle$$

### A. Premise Partitioning
Each premise $p \in \mathcal{P}$ evaluated by an NLI cross-encoder $\psi_{\text{NLI}}(p, c_i)$ produces a probability distribution over labels $\mathbf{p} = [p_{\text{con}}, p_{\text{ent}}, p_{\text{neu}}] \in \Delta^2$.
Premises are partitioned into mutually disjoint sets:

1. **Supporting Evidence** $\mathcal{E}^+(c_i)$:
   $$\mathcal{E}^+(c_i) = \left\{ p \in \mathcal{P} \;\middle|\; p_{\text{ent}}(p, c_i) \ge \tau_{\text{sup}} \right\}$$

2. **Contradicting Evidence** $\mathcal{E}^-(c_i)$:
   $$\mathcal{E}^-(c_i) = \left\{ p \in \mathcal{P} \;\middle|\; p_{\text{con}}(p, c_i) \ge \tau_{\text{con}} \;\wedge\; p_{\text{con}} > p_{\text{ent}} \;\wedge\; p_{\text{con}} > p_{\text{neu}} \right\}$$

3. **Neutral / Insufficient Evidence** $\mathcal{E}^{\emptyset}(c_i)$:
   $$\mathcal{E}^{\emptyset}(c_i) = \mathcal{P} \setminus \left( \mathcal{E}^+(c_i) \cup \mathcal{E}^-(c_i) \right)$$

### B. Independent Extremal Scores
Unlike heuristic models that calculate a single net difference $(p_{\text{ent}} - p_{\text{con}})$, MedRAG tracks support and conflict independently:

$$s_{\text{sup}}(c_i) = \max_{p \in \mathcal{P}} p_{\text{ent}}(p, c_i)$$

$$s_{\text{con}}(c_i) = \max_{p \in \mathcal{E}^-(c_i)} p_{\text{con}}(p, c_i) \quad (\text{or } 0.0 \text{ if } \mathcal{E}^-(c_i) = \emptyset)$$

### C. Source Quality and Temporal Reliability
For premise $p$ originating from document $d$, reliability is modeled as a convex combination of intrinsic peer-review source tier $\sigma(d) \in [0, 1]$ and annual recency decay:

$$w(d) = \alpha \cdot \sigma(d) + (1 - \alpha) \cdot \max\left(0, \; 1 - \beta \cdot (Y_{\text{ref}} - Y_d)\right)$$

where $\alpha \in [0, 1]$ (default 0.6) balances authority and freshness, $\beta = 0.02$ represents a 2% annual decay rate, and $Y_{\text{ref}}$ is the evaluation reference year.

### D. Four-Way Verification Status
$$\text{status}(c_i) = \begin{cases}
\text{CONFLICTED} & \text{if } s_{\text{con}}(c_i) \ge \tau_{\text{con}} \;\wedge\; s_{\text{sup}}(c_i) \ge \tau_{\text{sup}} \\
\text{CONTRADICTED} & \text{if } s_{\text{con}}(c_i) \ge \tau_{\text{con}} \;\wedge\; s_{\text{sup}}(c_i) < \tau_{\text{sup}} \\
\text{SUPPORTED} & \text{if } s_{\text{sup}}(c_i) \ge \tau_{\text{sup}} \;\wedge\; s_{\text{con}}(c_i) < \tau_{\text{con}} \;\wedge\; \mathcal{H}(c_i) \ge \tau_{\text{conf}} \\
\text{MISSING} & \text{if } s_{\text{sup}}(c_i) < \tau_{\text{mis}} \;\wedge\; s_{\text{con}}(c_i) < \tau_{\text{con}} \\
\text{UNCERTAIN} & \text{otherwise}
\end{cases}$$

---

## 4. Probabilistic Reformulation: Evidence Accumulation

Rather than relying purely on an empirical heuristic index, evidence accumulation across multiple independent retrieved sources $d_1, \dots, d_K$ can be formalized as updating belief mass.

Let $H_{c_i} \in \{+1, -1, 0\}$ denote the true clinical validity of claim $c_i$ ($+1$: true/indicated, $-1$: false/contraindicated, $0$: unverified).
Each retrieved premise $p_k$ provides a noisy observation with likelihood:
$$P(p_k \mid H_{c_i} = +1) \propto p_{\text{ent}}(p_k, c_i) \cdot w(d_k)$$
$$P(p_k \mid H_{c_i} = -1) \propto p_{\text{con}}(p_k, c_i) \cdot w(d_k)$$

Under conditional independence across distinct medical sources:
$$\log \frac{P(H_{c_i} = +1 \mid \mathcal{P})}{P(H_{c_i} = -1 \mid \mathcal{P})} = \log \frac{P(H_{c_i} = +1)}{P(H_{c_i} = -1)} + \sum_{k=1}^K w(d_k) \cdot \log \frac{p_{\text{ent}}(p_k, c_i) + \delta}{p_{\text{con}}(p_k, c_i) + \delta}$$

This probabilistic formulation demonstrates why **contradictions must not be masked by raw document count**: a single high-reliability contradicting passage with high $p_{\text{con}}$ exerts a large negative log-likelihood ratio that cannot be cancelled by low-reliability neutral documents.

---

## 5. Selective Answering and Risk-Coverage Optimization

In standard classification, an algorithm maps input $x$ to prediction $\hat{y}$. In selective prediction (Geifman & El-Yaniv, 2017), the system outputs a pair $(\hat{y}, g(x))$, where $g(x) \in \{0, 1\}$ is a binary selection function indicating whether to answer ($g(x)=1$) or abstain ($g(x)=0$).

### Selective Metrics
- **Coverage**:
  $$\text{Coverage}(g) = \mathbb{E}_{x \sim \mathcal{D}}[g(x)] = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} g(x_i)$$

- **Selective Accuracy**:
  $$\text{SelectiveAccuracy}(f, g) = \frac{\sum_{i=1}^{|\mathcal{D}|} g(x_i) \cdot \mathbb{I}(f(x_i) = y_i^*)}{\sum_{i=1}^{|\mathcal{D}|} g(x_i)}$$

- **Selective Risk**:
  $$\mathcal{R}(f, g) = 1 - \text{SelectiveAccuracy}(f, g) = \frac{\sum_{i=1}^{|\mathcal{D}|} g(x_i) \cdot \mathbb{I}(f(x_i) \neq y_i^*)}{\sum_{i=1}^{|\mathcal{D}|} g(x_i)}$$

### Optimization Objective
Given user-defined maximum risk tolerance $\epsilon^*$ (e.g., $\epsilon^* = 0.05$, corresponding to $\ge 95\%$ clinical accuracy on answered questions), find selection threshold $\tau^*$ that maximizes coverage:

$$\max_{\tau} \text{Coverage}(g_\tau) \quad \text{subject to} \quad \mathcal{R}(f, g_\tau) \le \epsilon^*$$

MedRAG achieves this by tuning decision thresholds strictly on validation data $\mathcal{D}_{\text{val}}$ and freezing them for test evaluation.
