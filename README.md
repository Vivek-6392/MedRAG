# MedRag — Adaptive Evidence-Aware Retrieval and Verification Framework

**Reliable Medical Question Answering via Structured Evidence Reasoning**


---

## Architecture Overview

```
Medical Question
      │
      ▼
┌──────────────────────────────────┐
│  Phase 1: Clinical Query         │  Entity • Intent • Context • Risk
│  Understanding                   │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  Phase 2: Evidence-Aware         │  BM25 + Dense → RRF → Reranker
│  Retrieval                       │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  Phase 3: Generate Answer        │  LLM → Answer + Atomic Claims
│  + Atomic Claims                 │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  Phase 4: Claim-Level            │  NLI per (claim, document) pair
│  Verification                    │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐  Support • Conflict • Missing
│  Phase 5: Evidence State         │  Recency • Source Quality • Risk
└──────────────────┬───────────────┘
                   │
                   ▼
          ┌────────┴────────┐
          │ Next-Action     │
          │ Decision        │
          └──┬──────┬───┬───┘
             │      │   │
           STOP  ABSTAIN SEARCH ──► targeted gap query ──► Phase 2
             │      │
          Final  Abstention
          Answer  Notice
```

### Decision Rules

| Action    | Condition |
|-----------|-----------|
| **STOP**    | All claims meet the configured entailment and clinical-confidence thresholds, with no material contradiction |
| **PARTIAL** | Some claims meet those thresholds, remaining claims are explicitly unresolved, and maximum clinical risk is below the configured high-risk threshold |
| **CONFLICTED** | Material contradictory evidence remains; report both positions rather than selecting one |
| **SEARCH**  | Evidence gaps / conflicts remain, rounds available → targeted re-query |
| **ABSTAIN** | Evidence is insufficient or unreliable, no specific gap query can be formed, or high-risk claims remain unresolved |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Interactive Streamlit Web UI (Recommended)
streamlit run app.py

# 3. Terminal CLI mode
python main.py -q "Is metformin safe for patients with renal impairment?"

# 4. Built-in demo questions
python main.py --demo
```


---

## Project Structure

```
MedRag/
├── main.py                     # CLI entry point + sample corpus
├── config.py                   # MedRagConfig dataclass
├── requirements.txt
├── medrag/
│   ├── __init__.py
│   ├── query_understanding.py  # Phase 1: Entity / intent / risk extraction
│   ├── retriever.py            # Phase 2: BM25 + Dense + Reranker (RRF)
│   ├── generator.py            # Phase 3: LLM answer + claim decomposition
│   ├── evidence_state.py       # Phase 5: ClaimEvidence + EvidenceState
│   ├── verifier.py             # Phase 4: NLI claim verification
│   ├── controller.py           # Phase 6: SEARCH / STOP / ABSTAIN
│   └── pipeline.py             # Full adaptive loop orchestration
└── data/                       # (place your corpus JSON/JSONL files here)
```

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_backend` | `"groq"` | `openai` / `groq` / `ollama` / `mock` |
| `llm_model` | `"gpt-4o-mini"` | Model name |
| `dense_model` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `reranker_model` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `nli_model` | `nli-MiniLM2-L6-H768` | NLI claim verifier |
| `top_k_retrieve` | `20` | Candidates per retrieval |
| `top_k_rerank` | `5` | Docs passed to LLM / NLI |
| `support_threshold` | `0.70` | Min entailment for STOP |
| `conflict_threshold` | `0.35` | Max contradiction for STOP |
| `missing_threshold` | `0.35` | Below this entailment score, a non-conflicted claim is missing rather than uncertain |
| `clinical_confidence_threshold` | `0.65` | Minimum heuristic confidence after source-quality and recency factors |

For Groq, `GROQ_MODEL` must name a model available to your account. The example configuration uses
`openai/gpt-oss-120b`, which Groq currently recommends as a migration target for
`llama-3.3-70b-versatile`. Check the [Groq model catalog](https://console.groq.com/docs/models)
if Groq returns `model_not_found`; availability can vary by account and can change over time.
Groq GPT-OSS requests use low reasoning effort and a separate completion-token budget so internal
reasoning is less likely to consume the entire allowance before producing answer text.
| `max_search_rounds` | `3` | Maximum retrieval/verification rounds before a final decision |
| `high_risk_threshold` | `0.75` | High-risk unresolved claims cannot produce a partial answer after search |

`high_risk_penalty` remains accepted for configuration compatibility but is not used to shift evidence thresholds.
Risk changes the final action policy instead, keeping verifier and controller thresholds consistent. Retrieval
relevance scores are not used as NLI support; source quality and recency affect a heuristic confidence
indicator separately. This indicator is not a calibrated clinical probability.
All citations point to passage IDs and include source metadata. Evidence status is relative to the supplied
corpus only: the built-in sample corpus is illustrative, not authoritative clinical literature.
The built-in sample documents are explicitly marked illustrative and cannot meet the clinical-confidence
gate for a **STOP / VERIFIED** result, regardless of their example quality scores.

Claim statuses use one definition across verifier, evidence state, and controller: **conflicted** when
contradiction meets its threshold; **supported** when entailment and heuristic clinical confidence both meet
their thresholds with traceable non-illustrative support and no material contradiction; **missing** when entailment is below the missing
threshold without a material conflict; otherwise **uncertain**.

---

## Corpus Format

JSON or JSONL files containing a list of documents:

```json
[
  {
    "id": "abc123",
    "title": "Metformin in Type 2 Diabetes",
    "text": "Full text of the abstract or article...",
    "source": "NEJM",
    "year": 2024,
    "quality_score": 0.95
  }
]
```

`quality_score` (0–1) represents journal / source credibility and influences the Evidence State.
Optional `metadata: {"illustrative": true}` marks non-authoritative example data that cannot satisfy
the clinical-confidence gate.

---

## LLM Backends

| Backend | How to activate |
|---------|-----------------|
| OpenAI  | `export OPENAI_API_KEY=sk-...` |
| Groq    | `export GROQ_API_KEY=gsk_...` |
| Ollama  | `python main.py --backend ollama --base-url http://localhost:11434/v1 --model llama3` |
| Mock    | `python main.py --backend mock` (no key needed, template responses) |

---

## Programmatic API

```python
from medrag import MedRagPipeline, Document, LLMClient
from config import MedRagConfig

docs = [
    Document("1", "Title", "Full text...", source="NEJM", year=2024, quality_score=0.95),
]

config = MedRagConfig.from_env()
config.max_search_rounds = 3

llm = LLMClient(backend="openai", model="gpt-4o-mini", api_key="sk-...")
pipeline = MedRagPipeline(documents=docs, llm_client=llm, config=config)

result = pipeline.answer("Is metformin safe in renal failure?")
result.print_summary()

# Access structured evidence
for ce in result.evidence_state.claim_evidences:
    print(ce.claim, ce.status, ce.support_score, ce.conflict_score)
```

---

## Evaluation Plan (from Slide 9)

| Metric | Description |
|--------|-------------|
| Accuracy / F1 / EM | Standard QA metrics on MedQA + PubMedQA |
| Claim faithfulness | % claims supported by retrieved evidence |
| Contradiction rate | % claims with conflict score > threshold |
| Evidence coverage | % claims with at least one relevant document |
| Latency | End-to-end seconds per question |
| Abstention quality | Quality of ABSTAIN decisions vs gold labels |
| Retrieval rounds | Mean rounds to reach STOP / ABSTAIN |

### Ablations

| Ablation | Variable |
|----------|----------|
| 1 | Evidence state OFF vs ON |
| 2 | Generic query vs targeted gap query |
| 3 | Fixed threshold vs adaptive (risk-aware) policy |

---

## References

1. Ryan et al. (2026). *Self-MedRAG*: A Self-Reflective Hybrid RAG Framework.
2. Yang et al. (2025). *SIM-RAG*: Knowing When to Continue Search. SIGIR 2025.
3. Huang et al. (2026). *SEMA-RAG*: Self-Evolving Multi-Agent RAG. ACL 2026.
4. Ji et al. (2026). *MedRAGChecker*: Claim-Level Verification for Biomedical RAG.
5. Xiong et al. (2024). Benchmarking RAG for Medicine. ACL 2024.
