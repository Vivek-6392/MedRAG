# MedRag — Adaptive Evidence-Aware Retrieval and Verification Framework

**Reliable Medical Question Answering via Structured Evidence Reasoning**

> IIIT Nagpur · Guided by Dr. Pooja Jain  
> Vivek Yadav · Jayant Dhaka · Ojaswa Awasthi

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
| **STOP**    | All claims: `support ≥ threshold` AND `conflict < threshold` |
| **ABSTAIN** | Round limit reached · OR · Very high clinical risk + unresolved conflict |
| **SEARCH**  | Evidence gaps / conflicts remain, rounds available → targeted re-query |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo (mock LLM, no API key needed)
python main.py --demo --backend mock --no-reranker

# 3. Interactive mode with OpenAI
export OPENAI_API_KEY=sk-...
python main.py

# 4. Single question with Groq
export GROQ_API_KEY=gsk_...
python main.py -q "Is metformin safe for patients with renal impairment?"

# 5. Custom corpus
python main.py --corpus data/pubmed.jsonl --max-rounds 4
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
| `llm_backend` | `"openai"` | `openai` / `groq` / `ollama` / `mock` |
| `llm_model` | `"gpt-4o-mini"` | Model name |
| `dense_model` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `reranker_model` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `nli_model` | `nli-MiniLM2-L6-H768` | NLI claim verifier |
| `top_k_retrieve` | `20` | Candidates per retrieval |
| `top_k_rerank` | `5` | Docs passed to LLM / NLI |
| `support_threshold` | `0.70` | Min entailment for STOP |
| `conflict_threshold` | `0.35` | Max contradiction for STOP |
| `max_search_rounds` | `3` | ABSTAIN after N rounds |
| `high_risk_threshold` | `0.75` | Stricter threshold above this risk |
| `high_risk_penalty` | `0.15` | Extra support required for high-risk |

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
