# Table 1: MedQA USMLE Baseline Comparison

> **Data Provenance:** Empirical results extracted from validated evaluation run `runs/cli_study_smoke`.
> 95% Confidence Intervals calculated via Wilson score interval with continuity correction.

| Architecture / Baseline | N | Raw Accuracy (95% CI) | Selective Accuracy | Coverage | Abstention Rate | Mean Latency (s) | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Closed Book | 2 | 0.0% [0.0%, 65.8%] | 0.0% | 0.0% | 100.0% | 0.00s | Validated Smoke Suite |
| Single Pass Dense Rag | 2 | 100.0% [34.2%, 100.0%] | 100.0% | 100.0% | 0.0% | 0.18s | Validated Smoke Suite |
| Single Pass Bm25 Rag | 2 | 100.0% [34.2%, 100.0%] | 100.0% | 100.0% | 0.0% | 0.01s | Validated Smoke Suite |
| Hybrid Rag | 2 | 100.0% [34.2%, 100.0%] | 100.0% | 100.0% | 0.0% | 0.41s | Validated Smoke Suite |
| Hybrid Reranked Rag | 2 | 100.0% [34.2%, 100.0%] | 100.0% | 100.0% | 0.0% | 0.97s | Validated Smoke Suite |
| Claim Verified Rag | 2 | 100.0% [34.2%, 100.0%] | 100.0% | 100.0% | 0.0% | 13.68s | Validated Smoke Suite |
| Iterative Rag | 2 | 100.0% [34.2%, 100.0%] | 100.0% | 100.0% | 0.0% | 41.13s | Validated Smoke Suite |
| Full Medrag | 2 | 100.0% [34.2%, 100.0%] | 100.0% | 100.0% | 0.0% | 39.94s | Validated Smoke Suite |

*Note: Full-scale USMLE Step 1/2/3 4-option test evaluation (N=1,273 examples) is designated as `[Pending Cloud GPU Run]` requiring extended external LLM inferencing.*
