# Table 2: PubMedQA Biomedical QA Evaluation

> **Data Provenance:** Empirical results extracted from validated runs `runs/cli_pqa_prov_smoke` and `runs/cli_pqa_ret_smoke`.
> Label space: {yes, no, maybe}. Closed-world provided context vs Open-world scientific retrieval.

| Evaluation Protocol | N | Raw Accuracy (95% CI) | Coverage | Abstention Rate | Macro-F1 | Mean Latency (s) | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `pubmedqa_provided_context` | 2 | 0.0% [0.0%, 65.8%] | 0.0% | 100.0% | 0.00 | 0.00s | Validated Smoke Suite |
| `pubmedqa_retrieval` | 2 | 0.0% [0.0%, 65.8%] | 0.0% | 100.0% | 0.00 | 0.44s | Validated Smoke Suite |

*Note: Full PQA-L expert test set evaluation (N=500) is designated as `[Pending Cloud GPU Run]`.*
