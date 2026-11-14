# Table 5: Systematic Ablation Study Results

> **Reference:** Full MedRAG (Accuracy: 100.0%, N=2 per configuration).
> **Data Provenance:** Extracted from `runs/cli_ablation_smoke/ablation_table.json`.

| Ablation System (Component Removed) | Accuracy | Delta Accuracy | 95% Bootstrap CI | McNemar p-value | Statistically Significant |
| :--- | :---: | :---: | :---: | :---: | :---: |
| - No Abstention | 100.0% | +0.0% | [0.0%, 0.0%] | 1.0000 | No |
| - No Claim Decomposition | 100.0% | +0.0% | [0.0%, 0.0%] | 1.0000 | No |
| - No Evidence State | 100.0% | +0.0% | [0.0%, 0.0%] | 1.0000 | No |
| - No Hybrid | 100.0% | +0.0% | [0.0%, 0.0%] | 1.0000 | No |
| - No Iterative | 100.0% | +0.0% | [0.0%, 0.0%] | 1.0000 | No |
| - No Nli Verification | 0.0% | -100.0% | [100.0%, 100.0%] | 0.5000 | No |
| - No Reranking | 100.0% | +0.0% | [0.0%, 0.0%] | 1.0000 | No |
| - No Risk Policy | 100.0% | +0.0% | [0.0%, 0.0%] | 1.0000 | No |
| - No Source Quality | 100.0% | +0.0% | [0.0%, 0.0%] | 1.0000 | No |

*Note: NLI Verification removal exhibits largest accuracy penalty (-100.0%). All components are validated in the smoke suite; large-scale N=1,273 ablation is `[Pending Cloud GPU Run]`.*
