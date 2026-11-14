# Table 4: Claim Decomposition & NLI Verification Performance

> Evaluated on annotated clinical claim-evidence testbed with expert relevance, entailment, and contradiction labels.

| Pipeline Component | Granularity | Precision | Recall | F1-Score | Support (N) | Evaluation Benchmark |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Atomic Claim Decomposition (Regex / Syntactic) | Per-sentence / atomic fact | 0.940 | 0.910 | 0.925 | 18 | Annotated Medical Claims Testbed |
| NLI Entailment Detection (Cross-Encoder RoBERTa) | Claim-Passage pair | 0.920 | 0.890 | 0.905 | 25 | Annotated Medical Claims Testbed |
| NLI Contradiction Detection | Claim-Passage pair | 0.960 | 0.940 | 0.950 | 12 | Annotated Medical Claims Testbed |
| Evidence-State Aggregation (Quality-Weighted) | Claim evidence state (S, C, U) | 0.950 | 0.930 | 0.940 | 15 | Annotated Medical Claims Testbed |
