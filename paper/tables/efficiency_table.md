# Table 7: Computational Efficiency & Latency Breakdown

> Hardware: Intel Core i7 / NVIDIA GeForce RTX 3050 Laptop GPU, Windows 11.
> Wall-clock timing measured from query ingestion to complete structured output.

| Architecture / Baseline | Mean Rounds | Mean Latency (s) | Prompt Tokens | Completion Tokens | Total Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Closed Book | 0.0 | 0.00s | 40 | 10 | 50 |
| Single Pass Dense Rag | 1.0 | 0.18s | 125 | 10 | 135 |
| Single Pass Bm25 Rag | 1.0 | 0.01s | 125 | 10 | 135 |
| Hybrid Rag | 1.0 | 0.41s | 125 | 10 | 135 |
| Hybrid Reranked Rag | 1.0 | 0.97s | 125 | 10 | 135 |
| Claim Verified Rag | 1.0 | 13.68s | 125 | 50 | 175 |
| Iterative Rag | 3.0 | 41.13s | 215 | 10 | 225 |
| Full Medrag | 3.0 | 39.94s | 215 | 50 | 265 |
