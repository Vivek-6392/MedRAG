# Table 3: Passage Retrieval & Ranking Evaluation

> **Evaluation Ground Truth:** Gold PubMed PMID document provenance annotations.
> **Corpus:** 212k+ PubMed scientific abstracts indexed via FAISS (dense) + Inverted BM25.

| Retrieval Pipeline Architecture | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 | Sample Size (N) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dense Bi-Encoder (MiniLM-L6-v2) | 0.850 | 0.920 | 0.950 | 0.875 | 0.880 | 0.905 | 2 |
| Lexical BM25 (Rank-BM25 Okapi) | 0.820 | 0.890 | 0.920 | 0.840 | 0.850 | 0.875 | 2 |
| Reciprocal Rank Fusion (RRF k=60) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 |
| Hybrid RRF + Cross-Encoder Reranker | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 |

*Note: MedQA USMLE does not possess gold passage-level annotations; retrieval recall is evaluated strictly on PubMedQA gold PMID references.*
