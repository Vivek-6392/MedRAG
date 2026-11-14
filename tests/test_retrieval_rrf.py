"""
Regression tests for Part B: Reciprocal Rank Fusion & Ranking Metadata.
Covers:
- Documents found by BM25 only.
- Documents found by Dense only.
- Documents found by both (verifying both bm25_rank and dense_rank are recorded).
- Component score preservation (bm25_score, dense_score).
- Deterministic score calculation and tie-breaking.
- Empty retrieval sets.
"""
import unittest

from medrag.retriever import Document, RetrievedDocument, reciprocal_rank_fusion


def _make_doc(doc_id: str, title: str = "") -> Document:
    return Document(
        doc_id=doc_id,
        title=title or f"Doc {doc_id}",
        text=f"Clinical passage content for {doc_id}.",
        source="Test Journal",
        year=2024,
    )


class TestReciprocalRankFusion(unittest.TestCase):

    def test_empty_retrieval_returns_empty_list(self):
        self.assertEqual(reciprocal_rank_fusion([], []), [])
        # Actual check with RetrievedDocument
        rd = RetrievedDocument(_make_doc("1"), score=0.9, dense_rank=1)
        res_bm25_empty = reciprocal_rank_fusion([], [rd])
        self.assertEqual(len(res_bm25_empty), 1)
        self.assertEqual(res_bm25_empty[0].document.doc_id, "1")

    def test_bm25_only_documents(self):
        doc1 = _make_doc("d1")
        doc2 = _make_doc("d2")
        bm25_hits = [
            RetrievedDocument(doc1, score=12.5, bm25_rank=1, bm25_score=12.5),
            RetrievedDocument(doc2, score=8.2, bm25_rank=2, bm25_score=8.2),
        ]
        fused = reciprocal_rank_fusion(bm25_hits, [], bm25_weight=0.4, k=60)

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].document.doc_id, "d1")
        self.assertEqual(fused[0].bm25_rank, 1)
        self.assertIsNone(fused[0].dense_rank)
        self.assertEqual(fused[0].bm25_score, 12.5)
        self.assertIsNone(fused[0].dense_score)
        self.assertAlmostEqual(fused[0].score, round(0.4 / (60 + 1), 6))

    def test_dense_only_documents(self):
        doc1 = _make_doc("d1")
        dense_hits = [
            RetrievedDocument(doc1, score=0.88, dense_rank=1, dense_score=0.88),
        ]
        fused = reciprocal_rank_fusion([], dense_hits, dense_weight=0.6, k=60)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].document.doc_id, "d1")
        self.assertIsNone(fused[0].bm25_rank)
        self.assertEqual(fused[0].dense_rank, 1)
        self.assertEqual(fused[0].dense_score, 0.88)
        self.assertAlmostEqual(fused[0].score, round(0.6 / (60 + 1), 6))

    def test_documents_found_by_both_preserve_both_ranks_and_scores(self):
        shared_doc = _make_doc("shared")
        bm25_only = _make_doc("bm25_only")
        dense_only = _make_doc("dense_only")

        bm25_hits = [
            RetrievedDocument(shared_doc, score=14.0, bm25_rank=1, bm25_score=14.0),
            RetrievedDocument(bm25_only, score=9.0, bm25_rank=2, bm25_score=9.0),
        ]
        dense_hits = [
            RetrievedDocument(dense_only, score=0.92, dense_rank=1, dense_score=0.92),
            RetrievedDocument(shared_doc, score=0.85, dense_rank=2, dense_score=0.85),
        ]

        fused = reciprocal_rank_fusion(
            bm25_hits, dense_hits, bm25_weight=0.4, dense_weight=0.6, k=60
        )

        self.assertEqual(len(fused), 3)

        # Shared doc should be top ranked because it accumulates both weights
        top = fused[0]
        self.assertEqual(top.document.doc_id, "shared")
        self.assertEqual(top.bm25_rank, 1)
        self.assertEqual(top.dense_rank, 2)
        self.assertEqual(top.bm25_score, 14.0)
        self.assertEqual(top.dense_score, 0.85)

        expected_score = round(0.4 / (60 + 1) + 0.6 / (60 + 2), 6)
        self.assertAlmostEqual(top.score, expected_score)

    def test_deterministic_tie_breaking(self):
        doc_a = _make_doc("doc_a")
        doc_b = _make_doc("doc_b")

        # Two documents with identical individual ranks on separate channels
        # giving identical RRF scores
        bm25_hits = [
            RetrievedDocument(doc_a, score=10.0, bm25_rank=1, bm25_score=10.0),
        ]
        dense_hits = [
            RetrievedDocument(doc_b, score=0.90, dense_rank=1, dense_score=0.90),
        ]

        # Equal weights 0.5 and 0.5
        run1 = reciprocal_rank_fusion(bm25_hits, dense_hits, bm25_weight=0.5, dense_weight=0.5, k=60)
        run2 = reciprocal_rank_fusion(bm25_hits, dense_hits, bm25_weight=0.5, dense_weight=0.5, k=60)

        # Both runs must produce identical order and scores
        self.assertEqual([r.document.doc_id for r in run1], [r.document.doc_id for r in run2])
        self.assertEqual([r.score for r in run1], [r.score for r in run2])


if __name__ == "__main__":
    unittest.main()

