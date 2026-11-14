"""
Regression tests for Part E: Document Validation.
Covers:
- Validation of document id, text length, title, source, publication year.
- URL and DOI format validation.
- Document type normalization.
- Source quality score bounds and default handling.
- Corpus validation with duplicate id detection.
"""
import unittest

from medrag.document_validator import validate_document, validate_corpus


class TestDocumentValidation(unittest.TestCase):

    def test_valid_document_parsing(self):
        raw = {
            "id": "pmid_12345",
            "title": "Clinical Trial on GLP-1",
            "text": "This is a detailed clinical trial abstract describing the safety and efficacy of the intervention.",
            "source": "New England Journal of Medicine",
            "year": 2024,
            "quality_score": 0.95,
            "doi": "10.1056/NEJMoa123456",
            "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa123456",
            "doc_type": "rct",
        }
        doc = validate_document(raw)
        self.assertEqual(doc.doc_id, "pmid_12345")
        self.assertEqual(doc.title, "Clinical Trial on GLP-1")
        self.assertEqual(doc.year, 2024)
        self.assertEqual(doc.quality_score, 0.95)
        self.assertEqual(doc.metadata["doi"], "10.1056/NEJMoa123456")
        self.assertEqual(doc.metadata["doc_type"], "rct")
        self.assertFalse(doc.metadata["is_default_quality"])

    def test_empty_id_rejected(self):
        raw = {"id": "", "text": "Valid length text for medical evaluation."}
        with self.assertRaises(ValueError) as ctx:
            validate_document(raw)
        self.assertIn("Document id cannot be empty", str(ctx.exception))

    def test_short_text_rejected(self):
        raw = {"id": "1", "text": "Too short"}
        with self.assertRaises(ValueError) as ctx:
            validate_document(raw)
        err = str(ctx.exception)
        self.assertTrue("at least 15 characters" in err or "Document text too short" in err)

    def test_invalid_publication_year_rejected(self):
        raw = {"id": "1", "text": "Valid length text for medical evaluation.", "year": 1700}
        with self.assertRaises(ValueError) as ctx:
            validate_document(raw)
        self.assertIn("Publication year", str(ctx.exception))

    def test_invalid_doi_format_rejected(self):
        raw = {
            "id": "1",
            "text": "Valid length text for medical evaluation.",
            "doi": "invalid-doi-format",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_document(raw)
        self.assertIn("Invalid DOI format", str(ctx.exception))

    def test_invalid_url_format_rejected(self):
        raw = {
            "id": "1",
            "text": "Valid length text for medical evaluation.",
            "url": "not-a-valid-url",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_document(raw)
        self.assertIn("Invalid URL format", str(ctx.exception))

    def test_missing_quality_score_defaults_to_safe_value(self):
        raw = {"id": "1", "text": "Valid length text for medical evaluation."}
        doc = validate_document(raw)
        self.assertEqual(doc.quality_score, 0.50)
        self.assertTrue(doc.metadata["is_default_quality"])

    def test_corpus_duplicate_id_detection(self):
        items = [
            {"id": "doc1", "text": "Passage text number one for clinical evaluation."},
            {"id": "doc1", "text": "Duplicate passage text with the same document id."},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_corpus(items)
        self.assertIn("Duplicate document id 'doc1'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

