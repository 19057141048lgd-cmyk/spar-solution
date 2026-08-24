import unittest

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.paperdoc import (
    PaperDocValidationError,
    canonical_paper_key,
    merge_paper_docs,
    validate_paper_doc,
)


class PaperDocTests(unittest.TestCase):
    def test_valid_doc_and_canonical_doi_key(self):
        doc = _paper("test", "abstract")
        self.assertIs(validate_paper_doc(doc), doc)
        self.assertEqual(canonical_paper_key(doc), "doi:10 1234 mock paper")

    def test_canonical_key_normalizes_doi_and_marks_ambiguous(self):
        doc = _paper("test", "abstract")
        doc["identifiers"]["doi"] = "https://doi.org/10.1234/EXAMPLE"
        self.assertEqual(canonical_paper_key(doc), "doi:10 1234 example")
        doc["identifiers"]["doi"] = None
        doc["bibliography"].update({"title": "", "year": None, "authors": []})
        self.assertTrue(canonical_paper_key(doc).startswith("ambiguous:"))

    def test_rejects_stringified_arrays(self):
        doc = _paper("test", "abstract")
        doc["bibliography"]["authors"] = "Ada Lovelace"
        with self.assertRaises(PaperDocValidationError):
            validate_paper_doc(doc)

    def test_rejects_unknown_evidence_status(self):
        doc = _paper("test", "abstract")
        doc["status"]["evidence_status"] = "pdf_guess"
        with self.assertRaises(PaperDocValidationError):
            validate_paper_doc(doc)

    def test_merges_same_doi_without_losing_arrays(self):
        first = _paper("a", "short")
        second = _paper("b", "a longer abstract")
        second["bibliography"]["authors"].append("Grace Hopper")
        second["bibliography"]["fields"].append("machine learning")
        merged = merge_paper_docs(first, second)
        self.assertEqual(merged["bibliography"]["abstract"], "a longer abstract")
        self.assertEqual(merged["bibliography"]["authors"], ["Ada Lovelace", "Grace Hopper"])
        self.assertEqual(merged["provenance"]["sources"], ["a", "b", "merged"])
        self.assertIsInstance(merged["bibliography"]["authors"], list)


if __name__ == "__main__":
    unittest.main()
