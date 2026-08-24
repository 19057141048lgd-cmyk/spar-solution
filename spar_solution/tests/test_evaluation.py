import unittest

from spar_solution.src.spar_baseline.metrics import (
    deduplicate_papers,
    evaluate_at_k,
    evaluate_modes,
    evaluate_queries,
)
from spar_solution.src.spar_baseline.mock_pipeline import _paper


def _doc(title, *, doi=None, arxiv_id=None, year=2024, author="A Author"):
    doc = _paper("fixture", "abstract")
    doc["paper_id"] = doi or arxiv_id or title
    doc["identifiers"].update({"doi": doi, "arxiv_id": arxiv_id})
    doc["bibliography"].update({"title": title, "year": year, "authors": [author]})
    return doc


class EvaluationTests(unittest.TestCase):
    def test_dedup_prefers_doi_and_arxiv_and_keeps_ambiguous(self):
        same_doi = _doc("One", doi="10.1/one")
        same_doi_2 = _doc("Different title", doi="https://doi.org/10.1/one")
        same_arxiv = _doc("Two", arxiv_id="arXiv:2401.00001v2")
        same_arxiv_2 = _doc("Two", arxiv_id="2401.00001")
        ambiguous = _doc("No metadata", doi=None, arxiv_id=None, year=None, author="")
        result = deduplicate_papers([same_doi, same_doi_2, same_arxiv, same_arxiv_2, ambiguous])
        self.assertEqual(result["duplicates_removed"], 2)
        self.assertEqual(len(result["records"]), 3)

    def test_tp_fp_fn_and_identity_trace(self):
        predictions = [_doc("A", doi="10.1/a"), _doc("wrong", doi="10.1/x")]
        gold = [_doc("A formal", doi="doi:10.1/a"), _doc("Missing", doi="10.1/m")]
        result = evaluate_at_k(predictions, gold, k=10)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["matches"][0]["matched_by"], "doi")

    def test_title_normalization_and_preprint_formal_duplicate(self):
        prediction = _doc("WiFi-Based Heart Rate Measurement!", year=2020, author="First Author", arxiv_id="1234.1v1")
        duplicate = _doc("wifi based heart rate measurement", year=2020, author="FIRST AUTHOR", arxiv_id="1234.1")
        gold = [_doc("WiFi-based heart rate measurement", year=2020, author="First Author", doi="10.1/formal")]
        result = evaluate_at_k([prediction, duplicate], gold, k=10)
        self.assertEqual(result["duplicates_removed"], 1)
        self.assertEqual(result["tp"], 1)

    def test_empty_gold_and_empty_prediction_are_safe(self):
        self.assertEqual(evaluate_at_k([], [], k=10)["f1"], 0.0)
        result = evaluate_at_k([_doc("Prediction", doi="10.1/p")], [], k=10)
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0.0, 0.0, 0.0))
        result = evaluate_at_k([], [_doc("Gold", doi="10.1/g")], k=10)
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0.0, 0.0, 0.0))

    def test_provider_error_is_audit_event_not_false_positive(self):
        result = evaluate_at_k([], [_doc("Gold", doi="10.1/g")], provider_errors=[{"code": "timeout"}])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
        self.assertEqual(result["source_errors_count"], 1)

    def test_batch_macro_micro_and_runtime_statistics(self):
        gold = {"q1": [_doc("A", doi="10.1/a")], "q2": [_doc("B", doi="10.1/b")]}
        runs = {
            "q1": {"papers": [_doc("A", doi="10.1/a"), _doc("A duplicate", doi="10.1/a")], "stats": {
                "providers": [{"source": "arxiv", "records": 2}], "source_errors": [{"code": "timeout"}]
            }},
            "q2": {"papers": [], "stats": {"providers": [{"source": "local", "records": 0}], "source_errors": []}},
        }
        result = evaluate_queries(runs, gold, run_metadata={"q1": {"latency_ms": 10, "api_calls": 1}, "q2": {"latency_ms": 30, "api_calls": 2}})
        self.assertAlmostEqual(result["macro_f1"]["10"], 0.5)
        self.assertAlmostEqual(result["micro_f1"]["10"], 2 / 3)
        self.assertEqual(result["average_latency_ms"], 20.0)
        self.assertEqual(result["api_call_count"], 3)
        self.assertEqual(result["source_return_counts"], {"arxiv": 2, "local": 0})
        self.assertEqual(result["dedup_count"], 1)
        self.assertEqual(result["source_errors_count"], 1)

    def test_dedup_count_prefers_search_layer_merge_count(self):
        gold = {"q": [_doc("A", doi="10.1/a")]}
        runs = {"q": {"papers": [_doc("A", doi="10.1/a")], "stats": {"dedup_count": 3}}}
        result = evaluate_queries(runs, gold)
        self.assertEqual(result["dedup_count"], 3)

    def test_four_modes_share_comparable_output(self):
        gold = {"q": [_doc("A", doi="10.1/a"), _doc("B", doi="10.1/b")]}
        modes = {
            "A_arxiv": {"q": [_doc("A", doi="10.1/a")]},
            "B_local": {"q": [_doc("B", doi="10.1/b")]},
            "C_fusion": {"q": [_doc("A", doi="10.1/a"), _doc("B", doi="10.1/b")]},
            "D_reranked": {"q": [_doc("B", doi="10.1/b"), _doc("A", doi="10.1/a")]},
        }
        output = evaluate_modes(modes, gold)
        self.assertEqual(set(output), set(modes))
        self.assertEqual(output["C_fusion"]["by_cutoff"]["10"]["f1"], 1.0)
        self.assertEqual(output["D_reranked"]["by_cutoff"]["10"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
