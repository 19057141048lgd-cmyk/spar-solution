"""DOI↔arXiv ID 等价的回归测试。

OpenAlex 只给 arXiv 预印本 DOI（10.48550/arxiv.<id>），AutoScholarQuery 的
Gold 是裸 arXiv ID。等价规则缺失时，已检回且排序靠前的 Gold 论文会被记为
FP（见 artifacts/p2/live-benchmark-q4-foundation）。
"""

import unittest

from spar_solution.src.spar_baseline.identity import arxiv_id_from_doi, match_papers
from spar_solution.src.spar_baseline.metrics import evaluate_at_k
from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.openalex_provider import OpenAlexProvider
from spar_solution.src.spar_baseline.paperdoc import canonical_paper_key


def _doc(paper_id: str, identifiers: dict, title: str = "A paper", year: int = 2020):
    doc = _paper("arxiv", "abstract")
    doc["paper_id"] = paper_id
    doc["identifiers"] = {
        "doi": None, "arxiv_id": None, "s2_id": None, "openalex_id": None,
        "pmid": None, "pmcid": None, "sciverse_doc_id": None, "unique_id": None,
    }
    doc["identifiers"].update(identifiers)
    doc["bibliography"]["title"] = title
    doc["bibliography"]["year"] = year
    doc["bibliography"]["authors"] = ["Ada Lovelace"]
    return doc


class ArxivDoiEquivalenceTests(unittest.TestCase):
    def test_arxiv_doi_derivation(self):
        self.assertEqual(arxiv_id_from_doi("10.48550/arxiv.2005.14165"), "2005.14165")
        self.assertEqual(arxiv_id_from_doi("10.48550/arxiv.2005.14165v2"), "2005.14165")
        self.assertEqual(arxiv_id_from_doi("https://doi.org/10.48550/ARXIV.1810.04805"), "1810.04805")
        self.assertIsNone(arxiv_id_from_doi("10.1109/globecom38437.2019.9014297"))
        self.assertIsNone(arxiv_id_from_doi(None))
        self.assertIsNone(arxiv_id_from_doi(""))

    def test_openalex_prediction_matches_bare_arxiv_gold(self):
        prediction = _doc("doi:10.48550/arxiv.2005.14165", {"doi": "10.48550/arxiv.2005.14165"})
        gold = _doc("arxiv:2005.14165", {"arxiv_id": "2005.14165"})
        result = match_papers(prediction, gold)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["matched_by"], "arxiv_doi_equivalence")

    def test_conflicting_arxiv_identity_is_ambiguous(self):
        prediction = _doc("doi:10.48550/arxiv.2005.14165", {"doi": "10.48550/arxiv.2005.14165"})
        gold = _doc("arxiv:1900.12345", {"arxiv_id": "1900.12345"})
        self.assertEqual(match_papers(prediction, gold)["status"], "ambiguous")

    def test_publisher_doi_still_wins_by_direct_doi_match(self):
        prediction = _doc("doi:10.1109/x.1", {"doi": "10.1109/x.1"})
        gold = _doc("doi:10.1109/x.1", {"doi": "10.1109/X.1"})
        result = match_papers(prediction, gold)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["matched_by"], "doi")

    def test_cross_source_dedup_key_is_unified(self):
        arxiv_sourced = _doc("arxiv:2005.14165", {"arxiv_id": "2005.14165"})
        openalex_sourced = _doc("doi:10.48550/arxiv.2005.14165", {"doi": "10.48550/arxiv.2005.14165"})
        self.assertEqual(canonical_paper_key(arxiv_sourced), canonical_paper_key(openalex_sourced))
        self.assertTrue(canonical_paper_key(openalex_sourced).startswith("arxiv_id:"))

    def test_publisher_doi_key_unchanged(self):
        doc = _doc("doi:10.1109/x.1", {"doi": "10.1109/x.1"})
        self.assertTrue(canonical_paper_key(doc).startswith("doi:"))

    def test_metrics_credit_tp_for_arxiv_doi_prediction(self):
        # Q4 的精确失败模式：预测只有 arXiv DOI，Gold 是裸 ID，必须计 TP。
        predictions = [
            _doc("doi:10.48550/arxiv.2005.14165", {"doi": "10.48550/arxiv.2005.14165"}, title="Language Models are Few-Shot Learners"),
            _doc("doi:10.48550/arxiv.1910.10683", {"doi": "10.48550/arxiv.1910.10683"}, title="Exploring the Limits of Transfer Learning"),
        ]
        gold = [
            {"paper_id": "2005.14165", "identifiers": {"arxiv_id": "2005.14165"}},
            {"paper_id": "1910.10683", "identifiers": {"arxiv_id": "1910.10683"}},
        ]
        result = evaluate_at_k(predictions, gold, k=10)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["recall"], 1.0)

    def test_openalex_provider_backfills_arxiv_id(self):
        provider = OpenAlexProvider({})
        record = {
            "id": "https://openalex.org/W3030163527",
            "doi": "https://doi.org/10.48550/arxiv.2005.14165",
            "title": "Language Models are Few-Shot Learners",
            "publication_year": 2020,
        }
        doc = provider._to_paper_doc(record, query_id="q_test", page=1, retrieved_at="2026-01-01T00:00:00Z")
        self.assertEqual(doc["identifiers"]["arxiv_id"], "2005.14165")
        self.assertEqual(doc["identifiers"]["doi"], "10.48550/arxiv.2005.14165")

    def test_openalex_publisher_doi_has_no_arxiv_id(self):
        provider = OpenAlexProvider({})
        record = {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1109/ccnc.2018.8319181",
            "title": "Design and implementation",
            "publication_year": 2018,
        }
        doc = provider._to_paper_doc(record, query_id="q_test", page=1, retrieved_at="2026-01-01T00:00:00Z")
        self.assertIsNone(doc["identifiers"]["arxiv_id"])


if __name__ == "__main__":
    unittest.main()
