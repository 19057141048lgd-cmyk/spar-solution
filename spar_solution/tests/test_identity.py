import copy
import unittest

from spar_solution.src.spar_baseline.identity import match_papers
from spar_solution.src.spar_baseline.mock_pipeline import _paper


def _doc() -> dict:
    doc = _paper("fixture", "abstract")
    doc["bibliography"].update({
        "title": "WiFi-Based Heart Rate Measurement: A Study",
        "authors": ["Yu Gu"],
        "year": 2019,
    })
    doc["identifiers"] = {
        "doi": None,
        "arxiv_id": None,
        "s2_id": None,
        "openalex_id": None,
        "pmid": None,
        "pmcid": None,
        "sciverse_doc_id": None,
        "unique_id": None,
    }
    return doc


class IdentityTests(unittest.TestCase):
    def test_matches_normalized_doi_before_other_fields(self):
        left, right = _doc(), _doc()
        left["identifiers"]["doi"] = "https://doi.org/10.1109/ABC.123"
        right["identifiers"]["doi"] = "doi:10.1109/abc.123"
        right["bibliography"]["title"] = "different title"
        result = match_papers(left, right)
        self.assertEqual((result["status"], result["matched_by"]), ("matched", "doi"))

    def test_matches_arxiv_id_without_version_suffix(self):
        left, right = _doc(), _doc()
        left["identifiers"]["arxiv_id"] = "arXiv:2401.01234v2"
        right["identifiers"]["arxiv_id"] = "https://arxiv.org/abs/2401.01234"
        self.assertEqual(match_papers(left, right)["matched_by"], "arxiv_id")

    def test_matches_openalex_stable_id(self):
        left, right = _doc(), _doc()
        left["identifiers"]["openalex_id"] = "https://openalex.org/W123"
        right["identifiers"]["openalex_id"] = "w123"
        self.assertEqual(match_papers(left, right)["matched_by"], "openalex_id")

    def test_matches_normalized_title_year_first_author(self):
        left, right = _doc(), _doc()
        right["bibliography"]["title"] = "wifi based heart rate measurement - a study"
        right["bibliography"]["authors"] = ["YU GU"]
        result = match_papers(left, right)
        self.assertEqual(result["matched_by"], "title_year_first_author")

    def test_preprint_and_formal_record_match_through_shared_arxiv_id(self):
        preprint, formal = _doc(), _doc()
        preprint["identifiers"]["arxiv_id"] = "2401.01234v1"
        formal["identifiers"]["arxiv_id"] = "2401.01234"
        formal["identifiers"]["doi"] = "10.1109/formal.2025"
        self.assertEqual(match_papers(preprint, formal)["matched_by"], "arxiv_id")

    def test_missing_fallback_metadata_is_ambiguous(self):
        left, right = _doc(), _doc()
        right = copy.deepcopy(right)
        right["bibliography"]["year"] = None
        self.assertEqual(match_papers(left, right)["status"], "ambiguous")

    def test_conflicting_doi_is_ambiguous_and_not_forced_to_title_match(self):
        left, right = _doc(), _doc()
        left["identifiers"]["doi"] = "10.1/one"
        right["identifiers"]["doi"] = "10.1/two"
        self.assertEqual(match_papers(left, right)["status"], "ambiguous")

    def test_conflicting_stable_ids_are_ambiguous(self):
        left, right = _doc(), _doc()
        left["identifiers"]["s2_id"] = "S2:one"
        right["identifiers"]["s2_id"] = "S2:two"
        self.assertEqual(match_papers(left, right)["status"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
