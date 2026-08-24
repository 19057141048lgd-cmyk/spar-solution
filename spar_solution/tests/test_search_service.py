import json
import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.providers.base import ProviderError, ProviderResult
from spar_solution.src.spar_baseline.search_service import search
from spar_solution.src.spar_baseline.cli import main


class _ResultProvider:
    name = "bohrium"

    def __init__(self, abstract: str):
        self.abstract = abstract

    def search(self, query, *, page_size=10):
        paper = _paper(self.name, self.abstract)
        paper["bibliography"]["title"] = query
        return ProviderResult(self.name, "search", [paper], total=1)


class _ListProvider:
    source = "openalex"

    def search(self, query, *, per_page=10):
        paper = _paper(self.source, "Longer evidence for " + query)
        paper["bibliography"]["title"] = query
        return [paper]


class _FailingProvider:
    name = "broken"

    def search(self, query, **kwargs):
        raise ProviderError(self.name, "rate", "test rate limit", retryable=True, status_code=429)


class _RecordsProvider:
    name = "records"

    def __init__(self, records):
        self.records = records

    def search(self, query, *, page_size=10):
        return ProviderResult(self.name, "search", self.records, total=len(self.records))


class SearchServiceTests(unittest.TestCase):
    def test_merges_provider_result_and_list_and_keeps_error(self):
        artifact = search(
            "WiFi heart rate monitoring",
            [_ResultProvider("short"), _ListProvider(), _FailingProvider()],
            mode="mock",
            run_id="run_test",
        )
        self.assertEqual(artifact["run_id"], "run_test")
        self.assertEqual(artifact["query_id"].startswith("q_"), True)
        self.assertEqual(artifact["papers"][0]["provenance"]["query_id"], artifact["query_id"])
        self.assertEqual(artifact["stats"]["input_records"], 2)
        self.assertEqual(artifact["stats"]["merged_records"], 1)
        self.assertEqual(artifact["source_errors"][0]["source"], "broken")
        self.assertEqual(artifact["source_errors"][0]["code"], "rate")
        self.assertIn("bohrium", artifact["papers"][0]["provenance"]["sources"])
        self.assertIn("openalex", artifact["papers"][0]["provenance"]["sources"])
        self.assertEqual(artifact["papers"][0]["bibliography"]["abstract"], "Longer evidence for WiFi heart rate monitoring")

    def test_cli_mock_writes_native_json_for_wifi_query(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wifi-heart-rate.json"
            self.assertEqual(
                main(["search", "--query", "WiFi heart rate monitoring", "--mock", "--output", str(output)]),
                0,
            )
            serialized = output.read_text(encoding="utf-8")
            artifact = json.loads(serialized)
        self.assertEqual(artifact["mode"], "mock")
        self.assertEqual(artifact["query"], "WiFi heart rate monitoring")
        self.assertIsInstance(artifact["papers"], list)
        self.assertIsInstance(artifact["source_errors"], list)
        self.assertNotIn("Found", serialized)

    def test_search_normalizes_doi_and_keeps_ambiguous_records_separate(self):
        first = _paper("records", "one")
        second = _paper("records", "two")
        first["identifiers"]["doi"] = "https://doi.org/10.1234/example"
        second["identifiers"]["doi"] = "doi:10.1234/example"
        ambiguous_a = _paper("records", "three")
        ambiguous_b = _paper("records", "four")
        for doc in (ambiguous_a, ambiguous_b):
            doc["identifiers"]["doi"] = None
            doc["bibliography"].update({"title": "", "year": None, "authors": []})
        result = search("wifi", [_RecordsProvider([first, second, ambiguous_a, ambiguous_b])])
        self.assertEqual(result["stats"]["merged_records"], 3)

    def test_provider_error_event_redacts_secret_text(self):
        class _SecretFailingProvider:
            name = "secret_source"

            def search(self, query, **kwargs):
                raise ProviderError(self.name, "network", "token=super-secret-value", details={"Authorization": "Bearer super-secret-value"})

        result = search("wifi", [_SecretFailingProvider()])
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("super-secret-value", serialized)
        self.assertIn("***", serialized)

    def test_invalid_record_is_not_counted_as_deduplication(self):
        result = search("wifi", [_RecordsProvider(["invalid-record"])])
        self.assertTrue(result["source_errors"])
        self.assertEqual(result["stats"]["dedup_count"], 0)


if __name__ == "__main__":
    unittest.main()
