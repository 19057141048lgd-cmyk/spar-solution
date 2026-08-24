import json
import unittest
from urllib.parse import parse_qs, urlsplit

from spar_solution.src.spar_baseline.openalex_provider import (
    OpenAlexProvider,
    ProviderError,
    TransportResponse,
)
from spar_solution.src.spar_baseline.paperdoc import validate_paper_doc


class OpenAlexProviderTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.api_key = "test-key-never-print"

    def _transport(self, status, payload):
        def send(method, url, headers, timeout):
            self.calls.append(
                {"method": method, "url": url, "headers": headers, "timeout": timeout}
            )
            body = payload if isinstance(payload, str) else json.dumps(payload)
            return TransportResponse(status=status, body=body)

        return send

    def test_search_builds_expected_request_and_paperdoc(self):
        work = {
            "id": "https://openalex.org/W123456",
            "doi": "https://doi.org/10.1234/WIFI.HEART",
            "title": "Contactless WiFi Heart Rate Monitoring",
            "publication_year": 2024,
            "authorships": [
                {"author": {"display_name": "Ada Lovelace"}},
                {"author": {"display_name": "Grace Hopper"}},
            ],
            "abstract_inverted_index": {
                "WiFi": [0],
                "heart": [1],
                "rate": [2],
                "monitoring": [3],
            },
            "primary_location": {
                "landing_page_url": "https://example.org/work",
                "pdf_url": "https://example.org/work.pdf",
                "source": {"display_name": "Sensors"},
            },
            "open_access": {
                "is_oa": True,
                "oa_url": "https://example.org/open-access",
            },
            "topics": [{"display_name": "Wireless Sensing"}],
            "concepts": [{"display_name": "Vital Signs"}],
            "referenced_works": [
                "https://openalex.org/W10",
                "https://openalex.org/W10",
                "https://openalex.org/W11",
            ],
            "related_works": ["https://openalex.org/W12"],
            "relevance_score": 31.5,
        }
        provider = OpenAlexProvider(
            {
                "base_url": "https://api.openalex.org",
                "api_key": self.api_key,
            },
            transport=self._transport(200, {"meta": {"count": 1}, "results": [work]}),
        )

        result = provider.search("WiFi heart rate monitoring", per_page=5)

        self.assertEqual(result.operation, "search")
        self.assertEqual(result.total, 1)
        papers = result.records
        self.assertEqual(len(papers), 1)
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["method"], "GET")
        parsed = urlsplit(call["url"])
        self.assertEqual(parsed.path, "/works")
        params = parse_qs(parsed.query)
        self.assertEqual(params["search"], ["WiFi heart rate monitoring"])
        self.assertEqual(params["per_page"], ["5"])
        self.assertEqual(params["api_key"], [self.api_key])

        paper = papers[0]
        self.assertIs(validate_paper_doc(paper), paper)
        self.assertEqual(paper["paper_id"], "doi:10.1234/wifi.heart")
        self.assertEqual(paper["identifiers"]["openalex_id"], "W123456")
        self.assertEqual(paper["bibliography"]["abstract"], "WiFi heart rate monitoring")
        self.assertEqual(paper["bibliography"]["authors"], ["Ada Lovelace", "Grace Hopper"])
        self.assertEqual(paper["bibliography"]["fields"], ["Wireless Sensing", "Vital Signs"])
        self.assertEqual(paper["status"]["evidence_status"], "abstract")
        self.assertEqual(
            paper["relations"]["references"],
            [
                {"id": "W10", "relation_source": "openalex"},
                {"id": "W11", "relation_source": "openalex"},
            ],
        )
        self.assertEqual(paper["provenance"]["endpoints"], ["https://api.openalex.org/works"])
        self.assertNotIn(self.api_key, json.dumps(paper))

    def test_http_error_is_explicit_and_does_not_disclose_key(self):
        provider = OpenAlexProvider(
            {"api_key": self.api_key}, transport=self._transport(401, {"error": "invalid key"})
        )

        with self.assertRaises(ProviderError) as raised:
            provider.search("WiFi heart rate")

        self.assertEqual(raised.exception.source, "openalex")
        self.assertEqual(raised.exception.code, "auth")
        self.assertEqual(raised.exception.status_code, 401)
        self.assertNotIn(self.api_key, str(raised.exception))

    def test_invalid_json_raises_parse_error(self):
        provider = OpenAlexProvider({}, transport=self._transport(200, "not-json"))

        with self.assertRaises(ProviderError) as raised:
            provider.search("WiFi heart rate")

        self.assertEqual(raised.exception.code, "parse")

    def test_api_error_payload_raises_business_error(self):
        provider = OpenAlexProvider(
            {}, transport=self._transport(200, {"error": "temporary backend failure"})
        )

        with self.assertRaises(ProviderError) as raised:
            provider.search("WiFi heart rate")

        self.assertEqual(raised.exception.code, "business")

    def test_empty_results_are_not_silently_treated_as_success(self):
        provider = OpenAlexProvider({}, transport=self._transport(200, {"results": []}))

        with self.assertRaises(ProviderError) as raised:
            provider.search("WiFi heart rate")

        self.assertEqual(raised.exception.code, "empty")


if __name__ == "__main__":
    unittest.main()
