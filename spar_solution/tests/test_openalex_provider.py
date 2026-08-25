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
    def test_relation_api_cost_counts_resolve_and_reference_detail_calls(self):
        self.assertEqual(OpenAlexProvider.relation_api_cost("W123", "citations"), 1)
        self.assertEqual(OpenAlexProvider.relation_api_cost("W123", "references"), 2)
        self.assertEqual(OpenAlexProvider.relation_api_cost("W123", "all"), 3)
        self.assertEqual(OpenAlexProvider.relation_api_cost("10.1000/example", "all"), 4)

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

    def _sequence_transport(self, *responses):
        queue = list(responses)

        def send(method, url, headers, timeout):
            self.calls.append(
                {"method": method, "url": url, "headers": headers, "timeout": timeout}
            )
            status, payload = queue.pop(0)
            body = payload if isinstance(payload, str) else json.dumps(payload)
            return TransportResponse(status=status, body=body)

        return send

    @staticmethod
    def _work(work_id, title="Related work"):
        return {
            "id": f"https://openalex.org/{work_id}",
            "title": title,
            "publication_year": 2024,
            "authorships": [{"author": {"display_name": "Test Author"}}],
        }

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

    def test_empty_results_are_a_successful_no_results_response(self):
        provider = OpenAlexProvider({}, transport=self._transport(200, {"results": []}))

        result = provider.search("WiFi heart rate")

        self.assertTrue(result.ok)
        self.assertEqual(result.records, [])
        self.assertEqual(result.total, 0)
        self.assertIn("no_results", result.warnings)
        self.assertTrue(result.provenance["no_results"])

    def test_search_encodes_year_filters(self):
        provider = OpenAlexProvider({}, transport=self._transport(200, {"results": []}))

        provider.search("wireless sensing", start_year=2018, end_year=2021)
        provider.search("wireless sensing", start_year=2018)
        provider.search("wireless sensing", end_year=2021)

        filters = [parse_qs(urlsplit(call["url"]).query)["filter"][0] for call in self.calls]
        self.assertEqual(
            filters,
            ["publication_year:2018-2021", "publication_year:>=2018", "publication_year:<=2021"],
        )

    def test_search_rejects_invalid_year_filters(self):
        provider = OpenAlexProvider({}, transport=self._transport(200, {"results": []}))

        for kwargs in ({"start_year": 1899}, {"end_year": 2201}, {"start_year": 2022, "end_year": 2021}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ProviderError) as raised:
                provider.search("wireless sensing", **kwargs)
            self.assertEqual(raised.exception.code, "config")
        self.assertEqual(self.calls, [])

    def test_relations_citations_returns_valid_paperdocs(self):
        provider = OpenAlexProvider(
            {}, transport=self._transport(200, {"meta": {"count": 1}, "results": [self._work("W20")]})
        )

        result = provider.relations("openalex:W10", relation="citations", limit=4)

        self.assertEqual(result.operation, "relations")
        self.assertEqual(result.total, 1)
        self.assertEqual(result.provenance["api_calls"], 1)
        self.assertEqual(result.records[0]["relation_type"], "citations")
        self.assertEqual(result.records[0]["provenance"]["parent_node_id"], "openalex:W10")
        self.assertIs(validate_paper_doc(result.records[0]), result.records[0])
        params = parse_qs(urlsplit(self.calls[0]["url"]).query)
        self.assertEqual(params["filter"], ["cites:W10"])
        self.assertEqual(params["per_page"], ["4"])

    def test_relations_references_fetches_ids_then_details(self):
        provider = OpenAlexProvider(
            {},
            transport=self._sequence_transport(
                (200, {"id": "https://openalex.org/W10", "referenced_works": ["https://openalex.org/W1", "W2"]}),
                (200, {"results": [self._work("W1"), self._work("W2")]}),
            ),
        )

        result = provider.relations("W10", relation="references", page_size=2)

        self.assertEqual(result.total, 2)
        self.assertEqual(result.provenance["api_calls"], 2)
        self.assertTrue(all(record["relation_type"] == "references" for record in result.records))
        first = urlsplit(self.calls[0]["url"])
        self.assertEqual(first.path, "/works/W10")
        self.assertEqual(parse_qs(first.query)["select"], ["id,referenced_works"])
        second = parse_qs(urlsplit(self.calls[1]["url"]).query)
        self.assertEqual(second["filter"], ["openalex_id:W1|W2"])

    def test_relations_resolves_doi_before_citation_lookup(self):
        provider = OpenAlexProvider(
            {},
            transport=self._sequence_transport(
                (200, {"id": "https://openalex.org/W99"}),
                (200, {"results": [self._work("W100")]}),
            ),
        )

        result = provider.relations("doi:10.1234/example", relation="citations")

        self.assertEqual(result.provenance["seed_openalex_id"], "W99")
        self.assertEqual(result.provenance["api_calls"], 2)
        self.assertIn("/works/https://doi.org/10.1234/example", urlsplit(self.calls[0]["url"]).path)
        self.assertEqual(
            parse_qs(urlsplit(self.calls[1]["url"]).query)["filter"], ["cites:W99"]
        )

    def test_relations_all_preserves_partial_success_and_error(self):
        provider = OpenAlexProvider(
            {},
            transport=self._sequence_transport(
                (200, {"results": [self._work("W20")]}),
                (503, {"error": "temporary"}),
            ),
        )

        result = provider.relations("W10", relation="all")

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["relation_type"], "citations")
        self.assertIn("references_failed:network", result.warnings)
        self.assertEqual(result.provenance["api_calls"], 2)
        self.assertFalse(result.provenance["calls"][1]["ok"])
        self.assertEqual(result.provenance["source_errors"][0]["status_code"], 503)
        self.assertNotIn(self.api_key, json.dumps(result.provenance))

    def test_relations_empty_is_success(self):
        provider = OpenAlexProvider({}, transport=self._transport(200, {"results": []}))

        result = provider.relations("W10", relation="citations")

        self.assertTrue(result.ok)
        self.assertEqual(result.records, [])
        self.assertIn("no_results", result.warnings)

    def test_relations_all_raises_when_every_branch_fails(self):
        provider = OpenAlexProvider(
            {}, transport=self._sequence_transport((503, {}), (429, {}))
        )

        with self.assertRaises(ProviderError) as raised:
            provider.relations("W10", relation="all")

        self.assertEqual(raised.exception.code, "network")
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(len(raised.exception.details["source_errors"]), 2)


if __name__ == "__main__":
    unittest.main()


class ArxivSeedResolutionTests(unittest.TestCase):
    """arXiv 来源种子必须能做引用扩展（arxiv:xxx paper_id 解析成 W-id）。"""

    def test_relations_resolves_arxiv_paper_id_via_doi(self):
        provider = OpenAlexProvider({"OPENALEX_BASE_URL": "https://api.openalex.org"})
        state = {"step": 0}

        def transport(method, url, headers, timeout):
            state["step"] += 1
            if state["step"] == 1:  # arXiv DOI 解析（版本号必须已剥离）
                self.assertIn("10.48550/arxiv.2301.12345", url)
                self.assertNotIn("v2", url)
                return (200, json.dumps({"id": "https://openalex.org/W99"}))
            if state["step"] == 2:  # references 第一步：work 对象
                return (200, json.dumps({"id": "https://openalex.org/W99", "referenced_works": []}))
            return (200, json.dumps({"results": [], "meta": {"count": 0}}))

        provider.transport = transport
        result = provider.relations("arxiv:2301.12345v2", relation="references", page_size=5)
        self.assertEqual(result.ok, True)

    def test_relations_rejects_unknown_arxiv_paper_id(self):
        provider = OpenAlexProvider({"OPENALEX_BASE_URL": "https://api.openalex.org"})

        def transport(method, url, headers, timeout):
            return (200, json.dumps({}))

        provider.transport = transport
        with self.assertRaises(ProviderError):
            provider.relations("arxiv:not-a-valid-id", relation="references")
