import json
import unittest

from spar_solution.src.spar_baseline.paperdoc import validate_paper_doc
from spar_solution.src.spar_baseline.providers.base import ProviderError
from spar_solution.src.spar_baseline.providers.bohrium import (
    BohriumProvider,
    CONTENT_PATH,
    SEARCH_PATH,
)


class FakeTransport:
    def __init__(self, response, status=200):
        self.response = response
        self.status = status
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return self.status, self.response


class BohriumProviderTests(unittest.TestCase):
    def test_search_posts_expected_payload_and_maps_paperdoc(self):
        transport = FakeTransport(
            {
                "code": 0,
                "message": "ok",
                "data": [
                    {
                        "paperId": "bohr-1",
                        "title": "WiFi-based Contactless Heart Rate Monitoring",
                        "authors": [{"name": "Alice"}, {"name": "Bob"}],
                        "publicationYear": "2024",
                        "abstract": "Heart rate is recovered from WiFi CSI.",
                        "doi": "https://doi.org/10.1000/wifi.hr",
                        "journal": "IEEE Sensors Journal",
                        "keywords": ["WiFi CSI", "heart rate"],
                        "url": "https://example.test/paper",
                    }
                ],
                "total": 1,
            }
        )
        provider = BohriumProvider("test-secret", transport=transport)

        result = provider.search(
            "WiFi heart rate monitoring",
            words=["WiFi CSI", "heart rate"],
            page_size=10,
        )

        self.assertEqual(result.operation, "search")
        self.assertEqual(result.total, 1)
        self.assertEqual(len(result.records), 1)
        paper = result.records[0]
        validate_paper_doc(paper)
        self.assertEqual(paper["paper_id"], "doi:10.1000/wifi.hr")
        self.assertEqual(paper["bibliography"]["authors"], ["Alice", "Bob"])
        self.assertEqual(paper["bibliography"]["year"], 2024)
        self.assertEqual(paper["access"]["full_text_status"], "abstract")
        self.assertEqual(paper["provenance"]["sources"], ["bohrium"])

        call = transport.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertTrue(call["url"].endswith(SEARCH_PATH))
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-secret")
        self.assertEqual(
            call["payload"],
            {
                "words": ["WiFi CSI", "heart rate"],
                "question": "WiFi heart rate monitoring",
                "type": 0,
                "pageSize": 10,
            },
        )
        self.assertNotIn("test-secret", result.to_dict().__repr__())

    def test_nested_data_list_and_explicit_next_cursor(self):
        transport = FakeTransport(
            {
                "code": "0",
                "data": {
                    "papers": [{"id": "p-2", "paperTitle": "RF Vital Sign Sensing"}],
                    "totalCount": "23",
                    "nextPage": 3,
                },
            }
        )
        result = BohriumProvider("key", transport=transport).search("RF sensing", cursor="2")
        self.assertEqual(result.total, 23)
        self.assertEqual(result.next_cursor, "3")
        self.assertEqual(result.records[0]["identifiers"]["unique_id"], "p-2")
        self.assertEqual(transport.calls[0]["payload"]["page"], 2)

    def test_batch_content_maps_fulltext_without_putting_text_in_agent_record(self):
        transport = FakeTransport(
            {
                "code": 0,
                "data": [
                    {
                        "paperId": "bohr-1",
                        "title": "WiFi Heart Rate",
                        "content": "Introduction\nA contactless WiFi heart-rate method.",
                    }
                ],
            }
        )
        result = BohriumProvider("key", transport=transport).read_many(["bohr-1"])
        paper = result.records[0]
        self.assertEqual(result.operation, "read")
        self.assertEqual(paper["access"]["full_text_status"], "fulltext")
        self.assertEqual(paper["status"]["evidence_status"], "fulltext")
        self.assertGreater(paper["content"]["char_count"], 0)
        self.assertEqual(len(paper["content"]["chunks"]), 1)
        self.assertNotIn("A contactless", json.dumps(paper, ensure_ascii=False))
        self.assertTrue(transport.calls[0]["url"].endswith(CONTENT_PATH))
        self.assertEqual(transport.calls[0]["payload"], {"paperIds": ["bohr-1"]})

    def test_missing_key_is_explicit_config_error_and_transport_is_not_called(self):
        transport = FakeTransport({"code": 0, "data": []})
        with self.assertRaises(ProviderError) as context:
            BohriumProvider("", transport=transport).search("wifi")
        self.assertEqual(context.exception.code, "config")
        self.assertEqual(transport.calls, [])

    def test_business_error_is_not_treated_as_empty_result(self):
        transport = FakeTransport({"code": 10003, "message": "request rejected", "data": []})
        with self.assertRaises(ProviderError) as context:
            BohriumProvider("key", transport=transport).search("wifi")
        self.assertEqual(context.exception.code, "business")
        self.assertEqual(context.exception.details, {"provider_code": 10003})

    def test_http_errors_are_classified_without_response_body_or_key(self):
        for status, expected, retryable in ((401, "auth", False), (429, "rate", True), (503, "http", True)):
            with self.subTest(status=status):
                with self.assertRaises(ProviderError) as context:
                    BohriumProvider("secret", transport=FakeTransport({}, status=status)).search("wifi")
                error = context.exception
                self.assertEqual(error.code, expected)
                self.assertEqual(error.status_code, status)
                self.assertEqual(error.retryable, retryable)
                self.assertNotIn("secret", str(error))

    def test_invalid_json_and_schema_are_parse_errors(self):
        bad_responses = [
            "not-json",
            {"message": "no code", "data": []},
            {"code": 0},
            {"code": 0, "data": ["not an object"]},
            {"code": 0, "data": [{}]},
        ]
        for response in bad_responses:
            with self.subTest(response=response):
                with self.assertRaises(ProviderError) as context:
                    BohriumProvider("key", transport=FakeTransport(response)).search("wifi")
                self.assertEqual(context.exception.code, "parse")

    def test_input_limits_are_validated_before_network(self):
        transport = FakeTransport({"code": 0, "data": []})
        provider = BohriumProvider("key", transport=transport)
        for kwargs in ({"page_size": 0}, {"page_size": 101}, {"cursor": "bad"}, {"words": []}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ProviderError) as context:
                    provider.search("wifi", **kwargs)
                self.assertEqual(context.exception.code, "config")
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
