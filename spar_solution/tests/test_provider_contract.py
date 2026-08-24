import unittest

from spar_solution.src.spar_baseline.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderResult,
    ensure_result,
)


class FakeProvider:
    name = "fake"

    def search(self, query, **kwargs):
        return ProviderResult("fake", "search", [{"title": query}], total=1)

    def read(self, paper_id, **kwargs):
        return ProviderResult("fake", "read", [{"paper_id": paper_id, "content_ref": "x"}])

    def relations(self, paper_id, **kwargs):
        return ProviderResult("fake", "relations", [{"paper_id": paper_id, "relation": "references"}])


class ProviderContractTests(unittest.TestCase):
    def test_minimum_interface_is_structured(self):
        provider = FakeProvider()
        self.assertEqual(ensure_result(provider.search("wifi"), source="fake", operation="search").records[0]["title"], "wifi")
        self.assertEqual(provider.read("p").operation, "read")
        self.assertEqual(provider.relations("p").operation, "relations")

    def test_result_rejects_unstructured_records(self):
        with self.assertRaises(TypeError):
            ProviderResult("fake", "search", ["long natural language"])

    def test_error_is_serializable_and_distinct(self):
        error = ProviderError("openalex", "rate", "request limited", retryable=True, status_code=429)
        self.assertEqual(error.to_dict()["code"], "rate")
        self.assertTrue(error.to_dict()["retryable"])
        self.assertNotIn("relevance", error.to_dict())


if __name__ == "__main__":
    unittest.main()
