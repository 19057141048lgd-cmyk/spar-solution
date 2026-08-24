import json
import unittest

from spar_solution.src.spar_baseline.providers.arxiv import ArxivProvider
from spar_solution.src.spar_baseline.providers.base import ProviderError


ATOM = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom' xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/' xmlns:arxiv='http://arxiv.org/schemas/atom'>
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <title>WiFi Heart Rate Monitoring</title>
    <summary>Contactless vital sign monitoring with CSI.</summary>
    <published>2024-01-10T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <link title='pdf' type='application/pdf' href='http://arxiv.org/pdf/2401.01234v2'/>
    <arxiv:doi>10.1234/example</arxiv:doi>
  </entry>
</feed>"""


class ArxivProviderTests(unittest.TestCase):
    def test_atom_transport_maps_paperdoc_and_query(self):
        calls = []

        def transport(method, url, headers, timeout):
            calls.append((method, url, headers))
            return 200, ATOM

        result = ArxivProvider(transport=transport).search("WiFi heart rate monitoring", page_size=5)
        self.assertEqual(result.source, "arxiv")
        self.assertEqual(result.total, 1)
        self.assertEqual(result.records[0]["identifiers"]["arxiv_id"], "2401.01234")
        self.assertEqual(result.records[0]["bibliography"]["year"], 2024)
        self.assertIn("search_query", calls[0][1])
        self.assertIn("all%3AWiFi+OR+all%3Aheart+OR+all%3Arate", calls[0][1])
        self.assertNotIn("all%3AWiFi+AND+all%3Aheart", calls[0][1])
        self.assertNotIn("Authorization", calls[0][2])

    def test_phrase_and_or_query_are_preserved_without_forced_and(self):
        calls = []

        def transport(method, url, headers, timeout):
            calls.append(url)
            return 200, ATOM

        ArxivProvider(transport=transport).search('"wifi heart rate" CSI OR vital', page_size=5)
        self.assertIn("all%3A%22wifi+heart+rate%22", calls[0])
        self.assertIn("all%3ACSI+OR+all%3Avital", calls[0])
        self.assertNotIn("all%3ACSI+AND+all%3Avital", calls[0])

    def test_year_filter_excludes_future_records_and_reaches_api(self):
        calls = []
        future = ATOM.replace("<published>2024-01-10", "<published>2026-01-10")

        def transport(method, url, headers, timeout):
            calls.append(url)
            return 200, future

        result = ArxivProvider(transport=transport).search("wifi", end_year=2025)
        self.assertEqual(result.records, [])
        self.assertEqual(result.total, 0)
        self.assertEqual(result.provenance["filtered_out"], 1)
        self.assertIn("submittedDate%3A%5B190001010000+TO+202512312359%5D", calls[0])

    def test_year_range_is_validated(self):
        with self.assertRaises(ProviderError):
            ArxivProvider(transport=lambda *args: (200, ATOM)).search("wifi", start_year=2025, end_year=2024)

    def test_bad_xml_is_parse_error(self):
        provider = ArxivProvider(transport=lambda *args: (200, "not xml"))
        with self.assertRaises(ProviderError) as raised:
            provider.search("wifi")
        self.assertEqual(raised.exception.code, "parse")

    def test_endpoint_redacts_query_secret(self):
        provider = ArxivProvider(
            base_url="https://export.arxiv.org/api/query?api_key=secret-value",
            transport=lambda *args: (200, ATOM),
        )
        doc = provider.search("wifi").records[0]
        self.assertNotIn("secret-value", doc["provenance"]["endpoints"][0])


if __name__ == "__main__":
    unittest.main()
