import time
import unittest

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p2_recall import RecallRunner, SourceRouter
from spar_solution.src.spar_baseline.providers.base import ProviderError, ProviderResult


class _Provider:
    def __init__(self, name, *, delay=0, fail=False):
        self.name = name
        self.delay = delay
        self.fail = fail
        self.calls = []

    def search(self, query, *, page_size=10):
        self.calls.append((query, page_size))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise ProviderError(self.name, "rate", "temporary failure", retryable=True)
        paper = _paper(self.name, "abstract for " + query)
        paper["paper_id"] = self.name + ":" + query
        paper["identifiers"]["doi"] = "10.1234/" + self.name + "." + query.replace(" ", "-")
        paper["bibliography"]["title"] = query
        return ProviderResult(self.name, "search", [paper], total=1)


class RecallTests(unittest.TestCase):
    def test_router_honors_source_constraint_and_skips_unavailable(self):
        available = _Provider("arxiv")
        unavailable = _Provider("local_library")
        unavailable.library_status = "unavailable"
        router = SourceRouter({"arxiv": available, "local_library": unavailable})
        decisions = router.route_plan({"subqueries": [{"subquery_id": "s1", "query": "heart rate", "sources": ["arxiv", "local_library"]}]})
        self.assertEqual([(item.subquery_id, item.source) for item in decisions], [("s1", "arxiv")])

    def test_runner_is_bounded_and_keeps_plan_order(self):
        first = _Provider("first", delay=0.02)
        second = _Provider("second")
        runner = RecallRunner(SourceRouter([first, second]), max_workers=2, page_size=3)
        result = runner.run({"subqueries": [
            {"subquery_id": "s1", "query": "one", "sources": ["first"]},
            {"subquery_id": "s2", "query": "two", "sources": ["second"]},
        ]}, iteration=1)
        self.assertEqual([call["subquery_id"] for call in result.calls], ["s1", "s2"])
        self.assertEqual([paper["provenance"]["subquery_id"] for paper in result.records], ["s1", "s2"])
        self.assertEqual(result.stats["api_calls"], 2)
        self.assertEqual(first.calls[0][1], 3)

    def test_runner_accepts_query_plan_field_names_and_preserves_tree(self):
        provider = _Provider("arxiv")
        result = RecallRunner(SourceRouter([provider])).run({"subqueries": [{
            "subquery_id": "child", "parent_id": "root", "query_text": "wifi csi",
            "source_capabilities": ["arxiv"], "iteration": 1,
        }]})
        self.assertEqual(result.stats["api_calls"], 1)
        self.assertEqual(result.records[0]["provenance"]["subquery_id"], "child")
        self.assertEqual(result.records[0]["provenance"]["parent_node_id"], "root")
        self.assertEqual(result.records[0]["provenance"]["iteration"], 1)

    def test_runner_records_provider_error_without_fake_paper(self):
        broken = _Provider("broken", fail=True)
        result = RecallRunner(SourceRouter([broken])).run({"subqueries": [{"query": "wifi", "sources": ["broken"]}]})
        self.assertEqual(result.records, [])
        self.assertEqual(result.stats["source_errors"], 1)
        self.assertEqual(result.source_errors[0]["code"], "rate")

    def test_runner_respects_call_budget(self):
        one = _Provider("one")
        two = _Provider("two")
        result = RecallRunner(SourceRouter([one, two]), max_calls=1).run({"subqueries": [{"query": "a"}, {"query": "b"}]})
        self.assertEqual(result.stats["api_calls"], 1)
        self.assertEqual(len(one.calls) + len(two.calls), 1)


if __name__ == "__main__":
    unittest.main()
