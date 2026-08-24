import unittest

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p2_citation import CitationExpander
from spar_solution.src.spar_baseline.providers.base import ProviderError, ProviderResult


class _RelationProvider:
    name = "mock"

    def __init__(self, child):
        self.child = child
        self.calls = []

    def relations(self, paper_id, *, relation="all", page_size=10):
        self.calls.append((paper_id, relation, page_size))
        record = dict(self.child)
        record["relation"] = "references"
        return ProviderResult(self.name, "relations", [record], total=1)


class _ErrorRelationProvider:
    name = "mock"

    def relations(self, paper_id, **kwargs):
        raise ProviderError(self.name, "network", "relation unavailable")


class _CostedRelationProvider(_RelationProvider):
    name = "openalex"

    @staticmethod
    def relation_api_cost(paper_id, relation):
        return 2

    def relations(self, paper_id, *, relation="all", page_size=10):
        result = super().relations(paper_id, relation=relation, page_size=page_size)
        result.provenance["api_calls"] = 2
        return result


def _seed(*, eligible=True):
    paper = _paper("mock", "seed abstract")
    paper["paper_id"] = "seed"
    paper["identifiers"]["doi"] = "10.1234/seed"
    paper["status"]["hard_constraints_pass"] = eligible
    paper["scores"]["relevance"] = 0.9 if eligible else 0.2
    return paper


class CitationTests(unittest.TestCase):
    def test_expands_gated_seed_with_parent_edge(self):
        child = _paper("mock", "child abstract")
        child["paper_id"] = "child"
        child["identifiers"]["doi"] = "10.1234/child"
        # Relation providers may return compact records; full_doc is intentionally
        # accepted by the fixture adapter below as a PaperDoc payload.
        provider = _RelationProvider(child)
        result = CitationExpander({"mock": provider}).expand([_seed()])
        self.assertEqual(result.stats["api_calls"], 1)
        self.assertEqual(result.edges[0]["parent_paper_id"], "seed")
        self.assertEqual(result.edges[0]["child_paper_id"], "child")
        self.assertEqual(result.edges[0]["depth"], 1)
        self.assertEqual(provider.calls[0][0], "seed")

    def test_ineligible_seed_is_not_called(self):
        provider = _ErrorRelationProvider()
        result = CitationExpander({"mock": provider}).expand([_seed(eligible=False)])
        self.assertEqual(result.stats["eligible_seeds"], 0)
        self.assertEqual(result.stats["api_calls"], 0)
        self.assertEqual(result.source_errors, [])

    def test_disabled_citation_is_explicit_ablation(self):
        provider = _ErrorRelationProvider()
        result = CitationExpander({"mock": provider}, enabled=False).expand([_seed()])
        self.assertEqual(result.stats["ablation"], "citation_disabled")
        self.assertEqual(result.stats["api_calls"], 0)

    def test_relation_provider_failure_is_structured(self):
        result = CitationExpander({"mock": _ErrorRelationProvider()}).expand([_seed()])
        self.assertEqual(result.stats["source_errors"], 1)
        self.assertEqual(result.source_errors[0]["code"], "network")

    def test_provider_http_cost_respects_budget_and_uses_stable_openalex_id(self):
        child = _paper("openalex", "child abstract")
        child["paper_id"] = "openalex:W2"
        provider = _CostedRelationProvider(child)
        seed = _seed()
        seed["provenance"]["sources"] = ["openalex"]
        seed["identifiers"]["openalex_id"] = "W1"
        blocked = CitationExpander({"openalex": provider}, max_api_calls=1).expand([seed])
        self.assertEqual(blocked.stats["api_calls"], 0)
        self.assertEqual(blocked.stats["budget_skipped"], 1)
        result = CitationExpander({"openalex": provider}, max_api_calls=2).expand([seed])
        self.assertEqual(result.stats["api_calls"], 2)
        self.assertEqual(provider.calls[-1][0], "W1")


if __name__ == "__main__":
    unittest.main()
