import unittest

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p2_metrics import compare_citation_ablation, evaluate_p2_run
from spar_solution.src.spar_baseline.p2_pipeline import run_p2_fixture


class P2MetricsTests(unittest.TestCase):
    def test_metrics_have_retrieval_and_cost_fields(self):
        run = run_p2_fixture(citation_enabled=False)
        metrics = evaluate_p2_run(run, gold_ids=["fixture:seed"])
        self.assertEqual(metrics["by_cutoff"]["10"]["tp"], 1)
        self.assertEqual(metrics["by_cutoff"]["10"]["recall"], 1.0)
        self.assertIn("latency_ms", metrics["stats"])
        self.assertIn("citation_coverage", metrics)

    def test_ablation_reports_relations_are_enabled_only_when_requested(self):
        enabled = run_p2_fixture(citation_enabled=True)
        disabled = run_p2_fixture(citation_enabled=False)
        comparison = compare_citation_ablation(enabled, disabled)
        self.assertTrue(comparison["acceptance"]["citation_called"])
        self.assertGreater(comparison["delta"]["citation_edges"], 0)

    def test_matches_prefixed_prediction_to_bare_arxiv_gold(self):
        paper = _paper("arxiv", "abstract")
        paper["paper_id"] = "arxiv:2301.12345"
        paper["identifiers"]["doi"] = None
        paper["identifiers"]["arxiv_id"] = "2301.12345v2"
        metrics = evaluate_p2_run({"papers": [paper]}, gold_ids=["2301.12345"])
        self.assertEqual(metrics["by_cutoff"]["10"]["tp"], 1)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_matches_doi_gold_mapping_through_shared_identity_rules(self):
        paper = _paper("openalex", "abstract")
        paper["paper_id"] = "openalex:W123"
        paper["identifiers"]["doi"] = "https://doi.org/10.1000/WIFI.HR"
        metrics = evaluate_p2_run(
            {"papers": [paper]},
            gold_ids=[{"paper_id": "gold:one", "identifiers": {"doi": "10.1000/wifi.hr"}}],
        )
        self.assertEqual(metrics["by_cutoff"]["10"]["tp"], 1)

    def test_cost_is_forwarded_from_manifest(self):
        metrics = evaluate_p2_run({"papers": [], "manifest": {"cost": {"provider_calls": {"arxiv": 2}, "llm_calls": 3, "prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10, "llm_failures": 1, "wall_ms": 20, "per_stage_ms": {"judge": 5}}}})
        self.assertEqual(metrics["stats"]["provider_calls"], {"arxiv": 2})
        self.assertEqual(metrics["stats"]["llm_calls"], 3)
        self.assertEqual(metrics["stats"]["total_tokens"], 10)

    def test_relation_http_calls_are_counted_not_only_wrapper_calls(self):
        run = {"papers": [], "recall": [], "citations": [{"calls": [{"source": "openalex", "ok": True, "api_calls": 3}], "edges": [], "papers": []}]}
        metrics = evaluate_p2_run(run)
        self.assertEqual(metrics["stats"]["api_calls"], 3)
        self.assertEqual(metrics["stats"]["citation_api_calls"], 3)
        self.assertEqual(metrics["stats"]["successful_calls"], 3)

    def test_replay_run_manifest_cost_is_forwarded(self):
        metrics = evaluate_p2_run({"papers": [], "run_manifest": {"cost": {"total_tokens": 7, "provider_calls": {"arxiv": 1}}}})
        self.assertEqual(metrics["stats"]["total_tokens"], 7)
        self.assertEqual(metrics["stats"]["provider_calls"], {"arxiv": 1})


if __name__ == "__main__":
    unittest.main()
