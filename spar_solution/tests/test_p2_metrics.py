import unittest

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


if __name__ == "__main__":
    unittest.main()
