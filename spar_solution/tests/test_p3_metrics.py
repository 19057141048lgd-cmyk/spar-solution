import unittest

from spar_solution.src.spar_baseline.p3_metrics import compare_p3_ablation, evaluate_p3_run
from spar_solution.src.spar_baseline.p3_pipeline import run_p3_fixture


class P3MetricsTests(unittest.TestCase):
    def test_fixture_metrics_and_empty_gold(self):
        run = run_p3_fixture(citation_enabled=True)
        metrics = evaluate_p3_run(run, gold_ids=["fixture:p3:seed"])
        self.assertEqual(metrics["tp"], 1)
        self.assertGreater(metrics["recall"], 0)
        empty = evaluate_p3_run(run, gold_ids=[])
        self.assertEqual(empty["precision"], 0.0)
        self.assertEqual(empty["recall"], 0.0)

    def test_ablation_has_citation_delta(self):
        enabled = run_p3_fixture(citation_enabled=True)
        disabled = run_p3_fixture(citation_enabled=False)
        comparison = compare_p3_ablation(enabled, disabled)
        self.assertEqual(comparison["delta"]["citation_edges"], 1)
        self.assertTrue(comparison["acceptance"]["citation_called"])


if __name__ == "__main__":
    unittest.main()
