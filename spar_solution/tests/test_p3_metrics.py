import unittest

from spar_solution.src.spar_baseline.p3_metrics import compare_communication, compare_p3_ablation, evaluate_p3_run, long_text_baseline
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

    def test_identity_and_communication_comparison(self):
        run = run_p3_fixture(citation_enabled=False)
        metrics = evaluate_p3_run(run, gold_ids=["10.1234/fixture.p3.seed"])
        self.assertEqual(metrics["tp"], 1)
        comparison = compare_communication(run, {"stats": {"message_bytes": 10000, "message_tokens_estimate": 2500}, "cost": {"llm_calls": 10, "total_tokens": 2500}})
        self.assertEqual(comparison["structured"]["message_bytes"], run.stats["protocol_message_bytes"])
        self.assertGreater(comparison["long_text"]["message_bytes"], comparison["structured"]["message_bytes"])
        self.assertGreater(evaluate_p3_run(run)["provider_calls"], 0)
        self.assertGreater(long_text_baseline(run)["stats"]["message_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
