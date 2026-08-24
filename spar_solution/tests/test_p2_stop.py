import unittest

from spar_solution.src.spar_baseline.p2_stop import StopController


class P2StopTests(unittest.TestCase):
    def test_budget_has_strong_reason(self):
        decision = StopController(max_provider_calls=2).decide(
            iteration=0, citation_depth=0, provider_calls=2, provider_successes=1,
            new_unique_papers=[2], new_relevant_papers=3, subquery_coverage=0.1, evidence_coverage=0.1,
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.decision_type, "strong")
        self.assertEqual(decision.reason_code, "BUDGET_EXHAUSTED")
        self.assertIn("provider_calls", decision.measurements)

    def test_all_provider_failed_precedes_iteration(self):
        decision = StopController().decide(
            iteration=2, citation_depth=0, provider_calls=3, provider_successes=0,
            new_unique_papers=[0], new_relevant_papers=0, subquery_coverage=0, evidence_coverage=0,
        )
        self.assertEqual(decision.reason_code, "ALL_PROVIDER_FAILED")

    def test_no_new_papers_for_two_rounds(self):
        decision = StopController().decide(
            iteration=0, citation_depth=0, provider_calls=2, provider_successes=2,
            new_unique_papers=[3, 0, 0], new_relevant_papers=0, subquery_coverage=0, evidence_coverage=0,
        )
        self.assertEqual(decision.reason_code, "NO_NEW_PAPER_2_ROUNDS")

    def test_first_iteration_does_not_soft_stop(self):
        decision = StopController().decide(
            iteration=0, citation_depth=0, provider_calls=1, provider_successes=1,
            new_unique_papers=[1], new_relevant_papers=1, subquery_coverage=0.9, evidence_coverage=0.8,
        )
        self.assertFalse(decision.should_stop)
        self.assertEqual(decision.decision_type, "continue")

    def test_soft_stop_requires_two_conditions_after_first_iteration(self):
        decision = StopController(max_iterations=3).decide(
            iteration=1, citation_depth=0, provider_calls=1, provider_successes=1,
            new_unique_papers=[1, 1], new_relevant_papers=1, subquery_coverage=0.9, evidence_coverage=0.8,
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.decision_type, "soft")
        self.assertEqual(decision.reason_code, "LOW_GAIN_SUFFICIENT_COVERAGE")

    def test_citation_depth_is_audited_without_stopping_iteration_zero(self):
        decision = StopController(max_citation_depth=1).decide(
            iteration=0, citation_depth=1, provider_calls=1, provider_successes=1,
            new_unique_papers=[3], new_relevant_papers=3, subquery_coverage=1, evidence_coverage=1,
        )
        self.assertFalse(decision.should_stop)
        self.assertIn("MAX_CITATION_DEPTH", decision.triggered_conditions)

    def test_max_iteration_remains_a_strong_stop(self):
        decision = StopController(max_iterations=2, max_citation_depth=1).decide(
            iteration=1, citation_depth=1, provider_calls=1, provider_successes=1,
            new_unique_papers=[3, 1], new_relevant_papers=3, subquery_coverage=0, evidence_coverage=0,
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.decision_type, "strong")
        self.assertEqual(decision.reason_code, "MAX_ITERATION")
        self.assertIn("MAX_CITATION_DEPTH", decision.triggered_conditions)

    def test_continue_when_soft_conditions_insufficient(self):
        decision = StopController().decide(
            iteration=0, citation_depth=0, provider_calls=1, provider_successes=1,
            new_unique_papers=[1], new_relevant_papers=3, subquery_coverage=0.9, evidence_coverage=0.2,
        )
        self.assertFalse(decision.should_stop)
        self.assertEqual(decision.reason_code, "CONTINUE")


if __name__ == "__main__":
    unittest.main()
