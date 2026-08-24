import unittest

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p2_evidence import ConstraintGate, EvidenceLoader
from spar_solution.src.spar_baseline.p2_scoring import DEFAULT_WEIGHTS, Scorer


class P2ScoringTests(unittest.TestCase):
    def _inputs(self, *, full_text_status="abstract"):
        paper = _paper("mock", "WiFi CSI is used for heart rate monitoring", full_text_status=full_text_status)
        plan = {"raw_query": "WiFi heart rate monitoring", "topic": "WiFi heart rate", "methods": [], "datasets": [], "tasks": [], "hard_constraints": []}
        gate = ConstraintGate().evaluate(plan, paper)
        evidence = EvidenceLoader().load(paper, "abstract")
        return paper, plan, gate, evidence

    def test_default_weights_and_six_components_are_traceable(self):
        paper, plan, gate, evidence = self._inputs()
        verdict = Scorer().score(paper, plan, gate, evidence)
        self.assertEqual(verdict.weights, DEFAULT_WEIGHTS)
        self.assertEqual(set(verdict.component_scores), {"relevance", "constraint", "evidence", "quality", "citation", "novelty"})
        self.assertIsNotNone(verdict.final_score)
        self.assertTrue(verdict.evidence_refs)
        self.assertEqual(verdict.evidence_status, "abstract")

    def test_failed_constraint_excludes_without_fake_zero_relevance(self):
        paper, plan, _, evidence = self._inputs()
        plan["hard_constraints"] = [{"name": "year", "value": "1900"}]
        gate = ConstraintGate().evaluate(plan, paper)
        verdict = Scorer().score(paper, plan, gate, evidence)
        self.assertTrue(verdict.excluded)
        self.assertIsNone(verdict.final_score)
        self.assertEqual(verdict.component_scores["relevance"], 1.0)

    def test_provider_error_is_warning_not_relevance_penalty(self):
        paper, plan, gate, evidence = self._inputs()
        paper["status"]["provider_errors"] = [{"source": "openalex", "code": "timeout", "message": "test"}]
        verdict = Scorer().score(paper, plan, gate, evidence)
        self.assertIn("provider_error_preserved", verdict.warnings)
        self.assertGreater(verdict.component_scores["relevance"], 0)

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            Scorer({name: 1 / 6 for name in DEFAULT_WEIGHTS} | {"quality": 0.2})


if __name__ == "__main__":
    unittest.main()
