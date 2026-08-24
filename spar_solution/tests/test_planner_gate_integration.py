import unittest

from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p2_evidence import ConstraintGate
from spar_solution.src.spar_baseline.query_planner import QueryPlanner


class PlannerGateIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.planner = QueryPlanner()
        self.gate = ConstraintGate()

    @staticmethod
    def _paper_for_year(year):
        paper = _paper("mock", "WiFi heart rate monitoring")
        paper["bibliography"]["year"] = year
        return paper

    def test_between_range_is_inclusive_and_rejects_outside_years(self):
        plan = self.planner.plan("WiFi heart rate monitoring published between 2018 and 2021")
        self.assertEqual(plan["hard_constraints"][0]["value"], "2018-2021")
        self.assertEqual(self.gate.evaluate(plan, self._paper_for_year(2017)).state, "fail")
        self.assertEqual(self.gate.evaluate(plan, self._paper_for_year(2019)).state, "pass")
        self.assertEqual(self.gate.evaluate(plan, self._paper_for_year(2021)).state, "pass")
        self.assertEqual(self.gate.evaluate(plan, self._paper_for_year(2022)).state, "fail")

    def test_since_year_uses_inclusive_lower_bound(self):
        plan = self.planner.plan("WiFi heart rate monitoring since 2018")
        self.assertEqual(plan["hard_constraints"][0]["value"], ">=2018")
        self.assertEqual(self.gate.evaluate(plan, self._paper_for_year(2017)).state, "fail")
        self.assertEqual(self.gate.evaluate(plan, self._paper_for_year(2018)).state, "pass")

    def test_before_year_uses_inclusive_upper_bound(self):
        plan = self.planner.plan("WiFi heart rate monitoring before 2021")
        self.assertEqual(plan["hard_constraints"][0]["value"], "<=2021")
        self.assertEqual(self.gate.evaluate(plan, self._paper_for_year(2021)).state, "pass")
        self.assertEqual(self.gate.evaluate(plan, self._paper_for_year(2022)).state, "fail")

    def test_legacy_colon_formats_remain_replayable(self):
        paper = self._paper_for_year(2019)
        for value in ("2018:2021", "2018:", ":2021"):
            with self.subTest(value=value):
                plan = {"hard_constraints": [{"name": "time_range", "value": value}]}
                self.assertEqual(self.gate.evaluate(plan, paper).state, "pass")


if __name__ == "__main__":
    unittest.main()
