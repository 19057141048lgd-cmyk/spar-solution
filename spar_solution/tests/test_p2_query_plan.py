import json
import unittest

from spar_solution.src.spar_baseline.query_plan import QueryPlanValidationError, validate_query_plan
from spar_solution.src.spar_baseline.query_planner import QueryPlanner


class QueryPlanTests(unittest.TestCase):
    def setUp(self):
        self.planner = QueryPlanner()

    def test_autoscholar_question_is_cleaned_not_whole_question_and(self):
        plan = self.planner.plan(
            "Can you tell me some papers about hybrid architectures in reconstruction-based techniques?"
        )
        self.assertEqual(plan["schema_version"], "query_plan.v1")
        self.assertNotIn("can you tell me", plan["topic"])
        self.assertNotIn("papers", plan["topic"].split())
        self.assertTrue(plan["topic"])
        self.assertTrue(all("?" not in item["query_text"] for item in plan["subqueries"]))
        self.assertLess(len(plan["topic"].split()), 10)

    def test_structured_fields_and_explicit_time_constraint(self):
        plan = self.planner.plan(
            "Which transformer methods for medical image segmentation were published between 2019 and 2022?"
        )
        self.assertIn("transformer", plan["methods"])
        self.assertIn("segmentation", plan["tasks"][0])
        self.assertEqual(plan["time_range"]["start_year"], 2019)
        self.assertEqual(plan["time_range"]["end_year"], 2022)
        self.assertTrue(plan["hard_constraints"])
        self.assertEqual(plan["budget"]["max_iterations"], 2)
        validate_query_plan(plan)

    def test_plan_is_json_serializable_and_has_stable_ids(self):
        first = self.planner.plan("WiFi heart rate monitoring")
        second = self.planner.plan("WiFi heart rate monitoring")
        self.assertEqual(first["query_id"], second["query_id"])
        encoded = json.dumps(first, ensure_ascii=False)
        self.assertEqual(json.loads(encoded)["schema_version"], "query_plan.v1")

    def test_llm_json_is_injected_but_still_validated(self):
        deterministic = self.planner.plan("WiFi heart rate monitoring")
        plan = self.planner.plan("another question", llm_json=deterministic)
        self.assertEqual(plan["raw_query"], "another question")
        with self.assertRaises(QueryPlanValidationError):
            self.planner.plan("q", llm_json={"schema_version": "wrong"})

    def test_gap_iteration_is_bounded_and_structured(self):
        plan = self.planner.plan("WiFi heart rate monitoring")
        next_plan = self.planner.next_iteration(plan, gaps=["missing_method", "missing_dataset", "missing_time_range"])
        additions = next_plan["subqueries"][len(plan["subqueries"]):]
        self.assertEqual(len(additions), 3)
        self.assertTrue(all(item["iteration"] == 1 for item in additions))
        self.assertEqual(next_plan["subqueries"][-1]["parent_id"], plan["subqueries"][-1]["subquery_id"])
        bounded = self.planner.next_iteration(next_plan, gaps=["missing_method"])
        self.assertEqual(len(bounded["subqueries"]), len(next_plan["subqueries"]))

    def test_application_gap_uses_supported_subquery_kind(self):
        plan = self.planner.plan("WiFi heart rate monitoring")
        next_plan = self.planner.next_iteration(plan, gaps=["missing_application"])
        self.assertEqual(next_plan["subqueries"][-1]["kind"], "comparison")
        validate_query_plan(next_plan)

    def test_invalid_time_range_is_rejected(self):
        plan = self.planner.plan("WiFi heart rate monitoring")
        plan["time_range"] = {"start_year": 2024, "end_year": 2020}
        with self.assertRaises(QueryPlanValidationError):
            validate_query_plan(plan)


if __name__ == "__main__":
    unittest.main()
