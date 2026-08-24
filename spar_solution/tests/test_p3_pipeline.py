import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.p3_pipeline import run_p3_fixture
from spar_solution.src.spar_baseline.p2_pipeline import FixtureProvider
from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p3_pipeline import P3Pipeline
from spar_solution.src.spar_baseline.query_planner import QueryPlanner


class FakeUnderstandingLayer:
    def __init__(self):
        self.plan_calls = 0
        self.judge_calls = 0

    def plan(self, query):
        self.plan_calls += 1
        return QueryPlanner().plan(query)

    def judge(self, plan, papers):
        self.judge_calls += 1
        return [{
            "paper_id": item["paper_id"], "relevance_score": 0.91,
            "relevance_label": "relevant", "hard_constraint_state": "pass",
            "reason": "fixture", "evidence_needed": [], "confidence": 0.9,
        } for item in papers]


class P3PipelineTests(unittest.TestCase):
    def test_fixture_roundtable_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            run = run_p3_fixture(output_dir=temp)
            self.assertEqual([item["sender"] for item in run.messages], ["planner", "retriever", "citation_explorer", "evidence_judge", "arbiter", "arbiter"])
            self.assertEqual([item["type"] for item in run.messages], ["QUERY_PLAN", "RESULT_BATCH", "RELATION_BATCH", "EVIDENCE_VERDICT", "STOP_DECISION", "FINAL_SELECTION"])
            self.assertEqual(len(run.messages), 6)
            self.assertEqual(run.citation["stats"]["edges"], 1)
            self.assertEqual(len(run.selected), 2)
            self.assertTrue(all(item["scores"]["final"] is not None for item in run.selected))
            self.assertTrue((Path(temp) / "protocol.jsonl").is_file())
            self.assertTrue((Path(temp) / "arbiter" / "final_selection.json").is_file())

    def test_citation_ablation(self):
        with tempfile.TemporaryDirectory() as temp:
            run = run_p3_fixture(output_dir=temp, citation_enabled=False)
            self.assertFalse(run.citation["stats"]["enabled"])
            self.assertEqual(run.citation["stats"]["edges"], 0)
            self.assertEqual(len(run.papers), 1)

    def test_deepseek_layer_is_optional_front_and_judge(self):
        layer = FakeUnderstandingLayer()
        seed = _paper("arxiv", "WiFi CSI heart rate monitoring")
        seed["paper_id"] = "fixture:deepseek"
        provider = FixtureProvider("arxiv", [seed], {})
        with tempfile.TemporaryDirectory() as temp:
            run = P3Pipeline({"arxiv": provider}, citation_enabled=False, understanding_layer=layer).run("WiFi heart rate monitoring", output_dir=temp)
        self.assertEqual(layer.plan_calls, 1)
        self.assertEqual(layer.judge_calls, 1)
        self.assertAlmostEqual(run.papers[0]["scores"]["relevance"], 0.91)


if __name__ == "__main__":
    unittest.main()
