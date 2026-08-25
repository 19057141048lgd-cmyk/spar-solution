import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.p3_pipeline import replay_p3, run_p3_comparison_fixture, run_p3_fixture
from spar_solution.src.spar_baseline.p2_pipeline import FixtureProvider
from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p3_pipeline import P3Pipeline
from spar_solution.src.spar_baseline.query_planner import QueryPlanner
from spar_solution.src.spar_baseline.deepseek_layer import DeepSeekCallError, blend_relevance


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
            self.assertTrue(all(item["receiver"] == "orchestrator" for item in run.messages if item["type"] in {"STOP_DECISION", "FINAL_SELECTION"}))
            self.assertEqual(run.messages[-1]["type"], "FINAL_SELECTION")
            self.assertGreaterEqual(len(run.messages), 6)
            self.assertEqual(run.citation["stats"]["edges"], 1)
            self.assertEqual(len(run.selected), 2)
            self.assertTrue(all(item["scores"]["final"] is not None for item in run.selected))
            self.assertTrue((Path(temp) / "protocol.jsonl").is_file())
            self.assertTrue((Path(temp) / "arbiter" / "final_selection.json").is_file())
            self.assertEqual(run.rounds[0]["action"], "NEXT_QUERY")
            self.assertGreaterEqual(len(run.rounds), 2)
            replay = replay_p3(temp)
            self.assertEqual(replay["final_selection"]["schema_version"], "spar.final.v2")
            self.assertEqual(replay["manifest"]["query_id"], run.query_plan["query_id"])
            self.assertTrue(all((Path(temp) / ref).read_text(encoding="utf-8").startswith(f"paper_id: {row['paper_id']}\n") for row in run.final_selection["results"] for ref in row["evidence_refs"]))
            self.assertEqual(run.final_selection["cost"]["provider_calls"], run.cost["provider_calls"])

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
        # LLM 分(0.91)与词法分(1.0)按 blend_relevance 融合，而非原样覆盖。
        self.assertAlmostEqual(run.papers[0]["scores"]["relevance"], blend_relevance(0.91, 1.0))

    def test_judge_is_batched_and_final_selection_contains_graph(self):
        class BatchLayer(FakeUnderstandingLayer):
            def judge(self, plan, papers):
                result = super().judge(plan, papers)
                self.last_batch = len(papers)
                return result

        layer = BatchLayer()
        seed = _paper("arxiv", "WiFi CSI heart rate monitoring")
        seed["paper_id"] = "fixture:batch"
        provider = FixtureProvider("arxiv", [seed], {})
        with tempfile.TemporaryDirectory() as temp:
            run = P3Pipeline({"arxiv": provider}, citation_enabled=False, understanding_layer=layer).run("WiFi heart rate monitoring", output_dir=temp)
            self.assertEqual(layer.judge_calls, 1)
            self.assertEqual(run.final_selection["schema_version"], "spar.final.v2")
            self.assertIn("relation_graph", run.final_selection)

    def test_no_relevance_seed_does_not_expand_citations(self):
        with tempfile.TemporaryDirectory() as temp:
            run = run_p3_fixture(query="quantum chromodynamics", output_dir=temp, citation_enabled=True)
            self.assertEqual(run.citation["stats"]["edges"], 0)
            self.assertEqual(run.citation["stats"]["api_calls"], 0)

    def test_replay_reads_evidence_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            run_p3_fixture(output_dir=temp)
            evidence = next((Path(temp) / "evidence_judge").rglob("evidence_*.txt"))
            evidence.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                replay_p3(temp)

    def test_failed_conflict_review_is_not_marked_reviewed(self):
        class FailingReviewLayer(FakeUnderstandingLayer):
            def __init__(self):
                super().__init__()
                self._review = False

            def judge(self, plan, papers):
                if self._review:
                    raise DeepSeekCallError("network", "review unavailable")
                self._review = True
                return [{"paper_id": item["paper_id"], "relevance_score": 0.0, "relevance_label": "irrelevant", "hard_constraint_state": "pass", "reason": "fixture conflict", "evidence_needed": [], "confidence": 0.8} for item in papers]

        layer = FailingReviewLayer()
        seed = _paper("arxiv", "WiFi CSI heart rate monitoring")
        seed["paper_id"] = "fixture:conflict"
        provider = FixtureProvider("arxiv", [seed], {})
        with tempfile.TemporaryDirectory() as temp:
            run = P3Pipeline({"arxiv": provider}, citation_enabled=False, understanding_layer=layer).run("WiFi heart rate monitoring", output_dir=temp)
        self.assertTrue(any(item.get("stage") == "arbiter_review" for item in run.errors))
        self.assertFalse(any(item.get("conflict_reviewed") for item in run.verdicts))

    def test_p2_p3_and_communication_fixture_comparison_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            comparison = run_p3_comparison_fixture(output_dir=temp)
            self.assertIn("p2", comparison["p2_p3"]["runs"])
            self.assertIn("p3", comparison["p2_p3"]["runs"])
            self.assertGreater(comparison["communication"]["long_text"]["message_bytes"], comparison["communication"]["structured"]["message_bytes"])
            self.assertTrue((Path(temp) / "comparison.json").is_file())


if __name__ == "__main__":
    unittest.main()
