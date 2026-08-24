import json
import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.deepseek_layer import DeepSeekCallError, DeepSeekUnderstandingLayer, blend_relevance
from spar_solution.src.spar_baseline.p2_pipeline import FixtureProvider, P2Pipeline, replay_p2, run_p2_fixture
from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.query_planner import QueryPlanner


class FakeUnderstandingLayer:
    def __init__(self):
        self.judged_ids = []

    def plan(self, query):
        return QueryPlanner().plan(query)

    def judge(self, plan, papers):
        self.judged_ids.extend(item["paper_id"] for item in papers)
        return [{"paper_id": item["paper_id"], "relevance_score": 0.93, "relevance_label": "relevant", "hard_constraint_state": "pass", "reason": "fixture", "evidence_needed": [], "confidence": 0.9} for item in papers]


class FailingUnderstandingLayer:
    def plan(self, query):
        return QueryPlanner().plan(query)

    def judge(self, plan, papers):
        raise DeepSeekCallError("network", "fixture failure", retryable=True)


class FailingPlanningLayer:
    def plan(self, query):
        raise DeepSeekCallError("parse", "fixture plan failure")

    def judge(self, plan, papers):
        return []


class FakeMeteredClient:
    def reset_usage(self, *, max_calls=None):
        self.max_calls = max_calls
        self._usage = {"calls": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0}

    @property
    def usage(self):
        return dict(self._usage)

    def complete_json(self, system_prompt, user_prompt, *, max_tokens=1600):
        self._usage["calls"] += 1
        self._usage["prompt_tokens"] += 5
        self._usage["completion_tokens"] += 2
        self._usage["total_tokens"] += 7
        payload = json.loads(user_prompt)
        if payload["task"] == "decompose_query":
            return {"queries": ["WiFi heart rate monitoring"], "source_capabilities": ["arxiv"]}
        return {"results": [{"paper_id": item["paper_id"], "relevance_score": 0.8, "relevance_label": "relevant", "hard_constraint_state": "pass", "reason": "fixture", "evidence_needed": [], "confidence": 0.8} for item in payload["candidates"]]}


class P2PipelineTests(unittest.TestCase):
    def test_fixture_writes_and_replays_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = run_p2_fixture(output_dir=directory)
            payload = replay_p2(directory)
            self.assertEqual(payload["query_plan"]["schema_version"], "query_plan.v1")
            self.assertTrue(payload["recall"])
            self.assertTrue(payload["verdicts"])
            self.assertTrue(payload["stop"])
            self.assertEqual(payload["run_manifest"]["schema_version"], "p2_run.v1")
            self.assertEqual(payload["final_selection"]["schema_version"], "spar.final.v1")
            self.assertEqual(run.manifest["query_id"], payload["run_manifest"]["query_id"])
            self.assertEqual(run.manifest["iterations"], 2)
            self.assertEqual(len(payload["stop"]), 2)
            self.assertFalse(payload["stop"][0]["should_stop"])
            self.assertTrue(payload["stop"][1]["should_stop"])
            papers = payload["papers"]["papers"]
            self.assertTrue(papers)
            self.assertTrue(all(paper["scores"]["final"] is not None for paper in papers))
            self.assertTrue(all(paper["status"]["hard_constraints_pass"] is not None for paper in papers))
            self.assertTrue(all(paper["evidence_refs"] for paper in papers))
            self.assertTrue(all((Path(directory) / ref).is_file() for paper in papers for ref in paper["evidence_refs"]))
            self.assertTrue(all(paper["provenance"]["query_id"] == payload["query_plan"]["query_id"] for paper in papers))

    def test_citation_ablation_changes_fixture_and_calls_relations(self):
        enabled = run_p2_fixture(citation_enabled=True)
        disabled = run_p2_fixture(citation_enabled=False)
        self.assertGreaterEqual(sum(len(item["edges"]) for item in enabled.citations), 1)
        self.assertEqual(sum(item["stats"]["api_calls"] for item in enabled.citations), 1)
        self.assertEqual(sum(len(item["edges"]) for item in disabled.citations), 0)
        self.assertEqual(disabled.citations[0]["stats"]["ablation"], "citation_disabled")

    def test_errors_do_not_turn_into_fake_papers(self):
        run = run_p2_fixture()
        self.assertTrue(all("paper_id" in paper for paper in run.papers))
        self.assertTrue(all("relevance" in verdict["component_scores"] for verdict in run.verdicts))

    def test_two_iterations_do_not_rescore_existing_papers(self):
        run = run_p2_fixture(citation_enabled=True)
        verdict_ids = [verdict["paper_id"] for verdict in run.verdicts]
        self.assertEqual(run.manifest["iterations"], 2)
        self.assertEqual(len(verdict_ids), len(set(verdict_ids)))

    def test_replay_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run_p2_fixture(output_dir=directory)
            evidence = next((Path(directory) / "evidence").rglob("*.txt"))
            evidence.unlink()
            with self.assertRaises(ValueError):
                replay_p2(directory)

    def test_optional_deepseek_layer_overrides_relevance_without_replacing_provider(self):
        paper = _paper("arxiv", "WiFi CSI heart rate monitoring")
        paper["paper_id"] = "fixture:p2:deepseek"
        provider = FixtureProvider("arxiv", [paper], {})
        run = P2Pipeline({"arxiv": provider}, citation_enabled=False, understanding_layer=FakeUnderstandingLayer()).run("WiFi heart rate monitoring")
        self.assertTrue(run.papers)
        # LLM 分(0.93)与词法分(1.0)按 blend_relevance 融合，而非原样覆盖。
        self.assertAlmostEqual(run.papers[0]["scores"]["relevance"], blend_relevance(0.93, 1.0))

    def test_deepseek_judges_only_non_excluded_candidates(self):
        eligible = _paper("arxiv", "WiFi CSI heart rate monitoring")
        eligible["paper_id"] = "fixture:p2:eligible"
        eligible["identifiers"]["doi"] = "10.1234/p2.eligible"
        eligible["bibliography"]["year"] = 2025
        excluded = _paper("arxiv", "WiFi CSI heart rate monitoring")
        excluded["paper_id"] = "fixture:p2:excluded"
        excluded["identifiers"]["doi"] = "10.1234/p2.excluded"
        excluded["bibliography"]["year"] = 2020
        layer = FakeUnderstandingLayer()
        run = P2Pipeline({"arxiv": FixtureProvider("arxiv", [eligible, excluded], {})}, citation_enabled=False, understanding_layer=layer).run("WiFi heart rate monitoring since 2024")
        self.assertIn(eligible["paper_id"], layer.judged_ids)
        self.assertNotIn(excluded["paper_id"], layer.judged_ids)
        self.assertIsNone(next(item for item in run.papers if item["paper_id"] == excluded["paper_id"])["scores"]["final"])

    def test_citation_children_are_judged_after_relation_expansion(self):
        parent = _paper("arxiv", "WiFi CSI heart rate monitoring")
        parent["paper_id"] = "fixture:p2:parent"
        parent["identifiers"]["doi"] = "10.1234/p2.parent"
        child = _paper("arxiv", "WiFi contactless heart rate measurement")
        child["paper_id"] = "fixture:p2:child"
        child["identifiers"]["doi"] = "10.1234/p2.child"
        layer = FakeUnderstandingLayer()
        provider = FixtureProvider("arxiv", [parent], {parent["paper_id"]: [child]})
        run = P2Pipeline({"arxiv": provider}, citation_enabled=True, understanding_layer=layer).run("WiFi heart rate monitoring")
        self.assertIn(parent["paper_id"], layer.judged_ids)
        self.assertIn(child["paper_id"], layer.judged_ids)
        # 子论文摘要只覆盖 3/4 查询词，词法分 0.75，与 LLM 分 0.93 融合。
        self.assertAlmostEqual(next(item for item in run.papers if item["paper_id"] == child["paper_id"])["scores"]["relevance"], blend_relevance(0.93, 0.75))
        child_verdict = next(item for item in run.verdicts if item["paper_id"] == child["paper_id"])
        self.assertEqual(child_verdict["llm_judgement"]["reason"], "fixture")
        self.assertEqual(child_verdict["llm_judgement"]["confidence"], 0.9)

    def test_failed_judge_preserves_lexical_relevance_and_marks_warning(self):
        paper = _paper("arxiv", "WiFi CSI heart rate monitoring")
        paper["paper_id"] = "fixture:p2:judge-failure"
        provider = FixtureProvider("arxiv", [paper], {})
        run = P2Pipeline({"arxiv": provider}, citation_enabled=False, understanding_layer=FailingUnderstandingLayer()).run("WiFi heart rate monitoring")
        verdict = run.verdicts[0]
        self.assertGreater(verdict["component_scores"]["relevance"], 0)
        self.assertIn("llm_judge_unavailable", verdict["warnings"])
        self.assertEqual(run.errors[-1]["stage"], "judge")

    def test_failed_deepseek_plan_is_explicit_rules_fallback(self):
        paper = _paper("arxiv", "WiFi CSI heart rate monitoring")
        provider = FixtureProvider("arxiv", [paper], {})
        run = P2Pipeline({"arxiv": provider}, citation_enabled=False, understanding_layer=FailingPlanningLayer()).run("WiFi heart rate monitoring")
        self.assertEqual(run.manifest["planner_source"], "llm_fallback_rules")
        self.assertEqual(run.errors[0]["stage"], "plan")

    def test_fixture_manifest_has_complete_zero_llm_cost(self):
        run = run_p2_fixture(citation_enabled=False)
        self.assertEqual(run.cost["llm_calls"], 0)
        self.assertEqual(run.cost["total_tokens"], 0)
        self.assertIn("arxiv", run.cost["provider_calls"])
        self.assertEqual(run.manifest["cost"], run.cost)
        self.assertEqual(set(run.cost["per_stage_ms"]), {"plan", "recall", "judge", "citation", "evidence"})

    def test_metered_fake_client_cost_is_exact_in_manifest(self):
        paper = _paper("arxiv", "WiFi CSI heart rate monitoring")
        paper["paper_id"] = "fixture:p2:metered"
        client = FakeMeteredClient()
        layer = DeepSeekUnderstandingLayer(client)
        run = P2Pipeline({"arxiv": FixtureProvider("arxiv", [paper], {})}, citation_enabled=False, understanding_layer=layer).run("WiFi heart rate monitoring")
        self.assertEqual(run.manifest["planner_source"], "deepseek")
        self.assertEqual(run.cost["llm_calls"], 2)
        self.assertEqual(run.cost["prompt_tokens"], 10)
        self.assertEqual(run.cost["completion_tokens"], 4)
        self.assertEqual(run.cost["total_tokens"], 14)


if __name__ == "__main__":
    unittest.main()
