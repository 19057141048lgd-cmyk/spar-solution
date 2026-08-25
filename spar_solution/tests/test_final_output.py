"""spar.final.v2 交付物测试：官方口径提交集合 + Recall@K 排序池。"""

import json
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from spar_solution.src.spar_baseline.final_output import (
    FINAL_SCHEMA,
    build_final_selection,
    validate_final_selection,
)
from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p2_cli import main
from spar_solution.src.spar_baseline.p2_pipeline import replay_p2, run_p2_fixture


def _scored(paper_id, score, *, hard_pass=True):
    paper = _paper("arxiv", "WiFi CSI heart-rate evidence")
    paper["paper_id"] = paper_id
    paper["identifiers"]["doi"] = f"10.1234/{paper_id}"
    paper["scores"].update({
        "relevance": score, "constraint": 1.0, "evidence": 0.55,
        "quality": 0.8, "citation": 0.0, "novelty": 0.5,
        "final": score, "confidence": 0.8,
    })
    paper["status"]["hard_constraints_pass"] = hard_pass
    return paper


def _tree_paper(paper_id, relevance):
    """搜索树风格论文：只有 relevance，没有 final（v1 会把它全部漏掉）。"""
    paper = _paper("arxiv", "Tree-style candidate abstract")
    paper["paper_id"] = paper_id
    paper["identifiers"]["doi"] = f"10.1234/tree.{paper_id}"
    paper["scores"].update({
        "relevance": relevance, "constraint": None, "evidence": None,
        "quality": None, "citation": None, "novelty": None,
        "final": None, "confidence": None,
    })
    return paper


class FinalOutputV2Tests(unittest.TestCase):
    def test_fixture_writes_valid_v2_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            run_p2_fixture(output_dir=directory)
            payload = replay_p2(directory)
            self.assertIn("final_selection", payload)
            self.assertEqual(payload["final_selection"]["schema_version"], FINAL_SCHEMA)

    def test_threshold_selects_set_not_fixed_topk(self):
        papers = [_scored("keep1", 0.9), _scored("keep2", 0.6), _scored("drop", 0.2), _scored("excluded", 0.95, hard_pass=False)]
        result = build_final_selection({"query": "q", "query_plan": {"query_id": "qid"}, "papers": papers, "citations": [], "manifest": {"cost": {}}})
        self.assertEqual([item["paper_id"] for item in result["results"]], ["keep1", "keep2"])
        self.assertEqual(result["selection_rule"]["mode"], "threshold")
        self.assertEqual(result["selection_rule"]["select_threshold"], 0.55)
        # ranked_pool 覆盖全部可评分论文（含被阈值排除的），供 Recall@K 计分。
        self.assertEqual([item["paper_id"] for item in result["ranked_pool"]][:3], ["keep1", "keep2", "drop"])
        self.assertEqual(result["summary"]["selected"], 2)
        self.assertEqual(result["summary"]["pool_size"], 3)

    def test_max_selected_cap(self):
        papers = [_scored(f"p{i}", 0.9 - 0.01 * i) for i in range(10)]
        result = build_final_selection({"query": "q", "query_plan": {"query_id": "qid"}, "papers": papers, "citations": [], "manifest": {"cost": {}}}, max_selected=3)
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(len(result["ranked_pool"]), 10)

    def test_tree_style_papers_use_relevance_basis(self):
        papers = [_tree_paper("t1", 0.8), _tree_paper("t2", 0.4)]
        result = build_final_selection({"query": "q", "query_plan": {"query_id": "qid"}, "papers": papers, "citations": [], "manifest": {"cost": {}}})
        self.assertEqual([item["paper_id"] for item in result["results"]], ["t1"])
        self.assertEqual(result["ranked_pool"][0]["score"], 0.8)

    def test_unscored_papers_stay_in_ranked_pool(self):
        # 末层引用捞回、未判分的论文不能从 Recall@K 排序池消失。
        papers = [_scored("scored", 0.9), _tree_paper("unscored", None)]
        papers[1]["scores"]["relevance"] = None
        result = build_final_selection({"query": "q", "query_plan": {"query_id": "qid"}, "papers": papers, "citations": [], "manifest": {"cost": {}}})
        self.assertEqual(result["ranked_pool"][0]["paper_id"], "scored")
        self.assertEqual(result["ranked_pool"][1]["paper_id"], "unscored")
        self.assertIsNone(result["ranked_pool"][1]["score"])
        self.assertEqual(len(result["results"]), 1)

    def test_legacy_top_k_mode(self):
        papers = [_scored(f"p{i}", 0.9) for i in range(5)]
        result = build_final_selection({"query": "q", "query_plan": {"query_id": "qid"}, "papers": papers, "citations": [], "manifest": {"cost": {}}}, top_k=2)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["selection_rule"]["mode"], "legacy_top_k")

    def test_zone_boundaries_and_excluded_papers(self):
        # partial 区间 [0.3,0.6) 中只有 >= 阈值 0.55 的才会入选。
        papers = [_scored("high", 0.6), _scored("partial", 0.56), _scored("below", 0.3), _scored("excluded", 0.9, hard_pass=False)]
        result = build_final_selection({"query": "q", "query_plan": {"query_id": "qid"}, "papers": papers, "citations": [], "manifest": {"cost": {}}})
        self.assertEqual([item["relevance_zone"] for item in result["results"]], ["high", "partial"])
        self.assertNotIn("excluded", [item["paper_id"] for item in result["results"]])
        self.assertEqual(result["summary"]["excluded"], 1)
        # 低于阈值的仍在排序池里供 Recall@K 计分。
        self.assertIn("below", [item["paper_id"] for item in result["ranked_pool"]])

    def test_graph_keeps_related_node_outside_selection(self):
        run = run_p2_fixture()
        result = build_final_selection(run, top_k=1)
        self.assertEqual(len(result["results"]), 1)
        self.assertGreater(len(result["relation_graph"]["edges"]), 0)
        self.assertTrue(any(node.get("outside_selection") or node.get("outside_topk") for node in result["relation_graph"]["nodes"]))

    def test_validation_rejects_invalid_zone_and_rule(self):
        result = build_final_selection(run_p2_fixture(citation_enabled=False))
        invalid = deepcopy(result)
        invalid["results"][0]["relevance_zone"] = "unknown"
        with self.assertRaises(ValueError):
            validate_final_selection(invalid)
        bad_rule = deepcopy(result)
        bad_rule["selection_rule"]["mode"] = "mystery"
        with self.assertRaises(ValueError):
            validate_final_selection(bad_rule)

    def test_validator_accepts_legacy_v1_artifacts(self):
        v1 = {"schema_version": "spar.final.v1", "results": [], "relation_graph": {"nodes": [], "edges": []}, "summary": {}, "cost": {}}
        self.assertEqual(validate_final_selection(v1)["schema_version"], "spar.final.v1")

    def test_finalize_cli_rebuilds_file_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            run_p2_fixture(output_dir=directory)
            target = Path(directory) / "final_selection.json"
            target.unlink()
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["finalize", "--input", directory, "--top-k", "1"]), 0)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], FINAL_SCHEMA)
            self.assertEqual(len(payload["results"]), 1)
            self.assertEqual(payload["selection_rule"]["mode"], "legacy_top_k")


if __name__ == "__main__":
    unittest.main()
