import json
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from spar_solution.src.spar_baseline.final_output import build_final_selection, validate_final_selection
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


class FinalOutputTests(unittest.TestCase):
    def test_fixture_writes_valid_final_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            run_p2_fixture(output_dir=directory)
            payload = replay_p2(directory)
            self.assertIn("final_selection", payload)
            self.assertEqual(payload["final_selection"]["schema_version"], "spar.final.v1")
            self.assertTrue(payload["final_selection"]["results"])

    def test_zone_boundaries_and_excluded_papers(self):
        papers = [_scored("high", 0.6), _scored("partial", 0.3), _scored("reserve", 0.299), _scored("excluded", 0.9, hard_pass=False)]
        result = build_final_selection({"query": "q", "query_plan": {"query_id": "qid"}, "papers": papers, "citations": [], "manifest": {"cost": {}}})
        self.assertEqual([item["relevance_zone"] for item in result["results"]], ["high", "partial", "reserve"])
        self.assertNotIn("excluded", [item["paper_id"] for item in result["results"]])
        self.assertEqual(result["summary"]["excluded"], 1)

    def test_graph_keeps_related_node_outside_top_k(self):
        run = run_p2_fixture()
        result = build_final_selection(run, top_k=1)
        self.assertEqual(len(result["results"]), 1)
        self.assertGreater(len(result["relation_graph"]["edges"]), 0)
        self.assertTrue(any(node["outside_topk"] for node in result["relation_graph"]["nodes"]))

    def test_validation_rejects_invalid_zone(self):
        result = build_final_selection(run_p2_fixture(citation_enabled=False))
        invalid = deepcopy(result)
        invalid["results"][0]["relevance_zone"] = "unknown"
        with self.assertRaises(ValueError):
            validate_final_selection(invalid)

    def test_finalize_cli_rebuilds_file_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            run_p2_fixture(output_dir=directory)
            target = Path(directory) / "final_selection.json"
            target.unlink()
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["finalize", "--input", directory, "--top-k", "1"]), 0)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "spar.final.v1")
            self.assertEqual(len(payload["results"]), 1)


if __name__ == "__main__":
    unittest.main()
