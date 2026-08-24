import json
import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.p2_pipeline import replay_p2, run_p2_fixture


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
            self.assertEqual(run.manifest["query_id"], payload["run_manifest"]["query_id"])
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
        self.assertEqual(sum(len(item["edges"]) for item in disabled.citations), 0)
        self.assertEqual(disabled.citations[0]["stats"]["ablation"], "citation_disabled")

    def test_errors_do_not_turn_into_fake_papers(self):
        run = run_p2_fixture()
        self.assertTrue(all("paper_id" in paper for paper in run.papers))
        self.assertTrue(all("relevance" in verdict["component_scores"] for verdict in run.verdicts))

    def test_replay_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run_p2_fixture(output_dir=directory)
            evidence = next((Path(directory) / "evidence").rglob("*.txt"))
            evidence.unlink()
            with self.assertRaises(ValueError):
                replay_p2(directory)


if __name__ == "__main__":
    unittest.main()
