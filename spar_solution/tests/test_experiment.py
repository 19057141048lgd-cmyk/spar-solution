import json
import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.experiment import (
    MODE_SPECS,
    build_fixture_providers,
    compare_regressions,
    run_wifi_fixture,
)


class ExperimentTests(unittest.TestCase):
    def test_fixture_has_explicit_provider_status_and_four_modes(self):
        gold_path = Path(__file__).parents[1] / "gold" / "wifi_heart_rate.json"
        providers, _ = build_fixture_providers(gold_path)
        self.assertEqual(set(providers), {"arxiv", "local"})
        self.assertTrue(all(getattr(provider, "availability_status") == "mock" for provider in providers.values()))
        self.assertEqual(set(MODE_SPECS), {"A_arxiv", "B_local", "C_fusion", "D_reranked"})

    def test_fixture_writes_traceable_artifacts(self):
        gold_path = Path(__file__).parents[1] / "gold" / "wifi_heart_rate.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_wifi_fixture(temp_dir, gold_path=gold_path)
            self.assertFalse(result["metrics"]["acceptance"]["at_10"]["fusion_regression"])
            self.assertFalse(result["metrics"]["acceptance"]["at_10"]["rerank_regression"])
            query_dir = Path(temp_dir) / "wifi_hr_q1"
            required = {
                "query.json", "gold.json", "results_arxiv.json", "results_local.json",
                "results_fusion.json", "results_reranked.json", "metrics.json", "errors.json", "run_manifest.json",
            }
            self.assertTrue(required.issubset({path.name for path in query_dir.iterdir()}))
            self.assertTrue((Path(temp_dir) / "query.json").exists())
            self.assertTrue((Path(temp_dir) / "results_arxiv.json").exists())
            self.assertTrue((Path(temp_dir) / "errors.json").exists())
            fusion = json.loads((query_dir / "results_fusion.json").read_text(encoding="utf-8"))
            self.assertTrue(all(paper["schema_version"] == "paperdoc.v1" for paper in fusion["papers"]))
            self.assertEqual(fusion["stats"]["provider_status"], {"arxiv": "mock", "local_library": "mock"})
            reranked = json.loads((query_dir / "results_reranked.json").read_text(encoding="utf-8"))
            self.assertEqual(reranked["stats"]["api_calls"], 0)
            self.assertEqual(
                [paper["paper_id"] for paper in reranked["papers"]],
                [paper["paper_id"] for paper in fusion["papers"]],
            )
            local = json.loads((query_dir / "results_local.json").read_text(encoding="utf-8"))
            self.assertEqual(local["papers"][0]["provenance"].get("library_status"), "mock")

    def test_regression_rule_is_strict(self):
        metrics = {
            mode: {"by_cutoff": {"10": {"recall": 0.0, "f1": 0.0}}}
            for mode in MODE_SPECS
        }
        metrics["A_arxiv"]["by_cutoff"]["10"]["recall"] = 0.5
        metrics["C_fusion"]["by_cutoff"]["10"]["recall"] = 0.4
        metrics["C_fusion"]["by_cutoff"]["10"]["f1"] = 0.8
        metrics["D_reranked"]["by_cutoff"]["10"]["f1"] = 0.7
        result = compare_regressions(metrics)
        self.assertTrue(result["fusion_regression"])
        self.assertTrue(result["rerank_regression"])

    def test_does_not_compare_incomplete_mode_set(self):
        with self.assertRaises(ValueError):
            compare_regressions({"A_arxiv": {"by_cutoff": {"10": {"recall": 0, "f1": 0}}}})


if __name__ == "__main__":
    unittest.main()
