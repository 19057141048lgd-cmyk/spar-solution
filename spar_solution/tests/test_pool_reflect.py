import unittest

from spar_solution.src.spar_baseline.pool_reflect import parse_course_correction, reflect_on_pool


class PoolReflectTests(unittest.TestCase):
    def test_wrong_street_returns_new_queries_not_same_cluster(self):
        payload = {
            "verdict": "wrong_street",
            "why": "all titles are medical MRI reconstruction",
            "field": "time-series anomaly detection",
            "queries": ["time series anomaly detection autoencoder"],
            "survey_queries": ["deep anomaly detection survey"],
        }
        out = parse_course_correction(payload, "hybrid architectures in reconstruction-based techniques")
        self.assertEqual(out["verdict"], "wrong_street")
        self.assertEqual(out["queries"][0], "deep anomaly detection survey")
        self.assertIn("time series anomaly detection autoencoder", out["queries"])

    def test_on_track_emits_no_new_queries(self):
        out = parse_course_correction(
            {"verdict": "on_track", "queries": ["should not be used"], "survey_queries": ["nor this"]},
            "fp8 inference",
        )
        self.assertEqual(out["verdict"], "on_track")
        self.assertEqual(out["queries"], [])

    def test_failed_llm_does_not_block(self):
        class Broken:
            def complete_json(self, system, user, max_tokens=700):
                raise RuntimeError("nope")

        out = reflect_on_pool(Broken(), "q", {"field": "x"}, [{"bibliography": {"title": "A"}}])
        self.assertEqual(out["verdict"], "on_track")
        self.assertEqual(out["queries"], [])
