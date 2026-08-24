import unittest

from spar_solution.src.spar_baseline.mock_pipeline import run_mock


class MockPipelineTests(unittest.TestCase):
    def test_mock_pipeline_merges_and_records_provider_error(self):
        result = run_mock()
        self.assertTrue(result["ok"])
        self.assertEqual(result["stats"]["input_records"], 2)
        self.assertEqual(result["stats"]["merged_records"], 1)
        self.assertEqual(result["stats"]["source_errors"][0]["code"], "config_missing")
        self.assertEqual(result["papers"][0]["scores"]["relevance"], None)


if __name__ == "__main__":
    unittest.main()
