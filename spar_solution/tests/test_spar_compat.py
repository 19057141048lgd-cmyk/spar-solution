import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.spar_compat import inspect_spar_checkout


class SparCompatTests(unittest.TestCase):
    def test_known_upstream_findings_are_explicitly_isolated(self):
        report = inspect_spar_checkout(Path(__file__).parents[2] / "repos" / "SPAR")
        self.assertFalse(report["upstream_modified"])
        self.assertTrue(report["findings"])
        self.assertTrue(all(item["status"] in {"isolated", "not_found", "not_checked"} for item in report["findings"]))
        self.assertTrue(any(item["status"] == "isolated" for item in report["findings"]))


if __name__ == "__main__":
    unittest.main()
