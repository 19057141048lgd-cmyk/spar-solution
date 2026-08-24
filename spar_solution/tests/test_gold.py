import copy
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.gold import GoldValidationError, load_gold, validate_gold


GOLD_PATH = Path(__file__).parents[1] / "gold" / "wifi_heart_rate.json"


class GoldTests(unittest.TestCase):
    def test_wifi_gold_is_provisional_and_contains_four_queries(self):
        gold = load_gold(GOLD_PATH)
        self.assertEqual(gold["annotation_status"], "provisional")
        self.assertEqual(len(gold["queries"]), 4)
        self.assertTrue(all(query["annotation_status"] == "provisional" for query in gold["queries"]))
        self.assertTrue(all(query["relevant_papers"] for query in gold["queries"]))

    def test_rejects_relevant_paper_without_stable_identifier(self):
        gold = load_gold(GOLD_PATH)
        invalid = copy.deepcopy(gold)
        invalid["queries"][0]["relevant_papers"][0]["identifiers"] = {}
        with self.assertRaises(GoldValidationError):
            validate_gold(invalid)

    def test_allows_explicit_empty_provisional_query_gold(self):
        gold = load_gold(GOLD_PATH)
        gold["queries"][0]["relevant_papers"] = []
        self.assertIs(validate_gold(gold), gold)


if __name__ == "__main__":
    unittest.main()
