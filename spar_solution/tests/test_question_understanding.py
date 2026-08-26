import unittest

from spar_solution.src.spar_baseline.question_understanding import collect_search_queries, parse_understanding


class ParseUnderstandingTests(unittest.TestCase):
    def test_survey_queries_come_first(self):
        parsed = parse_understanding(
            {
                "field": "contextual bandits",
                "queries": ["nonstationary bandit", "smooth contextual bandits"],
                "survey_queries": ["contextual bandits survey"],
                "confidence": 0.8,
            },
            "stationary distribution of rewards over contexts",
        )
        self.assertEqual(
            collect_search_queries(parsed),
            ["contextual bandits survey", "nonstationary bandit", "smooth contextual bandits"],
        )

    def test_invalid_payload_does_not_raise(self):
        parsed = parse_understanding(None, "wifi heart rate")
        self.assertEqual(parsed["field"], "")
        self.assertEqual(collect_search_queries(parsed), [])

    def test_reflect_keeps_draft_when_revision_omits_queries(self):
        draft = parse_understanding({"field": "wifi sensing", "queries": ["wifi csi heart rate"]}, "wifi")
        revised = parse_understanding({"doubts": ["maybe radar instead"], "revised": True}, "wifi", fallback=draft)
        self.assertEqual(revised["field"], "wifi sensing")
        self.assertEqual(revised["queries"], ["wifi csi heart rate"])
        self.assertEqual(revised["doubts"], ["maybe radar instead"])
        self.assertTrue(revised["revised"])

    def test_accepts_query_objects(self):
        parsed = parse_understanding(
            {"queries": [{"query_text": "graph attention anomaly detection"}], "survey_queries": [{"query": "anomaly detection survey"}]},
            "hybrid reconstruction",
        )
        self.assertEqual(
            collect_search_queries(parsed),
            ["anomaly detection survey", "graph attention anomaly detection"],
        )
