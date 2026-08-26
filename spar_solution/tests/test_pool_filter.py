import unittest

from spar_solution.src.spar_baseline.pool_filter import filter_papers


def _paper(paper_id, title, abstract=""):
    return {
        "paper_id": paper_id,
        "bibliography": {"title": title, "abstract": abstract, "year": 2020},
    }


class _FilterClient:
    def __init__(self, keep_ids):
        self.keep_ids = set(keep_ids)
        self.calls = 0

    def complete_json(self, system, user, max_tokens=400):
        self.calls += 1
        import json
        payload = json.loads(user)
        return {
            "results": [
                {"paper_id": item["paper_id"], "keep": item["paper_id"] in self.keep_ids, "reason": "fixture"}
                for item in payload.get("candidates") or []
            ]
        }


class PoolFilterTests(unittest.TestCase):
    def test_drops_wrong_field_keeps_in_field(self):
        bandit = _paper("p1", "A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity")
        sky = _paper("p2", "The Dark Energy Survey")
        kept, dropped = filter_papers(
            _FilterClient({"p1"}),
            {"field": "contextual bandits", "answer_looks_like": "bandit papers"},
            [bandit, sky],
        )
        self.assertEqual([paper["paper_id"] for paper in kept], ["p1"])
        self.assertEqual([paper["paper_id"] for paper in dropped], ["p2"])

    def test_failed_batch_is_kept(self):
        class Broken:
            def complete_json(self, system, user, max_tokens=400):
                raise RuntimeError("nope")

        paper = _paper("p3", "Contextual bandits with delayed rewards")
        kept, dropped = filter_papers(Broken(), {"field": "bandits"}, [paper])
        self.assertEqual([item["paper_id"] for item in kept], ["p3"])
        self.assertEqual(dropped, [])

    def test_missing_verdict_defaults_to_keep(self):
        class Empty:
            def complete_json(self, system, user, max_tokens=400):
                return {"results": []}

        paper = _paper("p4", "Smooth contextual bandits")
        kept, dropped = filter_papers(Empty(), {"field": "bandits"}, [paper])
        self.assertEqual([item["paper_id"] for item in kept], ["p4"])
        self.assertEqual(dropped, [])
