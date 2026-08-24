import json
import tempfile
import unittest
from pathlib import Path

from spar_solution.src.spar_baseline.p3_protocol import AgentMessage, ArtifactStore, estimate_message, make_message, validate_message


class P3ProtocolTests(unittest.TestCase):
    def test_message_is_short_and_artifact_is_external(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ArtifactStore(temp)
            ref = store.put("planner", {"query": "long body stored once"}, name="plan")
            message = AgentMessage.create("run", "planner", "retriever", "plan_ready", ref, {"subqueries": 2})
            self.assertNotIn("long body stored once", json.dumps(message.to_dict()))
            self.assertGreater(estimate_message(message)["bytes"], 0)
            self.assertEqual(store.read(ref)["query"], "long body stored once")

    def test_long_body_rejected(self):
        with self.assertRaises(ValueError):
            AgentMessage.create("run", "planner", "retriever", "bad", "planner/plan.json", {"body": "x"})

    def test_strict_query_plan_message_validates(self):
        message = make_message(
            run_id="run1", message_id="msg1", message_type="QUERY_PLAN",
            sender="planner", receiver="retriever", seq=0,
            payload={"query_id": "q1", "subqueries": [{"subquery_id": "sq1", "kind": "topic", "query_text": "wifi heart rate", "source_capabilities": ["arxiv"]}]},
        )
        self.assertEqual(validate_message(message)["type"], "QUERY_PLAN")


if __name__ == "__main__":
    unittest.main()
