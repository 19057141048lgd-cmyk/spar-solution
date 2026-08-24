import json
import unittest

from spar_solution.src.spar_baseline.deepseek_layer import (
    DEEPSEEK_JUDGEMENT_SCHEMA,
    DEEPSEEK_PLAN_SCHEMA,
    DeepSeekCallError,
    DeepSeekClient,
    DeepSeekSchemaError,
    DeepSeekUnderstandingLayer,
    TransportResponse,
)
from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.query_plan import validate_query_plan


def _envelope(content):
    return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}


def _plan_response():
    return {
        "schema_version": DEEPSEEK_PLAN_SCHEMA,
        "topic": "WiFi CSI heart rate monitoring",
        "keywords": ["WiFi CSI", "heart rate", "contactless monitoring"],
        "synonyms": ["vital signs", "remote photoplethysmography"],
        "methods": ["channel state information"],
        "datasets": [],
        "tasks": ["heart rate estimation"],
        "time_range": {"start_year": 2018, "end_year": None},
        "hard_constraints": [{"name": "time_range", "operator": "between", "value": "2018:"}],
        "soft_constraints": [{"name": "evidence", "operator": "prefer", "value": "abstract_or_fulltext"}],
        "source_capabilities": ["arxiv", "openalex", "bohrium"],
        "queries": [
            {"kind": "topic", "query_text": '"WiFi" "heart rate"', "source_capabilities": ["arxiv", "openalex"]},
            {"kind": "method", "query_text": '"channel state information" heart rate', "source_capabilities": ["arxiv", "bohrium"]},
        ],
        "gaps": ["missing_dataset"],
    }


class FakeTransport:
    def __init__(self, contents, *, status=200):
        self.contents = list(contents)
        self.status = status
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body, timeout))
        content = self.contents.pop(0) if self.contents else {}
        return TransportResponse(self.status, json.dumps(_envelope(content)))


class SequentialTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, method, url, headers, body, timeout):
        self.calls += 1
        return self.responses.pop(0)


class DeepSeekLayerTests(unittest.TestCase):
    def test_plan_is_structured_and_contains_multiple_queries_constraints_time_sources(self):
        transport = FakeTransport([_plan_response()])
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))
        plan = layer.plan("How can WiFi measure heart rate without contact?")
        validate_query_plan(plan)
        self.assertEqual(plan["planner"], "deepseek")
        self.assertEqual(len(plan["subqueries"]), 2)
        self.assertEqual(plan["time_range"]["start_year"], 2018)
        self.assertEqual(plan["source_capabilities"], ["arxiv", "openalex", "bohrium"])
        self.assertEqual(plan["keywords"][0], "WiFi CSI")

    def test_plan_schema_failure_is_explicit(self):
        transport = FakeTransport([{"schema_version": DEEPSEEK_PLAN_SCHEMA}])
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))
        with self.assertRaises(DeepSeekSchemaError):
            layer.plan("WiFi heart rate")

    def test_compact_plan_is_safely_completed(self):
        transport = FakeTransport([{"queries": ["wifi CSI heart rate"]}])
        plan = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport)).plan("WiFi heart rate")
        self.assertEqual(plan["planner"], "deepseek")
        self.assertEqual(plan["subqueries"][0]["query_text"], "wifi CSI heart rate")

    def test_research_plan_aliases_are_accepted(self):
        transport = FakeTransport([{"objective": "WiFi heart rate", "search_queries": ["wifi CSI heart rate"], "research_questions": ["heart rate estimation"], "databases": ["arxiv"]}])
        plan = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport)).plan("WiFi heart rate")
        self.assertEqual(plan["topic"], "WiFi heart rate")
        self.assertEqual(plan["source_capabilities"], ["arxiv"])

    def test_compact_judgement_keeps_unknown_constraints(self):
        paper = _paper("arxiv", "WiFi heart rate")
        paper["paper_id"] = "arxiv:1234.0001"
        transport = FakeTransport([_plan_response(), {"results": [{"paper_id": paper["paper_id"], "score": 0.7}]}])
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))
        result = layer.judge(layer.plan("WiFi heart rate"), [paper])
        self.assertEqual(result[0]["hard_constraint_state"], "unknown")
        self.assertEqual(result[0]["relevance_score"], 0.7)

    def test_judgement_matches_each_paper_and_preserves_unknown(self):
        first = _paper("arxiv", "WiFi CSI heart rate estimation")
        first["paper_id"] = "arxiv:1234.0001"
        second = _paper("openalex", "Unrelated image classification")
        second["paper_id"] = "openalex:W123"
        response = {
            "schema_version": DEEPSEEK_JUDGEMENT_SCHEMA,
            "results": [
                {"paper_id": first["paper_id"], "relevance_score": 0.95, "relevance_label": "relevant", "hard_constraint_state": "pass", "reason": "The abstract directly evaluates WiFi CSI heart rate.", "evidence_needed": ["full_text"], "confidence": 0.9},
                {"paper_id": second["paper_id"], "relevance_score": 0.2, "relevance_label": "irrelevant", "hard_constraint_state": "unknown", "reason": "No WiFi or heart-rate evidence is present.", "evidence_needed": [], "confidence": 0.8},
            ],
        }
        transport = FakeTransport([_plan_response(), response])
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))
        plan = layer.plan("WiFi heart rate")
        results = layer.judge(plan, [first, second])
        self.assertEqual([item["paper_id"] for item in results], [first["paper_id"], second["paper_id"]])
        self.assertEqual(results[0]["hard_constraint_state"], "pass")
        self.assertEqual(results[1]["hard_constraint_state"], "unknown")

    def test_judgement_rejects_missing_or_duplicate_candidate(self):
        paper = _paper("arxiv", "WiFi heart rate")
        paper["paper_id"] = "arxiv:1234.0001"
        invalid = {"schema_version": DEEPSEEK_JUDGEMENT_SCHEMA, "results": [{"paper_id": "wrong", "relevance_score": 1, "relevance_label": "relevant", "hard_constraint_state": "pass", "reason": "x", "evidence_needed": [], "confidence": 1}]}
        transport = FakeTransport([_plan_response(), invalid])
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))
        with self.assertRaises(DeepSeekSchemaError):
            layer.judge(layer.plan("WiFi heart rate"), [paper])

    def test_judgement_rejects_missing_candidate_and_invalid_score(self):
        first = _paper("arxiv", "WiFi heart rate")
        first["paper_id"] = "arxiv:one"
        second = _paper("arxiv", "WiFi pulse")
        second["paper_id"] = "arxiv:two"
        missing = {"results": [{"paper_id": first["paper_id"], "relevance_score": 0.8}]}
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=FakeTransport([_plan_response(), missing])))
        with self.assertRaises(DeepSeekSchemaError):
            layer.judge(layer.plan("WiFi heart rate"), [first, second])
        invalid = {"results": [{"paper_id": first["paper_id"], "relevance_score": "bad", "relevance_label": "nonsense"}]}
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=FakeTransport([_plan_response(), invalid])))
        with self.assertRaises(DeepSeekSchemaError):
            layer.judge(layer.plan("WiFi heart rate"), [first])

    def test_missing_key_is_explicit_and_does_not_leak_secret(self):
        secret = "sk-test-not-to-leak-123456789"
        client = DeepSeekClient(api_key="", base_url="https://example.invalid")
        with self.assertRaises(DeepSeekCallError) as context:
            client.complete_json("system", "user")
        self.assertEqual(context.exception.code, "config")
        self.assertNotIn(secret, str(context.exception))

    def test_injected_transport_does_not_require_key_and_header_is_absent(self):
        transport = FakeTransport([_plan_response()])
        client = DeepSeekClient(api_key="", transport=transport)
        client.complete_json("system", "user")
        self.assertNotIn("Authorization", transport.calls[0][2])

    def test_http_error_is_status_only(self):
        transport = FakeTransport([{}], status=401)
        with self.assertRaises(DeepSeekCallError) as context:
            DeepSeekClient(api_key="secret", transport=transport).complete_json("system", "user")
        self.assertEqual(context.exception.status_code, 401)
        self.assertNotIn("secret", str(context.exception))

    def test_usage_accumulates_provider_token_counts(self):
        first = _envelope(_plan_response())
        first["usage"] = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
        second = _envelope({"ok": True})
        second["usage"] = {"prompt_tokens": 3, "completion_tokens": 2}
        transport = SequentialTransport([TransportResponse(200, json.dumps(first)), TransportResponse(200, json.dumps(second))])
        client = DeepSeekClient(transport=transport)
        client.complete_json("system", "user")
        client.complete_json("system", "user")
        self.assertEqual(client.usage["calls"], 2)
        self.assertEqual(client.usage["prompt_tokens"], 13)
        self.assertEqual(client.usage["completion_tokens"], 6)
        self.assertEqual(client.usage["total_tokens"], 19)
        self.assertEqual(client.usage["failures"], 0)

    def test_retryable_http_status_retries_once(self):
        ok = _envelope({"ok": True})
        ok["usage"] = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        transport = SequentialTransport([TransportResponse(429, "{}"), TransportResponse(200, json.dumps(ok))])
        client = DeepSeekClient(transport=transport)
        self.assertEqual(client.complete_json("system", "user"), {"ok": True})
        self.assertEqual(client.usage["calls"], 2)
        self.assertEqual(client.usage["failures"], 1)

    def test_call_budget_prevents_retry_from_exceeding_limit(self):
        transport = SequentialTransport([TransportResponse(500, "{}")])
        client = DeepSeekClient(transport=transport)
        client.reset_usage(max_calls=1)
        with self.assertRaises(DeepSeekCallError) as context:
            client.complete_json("system", "user")
        self.assertEqual(context.exception.code, "budget")
        self.assertEqual(client.usage["calls"], 1)

    def test_invalid_response_json_counts_as_failure(self):
        client = DeepSeekClient(transport=SequentialTransport([TransportResponse(200, "not-json")]))
        with self.assertRaises(DeepSeekCallError):
            client.complete_json("system", "user")
        self.assertEqual(client.usage["calls"], 1)
        self.assertEqual(client.usage["failures"], 1)


if __name__ == "__main__":
    unittest.main()
