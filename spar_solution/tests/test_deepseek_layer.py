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
from spar_solution.src.spar_baseline.query_planner import QueryPlanner


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

    def test_judgement_drops_unknown_candidate_with_issue(self):
        paper = _paper("arxiv", "WiFi heart rate")
        paper["paper_id"] = "arxiv:1234.0001"
        invalid = {"schema_version": DEEPSEEK_JUDGEMENT_SCHEMA, "results": [{"paper_id": "wrong", "relevance_score": 1, "relevance_label": "relevant", "hard_constraint_state": "pass", "reason": "x", "evidence_needed": [], "confidence": 1}]}
        transport = FakeTransport([_plan_response(), invalid])
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))
        results = layer.judge(layer.plan("WiFi heart rate"), [paper])
        self.assertEqual(results, [])
        self.assertTrue(any("unknown_paper_id" in issue for issue in layer.last_judge_issues))

    def test_judgement_partially_accepts_and_retries_missing_candidate(self):
        first = _paper("arxiv", "WiFi heart rate")
        first["paper_id"] = "arxiv:one"
        second = _paper("arxiv", "WiFi pulse")
        second["paper_id"] = "arxiv:two"
        missing = {"results": [{"paper_id": first["paper_id"], "relevance_score": 0.8}]}
        # 第二次调用（重试缺失候选）返回空对象 → 单篇放弃，第一条保留。
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=FakeTransport([_plan_response(), missing, {}])))
        results = layer.judge(layer.plan("WiFi heart rate"), [first, second])
        self.assertEqual([item["paper_id"] for item in results], [first["paper_id"]])
        self.assertTrue(any("arxiv:two" in issue for issue in layer.last_judge_issues))

    def test_judgement_coerces_freeform_label_and_state(self):
        first = _paper("arxiv", "WiFi heart rate")
        first["paper_id"] = "arxiv:one"
        lenient = {"results": [{"paper_id": first["paper_id"], "relevance_score": "bad", "relevance_label": "nonsense", "hard_constraint_state": "partially"}]}
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=FakeTransport([_plan_response(), lenient])))
        results = layer.judge(layer.plan("WiFi heart rate"), [first])
        # 分数是唯一必需信号：坏分数兜底 0.5，自由文本标签/状态按保守值折算。
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["relevance_score"], 0.5)
        self.assertEqual(results[0]["relevance_label"], "borderline")
        self.assertEqual(results[0]["hard_constraint_state"], "unknown")

    def test_judgement_drops_item_without_score(self):
        first = _paper("arxiv", "WiFi heart rate")
        first["paper_id"] = "arxiv:one"
        no_score = {"results": [{"paper_id": first["paper_id"], "relevance_label": "relevant"}]}
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=FakeTransport([_plan_response(), no_score, {}])))
        results = layer.judge(layer.plan("WiFi heart rate"), [first])
        self.assertEqual(results, [])
        self.assertTrue(any("invalid_item" in issue for issue in layer.last_judge_issues))

    def test_judgement_retries_with_halved_batch_on_total_failure(self):
        papers = []
        for index in range(4):
            paper = _paper("arxiv", f"WiFi heart rate study {index}")
            paper["paper_id"] = f"arxiv:paper{index}"
            papers.append(paper)
        # 首批 4 篇全部无效 → 减半为 2+2 重试；第二批 2 篇正常返回。
        good = {"results": [
            {"paper_id": "arxiv:paper0", "relevance_score": 0.9, "reason": "direct"},
            {"paper_id": "arxiv:paper1", "relevance_score": 0.1, "reason": "off topic"},
        ]}
        transport = FakeTransport([_plan_response(), {"results": []}, good, {}])
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))
        results = layer.judge(layer.plan("WiFi heart rate"), papers)
        self.assertEqual([item["paper_id"] for item in results], ["arxiv:paper0", "arxiv:paper1"])
        # 至少发生了减半重试：总请求数 > 单批直接成功的情况。
        self.assertGreaterEqual(len(transport.calls), 3)

    def test_judgement_output_budget_scales_with_batch_size(self):
        papers = []
        for index in range(3):
            paper = _paper("arxiv", f"WiFi heart rate study {index}")
            paper["paper_id"] = f"arxiv:paper{index}"
            papers.append(paper)
        ok = {"results": [{"paper_id": f"arxiv:paper{i}", "relevance_score": 0.5} for i in range(3)]}
        transport = FakeTransport([_plan_response(), ok])
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))
        layer.judge(layer.plan("WiFi heart rate"), papers)
        body = json.loads(transport.calls[-1][3])
        self.assertEqual(body["max_tokens"], min(8000, 150 * 3 + 400))

    def test_judgement_stops_on_llm_budget_exhaustion(self):
        paper = _paper("arxiv", "WiFi heart rate")
        paper["paper_id"] = "arxiv:1234.0001"

        class RateLimitTransport:
            def __init__(self):
                self.calls = 0

            def __call__(self, method, url, headers, body, timeout):
                self.calls += 1
                return TransportResponse(429, json.dumps({"error": {"message": "rate limited"}}))

        client = DeepSeekClient(api_key="k", transport=RateLimitTransport())
        client.reset_usage(max_calls=1)
        layer = DeepSeekUnderstandingLayer(client)
        plan = QueryPlanner().plan("WiFi heart rate")
        results = layer.judge(plan, [paper])
        self.assertEqual(results, [])
        self.assertTrue(any("budget_exhausted" in issue for issue in layer.last_judge_issues))


    def test_plan_tolerates_malformed_constraint_and_list_elements(self):
        # 真实失败案例：一条 name 为空的硬约束曾把整个计划打回规则规划器。
        plan_response = dict(_plan_response())
        plan_response["hard_constraints"] = [
            {"name": "", "operator": "between", "value": "2018-"},
            {"name": "time_range", "operator": "between", "value": "2018-2021"},
            "not-an-object",
        ]
        plan_response["keywords"] = ["foundation models", "", 42, "NLP"]
        plan_response["gaps"] = ["missing_dataset", "made_up_gap"]
        layer = DeepSeekUnderstandingLayer(DeepSeekClient(transport=FakeTransport([plan_response])))
        plan = layer.plan("foundation models for NLP")
        self.assertEqual(plan["hard_constraints"], [{"name": "time_range", "operator": "between", "value": "2018-2021"}])
        self.assertEqual(plan["keywords"], ["foundation models", "NLP"])
        self.assertEqual(plan["gaps"], ["missing_dataset"])

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
