"""SearchTreeRunner 的离线测试：fixture Provider + 假 DeepSeek 传输层。

fixture 注意：mock_pipeline._paper 自带共享 mock DOI/paper_id，构造每篇
论文时必须显式设置不同的 identifiers/paper_id，否则会被 P1 身份规则合并。
"""

import json
import unittest
from copy import deepcopy

from spar_solution.src.spar_baseline.deepseek_layer import DeepSeekClient, DeepSeekUnderstandingLayer, TransportResponse
from spar_solution.src.spar_baseline.mock_pipeline import _paper
from spar_solution.src.spar_baseline.p2_pipeline import FixtureProvider
from spar_solution.src.spar_baseline.providers.base import ProviderResult
from spar_solution.src.spar_baseline.search_tree import SearchTreeRunner


QUERY = "WiFi CSI heart rate monitoring"


def _fixture_paper(paper_id, doi, title, abstract):
    paper = _paper("arxiv", abstract)
    paper["paper_id"] = paper_id
    paper["identifiers"]["doi"] = doi
    paper["bibliography"]["title"] = title
    return paper


def _seed(paper_id="fixture:seed", doi="10.1234/tree.seed"):
    return _fixture_paper(
        paper_id,
        doi,
        "WiFi CSI heart rate monitoring",
        "WiFi CSI heart rate monitoring via contactless vital sign estimation using channel state information signals.",
    )


def _child(paper_id="fixture:child", doi="10.1234/tree.child"):
    child = _fixture_paper(
        paper_id,
        doi,
        "WiFi CSI heart rate measurement",
        "Reference paper on WiFi CSI heart rate monitoring measurement and contactless vital sign estimation.",
    )
    child["relation_type"] = "references"
    return child


class FakeTaskTransport:
    """按 task 分发的假 DeepSeek 传输层（FakeTransport 的脚本化变体）。

    plan 返回压缩查询计划；generate_queries 返回固定新检索式；
    judge_candidates 按请求内候选逐篇回填 relevance_score。
    """

    def __init__(self, *, plan_queries, generated_queries, relevance=0.9, disambiguate=None):
        self.plan_queries = list(plan_queries)
        self.generated_queries = list(generated_queries)
        self.relevance = relevance
        self.disambiguate = list(disambiguate or [])
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        request = json.loads(body)
        user = json.loads(request["messages"][1]["content"])
        task = user.get("task")
        self.calls.append(task)
        if task == "decompose_query":
            content = {"queries": list(self.plan_queries), "source_capabilities": ["arxiv"]}
        elif task == "disambiguate_queries":
            content = {"fields": [{"field": "fixture field", "query": item} for item in self.disambiguate]}
        elif task == "generate_queries":
            content = {"queries": [{"query_text": item} for item in self.generated_queries]}
        else:
            content = {"results": [
                {
                    "paper_id": item["paper_id"],
                    "relevance_score": self.relevance,
                    "relevance_label": "relevant",
                    "hard_constraint_state": "pass",
                    "reason": "scripted fixture judgement",
                    "evidence_needed": [],
                    "confidence": 0.9,
                }
                for item in user.get("candidates", [])
            ]}
        envelope = {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}
        return TransportResponse(200, json.dumps(envelope))


def _scripted_layer(transport):
    return DeepSeekUnderstandingLayer(DeepSeekClient(transport=transport))


class QueryAwareFixtureProvider(FixtureProvider):
    """按查询关键词返回不同记录的 fixture（FixtureProvider 对任意查询返回同一批）。"""

    def __init__(self, name, records, query_records=None, relations=None):
        super().__init__(name, records, relations)
        self.query_records = {str(key).casefold(): list(value) for key, value in (query_records or {}).items()}

    def search(self, query, *, page_size=10):
        for keyword, records in self.query_records.items():
            if keyword in str(query).casefold():
                self.search_calls += 1
                return ProviderResult(self.name, "search", [deepcopy(item) for item in records[:page_size]], total=len(records))
        return super().search(query, page_size=page_size)


class RelationCountingProvider(FixtureProvider):
    """记录 relations 调用目标的 fixture，用于死循环防护断言。"""

    def __init__(self, name, records, relations=None):
        super().__init__(name, records, relations)
        self.relation_targets = []

    def relations(self, paper_id, *, relation="all", page_size=10):
        self.relation_targets.append(paper_id)
        return super().relations(paper_id, relation=relation, page_size=page_size)


class SearchTreeRunnerTests(unittest.TestCase):
    def test_two_level_happy_path_with_llm(self):
        seed = _seed()
        child = _child()
        level1 = _fixture_paper(
            "fixture:level1",
            "10.1234/tree.level1",
            "Deep learning CSI vital signs",
            "Deep learning models for CSI based vital sign estimation and heart rate monitoring.",
        )
        transport = FakeTaskTransport(
            plan_queries=["wifi csi heart rate monitoring", "contactless vital signs monitoring"],
            generated_queries=["CSI vital sign deep learning estimation"],
        )
        provider = QueryAwareFixtureProvider(
            "arxiv",
            [seed],
            query_records={"deep learning": [level1]},
            relations={seed["paper_id"]: [child]},
        )
        result = SearchTreeRunner({"arxiv": provider}, _scripted_layer(transport)).run(QUERY)

        self.assertEqual(result["schema_version"], "search_tree_run.v1")
        self.assertIn("papers", result)
        self.assertIn("nodes", result)
        self.assertIn("stats", result)
        self.assertIn("stop_reason", result)
        self.assertEqual(result["stop_reason"], "max_depth")
        self.assertEqual([node["level"] for node in result["nodes"]], [0, 1])
        # 第 0 层：检索到种子 → 引用扩展出子论文。
        self.assertEqual(result["nodes"][0]["citation_calls"], 1)
        self.assertEqual(result["nodes"][0]["new_papers"], 2)
        self.assertEqual(
            [(edge["parent"], edge["child"], edge["depth"]) for edge in result["edges"]],
            [(seed["paper_id"], child["paper_id"], 1)],
        )
        # 第 1 层：LLM 从种子生成的新检索式命中了另一篇论文。
        self.assertEqual(result["nodes"][1]["queries"], ["CSI vital sign deep learning estimation"])
        self.assertEqual(result["nodes"][1]["new_papers"], 1)
        papers = {paper["paper_id"]: paper for paper in result["papers"]}
        # 子论文在第 1 层与检索新增论文一起判断。
        self.assertAlmostEqual(papers[child["paper_id"]]["scores"]["relevance"], 0.9)
        self.assertAlmostEqual(papers[level1["paper_id"]]["scores"]["relevance"], 0.9)
        self.assertEqual(papers[child["paper_id"]]["provenance"]["search_node"], {"level": 1, "parent_paper_id": seed["paper_id"], "relation_type": "references"})
        self.assertEqual(papers[seed["paper_id"]]["provenance"]["search_node"]["level"], 0)
        self.assertEqual(papers[seed["paper_id"]]["provenance"]["search_node"]["query"], "wifi csi heart rate monitoring")
        self.assertTrue(all(paper["provenance"].get("search_node") is not None for paper in result["papers"]))
        # 成本：2 次检索 + 1 次引用（L0）+ 1 次检索 + 2 次引用（L1）；5 次 LLM
        # （plan + 领域消歧 + 两轮判断 + 深层查询生成；消歧未脚本化时返回空
        # 领域列表，L0 仍用计划查询）。
        self.assertEqual(result["stats"]["provider_calls"], 6)
        self.assertEqual(result["stats"]["llm_calls"], 5)
        self.assertEqual(result["stats"]["planner_source"], "llm")
        self.assertEqual(transport.calls, ["decompose_query", "disambiguate_queries", "judge_candidates", "generate_queries", "judge_candidates"])

    def test_rules_fallback_without_llm(self):
        seed = _seed()
        child = _child()
        provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: [child]})
        result = SearchTreeRunner({"arxiv": provider}).run(QUERY)

        self.assertEqual(result["stats"]["planner_source"], "rules")
        self.assertEqual(result["stats"]["llm_calls"], 0)
        self.assertEqual(result["nodes"][0]["queries"], ["wifi csi heart rate monitoring"])
        # 深层查询退到 next_iteration 的 gap 模板（宽度 4 条/层）。
        self.assertEqual(len(result["nodes"][1]["queries"]), 4)
        self.assertTrue(all(query.startswith("wifi csi heart rate monitoring") for query in result["nodes"][1]["queries"]))
        # 无 LLM 时子论文用词法分：原始 1.0，排序分打 0.7 折；来源可审计。
        papers = {paper["paper_id"]: paper for paper in result["papers"]}
        self.assertAlmostEqual(papers[child["paper_id"]]["scores"]["relevance"], 0.7)
        self.assertEqual(papers[child["paper_id"]]["provenance"]["relevance_source"], "lexical")
        self.assertAlmostEqual(papers[child["paper_id"]]["provenance"]["lexical_relevance"], 1.0)
        self.assertEqual(result["stop_reason"], "no_new_papers")
        self.assertEqual(len(result["edges"]), 1)

    def test_provider_budget_hard_cap_stops_expansion(self):
        seed = _seed()
        child = _child()
        transport = FakeTaskTransport(
            plan_queries=["wifi csi heart rate monitoring", "contactless vital signs"],
            generated_queries=["csi heart rate deep learning"],
        )
        provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: [child]})
        runner = SearchTreeRunner({"arxiv": provider}, _scripted_layer(transport), max_provider_calls=2)
        result = runner.run(QUERY)

        # 两条查询 × 1 个 Provider 恰好耗尽预算：判断照常，引用扩展被硬顶拦下。
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertEqual(result["stats"]["provider_calls"], 2)
        self.assertEqual(result["stats"]["levels"], 1)
        self.assertEqual(result["nodes"][0]["citation_calls"], 0)
        self.assertEqual(result["nodes"][0]["new_relevant"], 1)
        self.assertEqual(provider.relation_calls, 0)
        self.assertEqual(result["edges"], [])

    def test_stops_when_level_adds_no_new_papers(self):
        seed = _seed()
        transport = FakeTaskTransport(plan_queries=["wifi csi heart rate"], generated_queries=["csi heart rate estimation"])
        provider = FixtureProvider("arxiv", [seed], {})
        result = SearchTreeRunner({"arxiv": provider}, _scripted_layer(transport)).run(QUERY)

        self.assertEqual(result["stop_reason"], "no_new_papers")
        self.assertEqual(result["stats"]["levels"], 2)
        self.assertEqual(result["nodes"][1]["new_papers"], 0)
        self.assertEqual(result["nodes"][1]["citation_calls"], 0)

    def test_no_relevant_papers_skips_expansion(self):
        seed = _fixture_paper(
            "fixture:irrelevant",
            "10.1234/tree.irrelevant",
            "Grape vineyard history",
            "Medieval history of grape vineyard cultivation and monastery wine production in Europe.",
        )
        provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: [_child()]})
        result = SearchTreeRunner({"arxiv": provider}).run("quantum error correction benchmark")

        self.assertEqual(result["stop_reason"], "no_relevant_papers")
        self.assertEqual(result["stats"]["levels"], 1)
        self.assertEqual(provider.relation_calls, 0)
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["nodes"][0]["new_relevant"], 0)
        self.assertAlmostEqual(result["papers"][0]["scores"]["relevance"], 0.0)

    def test_edges_trace_parent_papers(self):
        first = _seed("fixture:a", "10.1234/tree.a")
        second = _seed("fixture:b", "10.1234/tree.b")
        child_a = _child("fixture:child-a", "10.1234/tree.child-a")
        child_b = _child("fixture:child-b", "10.1234/tree.child-b")
        provider = FixtureProvider("arxiv", [first, second], {first["paper_id"]: [child_a], second["paper_id"]: [child_b]})
        result = SearchTreeRunner({"arxiv": provider}).run(QUERY)

        paper_ids = {paper["paper_id"] for paper in result["papers"]}
        self.assertEqual(len(result["edges"]), 2)
        self.assertEqual({edge["parent"] for edge in result["edges"]}, {first["paper_id"], second["paper_id"]})
        for edge in result["edges"]:
            self.assertEqual(set(edge), {"parent", "child", "relation_type", "depth"})
            self.assertEqual(edge["relation_type"], "references")
            self.assertIn(edge["parent"], paper_ids)
            self.assertIn(edge["child"], paper_ids)
        papers = {paper["paper_id"]: paper for paper in result["papers"]}
        self.assertEqual(papers[child_a["paper_id"]]["provenance"]["search_node"]["parent_paper_id"], first["paper_id"])
        self.assertEqual(papers[child_b["paper_id"]]["provenance"]["search_node"]["parent_paper_id"], second["paper_id"])

    def test_citation_cycle_does_not_reexpand_children(self):
        first = _seed("fixture:a", "10.1234/tree.a")
        second = _child("fixture:b", "10.1234/tree.b")
        provider = RelationCountingProvider("arxiv", [first], {first["paper_id"]: [second], second["paper_id"]: [first]})
        result = SearchTreeRunner({"arxiv": provider}).run(QUERY)

        # A→B→A 循环：每篇只被扩展一次，子论文不重复入池。
        self.assertEqual(provider.relation_targets, [first["paper_id"], second["paper_id"]])
        self.assertEqual(provider.relation_targets.count(first["paper_id"]), 1)
        self.assertEqual({paper["paper_id"] for paper in result["papers"]}, {first["paper_id"], second["paper_id"]})
        edge_pairs = {(edge["parent"], edge["child"]) for edge in result["edges"]}
        self.assertEqual(edge_pairs, {(first["paper_id"], second["paper_id"]), (second["paper_id"], first["paper_id"])})
        self.assertEqual([edge["depth"] for edge in result["edges"]], [1, 2])
        self.assertEqual(result["stop_reason"], "no_new_papers")

    def test_llm_budget_cap_falls_back_to_rules(self):
        seed = _seed()
        child = _child()
        transport = FakeTaskTransport(
            plan_queries=["wifi csi heart rate monitoring", "contactless vital signs"],
            generated_queries=["should never be requested"],
        )
        provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: [child]})
        result = SearchTreeRunner({"arxiv": provider}, _scripted_layer(transport), max_llm_calls=1).run(QUERY)

        # 第 0 层计划用掉唯一一次 LLM 调用：判断退词法分、深层查询退 gap 模板。
        self.assertEqual(transport.calls, ["decompose_query"])
        self.assertEqual(result["stats"]["llm_calls"], 1)
        self.assertEqual(result["stats"]["planner_source"], "llm")
        papers = {paper["paper_id"]: paper for paper in result["papers"]}
        self.assertAlmostEqual(papers[seed["paper_id"]]["scores"]["relevance"], 0.7)
        self.assertEqual(papers[seed["paper_id"]]["provenance"]["relevance_source"], "lexical")
        self.assertEqual(len(result["nodes"][1]["queries"]), 4)
        self.assertTrue(all(query.startswith("wifi csi heart rate monitoring") for query in result["nodes"][1]["queries"]))
        self.assertEqual(result["stop_reason"], "no_new_papers")


if __name__ == "__main__":
    unittest.main()


class RecallPackageTests(unittest.TestCase):
    """召回修复包（净化/垃圾池重搜/兜底种子）的专项回归。"""

    def test_sanitize_query_strips_question_shell_only(self):
        from spar_solution.src.spar_baseline.search_tree import _sanitize_query

        shell = "Can you tell me some papers about hybrid architectures in reconstruction-based techniques?"
        cleaned = _sanitize_query(shell)
        self.assertNotIn("tell", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertIn("hybrid", cleaned)
        keyword = "CSI Vital Sign Deep Learning Estimation"
        self.assertEqual(_sanitize_query(keyword), keyword)  # 关键词式原样放行（保留大小写）

    def test_best_effort_seeds_expand_without_high_relevance(self):
        # 词法相关约 0.4（一半查询词命中）且带摘要：不到 0.75 门槛也必须扩引用。
        seed = _fixture_paper(
            "fixture:be:seed",
            "10.1234/be.seed",
            "WiFi heart rate partial match",
            "Contactless vital sign estimation study with only partial WiFi coverage of the topic words.",
        )
        child = _fixture_paper(
            "fixture:be:child",
            "10.1234/be.child",
            "WiFi heart rate measurement",
            "Reference paper on WiFi CSI heart rate monitoring measurement.",
        )
        child["relation_type"] = "references"
        provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: [child]})
        runner = SearchTreeRunner({"arxiv": provider}, max_depth=1, max_provider_calls=10)
        result = runner.run("WiFi heart rate monitoring")
        self.assertGreater(len(result["edges"]), 0, "兜底种子未触发引用扩展")
        self.assertTrue(any(p["paper_id"] == child["paper_id"] for p in result["papers"]))

    def test_best_effort_seeds_skip_offtopic_pool(self):
        # 词法分低于 0.3 地板：整体错领域，不应浪费引用调用。
        junk = _fixture_paper(
            "fixture:junk",
            "10.1234/junk.1",
            "Dynamics of controlled hybrid systems",
            "Control theory stability analysis for switched systems with lyapunov functions and no overlap words here.",
        )
        provider = FixtureProvider("arxiv", [junk], {junk["paper_id"]: []})
        runner = SearchTreeRunner({"arxiv": provider}, max_depth=1, max_provider_calls=10)
        result = runner.run("WiFi heart rate monitoring")
        self.assertEqual(len(result["edges"]), 0)

    def test_rephrase_fallback_without_llm(self):
        from spar_solution.src.spar_baseline.search_tree import _plan_queries, _sanitize_query

        runner = SearchTreeRunner({"arxiv": FixtureProvider("arxiv", [], {})}, max_depth=2)
        plan = runner.planner.plan("WiFi heart rate monitoring")
        queries = runner._rephrase_queries("WiFi heart rate monitoring", set(), plan, plan, [], 1)
        self.assertTrue(queries, "规则兜底应返回可检索查询")
        self.assertTrue(all(_sanitize_query(q) for q in queries))
        self.assertEqual([q for q in queries], [q for q in _plan_queries(plan, runner.queries_per_level)] if set() == set() else queries)


def _grandchild(paper_id="fixture:grandchild", doi="10.1234/tree.grandchild"):
    return _fixture_paper(
        paper_id,
        doi,
        "WiFi CSI heart rate follow-up measurement",
        "Follow-up work on WiFi CSI heart rate monitoring measurement and contactless vital sign estimation.",
    )


class FinalJudgePassTests(unittest.TestCase):
    """P0-1：末层引用扩展入池的子论文必须在出池前拿到判分（LLM 或词法）。"""

    @staticmethod
    def _provider():
        seed = _seed()
        child = _child()
        grandchild = _grandchild()
        return QueryAwareFixtureProvider(
            "arxiv",
            [seed],
            query_records={"deep learning": [_fixture_paper(
                "fixture:level1",
                "10.1234/tree.level1",
                "Deep learning CSI vital signs",
                "Deep learning models for CSI based vital sign estimation and heart rate monitoring.",
            )]},
            relations={seed["paper_id"]: [child], child["paper_id"]: [grandchild]},
        )

    def test_final_level_children_get_llm_judgement(self):
        transport = FakeTaskTransport(
            plan_queries=["wifi csi heart rate monitoring", "contactless vital signs monitoring"],
            generated_queries=["CSI vital sign deep learning estimation"],
        )
        result = SearchTreeRunner({"arxiv": self._provider()}, _scripted_layer(transport)).run(QUERY)
        papers = {paper["paper_id"]: paper for paper in result["papers"]}
        self.assertIn("fixture:grandchild", papers)
        # 末层子论文由循环后的补判轮拿到 LLM 分，而不是 relevance=None 沉底。
        self.assertAlmostEqual(papers["fixture:grandchild"]["scores"]["relevance"], 0.9)
        self.assertTrue(all(paper["scores"].get("relevance") is not None for paper in result["papers"]))
        # plan + 消歧 + L0 judge + generate + L1 judge + final judge = 6 次 LLM 调用。
        self.assertEqual(
            transport.calls,
            ["decompose_query", "disambiguate_queries", "judge_candidates", "generate_queries", "judge_candidates", "judge_candidates"],
        )
        self.assertEqual(result["stats"]["llm_calls"], 6)

    def test_final_level_children_lexical_fallback_without_llm(self):
        result = SearchTreeRunner({"arxiv": self._provider()}, max_depth=2).run(QUERY)
        papers = {paper["paper_id"]: paper for paper in result["papers"]}
        self.assertIn("fixture:grandchild", papers)
        score = papers["fixture:grandchild"]["scores"]["relevance"]
        self.assertIsNotNone(score)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertTrue(all(paper["scores"].get("relevance") is not None for paper in result["papers"]))


class LlmTokenStatsTests(unittest.TestCase):
    """P0-2：tree 路径把 LLM client 的 usage 累计写进 stats。"""

    def test_stats_include_llm_token_usage(self):
        class StubClient:
            usage = {"calls": 3, "failures": 1, "prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500, "latency_ms": 1.0}

        class StubLayer:
            def __init__(self):
                self.client = StubClient()

            def judge(self, plan, papers):
                return []

        provider = FixtureProvider("arxiv", [_seed()], {_seed()["paper_id"]: [_child()]})
        result = SearchTreeRunner({"arxiv": provider}, StubLayer(), max_depth=1).run(QUERY)
        stats = result["stats"]
        self.assertEqual(stats["llm_prompt_tokens"], 1200)
        self.assertEqual(stats["llm_completion_tokens"], 300)
        self.assertEqual(stats["llm_total_tokens"], 1500)
        self.assertEqual(stats["llm_failures"], 1)


class JudgeBudgetCapTests(unittest.TestCase):
    """判分预算调度：max_judge_papers 内的候选走 LLM，其余退词法折扣分。"""

    def test_judge_budget_caps_llm_candidates(self):
        seed = _seed()
        children = [
            _fixture_paper(
                f"fixture:child{i}",
                f"10.1234/cap.child{i}",
                f"WiFi CSI heart rate measurement {i}",
                f"Reference paper {i} on WiFi CSI heart rate monitoring measurement and vital sign estimation.",
            )
            for i in range(5)
        ]
        for child in children:
            child["relation_type"] = "references"
        provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: children})
        transport = FakeTaskTransport(plan_queries=["wifi csi heart rate monitoring"], generated_queries=[])
        result = SearchTreeRunner(
            {"arxiv": provider}, _scripted_layer(transport), max_depth=1, max_judge_papers=3
        ).run(QUERY)
        sources = [paper.get("provenance", {}).get("relevance_source") for paper in result["papers"]]
        # L0 判 seed 1 篇 + 末轮补判预算剩 2 篇 → 恰好 3 篇 LLM，其余词法。
        self.assertEqual(sources.count("llm"), 3)
        self.assertEqual(sources.count("lexical"), 3)
        self.assertTrue(all(source in {"llm", "lexical"} for source in sources))
        stats = result["stats"]
        self.assertEqual(stats["llm_judge_papers"], 3)
        self.assertEqual(stats["judge_capped"], 3)
        # 被预算挤掉的论文仍出池、仍带分，不出现 relevance=None。
        self.assertTrue(all(paper["scores"].get("relevance") is not None for paper in result["papers"]))

    def test_zero_judge_budget_is_all_lexical(self):
        seed = _seed()
        provider = FixtureProvider("arxiv", [seed], {})
        transport = FakeTaskTransport(plan_queries=["wifi csi heart rate monitoring"], generated_queries=[])
        result = SearchTreeRunner(
            {"arxiv": provider}, _scripted_layer(transport), max_depth=1, max_judge_papers=0
        ).run(QUERY)
        sources = [paper.get("provenance", {}).get("relevance_source") for paper in result["papers"]]
        self.assertEqual(sources, ["lexical"])
        self.assertEqual(result["stats"]["llm_judge_papers"], 0)
        # 判分预算为 0 不影响规划调用；judge_candidates 不应出现。
        self.assertNotIn("judge_candidates", transport.calls)


class DisambiguateQueriesTests(unittest.TestCase):
    """L0 领域消歧：领域术语查询排在照抄问题的计划查询之前；失败退回计划查询。"""

    def test_disambiguated_queries_lead_level0(self):
        transport = FakeTaskTransport(
            plan_queries=["wifi csi heart rate monitoring"],
            generated_queries=[],
            disambiguate=["reconstruction error anomaly detection", "contactless csi vital signs"],
        )
        provider = FixtureProvider("arxiv", [_seed()], {_seed()["paper_id"]: [_child()]})
        result = SearchTreeRunner({"arxiv": provider}, _scripted_layer(transport), max_depth=1).run(QUERY)
        # 消歧查询在前，计划查询补位；L0 宽度上限 5 全部保留。
        self.assertEqual(
            result["nodes"][0]["queries"],
            ["reconstruction error anomaly detection", "contactless csi vital signs", "wifi csi heart rate monitoring"],
        )
        self.assertIn("disambiguate_queries", transport.calls)

    def test_disambiguation_failure_falls_back_to_plan_queries(self):
        # 未脚本化消歧 → 返回空领域列表 → L0 保持计划查询，不报错。
        transport = FakeTaskTransport(plan_queries=["wifi csi heart rate monitoring"], generated_queries=[])
        provider = FixtureProvider("arxiv", [_seed()], {_seed()["paper_id"]: [_child()]})
        result = SearchTreeRunner({"arxiv": provider}, _scripted_layer(transport), max_depth=1).run(QUERY)
        self.assertEqual(result["nodes"][0]["queries"], ["wifi csi heart rate monitoring"])
        self.assertIn("disambiguate_queries", transport.calls)

    def test_no_llm_means_no_disambiguation_call(self):
        provider = FixtureProvider("arxiv", [_seed()], {_seed()["paper_id"]: [_child()]})
        result = SearchTreeRunner({"arxiv": provider}, max_depth=1).run(QUERY)
        self.assertEqual(result["nodes"][0]["queries"], ["wifi csi heart rate monitoring"])
        self.assertEqual(result["stats"]["llm_calls"], 0)
