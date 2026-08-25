"""SPAR 式逐层搜索树检索（SearchTreeRunner）。

与 P2 的“两轮迭代 + 一跳引用”不同，本模块实现 SPAR 原版的逐层树打法：
每层先执行当前层检索式，对新增论文做相关性判断，再对高相关论文做引用
扩展（子论文入池、下一层一起判断），最后从 top 相关论文生成下一层检索式，
直到预算耗尽、该层无新增论文、无相关论文可扩展或达到 max_depth。

本模块只编排既有组件：检索复用 ``p2_recall.RecallRunner/SourceRouter``，
去重复用 ``p2_pipeline._deduplicate``，引用记录解析复用 ``p2_citation`` 的
关系调用协议，词法兜底复用 ``p2_scoring.Scorer.preliminary_relevance``，
深层查询兜底复用 ``query_planner.QueryPlanner.next_iteration``。LLM 缺席
或失败时全部退到确定性规则路径，不会产生虚假论文或虚假分数。
"""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from .deepseek_layer import DeepSeekCallError, DeepSeekSchemaError
from .p2_citation import _child_id, _nested_paper, _relation_call, _relation_type
from .p2_pipeline import _deduplicate, _group_by_source
from .p2_recall import RecallRunner, SourceRouter
from .p2_scoring import Scorer
from .paperdoc import validate_paper_doc
from .query_planner import QueryPlanner, _clean_query
from .rank_fusion import rrf_fuse


_QUESTION_SHELL_RE = re.compile(
    r"(can you|could you|tell me|papers? about|studies? that|are there|what papers|"
    r"any resources|any studies|list (?:of|the) papers|i want|looking for|\?)",
    re.IGNORECASE,
)


def _sanitize_query(text: str) -> str:
    """把送进 Provider 的检索式洗净：剥掉疑问壳/礼貌词。

    真实运行中 LLM 计划偶尔原样回显完整问句（"Can you tell me some papers
    about ..."），直接进 arXiv/OpenAlex 会返回大量错领域垃圾并饿死整棵树
    （见 autoscholar/tree-n10 的 test_0/test_9）。只清洗"像问句"的查询
    （含疑问壳或超过 6 个词），关键词式短查询原样放行以保留大小写。
    """

    text = str(text or "").strip()
    if not text:
        return text
    if len(text.split()) <= 6 and not _QUESTION_SHELL_RE.search(text):
        return text
    cleaned = _clean_query(text)
    return cleaned if len(cleaned.split()) >= 2 else text


# 层停止原因（枚举字符串）；判断优先级与 run() 内的检查顺序一致。
STOP_BUDGET_EXHAUSTED = "budget_exhausted"
STOP_NO_NEW_PAPERS = "no_new_papers"
STOP_NO_RELEVANT_PAPERS = "no_relevant_papers"
STOP_NO_NEW_QUERIES = "no_new_queries"
STOP_MAX_DEPTH = "max_depth"
STOP_REASONS = frozenset({
    STOP_BUDGET_EXHAUSTED,
    STOP_NO_NEW_PAPERS,
    STOP_NO_RELEVANT_PAPERS,
    STOP_NO_NEW_QUERIES,
    STOP_MAX_DEPTH,
})

# LLM 交互可能抛出的协议/调用异常；命中即退规则兜底，绝不中断整棵树。
_LLM_ERRORS = (DeepSeekCallError, DeepSeekSchemaError, ValueError, TypeError, KeyError)


def _norm_query(query: str) -> str:
    """查询去重键：忽略大小写与空白差异。"""

    return " ".join(str(query or "").casefold().split())


def _relevance_of(paper: Mapping[str, Any]) -> float:
    """读取已判断的相关性分；未判断记为 -1，排序时沉底。"""

    value = (paper.get("scores") or {}).get("relevance")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return -1.0
    return float(value)


def _plan_queries(plan: Mapping[str, Any] | None, limit: int) -> list[str]:
    """从 QueryPlan 提取去重后的检索式文本（保持原顺序）。"""

    queries: list[str] = []
    seen: set[str] = set()
    for subquery in ((plan or {}).get("subqueries") or []):
        if not isinstance(subquery, Mapping):
            continue
        text = str(subquery.get("query_text") or subquery.get("query") or "").strip()
        key = _norm_query(text)
        if text and key and key not in seen:
            seen.add(key)
            queries.append(text)
    return queries[:limit]


def _parse_generated_queries(payload: Mapping[str, Any]) -> list[str]:
    """宽容解析 {"queries":[{"query_text":"..."}]} 协议；畸形条目直接丢弃。"""

    values = payload.get("queries")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    texts: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            text = str(item.get("query_text") or item.get("query") or item.get("search_query") or "").strip()
        elif isinstance(item, str):
            text = item.strip()
        else:
            continue
        if text:
            texts.append(text)
    return texts


class SearchTreeRunner:
    """SPAR 式搜索树检索器。

    参数硬顶：``max_provider_calls`` 全局限制 search + relations 调用总数，
    ``max_llm_calls`` 全局限制 plan/judge/查询生成调用次数；两层预算耗尽后
    相应阶段退规则路径或直接停止，不会悄悄超支。
    """

    def __init__(
        self,
        providers: Mapping[str, Any] | Iterable[Any],
        understanding_layer: Any | None = None,
        *,
        max_depth: int = 2,
        queries_per_level: int = 2,
        docs_to_expand: int = 8,
        relevance_threshold: float = 0.75,
        page_size: int = 10,
        max_provider_calls: int = 30,
        max_llm_calls: int = 20,
    ) -> None:
        if max_depth < 1 or queries_per_level < 1 or page_size < 1:
            raise ValueError("max_depth, queries_per_level and page_size must be positive")
        if docs_to_expand < 0 or max_provider_calls < 0 or max_llm_calls < 0:
            raise ValueError("docs_to_expand and call budgets must be non-negative")
        if not 0 <= relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be between 0 and 1")
        self.providers = providers
        self.understanding_layer = understanding_layer
        self.max_depth = int(max_depth)
        self.queries_per_level = int(queries_per_level)
        self.docs_to_expand = int(docs_to_expand)
        self.relevance_threshold = float(relevance_threshold)
        self.page_size = int(page_size)
        self.max_provider_calls = int(max_provider_calls)
        self.max_llm_calls = int(max_llm_calls)
        self.router = SourceRouter(providers)
        self.recall_runner = RecallRunner(self.router, page_size=self.page_size)
        self.planner = QueryPlanner()
        self.scorer = Scorer()
        self._llm_calls = 0

    # ------------------------------------------------------------------
    # LLM 计量与预算
    # ------------------------------------------------------------------

    def _usage_calls(self) -> int | None:
        """读取理解层客户端的调用计数；无计量能力时返回 None。"""

        if self.understanding_layer is None:
            return None
        usage = getattr(getattr(self.understanding_layer, "client", None), "usage", None)
        if isinstance(usage, Mapping) and isinstance(usage.get("calls"), int):
            return int(usage["calls"])
        return None

    def _llm_budget_left(self) -> bool:
        return self._llm_calls < self.max_llm_calls

    def _call_llm(self, operation: Callable[[], Any]) -> Any:
        """执行一次 LLM 操作并按 usage 增量计费；无 usage 的假层记 1 次。"""

        before = self._usage_calls()
        try:
            return operation()
        finally:
            after = self._usage_calls()
            if before is None or after is None:
                self._llm_calls += 1
            else:
                self._llm_calls += max(0, after - before)

    # ------------------------------------------------------------------
    # 查询生成
    # ------------------------------------------------------------------

    def _rephrase_queries(
        self,
        query: str,
        searched_norm: set[str],
        base_plan: Mapping[str, Any] | None,
        lexical_plan: Mapping[str, Any] | None,
        errors: list[dict[str, Any]],
        level: int,
    ) -> list[str]:
        """垃圾池重写：第 0 层无高相关论文时，让 LLM 换术语重写检索式。

        与 _generate_deep_queries 的区别：不从已找到论文派生（垃圾池没有
        可派生的对象），而是要求模型换角度/换术语/更具体地重述问题。
        失败退规则 gap 模板。
        """

        layer = self.understanding_layer
        client = getattr(layer, "client", None)
        if layer is not None and self._llm_budget_left() and callable(getattr(client, "complete_json", None)):
            system = (
                "You are an academic literature search expert. The previous queries retrieved "
                "nothing relevant. Rephrase the question into short retrieval queries using "
                "DIFFERENT terminology, more specific technical terms, and remove any "
                "conversational phrasing. Return JSON only: "
                '{"queries": [{"query_text": "..."}]}. Never invent paper facts.'
            )
            user = json.dumps(
                {
                    "task": "rephrase_query",
                    "query": query,
                    "searched_queries": sorted(searched_norm),
                    "required": {"queries": "1-2 objects, each with a short keyword-style query_text"},
                },
                ensure_ascii=False,
            )
            try:
                payload = self._call_llm(lambda: client.complete_json(system, user, max_tokens=400))
                fresh = [
                    str(item.get("query_text") or item.get("query") or "").strip()
                    for item in (payload.get("queries") or [])
                    if isinstance(item, Mapping)
                ]
                fresh = [text for text in fresh if text and _norm_query(text) not in searched_norm]
                if fresh:
                    return fresh[: self.queries_per_level]
            except _LLM_ERRORS as exc:
                errors.append({"source": "search_tree", "code": str(getattr(exc, "code", "rephrase_fallback")), "message": str(exc)[:200], "stage": f"rephrase_L{level}"})
        # 规则兜底：计划里未被搜过的其他子查询，或 gap 模板。
        fallback_plan = lexical_plan or base_plan
        if fallback_plan is not None:
            texts = [
                str(item.get("query_text") or "").strip()
                for item in (fallback_plan.get("subqueries") or [])
                if isinstance(item, Mapping)
            ]
            texts = [text for text in texts if text and _norm_query(text) not in searched_norm]
            if texts:
                return texts[: self.queries_per_level]
        try:
            evolved = self.planner.next_iteration(fallback_plan) if fallback_plan is not None else None
        except Exception:
            evolved = None
        return _plan_queries(evolved, self.queries_per_level) if evolved is not None else []

    def _generate_deep_queries(
        self,
        query: str,
        top_papers: list[dict[str, Any]],
        searched_norm: set[str],
        base_plan: Mapping[str, Any] | None,
        lexical_plan: Mapping[str, Any] | None,
        errors: list[dict[str, Any]],
        level: int,
    ) -> list[str]:
        """深层检索式：LLM 从 top 论文标题+摘要生成，失败退 gap 模板。"""

        layer = self.understanding_layer
        client = getattr(layer, "client", None)
        if layer is not None and self._llm_budget_left() and callable(getattr(client, "complete_json", None)):
            system = (
                "You are an academic literature search expert. Generate diverse retrieval "
                "queries from the top papers to cover aspects the searched queries missed. "
                'Return JSON only: {"queries": [{"query_text": "..."}]}. Never invent paper facts.'
            )
            user = json.dumps(
                {
                    "task": "generate_queries",
                    "query": query,
                    "searched_queries": sorted(searched_norm),
                    "top_papers": [
                        {
                            "title": str((paper.get("bibliography") or {}).get("title") or ""),
                            "abstract": str((paper.get("bibliography") or {}).get("abstract") or "")[:1200],
                        }
                        for paper in top_papers
                    ],
                    "required": {"queries": "1-2 objects, each with a non-empty query_text string"},
                },
                ensure_ascii=False,
            )
            try:
                payload = self._call_llm(lambda: client.complete_json(system, user, max_tokens=600))
                fresh = [
                    text
                    for text in _parse_generated_queries(payload)
                    if _norm_query(text) and _norm_query(text) not in searched_norm
                ][: self.queries_per_level]
                if fresh:
                    return fresh
                errors.append({"source": "search_tree", "code": "empty_generated_queries", "message": "LLM generated no unsearched queries", "stage": f"queries_L{level}"})
            except _LLM_ERRORS as exc:
                errors.append({"source": "search_tree", "code": str(getattr(exc, "code", "query_fallback")), "message": str(exc)[:200], "stage": f"queries_L{level}"})
        # 规则兜底：优先本层基础计划的 gap 模板；LLM 计划未给 gaps 时退规则计划的 gap。
        return self._gap_queries(base_plan, searched_norm) or self._gap_queries(lexical_plan, searched_norm)

    def _gap_queries(self, base_plan: Mapping[str, Any] | None, searched_norm: set[str]) -> list[str]:
        """规则兜底：query_planner.next_iteration 的 gap 模板（已搜去重）。"""

        if not isinstance(base_plan, Mapping):
            return []
        try:
            evolved = self.planner.next_iteration(base_plan, gaps=base_plan.get("gaps") if base_plan else None)
            fresh = [subquery for subquery in evolved.get("subqueries") or [] if int(subquery.get("iteration") or 0) > 0]
        except Exception:
            return []
        return [
            text
            for text in (str(subquery.get("query_text") or "").strip() for subquery in fresh)
            if text and _norm_query(text) not in searched_norm
        ][: self.queries_per_level]

    # ------------------------------------------------------------------
    # 引用扩展
    # ------------------------------------------------------------------

    def _relations_provider(self, seed: Mapping[str, Any]) -> tuple[str, Any] | None:
        """按论文来源优先选择带 relations 方法的 Provider。"""

        sources = [str(item).casefold() for item in ((seed.get("provenance") or {}).get("sources") or [])]
        ordered = [name for name in sources if name in self.router.providers] + sorted(
            name for name in self.router.providers if name not in sources
        )
        for name in ordered:
            if name in self._relations_unsupported:
                continue
            provider = self.router.providers[name]
            if getattr(provider, "library_status", None) == "unavailable":
                continue
            if callable(getattr(provider, "relations", None)):
                return name, provider
        return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        wall_started = perf_counter()
        self._llm_calls = 0
        self._relations_unsupported: set[str] = set()
        errors: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        edge_keys: set[tuple[str, str, str]] = set()
        pool: list[dict[str, Any]] = []
        pool_ids: set[str] = set()
        judged_ids: set[str] = set()
        expanded_ids: set[str] = set()  # 死循环防护：每篇论文至多被引用扩展一次
        query_seed_ids: set[str] = set()  # 已用于生成深层查询的论文
        searched_norm: set[str] = set()
        provider_calls = 0
        search_calls = 0
        relation_calls = 0

        # 让真实 DeepSeek 客户端同步执行 LLM 硬顶（内部重试也不会超支）。
        client = getattr(self.understanding_layer, "client", None) if self.understanding_layer is not None else None
        if callable(getattr(client, "reset_usage", None)):
            client.reset_usage(max_calls=self.max_llm_calls)

        # 词法兜底计划：规则版 QueryPlanner；无可检索词时退最小计分视图。
        try:
            lexical_plan: Mapping[str, Any] | None = self.planner.plan(query)
        except ValueError as exc:
            lexical_plan = None
            errors.append({"source": "search_tree", "code": "lexical_plan_unavailable", "message": str(exc)[:200], "stage": "plan"})
        scoring_plan: Mapping[str, Any] = lexical_plan or {"raw_query": query, "topic": query, "methods": [], "datasets": [], "tasks": []}
        base_plan: Mapping[str, Any] = lexical_plan or scoring_plan
        planner_source = "rules"

        # 第 0 层查询：优先 understanding_layer.plan，失败退规则计划。
        level_queries: list[str] = []
        if self.understanding_layer is not None and callable(getattr(self.understanding_layer, "plan", None)) and self._llm_budget_left():
            candidate = None
            # 规划失败重试一次（瞬时网络/限流常见；402 欠费等硬错误两次都失败）。
            for attempt in range(2):
                try:
                    candidate = self._call_llm(lambda: self.understanding_layer.plan(query))
                    texts = _plan_queries(candidate, self.queries_per_level)
                    break
                except _LLM_ERRORS as exc:
                    texts = []
                    if attempt == 1:
                        errors.append({"source": "search_tree", "code": str(getattr(exc, "code", "plan_fallback")), "message": str(exc)[:200], "stage": "plan"})
            if texts:
                base_plan = candidate
                planner_source = "llm"
                level_queries = texts
        if not level_queries:
            level_queries = _plan_queries(base_plan, self.queries_per_level) or [query.strip()]
        # 任何来源的第 0 层查询（LLM 回显问句/规则兜底）都必须先净化；
        # 单词垃圾查询（规则兜底曾产出 "models"）直接丢弃。
        level_queries = [q for q in dict.fromkeys(_sanitize_query(t) for t in level_queries if _sanitize_query(t)) if len(q.split()) >= 2]
        if not level_queries:
            fallback = _sanitize_query(query)
            level_queries = [fallback] if len(fallback.split()) >= 2 else [query.strip()]

        judge_plan = base_plan.to_dict() if hasattr(base_plan, "to_dict") else dict(base_plan)
        stop_reason = STOP_MAX_DEPTH

        for level in range(self.max_depth):
            # -- 1. 本层查询（第 0 层已在上方生成；深层查询由 top 论文派生）--
            if level:
                relevant_pool = [paper for paper in pool if _relevance_of(paper) >= self.relevance_threshold]
                if relevant_pool:
                    relevant_pool.sort(key=lambda paper: (-_relevance_of(paper), str(paper.get("paper_id"))))
                    unused = [paper for paper in relevant_pool if str(paper.get("paper_id")) not in query_seed_ids]
                    top_papers = (unused or relevant_pool)[:2]
                    query_seed_ids.update(str(paper.get("paper_id")) for paper in top_papers)
                    level_queries = self._generate_deep_queries(query, top_papers, searched_norm, base_plan, lexical_plan, errors, level)
                else:
                    # 垃圾池二 chance：第 0 层没有任何高相关论文时，换术语重写
                    # 查询再搜一层，而不是直接 no_relevant_papers 停死。
                    level_queries = self._rephrase_queries(query, searched_norm, base_plan, lexical_plan, errors, level)
                level_queries = [_sanitize_query(text) for text in level_queries]
            level_queries = [
                text
                for text in level_queries
                if text.strip() and _norm_query(text) not in searched_norm
            ][: self.queries_per_level]
            if not level_queries:
                stop_reason = STOP_NO_NEW_QUERIES
                break

            # -- 2. 检索：每条查询对有 search 方法的 Provider 各调一次 --
            remaining = self.max_provider_calls - provider_calls
            if remaining <= 0:
                stop_reason = STOP_BUDGET_EXHAUSTED
                break
            level_start_ids = set(pool_ids)
            plan_nodes = [
                {"subquery_id": f"st_L{level}_{index:02d}", "query_text": text, "iteration": level}
                for index, text in enumerate(level_queries)
            ]
            # 第 0 层加宽一倍（首次召回决定整棵树的种子质量），并做 RRF 融合。
            recall = (
                RecallRunner(self.recall_runner.router, max_workers=self.recall_runner.max_workers, page_size=self.page_size * 2)
                .run(plan_nodes, iteration=level, max_calls=remaining)
                if level == 0
                else self.recall_runner.run(plan_nodes, iteration=level, max_calls=remaining)
            )
            provider_calls += int(recall.stats.get("api_calls", 0))
            search_calls += int(recall.stats.get("api_calls", 0))
            errors.extend(dict(error, stage=f"recall_L{level}") for error in recall.source_errors)
            searched_norm.update(_norm_query(text) for text in level_queries)
            subquery_map = {node["subquery_id"]: node for node in plan_nodes}
            fresh_records = rrf_fuse(_group_by_source(recall.records)) if recall.records else []

            # -- 3. 合并去重（P1 身份规则）并标记 search_node --
            merged, dedup_errors = _deduplicate([*pool, *fresh_records])
            errors.extend(dict(error, stage=f"dedup_L{level}") for error in dedup_errors)
            pool = merged
            pool_ids = {str(paper.get("paper_id")) for paper in pool}
            for paper in pool:
                if str(paper.get("paper_id")) in pool_ids - level_start_ids:
                    node_info = subquery_map.get(str((paper.get("provenance") or {}).get("subquery_id") or ""))
                    paper.setdefault("provenance", {})["search_node"] = (
                        {"level": level, "subquery_id": node_info["subquery_id"], "query": node_info["query_text"]}
                        if node_info
                        else {"level": level}
                    )

            # -- 4. 判断：新增论文批量交给 understanding_layer.judge --
            to_judge = [paper for paper in pool if str(paper.get("paper_id")) not in judged_ids]
            judgements: dict[str, Mapping[str, Any]] = {}
            if self.understanding_layer is not None and to_judge and self._llm_budget_left():
                unique: dict[str, dict[str, Any]] = {}
                for paper in to_judge:
                    unique.setdefault(str(paper["paper_id"]), paper)
                try:
                    results = self._call_llm(lambda: self.understanding_layer.judge(judge_plan, list(unique.values())))
                    judgements = {str(item.get("paper_id")): item for item in (results or []) if isinstance(item, Mapping)}
                except _LLM_ERRORS as exc:
                    errors.append({"source": "search_tree", "code": str(getattr(exc, "code", "judge_fallback")), "message": str(exc)[:200], "stage": f"judge_L{level}"})
            new_relevant_ids: list[str] = []
            for paper in to_judge:
                paper_id = str(paper["paper_id"])
                judged_ids.add(paper_id)
                judgement = judgements.get(paper_id)
                score: float
                try:
                    score = float(judgement["relevance_score"]) if judgement is not None else float("nan")
                except (KeyError, TypeError, ValueError):
                    score = float("nan")
                if score != score:  # 无判断或畸形判断 → 词法分兜底
                    score = float(self.scorer.preliminary_relevance(paper, scoring_plan))
                paper.setdefault("scores", {})["relevance"] = score
                if score >= self.relevance_threshold:
                    new_relevant_ids.append(paper_id)

            # -- 5. 引用扩展：高相关论文按分取前 docs_to_expand 篇 --
            candidates = [paper for paper in pool if _relevance_of(paper) >= self.relevance_threshold and str(paper.get("paper_id")) not in expanded_ids]
            candidates.sort(key=lambda paper: (-_relevance_of(paper), str(paper.get("paper_id"))))
            if not candidates:
                # 兜底种子：没有 >=0.75 的论文时，只要池子大体在题（最高分
                # >=0.3）且有摘要，就取融合分 top-3 扩引用，避免引用链饿死；
                # 最高分连 0.3 都不到说明池子整体错领域，不浪费调用。
                best_effort = [
                    paper
                    for paper in pool
                    if _relevance_of(paper) >= 0.3
                    and str((paper.get("bibliography") or {}).get("abstract") or "").strip()
                    and str(paper.get("paper_id")) not in expanded_ids
                ]
                best_effort.sort(key=lambda paper: (-_relevance_of(paper), str(paper.get("paper_id"))))
                candidates = best_effort[:3]
            candidates = candidates[: self.docs_to_expand]
            citation_calls = 0
            child_records: list[dict[str, Any]] = []
            child_meta: dict[str, dict[str, Any]] = {}
            for seed in candidates:
                if provider_calls >= self.max_provider_calls:
                    break
                seed_id = str(seed["paper_id"])
                expanded_ids.add(seed_id)
                selected = self._relations_provider(seed)
                if selected is None:
                    errors.append({"source": "search_tree", "code": "config", "message": "no relations provider for relevant paper", "stage": f"expand_L{level}", "details": {"paper_id": seed_id}})
                    continue
                source, provider = selected
                provider_calls += 1
                citation_calls += 1
                relation_calls += 1
                try:
                    result = _relation_call(provider, seed_id, "references", self.page_size)
                except Exception as exc:
                    errors.append({"source": source, "code": str(getattr(exc, "code", "unknown")), "message": str(exc)[:200], "stage": f"expand_L{level}", "details": {"paper_id": seed_id}})
                    # arXiv 等来源的 relations 是显式 unsupported 存根：记住它，
                    # 本 run 后续种子直接换下一家，不再浪费种子槽位。
                    if str(getattr(exc, "code", "")) == "unsupported":
                        self._relations_unsupported.add(source)
                    continue
                for record in result.records:
                    child_id = _child_id(record)
                    if not child_id:
                        errors.append({"source": source, "code": "parse", "message": "relation record has no stable child identifier", "stage": f"expand_L{level}", "details": {"parent_paper_id": seed_id}})
                        continue
                    relation_type = _relation_type(record, "references")
                    edge_key = (seed_id, child_id, relation_type)
                    if edge_key not in edge_keys:
                        edge_keys.add(edge_key)
                        edges.append({"parent": seed_id, "child": child_id, "relation_type": relation_type, "depth": level + 1})
                    nested = _nested_paper(record)
                    if nested is None:
                        continue
                    child = dict(nested)
                    child.setdefault("provenance", {})["parent_node_id"] = seed_id
                    child.setdefault("provenance", {})["citation_depth"] = level + 1
                    try:
                        validate_paper_doc(child)
                    except Exception as exc:
                        errors.append({"source": source, "code": "parse", "message": f"invalid child PaperDoc: {exc}"[:200], "stage": f"expand_L{level}", "details": {"parent_paper_id": seed_id}})
                        continue
                    child_records.append(child)
                    meta = {"parent_paper_id": seed_id, "relation_type": relation_type}
                    child_meta[child_id] = meta
                    child_meta.setdefault(str(child.get("paper_id") or ""), meta)
            if child_records:
                ids_before_children = set(pool_ids)
                merged, merge_errors = _deduplicate([*pool, *child_records])
                errors.extend(dict(error, stage=f"expand_L{level}") for error in merge_errors)
                pool = merged
                pool_ids = {str(paper.get("paper_id")) for paper in pool}
                for paper in pool:
                    paper_id = str(paper.get("paper_id"))
                    if paper_id in pool_ids - ids_before_children:
                        meta = child_meta.get(paper_id, {})
                        paper.setdefault("provenance", {})["search_node"] = {"level": level + 1, **meta}

            # -- 6. 层记录与停止判断（顺序：预算 > 无新增 > 无相关 > 深度）--
            nodes.append({
                "level": level,
                "queries": list(level_queries),
                "new_papers": len(pool_ids - level_start_ids),
                "new_relevant": len(new_relevant_ids),
                "citation_calls": citation_calls,
            })
            if provider_calls >= self.max_provider_calls:
                stop_reason = STOP_BUDGET_EXHAUSTED
                break
            if not (pool_ids - level_start_ids):
                stop_reason = STOP_NO_NEW_PAPERS
                break
            if not any(_relevance_of(paper) >= self.relevance_threshold for paper in pool):
                stop_reason = STOP_NO_RELEVANT_PAPERS
                break
        else:
            stop_reason = STOP_MAX_DEPTH

        papers = sorted(pool, key=lambda paper: (-_relevance_of(paper), str(paper.get("paper_id"))))
        stats = {
            "provider_calls": provider_calls,
            "search_calls": search_calls,
            "relation_calls": relation_calls,
            "llm_calls": self._llm_calls,
            "levels": len(nodes),
            "papers": len(papers),
            "edges": len(edges),
            "planner_source": planner_source,
            "wall_ms": round((perf_counter() - wall_started) * 1000, 3),
        }
        return {
            "schema_version": "search_tree_run.v1",
            "query": query,
            "papers": papers,
            "nodes": nodes,
            "edges": edges,
            "stats": stats,
            "stop_reason": stop_reason,
            "errors": errors,
        }


__all__ = [
    "STOP_BUDGET_EXHAUSTED",
    "STOP_MAX_DEPTH",
    "STOP_NO_NEW_PAPERS",
    "STOP_NO_NEW_QUERIES",
    "STOP_NO_RELEVANT_PAPERS",
    "STOP_REASONS",
    "SearchTreeRunner",
]
