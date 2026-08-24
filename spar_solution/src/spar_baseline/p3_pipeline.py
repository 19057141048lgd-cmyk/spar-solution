"""P3 五角色结构化圆桌流水线。

P3 与 P2 共用 Provider、PaperDoc、证据和评分实现；本模块只增加受限的
两轮调度、冲突复核、结构化消息和可回放 artifact。长正文永远写入 artifact，
Agent 消息只携带短字段和引用。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

from .deepseek_layer import DeepSeekCallError, DeepSeekSchemaError, DeepSeekUnderstandingLayer
from .final_output import build_final_selection, validate_final_selection
from .p2_citation import CitationExpander
from .p2_evidence import ConstraintGate, ConstraintVerdict
from .p2_pipeline import FixtureProvider, _apply_prepared_scores, _deduplicate, _prepare_papers
from .p2_recall import RecallRunner, SourceRouter
from .p2_scoring import EvidenceVerdict, Scorer
from .p2_stop import StopController
from .p3_protocol import ArtifactStore, estimate_bytes, estimate_tokens, make_message, validate_message
from .query_planner import QueryPlanner


def _run_id(query: str, citation_enabled: bool) -> str:
    suffix = "cit" if citation_enabled else "nocit"
    return "p3_" + hashlib.sha256(f"{query}|{suffix}".encode("utf-8")).hexdigest()[:12]


def _safe_error(source: str, exc: Exception, *, stage: str) -> dict[str, Any]:
    message = str(getattr(exc, "message", str(exc)))
    for secret_word in ("authorization", "bearer", "api_key", "access_key", "token", "password", "secret", "email", "mailto"):
        if secret_word in message.casefold():
            message = f"{source} {stage} failed"
            break
    return {"source": source, "stage": stage, "code": str(getattr(exc, "code", "unknown")), "message": message[:200]}


def _iteration_plan(plan: Mapping[str, Any], iteration: int) -> dict[str, Any]:
    """Create a route-only view; parent queries remain in the full plan artifact."""

    payload = deepcopy(dict(plan))
    rows = [dict(item) for item in plan.get("subqueries", []) if int(item.get("iteration", 0)) == iteration]
    if not rows:
        rows = [dict(item) for item in plan.get("subqueries", []) if int(item.get("iteration", 0)) <= iteration]
    for item in rows:
        item["sources"] = list(item.get("source_capabilities") or [])
    payload["subqueries"] = rows
    return payload


def _call_count(groups: Iterable[Mapping[str, Any]]) -> int:
    return sum(int(call.get("api_calls") or 1) for group in groups for call in group.get("calls") or [])


def _provider_names(providers: Mapping[str, Any] | Iterable[Any]) -> list[str]:
    if isinstance(providers, Mapping):
        return sorted(str(name) for name in providers)
    return sorted(str(getattr(item, "name", getattr(item, "source", type(item).__name__))) for item in providers)


def _write_evidence_refs(root: Path, papers: list[dict[str, Any]], evidence: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> list[str]:
    """Materialize deterministic evidence files keyed by paper identity."""

    paper_by_id = {str(item.get("paper_id")): item for item in papers}
    rewrites: dict[str, str] = {}
    written: list[str] = []
    for index, item in enumerate(evidence):
        old_ref = str(item.get("evidence_ref") or item.get("content_ref") or "")
        paper_id = str(item.get("paper_id") or "")
        if not old_ref or not paper_id or item.get("evidence_status") in {None, "unavailable"}:
            continue
        safe_paper = "".join(char if char.isalnum() or char in "._-" else "_" for char in paper_id)[:120] or "paper"
        safe_ref = f"evidence_judge/{safe_paper}/evidence_{index:04d}.txt"
        path = root / safe_ref
        path.parent.mkdir(parents=True, exist_ok=True)
        abstract = str((paper_by_id.get(paper_id) or {}).get("bibliography", {}).get("abstract") or "")
        path.write_text(f"paper_id: {paper_id}\nevidence_id: {item.get('evidence_id', '')}\n\n{abstract}", encoding="utf-8")
        rewrites[old_ref] = safe_ref
        item["evidence_ref"] = safe_ref
        written.append(safe_ref)
    for paper in papers:
        paper["evidence_refs"] = [rewrites.get(str(ref), str(ref)) for ref in paper.get("evidence_refs") or []]
    for verdict in verdicts:
        verdict["evidence_refs"] = [rewrites.get(str(ref), str(ref)) for ref in verdict.get("evidence_refs") or []]
    # Judgement artifacts are referenced by EVIDENCE_VERDICT. Rewrite their
    # nested references too, otherwise replay would point at temporary paths.
    for artifact in sorted((root / "evidence_judge").glob("judgement_iter*.json")):
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        for row in payload.get("evidence") or []:
            row["evidence_ref"] = rewrites.get(str(row.get("evidence_ref") or ""), str(row.get("evidence_ref") or ""))
        for row in payload.get("verdicts") or []:
            row["evidence_refs"] = [rewrites.get(str(ref), str(ref)) for ref in row.get("evidence_refs") or []]
        artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written


@dataclass
class P3Run:
    query: str
    run_id: str
    query_plan: dict[str, Any]
    recall: dict[str, Any] = field(default_factory=dict)
    citation: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    papers: list[dict[str, Any]] = field(default_factory=list)
    selected: list[dict[str, Any]] = field(default_factory=list)
    stop: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    final_selection: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.__dict__)


class P3Pipeline:
    """执行 Planner → Retriever → CitationExplorer → EvidenceJudge → Arbiter。"""

    def __init__(
        self,
        providers: Mapping[str, Any] | Iterable[Any],
        *,
        citation_provider: Mapping[str, Any] | Iterable[Any] | None = None,
        citation_enabled: bool = True,
        page_size: int = 10,
        max_workers: int = 4,
        max_selected: int = 10,
        understanding_layer: DeepSeekUnderstandingLayer | None = None,
    ) -> None:
        if page_size < 1 or max_workers < 1 or max_selected < 1:
            raise ValueError("page_size, max_workers and max_selected must be positive")
        self.providers = providers
        self.citation_providers = citation_provider if citation_provider is not None else providers
        self.citation_enabled = bool(citation_enabled)
        self.page_size = page_size
        self.max_workers = max_workers
        self.max_selected = max_selected
        self.understanding_layer = understanding_layer
        self.planner = QueryPlanner()
        self.gate = ConstraintGate()
        self.scorer = Scorer()

    @staticmethod
    def _message(run_id: str, seq: int, sender: str, receiver: str, message_type: str, ref: str, payload: Mapping[str, Any], *, diagnostic_code: str = "OK") -> dict[str, Any]:
        return make_message(run_id=run_id, message_id=f"msg_{run_id}_{seq}", message_type=message_type, sender=sender, receiver=receiver, seq=seq, payload=payload, payload_ref=ref, diagnostic_code=diagnostic_code)

    def _prepare(self, papers: Iterable[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], ConstraintVerdict, list[Any], float]]:
        return _prepare_papers(papers, plan, self.gate, self.scorer)

    def _score(self, prepared: Mapping[str, tuple[dict[str, Any], ConstraintVerdict, list[Any], float]], judgements: Mapping[str, Mapping[str, Any]], plan: Mapping[str, Any], unavailable_ids: Iterable[str] = ()) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return _apply_prepared_scores(prepared, judgements, plan, self.scorer, stage="p3", unavailable_ids=unavailable_ids)

    def _judge(self, plan: Mapping[str, Any], prepared: Mapping[str, tuple[dict[str, Any], ConstraintVerdict, list[Any], float]], errors: list[dict[str, Any]], *, budget: int | None) -> tuple[dict[str, dict[str, Any]], int, list[str]]:
        if self.understanding_layer is None or not prepared:
            return {}, 0, []
        eligible = [item for item in prepared.values() if item[1].state != "fail"]
        eligible.sort(key=lambda item: (item[3], str(item[0].get("paper_id") or "")), reverse=True)
        configured_limit = int(plan.get("budget", {}).get("max_judge_candidates", 20))
        if configured_limit < 1:
            return {}, 0, []
        candidates = [item[0] for item in eligible[:configured_limit]]
        skipped = [str(item[0].get("paper_id")) for item in eligible[configured_limit:]]
        if skipped:
            errors.append({"source": "deepseek", "stage": "judge", "code": "judge_budget_truncated", "message": "candidate limit is configured in QueryPlan budget", "count": len(skipped)})
        if budget is not None and budget <= 0:
            return {}, 0, skipped
        try:
            output = self.understanding_layer.judge(plan, candidates)
            for issue in getattr(self.understanding_layer, "last_judge_issues", []):
                errors.append({"source": "deepseek", "stage": "judge", "code": "judge_partial", "message": str(issue)[:200]})
            return {str(item["paper_id"]): item for item in output}, 1, skipped
        except (DeepSeekCallError, DeepSeekSchemaError, ValueError) as exc:
            errors.append(_safe_error("deepseek", exc, stage="judge"))
            return {}, 1, [*skipped, *[str(item["paper_id"]) for item in candidates]]

    def _review_conflicts(self, plan: Mapping[str, Any], prepared: Mapping[str, tuple[dict[str, Any], ConstraintVerdict, list[Any], float]], judgements: dict[str, dict[str, Any]], errors: list[dict[str, Any]], *, budget: int | None) -> tuple[int, list[str], list[str]]:
        conflicts = [paper_id for paper_id, judgement in judgements.items() if paper_id in prepared and abs(float(judgement.get("relevance_score") or 0.0) - prepared[paper_id][3]) > 0.25]
        if not conflicts:
            return 0, [], []
        if budget is not None and budget <= 0:
            errors.append({"source": "deepseek", "stage": "arbiter", "code": "review_budget_exhausted", "message": "conflict review skipped"})
            return 0, conflicts, []
        try:
            reviewed = self.understanding_layer.judge(plan, [prepared[item][0] for item in conflicts]) if self.understanding_layer is not None else []
            for issue in getattr(self.understanding_layer, "last_judge_issues", []):
                errors.append({"source": "deepseek", "stage": "arbiter_review", "code": "judge_partial", "message": str(issue)[:200]})
            for item in reviewed:
                judgements[str(item["paper_id"])] = item
            reviewed_ids = [str(item["paper_id"]) for item in reviewed]
            return 1, conflicts, reviewed_ids
        except (DeepSeekCallError, DeepSeekSchemaError, ValueError) as exc:
            errors.append(_safe_error("deepseek", exc, stage="arbiter_review"))
            return 1, conflicts, []

    def run(self, query: str, *, output_dir: str | Path | None = None) -> P3Run:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        wall_started = perf_counter()
        run_id = _run_id(query, self.citation_enabled)
        root = Path(output_dir) if output_dir is not None else Path("spar_solution/artifacts/p3") / run_id
        store = ArtifactStore(root)
        errors: list[dict[str, Any]] = []
        initial_plan = self.planner.plan(query)
        planner_source = "rules"
        llm_client = getattr(self.understanding_layer, "client", None) if self.understanding_layer is not None else None
        if llm_client is not None and callable(getattr(llm_client, "reset_usage", None)):
            llm_client.reset_usage(max_calls=int(initial_plan["budget"].get("max_llm_calls", 10)))
        if self.understanding_layer is not None:
            try:
                initial_plan = self.understanding_layer.plan(query)
                planner_source = "deepseek"
            except (DeepSeekCallError, DeepSeekSchemaError, ValueError) as exc:
                planner_source = "llm_fallback_rules"
                errors.append(_safe_error("deepseek", exc, stage="plan"))
        plan = initial_plan
        max_iterations = max(1, int(plan["budget"].get("max_iterations", 2)))
        max_provider_calls = int(plan["budget"].get("max_provider_calls", 100))
        stop_controller = StopController.from_query_plan(plan)
        messages: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        seen: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        all_verdicts: list[dict[str, Any]] = []
        all_evidence: list[dict[str, Any]] = []
        all_recall: list[dict[str, Any]] = []
        all_citation: list[dict[str, Any]] = []
        all_errors: list[dict[str, Any]] = []
        next_gaps = list(plan.get("gaps") or [])
        provider_calls_used = 0
        judge_batches = 0
        sequence = 0
        final_stop: dict[str, Any] = {}

        for iteration in range(max_iterations):
            if iteration > 0:
                plan = self.planner.next_iteration(plan, gaps=next_gaps)
            iteration_plan = _iteration_plan(plan, iteration)
            plan_ref = store.put("planner", iteration_plan, name=f"query_plan_iter{iteration}")
            messages.append(self._message(run_id, sequence, "planner", "retriever", "QUERY_PLAN", plan_ref, {"query_id": plan["query_id"], "iteration": iteration, "subqueries": iteration_plan["subqueries"]})); sequence += 1
            remaining = max(0, max_provider_calls - provider_calls_used)
            recall_result = RecallRunner(SourceRouter(self.providers), max_workers=self.max_workers, page_size=self.page_size).run(iteration_plan, iteration=iteration, max_calls=remaining)
            recall = recall_result.to_dict()
            all_recall.append(recall)
            provider_calls_used += _call_count([recall])
            recall_ref = store.put("retriever", recall, name=f"recall_iter{iteration}")
            source = str(recall_result.calls[0].get("source") if recall_result.calls else "arxiv")
            messages.append(self._message(run_id, sequence, "retriever", "citation_explorer", "RESULT_BATCH", recall_ref, {"batch_id": f"batch_{run_id}_{iteration}", "query_id": plan["query_id"], "iteration": iteration, "source": source, "paper_ids": [str(item["paper_id"]) for item in recall_result.records], "records_ref": recall_ref, "provenance_ref": recall_ref}, diagnostic_code="DEGRADED" if recall_result.source_errors else "OK")); sequence += 1
            iteration_records, dedup_errors = _deduplicate(recall_result.records)
            new_records = [item for item in iteration_records if str(item.get("paper_id")) not in seen_ids]
            for item in new_records:
                seen_ids.add(str(item["paper_id"]))
            merged, merge_errors = _deduplicate([*seen, *new_records])
            seen = merged
            all_errors.extend([*dedup_errors, *merge_errors, *recall_result.source_errors])
            prepared = self._prepare(new_records, plan)
            for item in prepared.values():
                item[0].setdefault("provenance", {})["query_id"] = plan["query_id"]
            citation_result = CitationExpander(self.citation_providers, enabled=self.citation_enabled, max_depth=int(plan["budget"].get("max_citation_depth", 1)), max_seeds=min(5, len(prepared)), page_size=self.page_size, max_workers=self.max_workers, max_api_calls=max(0, max_provider_calls - provider_calls_used)).expand([item[0] for item in prepared.values()], iteration=iteration)
            citation = citation_result.to_dict()
            all_citation.append(citation)
            provider_calls_used += _call_count([citation])
            citation_ref = store.put("citation_explorer", citation, name=f"citation_iter{iteration}")
            all_errors.extend(citation_result.source_errors)
            relation_source = str(citation_result.calls[0].get("source") if citation_result.calls else "arxiv")
            messages.append(self._message(run_id, sequence, "citation_explorer", "evidence_judge", "RELATION_BATCH", citation_ref, {"relation_batch_id": f"rel_{run_id}_{iteration}", "query_id": plan["query_id"], "iteration": iteration, "source": relation_source, "edges": citation_result.edges})); sequence += 1
            expanded, child_errors = _deduplicate([*new_records, *citation_result.papers])
            all_errors.extend(child_errors)
            candidate_records = [item for item in expanded if str(item.get("paper_id")) not in seen_ids]
            for item in candidate_records:
                seen_ids.add(str(item["paper_id"]))
            seen.extend(candidate_records)
            prepared.update(self._prepare(candidate_records, plan))
            judgements, batch_count, _ = self._judge(plan, prepared, all_errors, budget=max_provider_calls - provider_calls_used)
            judge_batches += batch_count
            review_count, conflict_ids, reviewed_conflict_ids = self._review_conflicts(plan, prepared, judgements, all_errors, budget=1)
            judge_batches += review_count
            judged, verdicts, evidence = self._score(prepared, judgements, plan, unavailable_ids=_)
            for item in judged:
                existing_index = next((index for index, current in enumerate(seen) if str(current.get("paper_id")) == str(item.get("paper_id"))), None)
                if existing_index is None:
                    seen.append(item)
                else:
                    seen[existing_index] = item
            all_verdicts.extend([{**item, "iteration": iteration, "conflict_detected": str(item.get("paper_id")) in conflict_ids, "conflict_reviewed": str(item.get("paper_id")) in reviewed_conflict_ids} for item in verdicts])
            all_evidence.extend([{**item, "iteration": iteration} for item in evidence])
            judgement_ref = store.put("evidence_judge", {"query_id": plan["query_id"], "papers": judged, "evidence": evidence, "verdicts": verdicts, "conflicts": conflict_ids, "reviewed_conflicts": reviewed_conflict_ids}, name=f"judgement_iter{iteration}")
            messages.append(self._message(run_id, sequence, "evidence_judge", "arbiter", "EVIDENCE_VERDICT", judgement_ref, {"query_id": plan["query_id"], "iteration": iteration, "verdicts_ref": judgement_ref, "candidate_count": len(verdicts), "top_paper_id": str(judged[0]["paper_id"]) if judged else "empty", "conflict_count": len(conflict_ids)}, diagnostic_code="CONFLICT" if conflict_ids else "OK")); sequence += 1
            selected = sorted([item for item in seen if item.get("scores", {}).get("final") is not None], key=lambda item: (float(item["scores"]["final"]), str(item.get("paper_id") or "")), reverse=True)[: self.max_selected]
            new_ids = {str(item.get("paper_id")) for item in new_records + candidate_records}
            new_relevant = sum(1 for item in selected if str(item.get("paper_id")) in new_ids and float(item.get("scores", {}).get("relevance") or 0) >= 0.6)
            successful = sum(1 for call in recall_result.calls + citation_result.calls if call.get("ok"))
            total_round_calls = _call_count([recall, citation])
            evidence_ids = {str(item.get("paper_id")) for item in all_evidence if item.get("evidence_status") not in {None, "unavailable"}}
            coverage = sum(1 for call in recall_result.calls if call.get("ok")) / max(1, len(recall_result.calls))
            evidence_coverage = len(evidence_ids) / max(1, len(seen))
            next_gaps = [gap for gap in plan.get("gaps") or [] if gap != "citation_neighbor_gain"]
            if not next_gaps and new_relevant < int(plan["stop_strategy"].get("min_new_relevant", 2)):
                next_gaps = ["citation_neighbor_gain"]
            decision = stop_controller.decide(iteration=iteration, citation_depth=0, provider_calls=provider_calls_used, provider_successes=successful, new_unique_papers=[len(new_records) + len(candidate_records)], new_relevant_papers=new_relevant, subquery_coverage=min(1.0, coverage), evidence_coverage=min(1.0, evidence_coverage), budget_exhausted=provider_calls_used >= max_provider_calls)
            if iteration + 1 >= max_iterations and not decision.should_stop:
                decision = stop_controller.decide(iteration=iteration + 1, citation_depth=0, provider_calls=provider_calls_used, provider_successes=successful, new_unique_papers=[len(new_records) + len(candidate_records)], new_relevant_papers=new_relevant, subquery_coverage=min(1.0, coverage), evidence_coverage=min(1.0, evidence_coverage), budget_exhausted=False)
            final_stop = decision.to_dict()
            stop_ref = store.put("arbiter", final_stop, name=f"stop_iter{iteration}")
            action = "STOP" if decision.should_stop else "NEXT_QUERY"
            messages.append(self._message(run_id, sequence, "arbiter", "orchestrator", "STOP_DECISION", stop_ref, {"query_id": plan["query_id"], "iteration": iteration, "action": action, "reason_code": decision.reason_code}, diagnostic_code="OK" if decision.reason_code == "CONTINUE" else "DEGRADED" if decision.decision_type == "soft" else "OK")); sequence += 1
            rounds.append({"iteration": iteration, "query_plan_ref": plan_ref, "recall_ref": recall_ref, "citation_ref": citation_ref, "judgement_ref": judgement_ref, "stop_ref": stop_ref, "new_records": len(new_records) + len(candidate_records), "provider_calls": total_round_calls, "conflict_count": len(conflict_ids), "action": action})
            if decision.should_stop:
                break

        all_edges = [edge for batch in all_citation for edge in batch.get("edges") or []]
        final_run = P3Run(query=query, run_id=run_id, query_plan=plan.to_dict(), recall={"rounds": all_recall}, citation={"rounds": all_citation, "edges": all_edges, "stats": {"enabled": self.citation_enabled, "edges": len(all_edges), "api_calls": sum(_call_count([item]) for item in all_citation)}}, evidence=all_evidence, verdicts=all_verdicts, papers=sorted(seen, key=lambda item: (item.get("scores", {}).get("final") is not None, item.get("scores", {}).get("final") or -1, str(item.get("paper_id") or "")), reverse=True), selected=[], stop=final_stop, messages=[], errors=all_errors, rounds=rounds)
        evidence_files = _write_evidence_refs(root, final_run.papers, final_run.evidence, final_run.verdicts)
        final_run.selected = [
            item for item in final_run.papers
            if item.get("status", {}).get("hard_constraints_pass") is not False
            and isinstance(item.get("scores", {}).get("final"), (int, float))
        ][: self.max_selected]
        usage = getattr(llm_client, "usage", {}) if llm_client is not None else {}
        usage = usage if isinstance(usage, Mapping) else {}
        final_run.cost = {
            "provider_calls": provider_calls_used,
            "llm_calls": int(usage.get("calls", 0)),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "llm_failures": int(usage.get("failures", 0)),
            "latency_ms": round(float(usage.get("latency_ms", 0.0) or 0.0), 3),
            "wall_ms": round((perf_counter() - wall_started) * 1000, 3),
        }
        final_selection = build_final_selection(final_run, top_k=self.max_selected)
        selection_ref = store.put("arbiter", final_selection, name="final_selection")
        graph_ref = store.put("arbiter", final_selection["relation_graph"], name="relation_graph")
        messages.append(self._message(run_id, sequence, "arbiter", "orchestrator", "FINAL_SELECTION", selection_ref, {"query_id": plan["query_id"], "selection_ref": selection_ref, "relation_graph_ref": graph_ref, "selected_count": len(final_selection["results"]), "selections": [{"paper_id": item["paper_id"], "final_score": item["final_score"], "evidence_refs": item["evidence_refs"]} for item in final_selection["results"]]})); sequence += 1
        final_run.messages = [validate_message(item) for item in messages]
        protocol_ref = store.write_jsonl("protocol.jsonl", final_run.messages)
        final_run.final_selection = final_selection
        final_run.cost["wall_ms"] = round((perf_counter() - wall_started) * 1000, 3)
        final_selection["cost"] = deepcopy(final_run.cost)
        final_selection = validate_final_selection(final_selection)
        store.put("arbiter", final_selection, name="final_selection")
        final_run.stats = self._stats(final_run, root)
        final_run.manifest = {"schema_version": "p3_run.v2", "run_id": run_id, "query": query, "query_id": plan["query_id"], "citation_enabled": self.citation_enabled, "planner_source": planner_source, "deepseek_judge_batches": judge_batches, "generated_at": datetime.now(timezone.utc).isoformat(), "iterations": len(rounds), "roles": ["planner", "retriever", "citation_explorer", "evidence_judge", "arbiter"], "providers": _provider_names(self.providers), "cost": final_run.cost, "stats": final_run.stats, "evidence_files": evidence_files, "rounds": rounds, "artifacts": {"query_plan": rounds[0]["query_plan_ref"] if rounds else "", "protocol": protocol_ref, "final_selection": selection_ref, "relation_graph": graph_ref}}
        store.put("arbiter", final_run.stats, name="metrics")
        store.put("arbiter", final_run.manifest, name="run_manifest")
        # Recompute after bookkeeping artifacts are present so artifact_count
        # describes the complete replay directory, not only stage outputs.
        final_run.stats = self._stats(final_run, root)
        final_run.manifest["stats"] = final_run.stats
        store.put("arbiter", final_run.stats, name="metrics")
        store.put("arbiter", final_run.manifest, name="run_manifest")
        return final_run

    @staticmethod
    def _stats(run: P3Run, root: Path) -> dict[str, Any]:
        message_bytes = sum(estimate_bytes(item) for item in run.messages)
        files = [path for path in root.rglob("*") if path.is_file()]
        provider_calls_by_source: dict[str, int] = {}
        provider_latency_by_source: dict[str, float] = {}
        for group in [*run.recall.get("rounds", []), *run.citation.get("rounds", [])]:
            for call in group.get("calls") or []:
                source = str(call.get("source") or "unknown")
                provider_calls_by_source[source] = provider_calls_by_source.get(source, 0) + int(call.get("api_calls") or 1)
                provider_latency_by_source[source] = provider_latency_by_source.get(source, 0.0) + float(call.get("latency_ms") or 0.0)
        return {"agent_count": 5, "message_count": len(run.messages), "protocol_message_bytes": message_bytes, "protocol_message_tokens_estimate": estimate_tokens(run.messages), "long_text_message_bytes_estimate": sum(estimate_bytes({"message_type": item.get("type"), "papers": run.papers}) for item in run.messages), "long_text_message_tokens_estimate": estimate_tokens({"papers": run.papers}) * max(1, len(run.messages)), "artifact_count": len(files), "artifact_bytes": sum(path.stat().st_size for path in files), "candidate_count": len(run.papers), "selected_count": len(run.selected), "source_errors": len(run.errors), "provider_calls_by_source": provider_calls_by_source, "provider_latency_ms_by_source": provider_latency_by_source, "citation_enabled": bool(run.citation.get("stats", {}).get("enabled")), "citation_edges": sum(len(batch.get("edges") or []) for batch in run.citation.get("rounds", []))}


def replay_p3(output_dir: str | Path) -> dict[str, Any]:
    """Replay and validate a P3 artifact directory without calling providers."""

    root = Path(output_dir)
    manifest_path = root / "arbiter" / "run_manifest.json"
    protocol_path = root / "protocol.jsonl"
    final_path = root / "arbiter" / "final_selection.json"
    if not manifest_path.is_file() or not protocol_path.is_file() or not final_path.is_file():
        raise FileNotFoundError("P3 manifest, protocol.jsonl or final_selection is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    messages = [validate_message(json.loads(line)) for line in protocol_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    final_selection = validate_final_selection(json.loads(final_path.read_text(encoding="utf-8")))
    query_id = str(manifest.get("query_id") or "")
    if not query_id or final_selection.get("query_id") != query_id:
        raise ValueError("P3 query_id does not match final_selection")
    expected_seq = list(range(len(messages)))
    actual_seq = [int(message.get("seq")) for message in messages]
    if actual_seq != expected_seq:
        raise ValueError("protocol seq must be contiguous and ordered")
    for message in messages:
        if message.get("run_id") != manifest.get("run_id"):
            raise ValueError("protocol message run_id does not match manifest")
        if message.get("payload", {}).get("query_id") not in {None, query_id}:
            raise ValueError("protocol payload query_id does not match manifest")
        ref = message.get("payload_ref")
        if ref:
            path = (root / ref).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"missing or out-of-root payload_ref: {ref}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and payload.get("query_id") not in {None, query_id}:
                raise ValueError(f"artifact query_id does not match manifest: {ref}")
            short = message.get("payload") or {}
            if message.get("type") == "EVIDENCE_VERDICT":
                if not isinstance(payload, Mapping) or len(payload.get("verdicts") or []) != int(short.get("candidate_count", -1)):
                    raise ValueError("EVIDENCE_VERDICT candidate_count does not match artifact")
            if message.get("type") == "FINAL_SELECTION":
                if not isinstance(payload, Mapping) or len(payload.get("results") or []) != int(short.get("selected_count", -1)):
                    raise ValueError("FINAL_SELECTION selected_count does not match artifact")
                if payload != final_selection:
                    raise ValueError("FINAL_SELECTION payload does not match final_selection artifact")
        for ref_key in ("records_ref", "provenance_ref", "verdicts_ref", "selection_ref", "relation_graph_ref"):
            ref_value = (message.get("payload") or {}).get(ref_key)
            if ref_value:
                path = (root / ref_value).resolve()
                if root.resolve() not in path.parents or not path.is_file():
                    raise ValueError(f"missing or out-of-root payload reference: {ref_value}")
                if message.get("type") == "FINAL_SELECTION" and ref_key == "relation_graph_ref":
                    relation_graph = json.loads(path.read_text(encoding="utf-8"))
                    if relation_graph != final_selection.get("relation_graph"):
                        raise ValueError("relation_graph_ref does not match final selection")
    for row in final_selection["results"]:
        for ref in row.get("evidence_refs") or []:
            path = (root / ref).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"missing or out-of-root evidence_ref: {ref}")
            text = path.read_text(encoding="utf-8")
            if not text.startswith(f"paper_id: {row['paper_id']}\n"):
                raise ValueError(f"evidence_ref does not identify its paper: {ref}")
    for judgement_path in sorted((root / "evidence_judge").glob("judgement_iter*.json")):
        judgement = json.loads(judgement_path.read_text(encoding="utf-8"))
        if judgement.get("query_id") not in {None, query_id}:
            raise ValueError(f"judgement query_id does not match manifest: {judgement_path.name}")
        for row in judgement.get("evidence") or []:
            ref = row.get("evidence_ref")
            if not ref:
                continue
            path = (root / ref).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"judgement evidence_ref is missing: {ref}")
            if not path.read_text(encoding="utf-8").startswith(f"paper_id: {row.get('paper_id')}\n"):
                raise ValueError(f"judgement evidence_ref paper mismatch: {ref}")
    return {"manifest": manifest, "messages": messages, "final_selection": final_selection}


def run_p3_fixture(query: str = "WiFi heart rate monitoring", *, output_dir: str | Path | None = None, citation_enabled: bool = True) -> P3Run:
    from .mock_pipeline import _paper

    seed = _paper("arxiv", "WiFi CSI heart rate monitoring using contactless vital sign estimation.")
    seed["paper_id"] = "fixture:p3:seed"
    seed["identifiers"]["doi"] = "10.1234/fixture.p3.seed"
    seed["provenance"]["execution_status"] = "mock"
    child = _paper("arxiv", "Reference paper for WiFi CSI heart rate measurement.")
    child["paper_id"] = "fixture:p3:child"
    child["identifiers"]["doi"] = "10.1234/fixture.p3.child"
    child["provenance"]["execution_status"] = "mock"
    child["relation_type"] = "references"
    provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: [child]})
    return P3Pipeline({"arxiv": provider}, citation_enabled=citation_enabled).run(query, output_dir=output_dir)


def run_p3_comparison_fixture(query: str = "WiFi heart rate monitoring", *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run P2/P3 and structured/long-text fixture comparisons into one artifact."""

    from .p2_pipeline import run_p2_fixture
    from .p3_metrics import compare_communication, compare_p2_p3, long_text_baseline

    root = Path(output_dir) if output_dir is not None else Path("spar_solution/artifacts/p3") / "comparison"
    p2 = run_p2_fixture(query, output_dir=root / "p2", citation_enabled=True)
    p3 = run_p3_fixture(query, output_dir=root / "p3", citation_enabled=True)
    comparison = {"query": query, "p2_p3": compare_p2_p3(p2, p3), "communication": compare_communication(p3, long_text_baseline(p3))}
    (root / "comparison.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparison


__all__ = ["P3Pipeline", "P3Run", "replay_p3", "run_p3_comparison_fixture", "run_p3_fixture"]
