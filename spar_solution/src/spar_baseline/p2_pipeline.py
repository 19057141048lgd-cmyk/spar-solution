"""P2 可回放流水线与无 Key fixture。

本模块只编排已有 P2 组件，不复制 Provider、PaperDoc 或评分规则。每个阶段
均输出 JSON 兼容对象，便于审计、回放和 citation enabled/disabled 消融。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paperdoc import canonical_paper_key, merge_paper_docs, validate_paper_doc
from .p2_citation import CitationExpander
from .p2_evidence import ConstraintGate, EvidenceLoader
from .p2_recall import RecallRunner, SourceRouter
from .p2_scoring import EvidenceVerdict, Scorer
from .p2_stop import StopController, StopDecision
from .providers.base import ProviderError, ProviderResult
from .query_plan import QueryPlanValidationError, validate_query_plan
from .query_planner import QueryPlanner
from .deepseek_layer import DeepSeekCallError, DeepSeekSchemaError, DeepSeekUnderstandingLayer
from .p2_evidence import ConstraintResult, ConstraintVerdict


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _deduplicate(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 P1 身份规则合并 PaperDoc；无效/冲突记录保留为错误。"""

    output: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    by_key: dict[str, int] = {}
    for index, record in enumerate(records):
        item = deepcopy(dict(record))
        try:
            validate_paper_doc(item)
            key = canonical_paper_key(item)
            if key.startswith("ambiguous:"):
                key = f"{key}:record:{index}"
            if key not in by_key:
                by_key[key] = len(output)
                output.append(item)
                continue
            position = by_key[key]
            try:
                output[position] = merge_paper_docs(output[position], item)
            except Exception as exc:
                errors.append({"code": "identity_conflict", "message": str(exc), "record_index": index})
        except Exception as exc:
            errors.append({"code": "parse", "message": str(exc), "record_index": index})
    return output, errors


def _annotate(doc: dict[str, Any], verdict: EvidenceVerdict, constraint: Any, evidence: list[Any]) -> dict[str, Any]:
    item = deepcopy(doc)
    item.setdefault("scores", {}).update({**verdict.component_scores, "final": verdict.final_score, "confidence": verdict.confidence})
    item.setdefault("status", {}).update({"hard_constraints_pass": constraint.passed, "evidence_status": verdict.evidence_status})
    item["evidence_refs"] = list(dict.fromkeys(item.get("evidence_refs", []) + list(verdict.evidence_refs)))
    evidence_ids = []
    for evidence_item in evidence:
        if hasattr(evidence_item, "evidence_id"):
            evidence_ids.append(str(evidence_item.evidence_id))
        elif isinstance(evidence_item, Mapping) and evidence_item.get("evidence_id"):
            evidence_ids.append(str(evidence_item["evidence_id"]))
    item.setdefault("provenance", {}).setdefault("p2", {})["evidence_ids"] = evidence_ids
    validate_paper_doc(item)
    return item


@dataclass
class P2Run:
    query: str
    query_plan: dict[str, Any]
    recall: list[dict[str, Any]] = field(default_factory=list)
    papers: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    stops: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


class P2Pipeline:
    """执行固定 P2 流程：计划、召回、去重、门控、证据、评分、引用、停止。"""

    def __init__(self, providers: Mapping[str, Any] | Iterable[Any], *, citation_provider: Mapping[str, Any] | Iterable[Any] | None = None, citation_enabled: bool = True, page_size: int = 10, max_workers: int = 4, understanding_layer: DeepSeekUnderstandingLayer | None = None) -> None:
        self.providers = providers
        self.citation_providers = citation_provider if citation_provider is not None else providers
        self.citation_enabled = bool(citation_enabled)
        self.page_size = page_size
        self.max_workers = max_workers
        self.planner = QueryPlanner()
        self.understanding_layer = understanding_layer
        self.gate = ConstraintGate()
        self.scorer = Scorer()

    def run(self, query: str, *, output_dir: str | Path | None = None) -> P2Run:
        errors: list[dict[str, Any]] = []
        plan = self.planner.plan(query)
        if self.understanding_layer is not None:
            try:
                plan = self.understanding_layer.plan(query)
            except (DeepSeekCallError, DeepSeekSchemaError, ValueError) as exc:
                errors.append({"source": "deepseek", "code": getattr(exc, "code", "plan_fallback"), "message": str(exc)[:200], "stage": "plan"})
        run = P2Run(query=query, query_plan=plan.to_dict(), errors=errors)
        seen: list[dict[str, Any]] = []
        deepseek_judge_batches = 0
        previous_new: list[int] = []
        stop_controller = StopController.from_query_plan(plan)
        citation_depth = 0
        max_iterations = int(plan["budget"]["max_iterations"])
        for iteration in range(max_iterations):
            prior_ids = {str(item["paper_id"]) for item in seen}
            current_plan = self.planner.next_iteration(plan, gaps=plan.get("gaps")) if iteration and iteration < max_iterations else plan
            if iteration:
                plan = current_plan
                run.query_plan = plan.to_dict()
            iteration_plan = plan.to_dict()
            iteration_plan["subqueries"] = [item for item in plan["subqueries"] if item["iteration"] == iteration]
            # QueryPlan uses ``source_capabilities``; SourceRouter accepts the
            # executable alias ``sources``. Keep the protocol fields untouched
            # and add only a local routing view.
            for subquery in iteration_plan["subqueries"]:
                subquery["sources"] = list(subquery.get("source_capabilities") or [])
            calls_before = sum(item.get("stats", {}).get("api_calls", 0) for item in run.recall) + sum(item.get("stats", {}).get("api_calls", 0) for item in run.citations)
            remaining_calls = max(0, int(plan["budget"]["max_provider_calls"]) - calls_before)
            recall = RecallRunner(SourceRouter(self.providers), max_workers=self.max_workers, page_size=self.page_size).run(iteration_plan, iteration=iteration, max_calls=remaining_calls)
            run.recall.append(recall.to_dict())
            run.errors.extend(recall.source_errors)
            unique, dedup_errors = _deduplicate([*seen, *recall.records])
            run.errors.extend(dedup_errors)
            new_count = max(0, len(unique) - len(seen))
            seen = unique
            for paper in seen:
                paper.setdefault("provenance", {})["query_id"] = plan["query_id"]
            constraints: list[Any] = []
            evidence_items: list[Any] = []
            scored: list[dict[str, Any]] = []
            judgements: dict[str, dict[str, Any]] = {}
            if self.understanding_layer is not None and seen:
                try:
                    for start in range(0, min(len(seen), 12)):
                        deepseek_judge_batches += 1
                        judgements.update({str(item["paper_id"]): item for item in self.understanding_layer.judge(plan, seen[start:start + 1])})
                except (DeepSeekCallError, DeepSeekSchemaError, ValueError) as exc:
                    run.errors.append({"source": "deepseek", "code": getattr(exc, "code", "judge_fallback"), "message": str(exc)[:200], "stage": "judge"})
            for paper in seen:
                constraint = self.gate.evaluate(plan, paper)
                judgement = judgements.get(str(paper.get("paper_id")))
                if judgement is not None:
                    state = str(judgement.get("hard_constraint_state") or "unknown")
                    if state == "fail":
                        constraint = ConstraintVerdict(state="fail", results=constraint.results + (ConstraintResult("deepseek", "explicit", "fail", "deepseek_explicit_fail"),), reason_codes=constraint.reason_codes + ("deepseek_explicit_fail",))
                    elif state == "unknown" and constraint.state == "pass":
                        constraint = ConstraintVerdict(state="unknown", results=constraint.results, reason_codes=constraint.reason_codes + ("deepseek_unknown",))
                items = EvidenceLoader().load(paper, required_status="abstract")
                overrides = {"relevance": judgement["relevance_score"]} if judgement is not None else None
                verdict = self.scorer.score(paper, plan, constraint, items, component_overrides=overrides)
                scored.append(_annotate(paper, verdict, constraint, items))
                constraints.append(constraint)
                evidence_items.extend(items)
                for evidence_item in items:
                    evidence_dict = evidence_item.to_dict()
                    evidence_dict["iteration"] = iteration
                    run.evidence.append(evidence_dict)
                verdict_dict = verdict.to_dict()
                verdict_dict["iteration"] = iteration
                run.verdicts.append(verdict_dict)
            # 引用扩展和最终输出必须使用已经附加分量分、约束状态及
            # evidence_refs 的 PaperDoc，而不能继续保留原始候选。
            seen = scored
            scored.sort(key=lambda p: (p.get("scores", {}).get("final") is not None, p.get("scores", {}).get("final") or -1, p.get("paper_id", "")), reverse=True)
            citation_budget = max(0, remaining_calls - recall.stats.get("api_calls", 0))
            citation = CitationExpander(self.citation_providers, enabled=self.citation_enabled, max_depth=1, max_seeds=min(5, citation_budget), page_size=self.page_size, max_workers=self.max_workers).expand(scored, iteration=iteration)
            run.citations.append(citation.to_dict())
            run.errors.extend(citation.source_errors)
            if citation.papers:
                scored_children: list[dict[str, Any]] = []
                for paper in citation.papers:
                    paper.setdefault("provenance", {})["query_id"] = plan["query_id"]
                    constraint = self.gate.evaluate(plan, paper)
                    items = EvidenceLoader().load(paper, required_status="abstract")
                    judgement = judgements.get(str(paper.get("paper_id")))
                    overrides = {"relevance": judgement["relevance_score"]} if judgement is not None else None
                    verdict = self.scorer.score(paper, plan, constraint, items, component_overrides=overrides)
                    scored_children.append(_annotate(paper, verdict, constraint, items))
                    for evidence_item in items:
                        evidence_dict = evidence_item.to_dict()
                        evidence_dict["iteration"] = iteration
                        run.evidence.append(evidence_dict)
                    verdict_dict = verdict.to_dict()
                    verdict_dict["iteration"] = iteration
                    run.verdicts.append(verdict_dict)
                expanded, expand_errors = _deduplicate([*seen, *scored_children])
                run.errors.extend(expand_errors)
                citation_depth = 1
                new_count = max(new_count, len(expanded) - len(seen))
                seen = expanded
            run.papers = sorted(seen, key=lambda p: (p.get("scores", {}).get("final") is not None, p.get("scores", {}).get("final") or -1, p.get("paper_id", "")), reverse=True)
            new_ids = {str(item["paper_id"]) for item in seen} - prior_ids
            new_count = len(new_ids)
            previous_new.append(new_count)
            successful = sum(item.get("stats", {}).get("successful_calls", 0) for item in run.recall)
            calls = sum(item.get("stats", {}).get("api_calls", 0) for item in run.recall) + sum(item.get("stats", {}).get("api_calls", 0) for item in run.citations)
            successful_subqueries = {item.get("subquery_id") for item in recall.calls if item.get("ok")}
            coverage = len(successful_subqueries) / max(1, len(iteration_plan["subqueries"]))
            evidence_papers = {str(item.get("paper_id")) for item in run.evidence if item.get("evidence_status") != "unavailable"}
            evidence_coverage = len(evidence_papers) / max(1, len(seen))
            new_relevant = sum(1 for item in seen if str(item["paper_id"]) in new_ids and (item.get("scores", {}).get("relevance") or 0) >= 0.6)
            decision = stop_controller.decide(iteration=iteration, citation_depth=citation_depth, provider_calls=calls, provider_successes=successful, new_unique_papers=previous_new, new_relevant_papers=new_relevant, subquery_coverage=min(1.0, coverage), evidence_coverage=min(1.0, evidence_coverage), budget_exhausted=calls >= plan["budget"]["max_provider_calls"])
            run.stops.append(decision.to_dict())
            if decision.should_stop:
                break
        run.manifest = {"schema_version": "p2_run.v1", "query": query, "query_id": plan["query_id"], "citation_enabled": self.citation_enabled, "deepseek_judge_batches": deepseek_judge_batches, "generated_at": datetime.now(timezone.utc).isoformat(), "iterations": len(run.recall), "status": "degraded" if run.errors else "ok"}
        if output_dir is not None:
            self.write_artifacts(run, output_dir)
        return run

    @staticmethod
    def write_artifacts(run: P2Run, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        evidence_root = root / "evidence"
        evidence_files: list[str] = []
        paper_by_id = {str(paper.get("paper_id")): paper for paper in run.papers}
        ref_rewrites: dict[str, str] = {}
        for item in run.evidence:
            if item.get("evidence_status") in {None, "unavailable"}:
                continue
            paper_id = str(item.get("paper_id") or "")
            abstract = str((paper_by_id.get(paper_id) or {}).get("bibliography", {}).get("abstract") or "")
            if not abstract:
                continue
            safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in paper_id)[:120] or "paper"
            evidence_id = str(item.get("evidence_id") or "evidence")
            safe_evidence_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in evidence_id)[:160] or "evidence"
            evidence_path = evidence_root / safe_id / f"{safe_evidence_id}.txt"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(abstract, encoding="utf-8")
            old_ref = str(item.get("evidence_ref") or "")
            relative_ref = str(evidence_path.relative_to(root)).replace("\\", "/")
            if old_ref:
                ref_rewrites[old_ref] = relative_ref
            item["evidence_ref"] = relative_ref
            evidence_files.append(relative_ref)
        for paper in run.papers:
            paper["evidence_refs"] = [ref_rewrites.get(str(ref), str(ref)) for ref in paper.get("evidence_refs") or []]
        for verdict in run.verdicts:
            verdict["evidence_refs"] = [ref_rewrites.get(str(ref), str(ref)) for ref in verdict.get("evidence_refs") or []]
        run.manifest["evidence_files"] = sorted(set(evidence_files))
        files = {"query_plan.json": run.query_plan, "recall.json": run.recall, "papers.json": {"papers": run.papers}, "citation.json": run.citations, "evidence.json": run.evidence, "verdicts.json": run.verdicts, "stop.json": run.stops, "errors.json": run.errors, "run_manifest.json": run.manifest}
        for name, value in files.items():
            _write_json(root / name, value)
        return root


def replay_p2(output_dir: str | Path) -> dict[str, Any]:
    """读取并校验 artifact，返回与运行结果同形状的回放对象。"""
    root = Path(output_dir)
    names = ("query_plan", "recall", "papers", "citation", "evidence", "verdicts", "stop", "errors", "run_manifest")
    output: dict[str, Any] = {}
    for name in names:
        path = root / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = json.loads(path.read_text(encoding="utf-8"))
    try:
        validate_query_plan(output["query_plan"])
    except (KeyError, TypeError, QueryPlanValidationError) as exc:
        raise ValueError("invalid query_plan artifact") from exc
    manifest = output["run_manifest"]
    if not isinstance(manifest, Mapping) or manifest.get("query_id") != output["query_plan"].get("query_id"):
        raise ValueError("run_manifest query_id does not match query_plan")
    papers = output["papers"].get("papers") if isinstance(output["papers"], Mapping) else None
    if not isinstance(papers, list):
        raise ValueError("papers artifact must contain a papers array")
    root_resolved = root.resolve()
    for paper in papers:
        try:
            validate_paper_doc(paper)
        except Exception as exc:
            raise ValueError(f"invalid PaperDoc artifact: {paper.get('paper_id', '<unknown>')}") from exc
        for ref in paper.get("evidence_refs") or []:
            evidence_path = (root / str(ref)).resolve()
            if root_resolved not in evidence_path.parents or not evidence_path.is_file():
                raise ValueError(f"missing or out-of-root evidence_ref: {ref}")
    return output


class FixtureProvider:
    """确定性 fixture Provider；用于无 Key P2 回放和消融。"""

    def __init__(self, name: str, records: list[dict[str, Any]], relations: Mapping[str, list[dict[str, Any]]] | None = None) -> None:
        self.name = name
        self.records = records
        self.relation_records = dict(relations or {})
        self.search_calls = 0
        self.relation_calls = 0

    def search(self, query: str, *, page_size: int = 10) -> ProviderResult:
        self.search_calls += 1
        return ProviderResult(self.name, "search", [deepcopy(item) for item in self.records[:page_size]], total=len(self.records))

    def relations(self, paper_id: str, *, relation: str = "all", page_size: int = 10) -> ProviderResult:
        self.relation_calls += 1
        return ProviderResult(self.name, "relations", [deepcopy(item) for item in self.relation_records.get(paper_id, [])[:page_size]], total=len(self.relation_records.get(paper_id, [])))


def run_p2_fixture(query: str = "WiFi heart rate monitoring", *, output_dir: str | Path | None = None, citation_enabled: bool = True) -> P2Run:
    from .mock_pipeline import _paper
    seed = _paper("arxiv", "WiFi heart rate monitoring using CSI signals and contactless vital sign estimation.")
    seed["paper_id"] = "fixture:seed"
    seed["identifiers"]["doi"] = "10.1234/fixture.seed"
    seed["provenance"]["execution_status"] = "mock"
    child = _paper("arxiv", "Reference paper on WiFi CSI heart rate measurement.")
    child["paper_id"] = "fixture:child"
    child["identifiers"]["doi"] = "10.1234/fixture.child"
    child["provenance"]["execution_status"] = "mock"
    child["relation_type"] = "references"
    provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: [child]})
    return P2Pipeline({"arxiv": provider}, citation_enabled=citation_enabled).run(query, output_dir=output_dir)


__all__ = ["FixtureProvider", "P2Pipeline", "P2Run", "replay_p2", "run_p2_fixture"]
