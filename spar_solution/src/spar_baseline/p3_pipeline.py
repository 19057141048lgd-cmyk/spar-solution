"""P3 五 Agent 结构化圆桌 fixture 流程。

五个角色只通过 :mod:`p3_protocol` 的 artifact 引用传递阶段结果；实际论文
对象保存在 JSON artifact 中，不复制到 Agent 消息正文。这里的 Agent 是职责
边界和可替换执行器，fixture 模式不需要 LLM Key。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .p2_citation import CitationExpander
from .p2_evidence import ConstraintGate, ConstraintResult, ConstraintVerdict, EvidenceLoader
from .p2_pipeline import FixtureProvider, _deduplicate
from .p2_recall import RecallRunner, SourceRouter
from .p2_scoring import EvidenceVerdict, Scorer
from .p2_stop import StopController
from .paperdoc import validate_paper_doc
from .p3_protocol import ArtifactStore, estimate_bytes, estimate_tokens, make_message, validate_message
from .query_planner import QueryPlanner
from .deepseek_layer import DeepSeekCallError, DeepSeekSchemaError, DeepSeekUnderstandingLayer


def _run_id(query: str, citation_enabled: bool) -> str:
    suffix = "cit" if citation_enabled else "nocit"
    return "p3_" + hashlib.sha256(f"{query}|{suffix}".encode("utf-8")).hexdigest()[:12]


def _annotate(doc: Mapping[str, Any], verdict: EvidenceVerdict, constraint: Any, evidence: list[Any]) -> dict[str, Any]:
    item = deepcopy(dict(doc))
    item.setdefault("scores", {}).update({**verdict.component_scores, "final": verdict.final_score, "confidence": verdict.confidence})
    item.setdefault("status", {}).update({"hard_constraints_pass": constraint.passed, "evidence_status": verdict.evidence_status})
    item["evidence_refs"] = list(dict.fromkeys(item.get("evidence_refs", []) + list(verdict.evidence_refs)))
    evidence_ids = []
    for evidence_item in evidence:
        if isinstance(evidence_item, Mapping):
            evidence_ids.append(str(evidence_item.get("evidence_id")))
        elif hasattr(evidence_item, "evidence_id"):
            evidence_ids.append(str(evidence_item.evidence_id))
        else:
            evidence_ids.append(str(evidence_item))
    item.setdefault("provenance", {}).setdefault("p3", {})["evidence_ids"] = evidence_ids
    validate_paper_doc(item)
    return item


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

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.__dict__)


class P3Pipeline:
    """按固定 Planner/Retriever/CitationExplorer/EvidenceJudge/Arbiter 顺序运行。"""

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

    def _message(self, run_id: str, seq: int, sender: str, receiver: str, message_type: str, ref: str, payload: Mapping[str, Any], *, diagnostic_code: str = "OK") -> dict[str, Any]:
        return make_message(
            run_id=run_id,
            message_id=f"msg_{run_id}_{seq}",
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            seq=seq,
            payload=payload,
            payload_ref=ref,
            diagnostic_code=diagnostic_code,
        )

    def run(self, query: str, *, output_dir: str | Path | None = None) -> P3Run:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        run_id = _run_id(query, self.citation_enabled)
        root = Path(output_dir) if output_dir is not None else Path("spar_solution/artifacts/p3") / run_id
        store = ArtifactStore(root)
        errors: list[dict[str, Any]] = []
        plan = self.planner.plan(query)
        if self.understanding_layer is not None:
            try:
                plan = self.understanding_layer.plan(query)
            except (DeepSeekCallError, DeepSeekSchemaError, ValueError) as exc:
                errors.append({"source": "deepseek", "code": getattr(exc, "code", "plan_fallback"), "message": str(exc)[:200], "stage": "plan"})
        messages: list[dict[str, Any]] = []
        deepseek_judge_batches = 0
        plan_ref = store.put("planner", plan.to_dict(), name="query_plan")
        messages.append(self._message(run_id, 0, "planner", "retriever", "QUERY_PLAN", plan_ref, {"query_id": plan["query_id"], "subqueries": plan["subqueries"]}))

        routed_plan = plan.to_dict()
        for subquery in routed_plan["subqueries"]:
            subquery["sources"] = list(subquery.get("source_capabilities") or [])
        recall_result = RecallRunner(SourceRouter(self.providers), max_workers=self.max_workers, page_size=self.page_size).run(routed_plan, iteration=0, max_calls=plan["budget"]["max_provider_calls"])
        recall = recall_result.to_dict()
        recall_ref = store.put("retriever", recall, name="recall")
        source = str(recall_result.calls[0].get("source") if recall_result.calls else "arxiv")
        messages.append(self._message(run_id, 1, "retriever", "citation_explorer", "RESULT_BATCH", recall_ref, {"batch_id": f"batch_{run_id}", "query_id": plan["query_id"], "source": source, "paper_ids": [str(item["paper_id"]) for item in recall_result.records], "records_ref": recall_ref, "provenance_ref": recall_ref}, diagnostic_code="DEGRADED" if recall_result.source_errors else "OK"))

        papers, dedup_errors = _deduplicate(recall_result.records)
        # CitationExplorer receives a preliminary deterministic gate result. The
        # full evidence judgment still runs after expansion; unknown constraints
        # remain eligible when no hard constraint failed.
        for paper in papers:
            preliminary = self.gate.evaluate(plan, paper)
            paper.setdefault("status", {})["hard_constraints_pass"] = preliminary.passed
            paper.setdefault("scores", {})["relevance"] = self.scorer._relevance(paper, plan)
        citation_result = CitationExpander(
            self.citation_providers,
            enabled=self.citation_enabled,
            max_depth=int(plan["budget"]["max_citation_depth"]),
            max_seeds=min(5, len(papers)),
            page_size=self.page_size,
            max_workers=self.max_workers,
        ).expand(papers, iteration=0)
        citation = citation_result.to_dict()
        citation_ref = store.put("citation_explorer", citation, name="citation")
        relation_source = str(citation_result.calls[0].get("source") if citation_result.calls else "arxiv")
        messages.append(self._message(run_id, 2, "citation_explorer", "evidence_judge", "RELATION_BATCH", citation_ref, {"relation_batch_id": f"rel_{run_id}", "query_id": plan["query_id"], "source": relation_source, "edges": citation_result.edges}))

        all_candidates, more_errors = _deduplicate([*papers, *citation_result.papers])
        judgements: dict[str, dict[str, Any]] = {}
        if self.understanding_layer is not None and all_candidates:
            try:
                judgements: dict[str, dict[str, Any]] = {}
                for start in range(0, min(len(all_candidates), 12)):
                    deepseek_judge_batches += 1
                    judgements.update({str(item["paper_id"]): item for item in self.understanding_layer.judge(plan, all_candidates[start:start + 1])})
            except (DeepSeekCallError, DeepSeekSchemaError, ValueError) as exc:
                errors.append({"source": "deepseek", "code": getattr(exc, "code", "judge_fallback"), "message": str(exc)[:200], "stage": "judge"})
        evidence: list[dict[str, Any]] = []
        verdicts: list[dict[str, Any]] = []
        judged: list[dict[str, Any]] = []
        for paper in all_candidates:
            paper.setdefault("provenance", {})["query_id"] = plan["query_id"]
            constraint = self.gate.evaluate(plan, paper)
            judgement = judgements.get(str(paper.get("paper_id")))
            if judgement is not None:
                state = str(judgement.get("hard_constraint_state") or "unknown")
                if state == "fail":
                    constraint = ConstraintVerdict(
                        state="fail",
                        results=constraint.results + (ConstraintResult("deepseek", "explicit", "fail", "deepseek_explicit_fail"),),
                        reason_codes=constraint.reason_codes + ("deepseek_explicit_fail",),
                    )
                elif state == "unknown" and constraint.state == "pass":
                    constraint = ConstraintVerdict(state="unknown", results=constraint.results, reason_codes=constraint.reason_codes + ("deepseek_unknown",))
            items = EvidenceLoader().load(paper, required_status="abstract")
            overrides = {"relevance": judgement["relevance_score"]} if judgement is not None else None
            verdict = self.scorer.score(paper, plan, constraint, items, component_overrides=overrides)
            judged.append(_annotate(paper, verdict, constraint, items))
            evidence.extend(item.to_dict() for item in items)
            verdicts.append(verdict.to_dict())
        judged.sort(key=lambda item: (item.get("scores", {}).get("final") is not None, item.get("scores", {}).get("final") or -1, item.get("paper_id", "")), reverse=True)
        # Normalize evidence references to safe, local artifact paths before
        # putting them into strict protocol payloads.
        ref_map: dict[str, str] = {}
        for index, evidence_item in enumerate(evidence):
            old_ref = str(evidence_item.get("evidence_ref") or "")
            if not old_ref:
                continue
            safe_ref = f"evidence_judge/evidence_{index:04d}.txt"
            path = root / safe_ref
            path.parent.mkdir(parents=True, exist_ok=True)
            paper_id = str(evidence_item.get("paper_id") or "")
            abstract = next((str(item.get("bibliography", {}).get("abstract") or "") for item in judged if str(item.get("paper_id")) == paper_id), "")
            path.write_text(abstract, encoding="utf-8")
            ref_map[old_ref] = safe_ref
            evidence_item["evidence_ref"] = safe_ref
        for paper in judged:
            paper["evidence_refs"] = [ref_map.get(str(ref), str(ref)) for ref in paper.get("evidence_refs") or []]
        for verdict in verdicts:
            verdict["evidence_refs"] = [ref_map.get(str(ref), str(ref)) for ref in verdict.get("evidence_refs") or []]
        evidence_payload = {"papers": judged, "evidence": evidence, "verdicts": verdicts}
        evidence_ref = store.put("evidence_judge", evidence_payload, name="judgement")
        first_verdict = verdicts[0] if verdicts else {"paper_id": "empty", "component_scores": {name: 0.0 for name in ("relevance", "constraint", "evidence", "quality", "citation", "novelty")}, "constraint_state": "unknown", "confidence": 0.0, "evidence_refs": [], "excluded": True}
        messages.append(self._message(run_id, 3, "evidence_judge", "arbiter", "EVIDENCE_VERDICT", evidence_ref, {"query_id": plan["query_id"], "paper_id": str(first_verdict["paper_id"]), "verdict": "degraded" if first_verdict.get("excluded") else "relevant", "constraint_state": first_verdict.get("constraint_state", "unknown"), "confidence": float(first_verdict.get("confidence") or 0.0), "evidence_refs": list(first_verdict.get("evidence_refs") or []), "component_scores": first_verdict["component_scores"]}))

        selected = [item for item in judged if item.get("scores", {}).get("final") is not None][: self.max_selected]
        successful_calls = sum(1 for call in recall_result.calls + citation_result.calls if call.get("ok"))
        total_calls = len(recall_result.calls) + len(citation_result.calls)
        relevant_new = sum(1 for item in selected if float(item.get("scores", {}).get("relevance") or 0) >= 0.6)
        stop = StopController.from_query_plan(plan).decide(
            iteration=0,
            citation_depth=1 if citation_result.stats.get("enabled") and citation_result.papers else 0,
            provider_calls=total_calls,
            provider_successes=successful_calls,
            new_unique_papers=[len(all_candidates)],
            new_relevant_papers=relevant_new,
            subquery_coverage=(sum(1 for call in recall_result.calls if call.get("ok")) / max(1, len(recall_result.calls))),
            evidence_coverage=(len({str(item.get("paper_id")) for item in evidence if item.get("evidence_status") not in {None, "unavailable"}}) / max(1, len(all_candidates))),
            budget_exhausted=total_calls >= int(plan["budget"]["max_provider_calls"]),
        ).to_dict()
        selection_payload = {"selected": selected, "stop": stop, "candidate_count": len(judged)}
        selection_ref = store.put("arbiter", selection_payload, name="final_selection")
        messages.append(self._message(run_id, 4, "arbiter", "arbiter", "STOP_DECISION", selection_ref, {"query_id": plan["query_id"], "action": "STOP", "reason_code": stop["reason_code"]}))
        messages.append(self._message(run_id, 5, "arbiter", "arbiter", "FINAL_SELECTION", selection_ref, {"query_id": plan["query_id"], "selections": [{"paper_id": str(item["paper_id"]), "final_score": float(item["scores"]["final"]), "evidence_refs": list(item.get("evidence_refs") or [])} for item in selected]}))

        message_dicts = [validate_message(message) for message in messages]
        protocol_ref = store.write_jsonl("protocol.jsonl", message_dicts)
        errors.extend([*recall_result.source_errors, *citation_result.source_errors, *dedup_errors, *more_errors])
        manifest = {"schema_version": "p3_run.v1", "run_id": run_id, "query": query, "query_id": plan["query_id"], "citation_enabled": self.citation_enabled, "deepseek_judge_batches": deepseek_judge_batches, "generated_at": datetime.now(timezone.utc).isoformat(), "roles": ["planner", "retriever", "citation_explorer", "evidence_judge", "arbiter"], "artifacts": {"query_plan": plan_ref, "recall": recall_ref, "citation": citation_ref, "judgement": evidence_ref, "final_selection": selection_ref, "protocol": protocol_ref}}
        result = P3Run(query, run_id, plan.to_dict(), recall, citation, evidence, verdicts, judged, selected, stop, message_dicts, errors, {}, manifest)
        result.stats = self._stats(result, root)
        store.put("arbiter", result.stats, name="metrics")
        result.manifest["stats_ref"] = "arbiter/metrics.json"
        store.put("arbiter", result.manifest, name="run_manifest")
        return result

    @staticmethod
    def _stats(run: P3Run, root: Path) -> dict[str, Any]:
        message_bytes = sum(estimate_bytes(item) for item in run.messages)
        files = [path for path in root.rglob("*") if path.is_file()]
        return {"agent_count": 5, "message_count": len(run.messages), "protocol_message_bytes": message_bytes, "protocol_message_tokens_estimate": estimate_tokens(run.messages), "artifact_count": len(files), "artifact_bytes": sum(path.stat().st_size for path in files), "candidate_count": len(run.papers), "selected_count": len(run.selected), "source_errors": len(run.errors), "citation_enabled": bool(run.citation.get("stats", {}).get("enabled")), "citation_edges": len(run.citation.get("edges") or {})}


def run_p3_fixture(query: str = "WiFi heart rate monitoring", *, output_dir: str | Path | None = None, citation_enabled: bool = True) -> P3Run:
    from .mock_pipeline import _paper

    seed = _paper("arxiv", "WiFi CSI heart rate monitoring using contactless vital sign estimation.")
    seed["paper_id"] = "fixture:p3:seed"
    seed["identifiers"]["doi"] = "10.1234/fixture.p3.seed"
    seed["provenance"]["execution_status"] = "mock"
    seed["scores"]["relevance"] = 0.9
    child = _paper("arxiv", "Reference paper for WiFi CSI heart rate measurement.")
    child["paper_id"] = "fixture:p3:child"
    child["identifiers"]["doi"] = "10.1234/fixture.p3.child"
    child["provenance"]["execution_status"] = "mock"
    child["relation_type"] = "references"
    provider = FixtureProvider("arxiv", [seed], {seed["paper_id"]: [child]})
    return P3Pipeline({"arxiv": provider}, citation_enabled=citation_enabled).run(query, output_dir=output_dir)


__all__ = ["P3Pipeline", "P3Run", "run_p3_fixture"]
