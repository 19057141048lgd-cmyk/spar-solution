"""P2 可解释分量评分和 EvidenceVerdict。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

from .p2_evidence import ConstraintVerdict, EvidenceItem
from .paperdoc import validate_paper_doc


COMPONENT_NAMES = ("relevance", "constraint", "evidence", "quality", "citation", "novelty")
DEFAULT_WEIGHTS: dict[str, float] = {
    "relevance": 0.30,
    "constraint": 0.25,
    "evidence": 0.20,
    "quality": 0.10,
    "citation": 0.10,
    "novelty": 0.05,
}
_EVIDENCE_SCORES = {"unavailable": 0.0, "metadata": 0.2, "abstract": 0.55, "partial_text": 0.8, "fulltext": 1.0}


def _unit(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def validate_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    if set(weights) != set(COMPONENT_NAMES):
        raise ValueError(f"weights must contain exactly {list(COMPONENT_NAMES)}")
    output = {name: _unit(weights[name], f"weights.{name}") for name in COMPONENT_NAMES}
    if not math.isclose(sum(output.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("weights must sum to 1")
    return output


@dataclass(frozen=True)
class EvidenceVerdict:
    paper_id: str
    component_scores: dict[str, float]
    weights: dict[str, float]
    final_score: float | None
    constraint_state: str
    evidence_status: str
    evidence_refs: tuple[str, ...]
    confidence: float
    excluded: bool
    reason_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if set(self.component_scores) != set(COMPONENT_NAMES):
            raise ValueError("EvidenceVerdict requires all six component scores")
        for name, value in self.component_scores.items():
            _unit(value, f"component_scores.{name}")
        validate_weights(self.weights)
        if self.final_score is not None:
            _unit(self.final_score, "final_score")
        _unit(self.confidence, "confidence")
        if self.constraint_state not in {"pass", "fail", "unknown"}:
            raise ValueError("invalid constraint_state")
        if self.constraint_state == "fail" and not self.excluded:
            raise ValueError("failed hard constraints must be excluded")
        if self.excluded and self.final_score is not None:
            raise ValueError("excluded verdict cannot have final_score")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        data["reason_codes"] = list(self.reason_codes)
        data["warnings"] = list(self.warnings)
        return data


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE))


def _precomputed(doc: Mapping[str, Any], name: str) -> float | None:
    value = (doc.get("scores") or {}).get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return None


class Scorer:
    """确定性 P2 基线评分器；默认权重仅用于消融，不宣称最优。"""

    def __init__(self, weights: Mapping[str, Any] | None = None) -> None:
        self.weights = validate_weights(weights or DEFAULT_WEIGHTS)

    def score(
        self,
        paper_doc: Mapping[str, Any],
        query_plan: Mapping[str, Any],
        constraint: ConstraintVerdict,
        evidence_items: Sequence[EvidenceItem | Mapping[str, Any]],
        *,
        component_overrides: Mapping[str, Any] | None = None,
    ) -> EvidenceVerdict:
        doc = dict(paper_doc)
        validate_paper_doc(doc)
        overrides = dict(component_overrides or {})
        unknown = set(overrides) - set(COMPONENT_NAMES)
        if unknown:
            raise ValueError(f"unknown component overrides: {sorted(unknown)}")
        items = [item.to_dict() if isinstance(item, EvidenceItem) else dict(item) for item in evidence_items]
        status = self._evidence_status(doc, items)
        scores = {
            "relevance": self._relevance(doc, query_plan),
            "constraint": {"pass": 1.0, "unknown": 0.5, "fail": 0.0}[constraint.state],
            "evidence": _EVIDENCE_SCORES[status],
            "quality": self._quality(doc),
            "citation": self._citation(doc),
            "novelty": self._novelty(doc),
        }
        for name, value in overrides.items():
            scores[name] = _unit(value, f"component_overrides.{name}")
        refs = tuple(dict.fromkeys(
            str(item.get("evidence_ref") or item.get("content_ref"))
            for item in items if item.get("evidence_ref") or item.get("content_ref")
        ))
        provider_errors = list((doc.get("status") or {}).get("provider_errors") or [])
        warnings = tuple("provider_error_preserved" for _ in provider_errors)
        excluded = constraint.state == "fail"
        final = None if excluded else round(sum(scores[name] * self.weights[name] for name in COMPONENT_NAMES), 6)
        evidence_confidence = max((float(item.get("confidence") or 0.0) for item in items), default=0.0)
        confidence = round((scores["relevance"] + scores["constraint"] + evidence_confidence) / 3, 6)
        reasons = list(constraint.reason_codes)
        if not refs:
            reasons.append("evidence_ref_missing")
        if status in {"metadata", "unavailable"}:
            reasons.append("evidence_pending_verification")
        return EvidenceVerdict(
            paper_id=str(doc["paper_id"]), component_scores=scores, weights=dict(self.weights),
            final_score=final, constraint_state=constraint.state, evidence_status=status,
            evidence_refs=refs, confidence=confidence, excluded=excluded,
            reason_codes=tuple(dict.fromkeys(reasons)), warnings=warnings,
        )

    @staticmethod
    def _evidence_status(doc: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> str:
        order = {name: index for index, name in enumerate(("unavailable", "metadata", "abstract", "partial_text", "fulltext"))}
        statuses = [str(item.get("evidence_status") or "unavailable") for item in items]
        if not statuses:
            statuses = [str((doc.get("status") or {}).get("evidence_status") or "unavailable")]
        valid = [status for status in statuses if status in order]
        return max(valid, key=order.get) if valid else "unavailable"

    @staticmethod
    def _relevance(doc: Mapping[str, Any], plan: Mapping[str, Any]) -> float:
        existing = _precomputed(doc, "relevance")
        if existing is not None:
            return existing
        bib = doc.get("bibliography") or {}
        query_tokens = _tokens(" ".join(str(plan.get(key) or "") for key in ("raw_query", "topic")))
        for field in ("methods", "datasets", "tasks"):
            query_tokens.update(_tokens(" ".join(str(item) for item in plan.get(field) or [])))
        paper_tokens = _tokens(" ".join(str(bib.get(key) or "") for key in ("title", "abstract", "venue")))
        paper_tokens.update(_tokens(" ".join(str(item) for item in bib.get("fields") or [])))
        if not query_tokens:
            return 0.0
        return round(len(query_tokens & paper_tokens) / len(query_tokens), 6)

    @staticmethod
    def _quality(doc: Mapping[str, Any]) -> float:
        existing = _precomputed(doc, "quality")
        if existing is not None:
            return existing
        bib = doc.get("bibliography") or {}
        present = [bib.get("title"), bib.get("authors"), bib.get("year"), bib.get("venue"), bib.get("abstract")]
        return sum(bool(value) for value in present) / len(present)

    @staticmethod
    def _citation(doc: Mapping[str, Any]) -> float:
        existing = _precomputed(doc, "citation")
        if existing is not None:
            return existing
        relations = doc.get("relations") or {}
        count = len(relations.get("citations") or []) + len(relations.get("references") or [])
        return round(1.0 - math.exp(-count / 10.0), 6) if count else 0.0

    @staticmethod
    def _novelty(doc: Mapping[str, Any]) -> float:
        existing = _precomputed(doc, "novelty")
        return existing if existing is not None else 0.5


__all__ = ["COMPONENT_NAMES", "DEFAULT_WEIGHTS", "EvidenceVerdict", "Scorer", "validate_weights"]
