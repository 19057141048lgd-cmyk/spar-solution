"""P2 的可回放停止决策。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


STRONG_REASON_CODES = {
    "BUDGET_EXHAUSTED",
    "MAX_ITERATION",
    "MAX_CITATION_DEPTH",
    "ALL_PROVIDER_FAILED",
    "NO_NEW_PAPER_2_ROUNDS",
}
SOFT_REASON_CODE = "LOW_GAIN_SUFFICIENT_COVERAGE"


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    decision_type: str
    reason_code: str
    triggered_conditions: tuple[str, ...]
    measurements: dict[str, Any]
    thresholds: dict[str, Any]

    def __post_init__(self) -> None:
        if self.decision_type not in {"strong", "soft", "continue"}:
            raise ValueError("decision_type must be strong, soft or continue")
        if self.decision_type == "continue" and (self.should_stop or self.reason_code != "CONTINUE"):
            raise ValueError("continue decision is inconsistent")
        if self.decision_type != "continue" and not self.should_stop:
            raise ValueError("stop decision is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["triggered_conditions"] = list(self.triggered_conditions)
        return data


class StopController:
    """按固定优先级计算强停止和 2/3 软停止。"""

    def __init__(
        self,
        *,
        max_iterations: int = 2,
        max_citation_depth: int = 1,
        max_provider_calls: int = 100,
        min_new_relevant: int = 2,
        min_subquery_coverage: float = 0.8,
        min_evidence_coverage: float = 0.7,
    ) -> None:
        if min(max_iterations, max_citation_depth, max_provider_calls, min_new_relevant) < 0:
            raise ValueError("stop thresholds must be non-negative")
        for name, value in (("min_subquery_coverage", min_subquery_coverage), ("min_evidence_coverage", min_evidence_coverage)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        self.thresholds = {
            "max_iterations": max_iterations,
            "max_citation_depth": max_citation_depth,
            "max_provider_calls": max_provider_calls,
            "min_new_relevant": min_new_relevant,
            "min_subquery_coverage": min_subquery_coverage,
            "min_evidence_coverage": min_evidence_coverage,
            "soft_conditions_required": 2,
        }

    @classmethod
    def from_query_plan(cls, plan: Mapping[str, Any]) -> "StopController":
        budget = plan.get("budget") or {}
        strategy = plan.get("stop_strategy") or {}
        return cls(
            max_iterations=int(budget.get("max_iterations", 2)),
            max_citation_depth=int(budget.get("max_citation_depth", 1)),
            max_provider_calls=int(budget.get("max_provider_calls", 100)),
            min_new_relevant=int(strategy.get("min_new_relevant", 2)),
            min_subquery_coverage=float(strategy.get("min_subquery_coverage", 0.8)),
            min_evidence_coverage=float(strategy.get("min_evidence_coverage", 0.7)),
        )

    def decide(
        self,
        *,
        iteration: int,
        citation_depth: int,
        provider_calls: int,
        provider_successes: int,
        new_unique_papers: Sequence[int],
        new_relevant_papers: int,
        subquery_coverage: float,
        evidence_coverage: float,
        budget_exhausted: bool = False,
    ) -> StopDecision:
        if min(iteration, citation_depth, provider_calls, provider_successes, new_relevant_papers) < 0:
            raise ValueError("stop measurements must be non-negative")
        if any(value < 0 for value in new_unique_papers):
            raise ValueError("new_unique_papers must be non-negative")
        if not 0 <= subquery_coverage <= 1 or not 0 <= evidence_coverage <= 1:
            raise ValueError("coverage must be between 0 and 1")
        measured = {
            "iteration": iteration,
            "citation_depth": citation_depth,
            "provider_calls": provider_calls,
            "provider_successes": provider_successes,
            "new_unique_papers": list(new_unique_papers),
            "new_relevant_papers": new_relevant_papers,
            "subquery_coverage": subquery_coverage,
            "evidence_coverage": evidence_coverage,
            "budget_exhausted": bool(budget_exhausted),
        }
        strong = []
        if budget_exhausted or provider_calls >= self.thresholds["max_provider_calls"]:
            strong.append("BUDGET_EXHAUSTED")
        if provider_calls > 0 and provider_successes == 0:
            strong.append("ALL_PROVIDER_FAILED")
        if len(new_unique_papers) >= 2 and list(new_unique_papers)[-2:] == [0, 0]:
            strong.append("NO_NEW_PAPER_2_ROUNDS")
        if iteration >= self.thresholds["max_iterations"]:
            strong.append("MAX_ITERATION")
        if citation_depth >= self.thresholds["max_citation_depth"]:
            strong.append("MAX_CITATION_DEPTH")
        if strong:
            # 预算/Provider 可用性优先于循环上限，便于定位真正阻塞原因。
            priority = ["BUDGET_EXHAUSTED", "ALL_PROVIDER_FAILED", "NO_NEW_PAPER_2_ROUNDS", "MAX_ITERATION", "MAX_CITATION_DEPTH"]
            reason = next(code for code in priority if code in strong)
            return StopDecision(True, "strong", reason, tuple(strong), measured, dict(self.thresholds))

        soft = []
        if new_relevant_papers < self.thresholds["min_new_relevant"]:
            soft.append("LOW_RELEVANT_GAIN")
        if subquery_coverage >= self.thresholds["min_subquery_coverage"]:
            soft.append("SUBQUERY_COVERAGE_MET")
        if evidence_coverage >= self.thresholds["min_evidence_coverage"]:
            soft.append("EVIDENCE_COVERAGE_MET")
        if len(soft) >= self.thresholds["soft_conditions_required"]:
            return StopDecision(True, "soft", SOFT_REASON_CODE, tuple(soft), measured, dict(self.thresholds))
        return StopDecision(False, "continue", "CONTINUE", tuple(soft), measured, dict(self.thresholds))


__all__ = ["SOFT_REASON_CODE", "STRONG_REASON_CODES", "StopController", "StopDecision"]
