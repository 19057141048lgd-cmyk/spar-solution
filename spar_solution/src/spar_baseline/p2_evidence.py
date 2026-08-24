"""P2 约束门控和证据加载。

本模块只处理可审计的结构化状态，不把 Provider 错误转换成论文不相关。
正文通过 ``evidence_ref``/``content_ref`` 标识；EvidenceItem 可包含短摘要，
但不会把长正文复制到 Agent 消息中。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any, Mapping, Sequence

from .paperdoc import EVIDENCE_STATUSES, validate_paper_doc
from .providers.base import ProviderError, ProviderResult


ConstraintState = str
_CONSTRAINT_STATES = {"pass", "fail", "unknown"}
_EVIDENCE_ORDER = {"unavailable": 0, "metadata": 1, "abstract": 2, "partial_text": 3, "fulltext": 4}


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    expected: str
    state: ConstraintState
    reason_code: str
    observed: Any = None

    def __post_init__(self) -> None:
        if self.state not in _CONSTRAINT_STATES:
            raise ValueError("constraint state must be pass, fail or unknown")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstraintVerdict:
    state: ConstraintState
    results: tuple[ConstraintResult, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in _CONSTRAINT_STATES:
            raise ValueError("constraint state must be pass, fail or unknown")

    @property
    def passed(self) -> bool | None:
        return {"pass": True, "fail": False, "unknown": None}[self.state]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "passed": self.passed,
            "results": [item.to_dict() for item in self.results],
            "reason_codes": list(self.reason_codes),
        }


def _text(value: Any) -> str:
    return str(value or "").casefold().strip()


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[\w]+", _text(value), flags=re.UNICODE))


def _paper_text(paper_doc: Mapping[str, Any]) -> str:
    bib = paper_doc.get("bibliography") or {}
    fields = " ".join(str(item) for item in bib.get("fields") or [])
    authors = " ".join(str(item) for item in bib.get("authors") or [])
    return " ".join(str(bib.get(key) or "") for key in ("title", "abstract", "venue")) + " " + fields + " " + authors


def _constraint_result(name: str, expected: str, state: str, reason: str, observed: Any = None) -> ConstraintResult:
    return ConstraintResult(name=name, expected=expected, state=state, reason_code=reason, observed=observed)


class ConstraintGate:
    """规则优先的三态硬约束判断器。

    识别常见的 year/time_range、full_text/evidence、open_access 和文本约束。
    无法从 PaperDoc 的元数据或摘要确认时返回 ``unknown``，绝不推断为 pass。
    """

    def evaluate(self, query_plan: Mapping[str, Any] | None, paper_doc: Mapping[str, Any]) -> ConstraintVerdict:
        validate_paper_doc(dict(paper_doc))
        constraints = list((query_plan or {}).get("hard_constraints") or [])
        results: list[ConstraintResult] = []
        for item in constraints:
            if not isinstance(item, Mapping):
                results.append(_constraint_result("invalid", "", "unknown", "invalid_constraint"))
                continue
            name = str(item.get("name") or "").strip()
            expected = str(item.get("value") or "").strip()
            results.append(self._evaluate_one(name, expected, paper_doc))
        states = [item.state for item in results]
        if "fail" in states:
            state = "fail"
        elif "unknown" in states:
            state = "unknown"
        else:
            state = "pass"
        return ConstraintVerdict(
            state=state,
            results=tuple(results),
            reason_codes=tuple(item.reason_code for item in results),
        )

    def _evaluate_one(self, name: str, expected: str, paper_doc: Mapping[str, Any]) -> ConstraintResult:
        lname = _text(name)
        bib = paper_doc.get("bibliography") or {}
        access = paper_doc.get("access") or {}
        status = paper_doc.get("status") or {}
        if lname in {"year", "publication_year", "time_range", "date", "publication_date"} or "year" in lname:
            year = bib.get("year")
            if year is None:
                return _constraint_result(name, expected, "unknown", "year_unavailable")
            observed_year = int(year)
            # Canonical planner format is YYYY-YYYY, >=YYYY, or <=YYYY.
            # The colon forms remain accepted for replaying older artifacts
            # (YYYY:YYYY, YYYY:, and :YYYY).
            range_match = re.search(r"(\d{4})\s*(?:-|:|to|~|至)\s*(\d{4})", expected, flags=re.I)
            lower_match = re.search(r">=\s*(\d{4})", expected)
            upper_match = re.search(r"<=\s*(\d{4})", expected)
            legacy_lower = re.fullmatch(r"\s*(\d{4})\s*:\s*", expected)
            legacy_upper = re.fullmatch(r"\s*:\s*(\d{4})\s*", expected)
            if range_match:
                ok = int(range_match.group(1)) <= observed_year <= int(range_match.group(2))
            elif lower_match or legacy_lower:
                lower = int((lower_match or legacy_lower).group(1))
                ok = observed_year >= lower
            elif upper_match or legacy_upper:
                upper = int((upper_match or legacy_upper).group(1))
                ok = observed_year <= upper
            else:
                years = {int(value) for value in re.findall(r"\d{4}", expected)}
                ok = observed_year in years if years else None
            return _constraint_result(name, expected, "unknown" if ok is None else ("pass" if ok else "fail"), "year_match" if ok else ("year_mismatch" if ok is False else "year_unparseable"), year)
        if "full" in lname or "全文" in lname or "evidence" in lname:
            required = expected.casefold()
            wanted = "fulltext" if any(word in required for word in ("full", "全文", "fulltext")) else "abstract"
            actual = status.get("evidence_status") or access.get("full_text_status")
            if actual not in EVIDENCE_STATUSES:
                return _constraint_result(name, expected, "unknown", "evidence_status_missing", actual)
            if actual == "unavailable":
                return _constraint_result(name, expected, "fail", "evidence_unavailable", actual)
            ok = _EVIDENCE_ORDER[actual] >= _EVIDENCE_ORDER[wanted]
            return _constraint_result(name, expected, "pass" if ok else "fail", "evidence_sufficient" if ok else "evidence_insufficient", actual)
        if "open" in lname or "oa" == lname or "open_access" in lname:
            available = bool(access.get("oa_url") or access.get("pdf_url"))
            if not available and access.get("landing_url") is None:
                return _constraint_result(name, expected, "unknown", "oa_metadata_missing")
            wanted = expected.casefold() not in {"false", "0", "no", "否"}
            return _constraint_result(name, expected, "pass" if available == wanted else "fail", "oa_match" if available == wanted else "oa_mismatch", available)
        text = _paper_text(paper_doc)
        expected_tokens = _tokens(expected)
        if not expected_tokens:
            return _constraint_result(name, expected, "unknown", "constraint_value_empty")
        if not text.strip():
            return _constraint_result(name, expected, "unknown", "paper_text_unavailable")
        text_tokens = _tokens(text)
        if expected_tokens <= text_tokens:
            return _constraint_result(name, expected, "pass", "text_constraint_match", expected)
        # Metadata/abstract only proves absence when the field is present and non-empty.
        has_evidence = bool(bib.get("title") or bib.get("abstract") or bib.get("fields"))
        return _constraint_result(name, expected, "fail" if has_evidence else "unknown", "text_constraint_mismatch" if has_evidence else "paper_text_unavailable", expected)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    paper_id: str
    evidence_status: str
    evidence_ref: str | None
    content_ref: str | None = None
    source: str | None = None
    section: str | None = None
    offset: int | None = None
    page: int | None = None
    char_count: int = 0
    confidence: float = 0.0
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_status not in EVIDENCE_STATUSES:
            raise ValueError("unknown evidence status")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.evidence_status == "unavailable" and not self.unavailable_reason:
            raise ValueError("unavailable evidence requires unavailable_reason")
        if self.evidence_status != "unavailable" and not (self.evidence_ref or self.content_ref):
            raise ValueError("available evidence requires evidence_ref or content_ref")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceLoader:
    """从 PaperDoc 或可选 Provider ``read`` 结果构造 EvidenceItem。"""

    def __init__(self, provider: Any = None, *, artifact_prefix: str = "artifacts/evidence") -> None:
        self.provider = provider
        self.artifact_prefix = artifact_prefix.rstrip("/")

    def load(
        self,
        paper_doc: Mapping[str, Any],
        required_status: str = "abstract",
        *,
        provider: Any = None,
    ) -> list[EvidenceItem]:
        doc = dict(paper_doc)
        validate_paper_doc(doc)
        if required_status not in EVIDENCE_STATUSES:
            raise ValueError(f"unsupported evidence status: {required_status}")
        current = doc.get("status", {}).get("evidence_status") or doc.get("access", {}).get("full_text_status")
        if provider is not None or self.provider is not None:
            current_doc = self._read_with_provider(doc, provider or self.provider)
            if current_doc is not None:
                doc = current_doc
                current = doc.get("status", {}).get("evidence_status") or doc.get("access", {}).get("full_text_status")
        if current not in EVIDENCE_STATUSES:
            current = "unavailable"
        if current == "unavailable":
            return [self._unavailable(doc, "evidence_unavailable")]
        # Metadata is deliberately not upgraded to abstract/fulltext merely because a URL exists.
        if _EVIDENCE_ORDER[current] < _EVIDENCE_ORDER[required_status]:
            return [self._unavailable(doc, f"insufficient_evidence:{current}<{required_status}")]
        chunks = doc.get("content", {}).get("chunks") or []
        refs = doc.get("evidence_refs") or []
        source = (doc.get("provenance", {}).get("sources") or [None])[0]
        output: list[EvidenceItem] = []
        if chunks:
            for index, chunk in enumerate(chunks):
                if not isinstance(chunk, Mapping):
                    continue
                ref = chunk.get("evidence_ref") or chunk.get("content_ref") or doc.get("content", {}).get("content_ref")
                if not ref:
                    continue
                output.append(EvidenceItem(
                    evidence_id=str(chunk.get("chunk_id") or f"{doc['paper_id']}:chunk:{index}"),
                    paper_id=str(doc["paper_id"]), evidence_status=current, evidence_ref=str(ref),
                    content_ref=chunk.get("content_ref"), source=source, section=chunk.get("section"),
                    offset=chunk.get("offset"), page=chunk.get("page"),
                    char_count=int(chunk.get("char_count") or 0), confidence=self._confidence(current),
                ))
        if not output and refs:
            for index, ref in enumerate(refs):
                if isinstance(ref, Mapping):
                    value = ref.get("evidence_ref") or ref.get("content_ref") or ref.get("id")
                else:
                    value = ref
                if value:
                    output.append(EvidenceItem(f"{doc['paper_id']}:evidence:{index}", str(doc["paper_id"]), current, str(value), source=source, confidence=self._confidence(current)))
        if not output and current in {"abstract", "partial_text", "fulltext"}:
            abstract = str(doc.get("bibliography", {}).get("abstract") or "")
            if abstract:
                digest = hashlib.sha256(abstract.encode("utf-8")).hexdigest()[:16]
                output.append(EvidenceItem(f"{doc['paper_id']}:abstract", str(doc["paper_id"]), "abstract" if current == "metadata" else current, f"{self.artifact_prefix}/{doc['paper_id']}/abstract-{digest}.txt", source=source, char_count=len(abstract), confidence=self._confidence("abstract")))
        if not output:
            output.append(self._unavailable(doc, "evidence_ref_missing"))
        return output

    def _read_with_provider(self, doc: dict[str, Any], provider: Any) -> dict[str, Any] | None:
        try:
            result = provider.read(str(doc["paper_id"]))
        except (ProviderError, OSError, TimeoutError):
            return None
        if not isinstance(result, ProviderResult) or not result.records:
            return None
        candidate = result.records[0]
        return candidate if isinstance(candidate, dict) and candidate.get("schema_version") else None

    @staticmethod
    def _confidence(status: str) -> float:
        return {"metadata": 0.2, "abstract": 0.55, "partial_text": 0.8, "fulltext": 1.0}.get(status, 0.0)

    @staticmethod
    def _unavailable(doc: Mapping[str, Any], reason: str) -> EvidenceItem:
        return EvidenceItem(f"{doc['paper_id']}:unavailable", str(doc["paper_id"]), "unavailable", None, confidence=0.0, unavailable_reason=reason)


__all__ = ["ConstraintGate", "ConstraintResult", "ConstraintVerdict", "EvidenceItem", "EvidenceLoader"]
