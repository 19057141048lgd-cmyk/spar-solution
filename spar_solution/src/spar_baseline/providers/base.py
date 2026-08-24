"""P1 Provider 最小结构化契约。

适配器只负责访问一个来源并返回结构化结果；查询分解、去重、排序和停止
由上层 SPAR 流程负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol


ErrorCode = Literal[
    "config", "auth", "rate", "timeout", "parse", "network", "empty", "unknown"
]
Operation = Literal["search", "read", "relations"]


class ProviderError(RuntimeError):
    """可序列化的 Provider 错误，不与论文相关性分数混淆。"""

    def __init__(
        self,
        source: str,
        code: ErrorCode | str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.details = dict(details or {})
        super().__init__(f"{source}:{code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "details": dict(self.details),
        }


@dataclass
class ProviderResult:
    """search/read/relations 共用的结构化返回对象。"""

    source: str
    operation: Operation
    records: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: str | None = None
    total: int | None = None
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    ok: bool = True

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source is required")
        if self.operation not in {"search", "read", "relations"}:
            raise ValueError("operation must be search, read or relations")
        if not isinstance(self.records, list) or any(not isinstance(item, dict) for item in self.records):
            raise TypeError("records must be a list of objects")
        if self.total is not None and (not isinstance(self.total, int) or self.total < 0):
            raise ValueError("total must be a non-negative integer or null")

    @property
    def data(self) -> list[dict[str, Any]]:
        """兼容上层协议的统一数据访问别名。"""

        return self.records

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "operation": self.operation,
            "ok": self.ok,
            "records": [dict(item) for item in self.records],
            "next_cursor": self.next_cursor,
            "total": self.total,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


class BaseProvider(Protocol):
    """所有学术 Provider 必须实现的最小接口。"""

    name: str

    def search(self, query: str, *, page_size: int = 10, cursor: str | None = None, **kwargs: Any) -> ProviderResult:
        ...

    def read(self, paper_id: str, *, cursor: str | None = None, **kwargs: Any) -> ProviderResult:
        ...

    def relations(self, paper_id: str, *, relation: str = "all", cursor: str | None = None, **kwargs: Any) -> ProviderResult:
        ...


def ensure_result(result: ProviderResult, *, source: str, operation: Operation) -> ProviderResult:
    """适配器边界校验，防止返回任意自然语言或空错误。"""

    if not isinstance(result, ProviderResult):
        raise ProviderError(source, "parse", "provider returned a non-structured result")
    if result.source != source or result.operation != operation:
        raise ProviderError(source, "parse", "provider result source/operation mismatch")
    return result


__all__ = ["BaseProvider", "ErrorCode", "Operation", "ProviderError", "ProviderResult", "ensure_result"]
