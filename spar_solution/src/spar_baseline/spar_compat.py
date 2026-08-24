"""SPAR 原始仓库的 P1 兼容性隔离检查。

P1 不直接改动上游仓库；先把已知的确定性风险转成可重复的静态检查结果，
避免上游导入失败或配置错误被误判为检索结果为空。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


KNOWN_CHECKS = (
    ("semantic_undefined_query", "search_engine.py", "Searching Semantic Scholar for '{query}'"),
    ("merge_str_json", "search_engine.py", '"|".json('),
    ("reference_wrong_doc_info", "search_engine.py", 'doc["info"]'),
)


def inspect_spar_checkout(root: str | Path) -> dict[str, Any]:
    """静态检查 SPAR 已知风险，输出显式 isolation report。"""

    checkout = Path(root)
    files = {name: checkout / name for _, name, _ in KNOWN_CHECKS}
    findings: list[dict[str, Any]] = []
    for code, relative, needle in KNOWN_CHECKS:
        path = files[relative]
        if not path.is_file():
            findings.append({"code": code, "status": "not_checked", "reason": "file_missing", "file": relative})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        findings.append({
            "code": code,
            "status": "isolated" if needle in text else "not_found",
            "file": relative,
            "action": "do_not_call_upstream_path_in_p1" if needle in text else "no_action",
        })
    return {
        "schema_version": "spar.compat.v1",
        "root": str(checkout),
        "upstream_modified": False,
        "findings": findings,
        "policy": "P1 uses spar_solution baseline adapters; upstream SPAR is isolated until P2 repair pass.",
    }


__all__ = ["inspect_spar_checkout"]
