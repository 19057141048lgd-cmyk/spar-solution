"""P1 四组论文检索对照实验与 artifact 编排。

本模块只负责实验控制面：Provider 返回 PaperDoc，``search`` 负责收敛和去重，
``metrics`` 负责身份匹配与指标。P1 的 D 模式是可复现的元数据排序，不把它
冒充语义质量模型，也不进入 P2/P3。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

from .gold import load_gold
from .metrics import evaluate_modes
from .mock_pipeline import _paper
from .paperdoc import validate_paper_doc
from .providers.base import ProviderError, ProviderResult
from .providers.local_library import FixtureLocalLibraryProvider
from .search_service import search


WIFI_QUERIES = (
    "WiFi heart rate monitoring",
    "WiFi-based heart rate measurement",
    "contactless heart rate monitoring using WiFi",
    "WiFi CSI vital signs heart rate",
)

MODE_SPECS: dict[str, dict[str, Any]] = {
    "A_arxiv": {"label": "A", "description": "仅使用 arXiv API", "providers": ("arxiv",)},
    "B_local": {"label": "B", "description": "仅使用自建论文库", "providers": ("local",)},
    "C_fusion": {"label": "C", "description": "arXiv + 自建论文库，合并后去重", "providers": ("arxiv", "local")},
    "D_reranked": {"label": "D", "description": "去重后统一排序", "providers": ("arxiv", "local")},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _query_id(query: str) -> str:
    return "q_" + hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:16]


def _provider_status(provider: Any) -> str:
    """读取 Provider 的可用性状态，禁止把 unavailable 静默变成空结果。"""

    for attr in ("availability_status", "config_status", "provider_status"):
        value = getattr(provider, attr, None)
        if isinstance(value, str) and value in {"mock", "configured", "unavailable"}:
            return value
    value = getattr(provider, "status", None)
    if isinstance(value, str) and value in {"mock", "configured", "unavailable"}:
        return value
    if bool(getattr(provider, "is_mock", False)):
        return "mock"
    return "configured"


def _provider_name(provider: Any) -> str:
    return str(getattr(provider, "name", None) or getattr(provider, "source", None) or provider.__class__.__name__).casefold()


def _record_count(artifact: Mapping[str, Any], source: str) -> int:
    for item in (artifact.get("stats") or {}).get("providers") or []:
        if isinstance(item, Mapping) and str(item.get("source", "")).casefold() == source.casefold():
            return int(item.get("records") or 0)
    return 0


def _sort_key(paper: Mapping[str, Any]) -> tuple[float, float, int, str]:
    scores = paper.get("scores") if isinstance(paper.get("scores"), Mapping) else {}
    bibliography = paper.get("bibliography") if isinstance(paper.get("bibliography"), Mapping) else {}
    retrieval = scores.get("retrieval")
    final = scores.get("final")
    year = bibliography.get("year")
    return (
        -(float(final) if isinstance(final, (int, float)) else -1.0),
        -(float(retrieval) if isinstance(retrieval, (int, float)) else -1.0),
        -(int(year) if isinstance(year, int) else 0),
        str(paper.get("paper_id") or ""),
    )


def _rank_papers(papers: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """P1 deterministic unified ranking; no LLM/质量判断 is implied."""

    ranked = [deepcopy(dict(item)) for item in papers]
    ranked.sort(key=_sort_key)
    for index, paper in enumerate(ranked, start=1):
        paper.setdefault("provenance", {}).setdefault("ranking", {})
        paper["provenance"]["ranking"].update({"mode": "p1_deterministic", "rank": index})
    return ranked


def run_mode(
    query: str,
    providers: Mapping[str, Any],
    mode: str,
    *,
    page_size: int = 20,
    run_id: str | None = None,
    base_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """运行一个模式并保存可追溯的 PaperDoc 结果。"""

    if mode not in MODE_SPECS:
        raise ValueError(f"unsupported experiment mode: {mode}")
    provider_names = MODE_SPECS[mode]["providers"]
    aliases = {"local": ("local", "local_library"), "arxiv": ("arxiv",)}
    selected: list[Any] = []
    missing: list[str] = []
    for name in provider_names:
        candidate = next((providers[key] for key in aliases.get(name, (name,)) if key in providers), None)
        if candidate is None:
            missing.append(name)
        else:
            selected.append(candidate)
    started = perf_counter()
    if mode == "D_reranked" and base_artifact is not None:
        artifact = deepcopy(dict(base_artifact))
        artifact["mode"] = mode
        artifact["run_id"] = run_id or artifact.get("run_id")
    else:
        artifact = search(query, selected, page_size=page_size, mode=mode, run_id=run_id)
    if missing:
        for name in missing:
            artifact.setdefault("source_errors", []).append({
                "source": name,
                "code": "config_missing",
                "message": "provider was not supplied to this experiment",
                "retryable": False,
                "status_code": None,
                "details": {},
            })
    if mode == "D_reranked":
        artifact["papers"] = _rank_papers(artifact.get("papers") or [])
        artifact["ranking"] = {"mode": "p1_deterministic", "input_mode": "C_fusion"}
    for paper in artifact.get("papers") or []:
        validate_paper_doc(paper)
    stats = artifact.setdefault("stats", {})
    stats["latency_ms"] = round((perf_counter() - started) * 1000, 3)
    stats["api_calls"] = 0 if mode == "D_reranked" and base_artifact is not None else len(selected)
    stats["provider_status"] = {
        _provider_name(provider): _provider_status(provider) for provider in selected
    }
    stats["missing_providers"] = missing
    stats["dedup_count"] = int(
        stats.get("dedup_count")
        if stats.get("dedup_count") is not None
        else max(0, int(stats.get("valid_records") or stats.get("input_records") or 0) - int(stats.get("merged_records") or 0))
    )
    stats["source_errors_count"] = len(artifact.get("source_errors") or [])
    return artifact


def compare_regressions(metrics: Mapping[str, Mapping[str, Any]], *, cutoff: int = 10) -> dict[str, Any]:
    """按固定规则比较 C 对 A/B、D 对 C；不把 provisional Gold 当成证明。"""

    missing = sorted(set(MODE_SPECS) - set(metrics))
    if missing:
        raise ValueError(f"comparison requires all modes: missing {missing}")
    key = str(cutoff)
    def value(mode: str, metric: str) -> float:
        return float(((metrics.get(mode) or {}).get("by_cutoff") or {}).get(key, {}).get(metric) or 0.0)

    a_recall, b_recall, c_recall = (value(mode, "recall") for mode in ("A_arxiv", "B_local", "C_fusion"))
    c_f1, d_f1 = value("C_fusion", "f1"), value("D_reranked", "f1")
    return {
        "cutoff": cutoff,
        "fusion_regression": c_recall < max(a_recall, b_recall),
        "rerank_regression": d_f1 < c_f1,
        "values": {"A_recall": a_recall, "B_recall": b_recall, "C_recall": c_recall, "C_f1": c_f1, "D_f1": d_f1},
    }


def evaluate_experiment(
    mode_results: Mapping[str, Mapping[str, Any]],
    gold_by_query: Mapping[str, list[Mapping[str, Any]]],
    *,
    k_values: tuple[int, ...] = (10, 20),
    gold_status: str = "provisional",
) -> dict[str, Any]:
    metrics = evaluate_modes(mode_results, gold_by_query, k_values=k_values)
    metrics["acceptance"] = {
        "at_10": compare_regressions(metrics, cutoff=10),
        "at_20": compare_regressions(metrics, cutoff=20),
        "gold_status": gold_status,
        "effect_claim": (
            "暂不能证明效果提升：当前 Gold 为 provisional 人工标注集"
            if gold_status == "provisional"
            else "仅输出对照指标，不自动宣称效果提升"
        ),
    }
    return metrics


def _gold_paper_doc(item: Mapping[str, Any], source: str) -> dict[str, Any]:
    """把 Gold 引用转换为 fixture PaperDoc；不调用网络，也不伪造本地库。"""

    abstract = str(item.get("judgment_basis") or "Provisional Gold fixture record.")
    doc = _paper(source, abstract)
    identifiers = doc["identifiers"]
    identifiers.update({key: value for key, value in (item.get("identifiers") or {}).items() if value})
    doc["paper_id"] = str(item.get("paper_id") or next(iter(identifiers.values()), item.get("title")))
    doc["bibliography"].update({
        "title": item.get("title") or "Untitled fixture paper",
        "year": item.get("year") or 0,
        "authors": [item.get("first_author") or "Unknown"],
        "abstract": abstract,
    })
    doc["scores"]["retrieval"] = 0.8
    return validate_paper_doc(doc)


class FixtureProvider:
    """无网络实验 Provider；状态显式为 mock。"""

    availability_status = "mock"

    def __init__(self, name: str, papers_by_query: Mapping[str, list[Mapping[str, Any]]]):
        self.name = name
        self.papers_by_query = {str(key): [deepcopy(dict(item)) for item in value] for key, value in papers_by_query.items()}

    def search(self, query: str, *, page_size: int = 10, **_: Any) -> ProviderResult:
        records = self.papers_by_query.get(query, [])[:page_size]
        return ProviderResult(self.name, "search", records=records, total=len(records), provenance={"execution": "mock"})


def build_fixture_providers(gold_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """从 provisional Gold 构建 A/B 对照 fixture，返回 Provider 与查询元数据。"""

    gold = load_gold(gold_path)
    arxiv: dict[str, list[dict[str, Any]]] = {}
    local: dict[str, list[dict[str, Any]]] = {}
    gold_by_query: dict[str, list[dict[str, Any]]] = {}
    for query_item in gold["queries"]:
        query = str(query_item["query"])
        docs = [_gold_paper_doc(item, "fixture_arxiv") for item in query_item["relevant_papers"]]
        # 两个来源各返回一部分，C 的 union 才能体现协议中的去重合并。
        arxiv[query] = docs[:2]
        local[query] = [deepcopy(doc) for doc in docs[1:]]
        for doc in local[query]:
            doc["provenance"]["sources"] = ["fixture_local"]
            doc["provenance"]["endpoints"] = ["mock://local-library/search"]
        gold_by_query[query_item["query_id"]] = [_gold_paper_doc(item, "gold") for item in query_item["relevant_papers"]]
    return {"arxiv": FixtureProvider("arxiv", arxiv), "local": FixtureLocalLibraryProvider(local)}, {
        "gold": gold,
        "gold_by_query": gold_by_query,
    }


def run_wifi_fixture(
    output_dir: str | Path,
    *,
    gold_path: str | Path | None = None,
    page_size: int = 20,
) -> dict[str, Any]:
    """运行四个 WiFi 查询的 A/B/C/D fixture 实验并写出完整目录。"""

    root = Path(output_dir)
    gold_file = Path(gold_path) if gold_path else Path(__file__).parents[2] / "gold" / "wifi_heart_rate.json"
    providers, fixture = build_fixture_providers(gold_file)
    gold = fixture["gold"]
    gold_by_query = fixture["gold_by_query"]
    mode_results: dict[str, dict[str, Any]] = {mode: {} for mode in MODE_SPECS}
    query_manifests: dict[str, Any] = {}
    for query_item in gold["queries"]:
        query_id = str(query_item["query_id"])
        query = str(query_item["query"])
        query_dir = root / query_id
        query_dir.mkdir(parents=True, exist_ok=True)
        for mode in MODE_SPECS:
            artifact = run_mode(
                query,
                providers,
                mode,
                page_size=page_size,
                run_id=f"fixture_{query_id}_{MODE_SPECS[mode]['label']}",
                base_artifact=mode_results["C_fusion"].get(query_id) if mode == "D_reranked" else None,
            )
            mode_results[mode][query_id] = artifact
            file_name = {"A_arxiv": "results_arxiv.json", "B_local": "results_local.json", "C_fusion": "results_fusion.json", "D_reranked": "results_reranked.json"}[mode]
            (query_dir / file_name).write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        query_metrics = evaluate_experiment(
            {mode: {query_id: mode_results[mode][query_id]} for mode in MODE_SPECS},
            {query_id: gold_by_query[query_id]},
            gold_status=query_item["annotation_status"],
        )
        (query_dir / "query.json").write_text(json.dumps({"query_id": query_id, "query": query}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (query_dir / "gold.json").write_text(json.dumps(query_item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (query_dir / "metrics.json").write_text(json.dumps(query_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        errors = {mode: mode_results[mode][query_id].get("source_errors", []) for mode in MODE_SPECS}
        (query_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {"query_id": query_id, "query": query, "execution": "fixture", "gold_status": query_item["annotation_status"], "provider_status": {name: _provider_status(provider) for name, provider in providers.items()}, "modes": list(MODE_SPECS), "created_at": _utc_now()}
        query_manifests[query_id] = manifest
        (query_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = evaluate_experiment(mode_results, gold_by_query, gold_status=gold["annotation_status"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "query.json").write_text(
        json.dumps(
            [{"query_id": item["query_id"], "query": item["query"]} for item in gold["queries"]],
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (root / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for mode, filename in {
        "A_arxiv": "results_arxiv.json",
        "B_local": "results_local.json",
        "C_fusion": "results_fusion.json",
        "D_reranked": "results_reranked.json",
    }.items():
        (root / filename).write_text(
            json.dumps(mode_results[mode], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (root / "errors.json").write_text(
        json.dumps(
            {
                mode: {
                    query_id: mode_results[mode][query_id].get("source_errors", [])
                    for query_id in mode_results[mode]
                }
                for mode in MODE_SPECS
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    manifest = {"schema_version": "spar.p1.experiment.v1", "execution": "fixture", "gold_status": gold["annotation_status"], "queries": list(query_manifests.values()), "modes": MODE_SPECS, "created_at": _utc_now()}
    (root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"metrics": metrics, "manifest": manifest, "mode_results": mode_results}


__all__ = ["FixtureProvider", "MODE_SPECS", "WIFI_QUERIES", "build_fixture_providers", "compare_regressions", "evaluate_experiment", "run_mode", "run_wifi_fixture"]
