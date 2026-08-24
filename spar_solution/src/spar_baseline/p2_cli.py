"""P2 fixture/live/replay 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .config import load_config
from .deepseek_layer import DeepSeekClient, DeepSeekUnderstandingLayer
from .final_output import build_final_selection
from .openalex_provider import OpenAlexProvider
from .p2_metrics import evaluate_p2_run
from .p2_pipeline import P2Pipeline, replay_p2, run_p2_fixture
from .providers.arxiv import ArxivProvider
from .providers.bohrium import BohriumProvider
from .providers.local_library import LocalLibraryProvider
from .search_tree import SearchTreeRunner


def build_live_pipeline(config: Mapping[str, Any] | None = None, *, citation_enabled: bool = True, page_size: int = 10, max_workers: int = 4) -> P2Pipeline:
    values = dict(load_config() if config is None else config)
    providers: dict[str, Any] = {
        "arxiv": ArxivProvider.from_config(values),
        "openalex": OpenAlexProvider.from_config(values),
    }
    if str(values.get("BOHR_ACCESS_KEY") or "").strip():
        providers["bohrium"] = BohriumProvider.from_config(values)
    local = LocalLibraryProvider.from_config(values)
    if local.library_status == "configured":
        providers["local_library"] = local
    understanding = None
    if str(values.get("DEEPSEEK_API_KEY") or "").strip():
        client = DeepSeekClient(
            api_key=str(values["DEEPSEEK_API_KEY"]),
            base_url=str(values.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"),
            model=str(values.get("DEEPSEEK_MODEL") or "deepseek-chat"),
        )
        understanding = DeepSeekUnderstandingLayer(client)
    return P2Pipeline(providers, citation_enabled=citation_enabled, page_size=page_size, max_workers=max_workers, understanding_layer=understanding)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or replay the P2 structured retrieval pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="run the no-key fixture")
    fixture.add_argument("--query", default="WiFi heart rate monitoring")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--no-citation", action="store_true")
    live = sub.add_parser("live", help="run configured arXiv/OpenAlex/Bohrium retrieval")
    live.add_argument("--query", required=True)
    live.add_argument("--output", required=True)
    live.add_argument("--page-size", type=int, default=10)
    live.add_argument("--max-workers", type=int, default=4)
    live.add_argument("--no-citation", action="store_true")
    replay = sub.add_parser("replay", help="read a P2 artifact directory")
    replay.add_argument("--input", required=True)
    replay.add_argument("--gold-id", action="append", default=[])
    finalize = sub.add_parser("finalize", help="rebuild final_selection.json without rerunning providers")
    finalize.add_argument("--input", required=True)
    finalize.add_argument("--top-k", type=int, default=20)
    tree = sub.add_parser("tree", help="run the SPAR-style search tree on configured providers")
    tree.add_argument("--query", required=True)
    tree.add_argument("--output", required=True)
    tree.add_argument("--page-size", type=int, default=10)
    tree.add_argument("--max-depth", type=int, default=2)
    tree.add_argument("--docs-to-expand", type=int, default=8)
    tree.add_argument("--max-provider-calls", type=int, default=30)
    tree.add_argument("--fulltext-topk", type=int, default=0, help="对前 K 篇做本地全文抽取增强（0=关闭）")
    return parser


def build_tree_runner(config: Mapping[str, Any] | None = None, *, page_size: int = 10, max_depth: int = 2, docs_to_expand: int = 8, max_provider_calls: int = 30) -> "SearchTreeRunner":
    """用与 P2 live 相同的 Provider/LLM 装配构造搜索树检索器。"""

    pipeline = build_live_pipeline(config, citation_enabled=True, page_size=page_size)
    return SearchTreeRunner(
        pipeline.providers,
        pipeline.understanding_layer,
        page_size=page_size,
        max_depth=max_depth,
        docs_to_expand=docs_to_expand,
        max_provider_calls=max_provider_calls,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fixture":
        run = run_p2_fixture(args.query, output_dir=args.output, citation_enabled=not args.no_citation)
        result = evaluate_p2_run(run)
    elif args.command == "live":
        pipeline = build_live_pipeline(citation_enabled=not args.no_citation, page_size=args.page_size, max_workers=args.max_workers)
        run = pipeline.run(args.query, output_dir=args.output)
        result = evaluate_p2_run(run)
    elif args.command == "tree":
        runner = build_tree_runner(page_size=args.page_size, max_depth=args.max_depth, docs_to_expand=args.docs_to_expand, max_provider_calls=args.max_provider_calls)
        tree_result = runner.run(args.query)
        if getattr(args, "fulltext_topk", 0) > 0:
            from .fulltext import augment_topk, query_terms

            tree_result["papers"], _ft_stats = augment_topk(tree_result["papers"], Path(args.output), query_terms(args.query), top_k=args.fulltext_topk)
            tree_result["papers"].sort(key=lambda p: (p.get("scores", {}).get("relevance") is not None, p.get("scores", {}).get("relevance") or -1.0), reverse=True)
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        for name, payload in (("papers", {"papers": tree_result["papers"]}), ("nodes", tree_result["nodes"]), ("edges", tree_result["edges"]), ("stats", tree_result["stats"]), ("errors", tree_result["errors"])):
            (root / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / "run_manifest.json").write_text(json.dumps({"schema_version": "search_tree_run.v1", "query": args.query, "stop_reason": tree_result["stop_reason"], "stats": tree_result["stats"]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = evaluate_p2_run({"papers": tree_result["papers"]})
    elif args.command == "replay":
        payload = replay_p2(args.input)
        result = evaluate_p2_run(payload, gold_ids=args.gold_id)
    else:
        payload = replay_p2(args.input, validate_final=False)
        result = build_final_selection(payload, top_k=args.top_k)
        path = Path(args.input) / "final_selection.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
