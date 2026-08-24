"""P1 检索命令行：默认 live，``--mock`` 永不访问网络。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import get_provider_config, load_config
from .mock_pipeline import _paper
from .openalex_provider import OpenAlexProvider
from .providers.bohrium import BohriumProvider
from .search_service import search


class _MockProvider:
    def __init__(self, name: str, abstract: str, *, fail: bool = False) -> None:
        self.name = name
        self.abstract = abstract
        self.fail = fail

    def search(self, query: str, *, page_size: int = 10) -> list[dict[str, Any]]:
        if self.fail:
            raise RuntimeError("mock provider intentionally unavailable")
        paper = _paper(self.name, self.abstract)
        paper["bibliography"]["title"] = f"{query}: WiFi heart-rate monitoring mock result"
        paper["bibliography"]["abstract"] = self.abstract
        paper["content"]["char_count"] = len(self.abstract)
        return [paper]


def _mock_providers() -> list[_MockProvider]:
    return [
        _MockProvider("bohrium", "WiFi CSI can estimate contactless heart rate from respiratory and cardiac motion."),
        _MockProvider("openalex", "A second source provides complementary metadata for WiFi-based vital-sign sensing."),
        _MockProvider("fixture_unavailable", "", fail=True),
    ]


def _live_providers(names: list[str]) -> list[Any]:
    config = load_config()
    providers: list[Any] = []
    for name in names:
        normalized = name.casefold()
        if normalized == "bohrium":
            providers.append(BohriumProvider.from_config(config))
        elif normalized == "openalex":
            providers.append(OpenAlexProvider(get_provider_config(config, "openalex")))
        else:
            raise ValueError(f"unsupported provider: {name}")
    return providers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spar-p1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("search", help="search academic papers")
    command.add_argument("--query", required=True)
    command.add_argument("--output", required=True, type=Path)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="use fixture providers without network")
    mode.add_argument("--live", action="store_true", help="use configured providers (default)")
    command.add_argument("--providers", nargs="+", default=["bohrium", "openalex"])
    command.add_argument("--page-size", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "search":
        return 2
    providers = _mock_providers() if args.mock else _live_providers(args.providers)
    artifact = search(args.query, providers, page_size=args.page_size, mode="mock" if args.mock else "live")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
