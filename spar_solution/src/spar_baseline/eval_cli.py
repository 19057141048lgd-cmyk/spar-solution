"""P1 检索评测命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .autoscholar_smoke import run_autoscholar_smoke
from .experiment import run_wifi_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spar-p1-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("wifi-fixture", help="run the four WiFi P1 modes without network")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--gold", type=Path)
    fixture.add_argument("--page-size", type=int, default=20)
    smoke = subparsers.add_parser("auto-scholar-smoke", help="run a bounded live arXiv AutoScholar smoke")
    smoke.add_argument("--dataset", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--offset", type=int, default=0)
    smoke.add_argument("--limit", type=int, default=5)
    smoke.add_argument("--page-size", type=int, default=10)
    smoke.add_argument("--sleep", type=float, default=3.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "wifi-fixture":
        run_wifi_fixture(args.output, gold_path=args.gold, page_size=args.page_size)
        return 0
    if args.command == "auto-scholar-smoke":
        payload = run_autoscholar_smoke(
            args.dataset,
            args.output,
            offset=args.offset,
            limit=args.limit,
            page_size=args.page_size,
            sleep_seconds=args.sleep,
        )
        print(json.dumps({"output": str(args.output), **payload["summary"]}, ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
