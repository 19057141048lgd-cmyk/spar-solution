"""P1 检索评测命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_wifi_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spar-p1-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("wifi-fixture", help="run the four WiFi P1 modes without network")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--gold", type=Path)
    fixture.add_argument("--page-size", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "wifi-fixture":
        run_wifi_fixture(args.output, gold_path=args.gold, page_size=args.page_size)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
