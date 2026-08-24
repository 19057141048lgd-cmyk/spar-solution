"""P2 fixture/replay 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .p2_metrics import evaluate_p2_run
from .p2_pipeline import replay_p2, run_p2_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or replay the P2 structured retrieval pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="run the no-key fixture")
    fixture.add_argument("--query", default="WiFi heart rate monitoring")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--no-citation", action="store_true")
    replay = sub.add_parser("replay", help="read a P2 artifact directory")
    replay.add_argument("--input", required=True)
    replay.add_argument("--gold-id", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fixture":
        run = run_p2_fixture(args.query, output_dir=args.output, citation_enabled=not args.no_citation)
        result = evaluate_p2_run(run)
    else:
        payload = replay_p2(args.input)
        result = evaluate_p2_run(payload, gold_ids=args.gold_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
