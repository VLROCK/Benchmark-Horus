"""CLI for sharded full-corpus theme annotation and derived tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from horus_benchmark.audit.theme_common import load_context
from horus_benchmark.audit.theme_llm_full import (
    finalize_full, prepare_full_first, prepare_full_second,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horus-theme-llm-full")
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-first")
    second = commands.add_parser("prepare-second")
    second.add_argument("--first-index", type=Path, required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--first-index", type=Path, required=True)
    final.add_argument("--second-index", type=Path, required=True)
    args = parser.parse_args(argv)
    context = load_context(args.config)
    if args.command == "prepare-first":
        path = prepare_full_first(context)
    elif args.command == "prepare-second":
        path = prepare_full_second(context, args.first_index)
    else:
        path = finalize_full(context, args.first_index, args.second_index)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
