"""Command line entry point for full-corpus audits and paired comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from horus_benchmark.audit.runner import (
    recheck_integrity, refresh_summaries, run_audit, verify_near_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horus-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Audit one versioned corpus.")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument(
        "--reuse-features", type=Path,
        help="Reuse features from a previous run only if all input hashes match.",
    )
    refresh_parser = subparsers.add_parser(
        "refresh", help="Regenerate coverage tables and figures from a saved audit."
    )
    refresh_parser.add_argument("--run", type=Path, required=True)
    refresh_parser.add_argument("--config", type=Path, required=True)
    recheck_parser = subparsers.add_parser(
        "recheck-integrity", help="Recheck stored measurements against immutable partitions."
    )
    recheck_parser.add_argument("--run", type=Path, required=True)
    recheck_parser.add_argument("--config", type=Path, required=True)
    verify_parser = subparsers.add_parser(
        "verify-near", help="Recheck near-pair metrics with literal word shingles."
    )
    verify_parser.add_argument("--run", type=Path, required=True)
    verify_parser.add_argument("--config", type=Path, required=True)
    compare_parser = subparsers.add_parser(
        "compare", help="Compare matched original and normalized audit runs."
    )
    compare_parser.add_argument("--before", type=Path, required=True)
    compare_parser.add_argument("--after", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        print(run_audit(args.config, reuse_features=args.reuse_features))
    elif args.command == "refresh":
        print(refresh_summaries(args.run, args.config))
    elif args.command == "recheck-integrity":
        print(recheck_integrity(args.run, args.config))
    elif args.command == "verify-near":
        print(verify_near_pairs(args.run, args.config))
    else:
        from horus_benchmark.audit.compare import compare_runs

        print(compare_runs(args.before, args.after, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
