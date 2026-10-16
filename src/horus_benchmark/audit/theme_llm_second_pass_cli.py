"""Prepare and evaluate the targeted LLM theme second pass."""

from __future__ import annotations

import argparse
from pathlib import Path

from horus_benchmark.audit.theme_common import load_context
from horus_benchmark.audit.theme_llm_second_pass import (
    evaluate_second_pass, prepare_second_pass,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horus-theme-llm-second-pass")
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--first-gemma", type=Path, required=True)
    prepare.add_argument("--first-gpt-oss", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--sample", type=Path, required=True)
    evaluate.add_argument("--first-gemma", type=Path, required=True)
    evaluate.add_argument("--first-gpt-oss", type=Path, required=True)
    evaluate.add_argument("--second-gemma", type=Path, required=True)
    evaluate.add_argument("--second-gpt-oss", type=Path, required=True)
    args = parser.parse_args(argv)
    context = load_context(args.config)
    if args.command == "prepare":
        path = prepare_second_pass(
            context, args.sample, args.first_gemma, args.first_gpt_oss,
        )
    else:
        path = evaluate_second_pass(
            context, args.sample, args.first_gemma, args.first_gpt_oss,
            args.second_gemma, args.second_gpt_oss,
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
