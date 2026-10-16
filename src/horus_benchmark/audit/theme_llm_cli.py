"""CLI for the three-model, 600-text theme agreement study."""

from __future__ import annotations

import argparse
from pathlib import Path

from horus_benchmark.audit.theme_common import load_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horus-theme-llm")
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Reuse 120 pilot IDs and add 480 human texts")
    sample.add_argument("--pilot", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Export blind JSONL prompts for both LLMs")
    prepare.add_argument("--sample", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Agreement, not accuracy")
    evaluate.add_argument("--sample", type=Path, required=True)
    evaluate.add_argument("--gemma", type=Path, required=True)
    evaluate.add_argument("--gpt-oss", type=Path, required=True)
    evaluate.add_argument("--bert", type=Path, required=True)
    compare = commands.add_parser(
        "compare-context", help="Two-LLM agreement and coverage on the same 600 texts",
    )
    compare.add_argument("--sample", type=Path, required=True)
    compare.add_argument("--baseline-config", type=Path, required=True)
    compare.add_argument("--baseline-gemma", type=Path, required=True)
    compare.add_argument("--baseline-gpt-oss", type=Path, required=True)
    compare.add_argument("--gemma", type=Path, required=True)
    compare.add_argument("--gpt-oss", type=Path, required=True)
    args = parser.parse_args(argv)
    context = load_context(args.config)
    if args.command == "sample":
        from horus_benchmark.audit.theme_llm_600 import create_sample

        path = create_sample(context, args.pilot)
    elif args.command == "prepare":
        from horus_benchmark.audit.theme_llm_600 import prepare_inference

        path = prepare_inference(context, args.sample)
    elif args.command == "evaluate":
        from horus_benchmark.audit.theme_llm_600 import evaluate_agreement

        path = evaluate_agreement(context, args.sample, args.gemma, args.gpt_oss, args.bert)
    else:
        from horus_benchmark.audit.theme_llm_600 import evaluate_llm_context

        baseline_context = load_context(args.baseline_config)
        path = evaluate_llm_context(
            context, baseline_context, args.sample, args.baseline_gemma,
            args.baseline_gpt_oss, args.gemma, args.gpt_oss,
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
