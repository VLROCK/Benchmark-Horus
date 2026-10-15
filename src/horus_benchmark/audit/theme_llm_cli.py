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
    infer = commands.add_parser("infer", help="Run one local vLLM model on shared texts")
    infer.add_argument("--sample", type=Path, required=True)
    infer.add_argument("--model-key", choices=("gemma4_31b", "gpt_oss_120b"), required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--limit", type=int, help="Only for a smoke; use a separate output directory")
    evaluate = commands.add_parser("evaluate", help="Agreement, not accuracy")
    evaluate.add_argument("--sample", type=Path, required=True)
    evaluate.add_argument("--gemma", type=Path, required=True)
    evaluate.add_argument("--gpt-oss", type=Path, required=True)
    evaluate.add_argument("--bert", type=Path, required=True)
    args = parser.parse_args(argv)
    context = load_context(args.config)
    if args.command == "sample":
        from horus_benchmark.audit.theme_llm_600 import create_sample

        path = create_sample(context, args.pilot)
    elif args.command == "infer":
        from horus_benchmark.audit.theme_llm_inference import infer_local

        path = infer_local(
            context, args.sample, args.model_key, args.checkpoint,
            args.output_dir, args.limit,
        )
    else:
        from horus_benchmark.audit.theme_llm_600 import evaluate_agreement

        path = evaluate_agreement(context, args.sample, args.gemma, args.gpt_oss, args.bert)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
