"""Command line stages for the six-area human theme audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from horus_benchmark.audit.theme_common import load_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horus-theme")
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory", help="List current human theme values and mapping status")
    commands.add_parser("pilot", help="Export a blind taxonomy pilot sample")
    annotate = commands.add_parser("annotate-web", help="Annotate a blind CSV in a local browser")
    annotate.add_argument("--sheet", type=Path, required=True)
    annotate.add_argument("--port", type=int, default=8765)
    annotate.add_argument("--no-browser", action="store_true")
    commands.add_parser("synthetic-review", help="Export a small prompt-theme adherence check")
    commands.add_parser("baseline", help="Train six TF-IDF linear reference models")
    baseline_predict = commands.add_parser("baseline-predict", help="Predict humans with TF-IDF reference")
    baseline_predict.add_argument("--baseline-run", type=Path, required=True)
    commands.add_parser("train", help="Fine-tune one BERT model per broad area")
    predict = commands.add_parser("predict", help="Predict all human themes")
    predict.add_argument("--model-run", type=Path, required=True)
    sample = commands.add_parser("sample", help="Export blind main review sheets")
    sample.add_argument("--predictions", type=Path, required=True)
    sample.add_argument("--pilot", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Calibrate and test the confidence threshold")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--sample-manifest", type=Path, required=True)
    evaluate.add_argument("--primary", type=Path, required=True)
    evaluate.add_argument("--second", type=Path)
    evaluate.add_argument("--adjudicated", type=Path)
    evaluate.add_argument("--baseline-predictions", type=Path)
    fixed = commands.add_parser(
        "evaluate-fixed", help="Evaluate the same 120 texts with two human annotators"
    )
    fixed.add_argument("--predictions", type=Path, required=True)
    fixed.add_argument("--pilot", type=Path, required=True)
    fixed.add_argument("--primary", type=Path, required=True)
    fixed.add_argument("--second", type=Path, required=True)
    fixed.add_argument("--baseline-predictions", type=Path)
    materialize = commands.add_parser("materialize", help="Create derived corpus and theme tables")
    materialize.add_argument("--predictions", type=Path, required=True)
    materialize.add_argument("--threshold", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = load_context(args.config)
    if args.command == "inventory":
        from horus_benchmark.audit.theme_workflow import export_theme_inventory

        output = export_theme_inventory(context)
    elif args.command == "train":
        from horus_benchmark.audit.theme_model import train_models

        output = train_models(context)
    elif args.command == "baseline":
        from horus_benchmark.audit.theme_model import train_baselines

        output = train_baselines(context)
    elif args.command == "baseline-predict":
        from horus_benchmark.audit.theme_model import predict_baselines

        output = predict_baselines(context, args.baseline_run)
    elif args.command == "predict":
        from horus_benchmark.audit.theme_model import predict_humans

        output = predict_humans(context, args.model_run)
    elif args.command == "pilot":
        from horus_benchmark.audit.theme_workflow import export_pilot

        output = export_pilot(context)
    elif args.command == "annotate-web":
        from horus_benchmark.audit.theme_annotation_ui import serve_annotation_ui

        output = serve_annotation_ui(
            context, args.sheet, port=args.port, open_browser=not args.no_browser,
        )
    elif args.command == "synthetic-review":
        from horus_benchmark.audit.theme_workflow import export_synthetic_review

        output = export_synthetic_review(context)
    elif args.command == "sample":
        from horus_benchmark.audit.theme_workflow import export_review_sample

        output = export_review_sample(context, args.predictions, args.pilot)
    elif args.command == "evaluate-fixed":
        from horus_benchmark.audit.theme_workflow import evaluate_fixed_pilot

        output = evaluate_fixed_pilot(
            context, args.predictions, args.pilot, args.primary, args.second,
            args.baseline_predictions,
        )
    elif args.command == "evaluate":
        from horus_benchmark.audit.theme_workflow import evaluate_annotations

        output = evaluate_annotations(
            context, args.predictions, args.sample_manifest,
            args.primary, args.second, args.adjudicated, args.baseline_predictions,
        )
    else:
        from horus_benchmark.audit.theme_workflow import materialize_themes

        output = materialize_themes(context, args.predictions, args.threshold)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
