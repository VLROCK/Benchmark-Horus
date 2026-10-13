"""Command-line inspection and structural validation of registered data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from horus_benchmark.config import load_yaml
from horus_benchmark.data import DatasetCatalog, DatasetLoader
from horus_benchmark.data.loader import compare_schema


def _inspect(catalog: DatasetCatalog, resource_name: str) -> int:
    spec = catalog.resource(resource_name)
    loader = DatasetLoader(spec)
    payload = {
        "dataset_id": catalog.dataset_id,
        "dataset_version": catalog.dataset_version,
        "status": catalog.status,
        "resource": spec.name,
        "role": spec.role,
        "expected_rows": spec.expected_rows,
        "actual_rows": loader.count_rows(),
        "physical": loader.physical_metadata(),
        "schema": {field.name: str(field.type) for field in loader.schema},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _validate(catalog: DatasetCatalog) -> int:
    issues: list[dict[str, Any]] = []
    schema_payload: dict[str, Any] = {}
    if catalog.schema_manifest_path is not None:
        schema_payload = load_yaml(catalog.schema_manifest_path)
    declared_columns = schema_payload.get("columns", {})
    schema_resources = set(schema_payload.get("applies_to", []))

    resources: dict[str, Any] = {}
    for name, spec in catalog.resources.items():
        if not spec.path.is_file():
            issues.append({"resource": name, "issue": "file not found"})
            continue

        loader = DatasetLoader(spec)
        actual_rows = loader.count_rows()
        resources[name] = {
            "path": str(spec.path),
            "rows": actual_rows,
            "format": spec.format,
        }
        if spec.expected_rows is not None and actual_rows != spec.expected_rows:
            issues.append(
                {
                    "resource": name,
                    "issue": "row count mismatch",
                    "expected": spec.expected_rows,
                    "actual": actual_rows,
                }
            )
        if name in schema_resources:
            for difference in compare_schema(loader.schema, declared_columns):
                issues.append({"resource": name, "issue": difference})

    payload = {
        "dataset_id": catalog.dataset_id,
        "dataset_version": catalog.dataset_version,
        "resources": resources,
        "valid": not issues,
        "issues": issues,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horus-data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/horus_rc.yaml"),
        help="Path to the dataset catalog YAML.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect one resource.")
    inspect_parser.add_argument("--resource", default="corpus")
    subparsers.add_parser("validate", help="Validate files, rows and schemas.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = DatasetCatalog.from_yaml(args.config)
    if args.command == "inspect":
        return _inspect(catalog, args.resource)
    return _validate(catalog)


if __name__ == "__main__":
    raise SystemExit(main())

