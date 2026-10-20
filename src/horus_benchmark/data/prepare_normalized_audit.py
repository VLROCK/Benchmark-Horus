"""Prepare, but do not execute, the existing audit against a derived corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from horus_benchmark.config import load_yaml, resolve_from
from horus_benchmark.data.normalize_corpus import _config


def prepare(normalization_config: Path, run_dir: Path, base_audit_config: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))
    if not verification.get("verified") or verification["rows"] != summary["source_rows"]:
        raise ValueError("Verified normalization required before audit preparation")
    _, catalog, rules, _ = _config(normalization_config)
    if summary["stage"] != rules.stage or summary["dataset_version"] != catalog.dataset_version:
        raise ValueError("Run summary and normalization config differ")
    target_dataset = run_dir / "audit_dataset.yaml"
    target_experiment = run_dir / "audit_experiment.yaml"
    if target_dataset.exists() or target_experiment.exists():
        raise FileExistsError("Audit configs already exist; refuse to overwrite")
    taxonomy = catalog.resource("taxonomy")
    dataset_payload = {
        "schema_version": 1,
        "dataset_id": catalog.dataset_id,
        "dataset_version": f"{catalog.dataset_version}-{rules.stage}-candidate",
        "status": "derived-candidate",
        "workspace_root": str(catalog.workspace_root),
        "schema_manifest": str(catalog.schema_manifest_path) if catalog.schema_manifest_path else None,
        "resources": {
            "corpus": {
                "path": str(run_dir / "normalized_corpus.parquet"), "format": "parquet",
                "role": "processed_canonical_corpus", "expected_rows": summary["source_rows"],
                "immutable": True,
            },
            "taxonomy": {
                "path": str(taxonomy.path), "format": taxonomy.format,
                "role": taxonomy.role, "expected_rows": taxonomy.expected_rows,
                "immutable": True,
            },
        },
    }
    base_file = base_audit_config.expanduser().resolve()
    base = load_yaml(base_file)
    base_root = resolve_from(base_file.parent, base.get("workspace_root", "."))
    audit = dict(base)
    audit.update({
        "experiment_id": f"quality-shortcuts-normalized-{rules.stage}-candidate",
        "workspace_root": str(base_root),
        "dataset_config": str(target_dataset),
        "corpus_resource": "corpus",
        "split_resources": [],
        "benchmark_resource": None,
        "taxonomy_resource": "taxonomy",
        "input_is_normalized_view": True,
        "output_root": str(resolve_from(base_root, base["output_root"])),
    })
    target_dataset.write_text(
        yaml.safe_dump(dataset_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    target_experiment.write_text(
        yaml.safe_dump(audit, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return target_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalization-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-audit-config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare(args.normalization_config, args.run_dir, args.base_audit_config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
