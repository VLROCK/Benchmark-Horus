"""Freeze a verified N2 candidate by recording hashes of its derived artifacts.

This does not alter the source corpus, normalized corpus, or trace. The freeze
is an immutable candidate reference, not approval of a final benchmark split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from horus_benchmark.data.normalize_corpus import sha256_file
from horus_benchmark.data.verify_normalization import verify


ARTIFACTS = (
    "normalized_corpus.parquet",
    "normalization_trace.parquet",
    "summary.json",
    "verification.json",
    "audit_dataset.yaml",
    "audit_experiment.yaml",
)
MANIFEST = "freeze_manifest.json"


def _inventory(run_dir: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for name in ARTIFACTS:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"N2 freeze requires {path}")
        result[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _provenance(run_dir: Path) -> dict[str, object]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))
    if summary.get("stage") != "n2" or verification.get("stage") != "n2":
        raise ValueError("Only a verified N2 run can be frozen")
    if not verification.get("verified") or verification.get("rows") != summary.get("output_rows"):
        raise ValueError("N2 verification is missing or disagrees with the summary")
    if verification.get("source_sha256") != summary.get("source_sha256"):
        raise ValueError("N2 source hashes disagree")
    return {
        "dataset_id": summary["dataset_id"],
        "dataset_version": summary["dataset_version"],
        "source_sha256": summary["source_sha256"],
        "config_sha256": summary["config_sha256"],
        "code_sha256": summary["code_sha256"],
        "rows": summary["output_rows"],
    }


def freeze(config_path: Path, run_dir: Path) -> Path:
    """Reverify all rows, then create a non-overwriting artifact inventory."""

    run_dir = run_dir.expanduser().resolve()
    target = run_dir / MANIFEST
    if target.exists():
        raise FileExistsError(f"Freeze manifest already exists: {target}")
    verify(config_path.expanduser().resolve(), run_dir)
    payload = {
        "schema_version": 1,
        "kind": "verified_n2_candidate_freeze",
        **_provenance(run_dir),
        "artifacts": _inventory(run_dir),
        "note": "Candidate text view only; no selection or leakage-free split is asserted.",
    }
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return target


def check(run_dir: Path) -> Path:
    """Reject any change to the frozen artifacts or their provenance."""

    run_dir = run_dir.expanduser().resolve()
    target = run_dir / MANIFEST
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("kind") != "verified_n2_candidate_freeze":
        raise ValueError("Unsupported N2 freeze manifest")
    if set(payload.get("artifacts", {})) != set(ARTIFACTS):
        raise ValueError("N2 freeze artifact inventory is incomplete")
    for key, value in _provenance(run_dir).items():
        if payload.get(key) != value:
            raise ValueError(f"Frozen provenance differs: {key}")
    if payload["artifacts"] != _inventory(run_dir):
        raise ValueError("A frozen N2 artifact changed or is missing")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--config", type=Path, required=True)
    freeze_parser.add_argument("--run-dir", type=Path, required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(freeze(args.config, args.run_dir) if args.command == "freeze" else check(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
