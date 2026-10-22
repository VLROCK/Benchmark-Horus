"""Audit corrected hashes without changing the immutable source corpus.

This checks every source-hash mismatch against all source and N1 records.
Equal hashes with equal text are duplicates, not cryptographic collisions.
The researcher runs this program on the cluster (AGENTS.md rule 19).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from horus_benchmark.data.normalize_corpus import _config, sha256_file
from horus_benchmark.data.verify_normalization import _rows


def _pairs(digest_to_targets: dict[str, set[str]], rows, field: str, phase: str):
    pairs: set[tuple[str, str, str]] = set()
    for row in rows:
        for target_id in digest_to_targets.get(row[field], ()):
            if target_id != row["record_id"]:
                a, b = sorted((target_id, row["record_id"]))
                pairs.add((phase, a, b))
    return pairs


def run(config_path: Path, diagnostics_dir: Path, n1_run_dir: Path, output_dir: Path) -> Path:
    _, catalog, rules, _ = _config(config_path)
    if rules.stage != "n1":
        raise ValueError("Hash repair audit requires the N1 configuration")
    source = catalog.resource("corpus").path
    diagnostics_dir = diagnostics_dir.expanduser().resolve()
    n1_run_dir = n1_run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_relative_to(source.parent.resolve()):
        raise ValueError("Use a fresh output directory outside the source directory")
    n1_summary = json.loads((n1_run_dir / "summary.json").read_text(encoding="utf-8"))
    diagnostic_summary = json.loads((diagnostics_dir / "summary.json").read_text(encoding="utf-8"))
    source_sha = sha256_file(source)
    if (n1_summary["stage"] != "n1" or n1_summary["source_sha256"] != source_sha
            or diagnostic_summary["source_sha256"] != source_sha
            or not (n1_run_dir / "verification.json").is_file()):
        raise ValueError("Diagnostics and verified N1 must derive from this source")
    with (diagnostics_dir / "hash_mismatch_records.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        listed = {row["record_id"]: row for row in csv.DictReader(stream)}
    if len(listed) != diagnostic_summary["counts"]["content_hash_mismatch"]:
        raise ValueError("Mismatch CSV count differs from diagnostic summary")

    source_targets: dict[str, set[str]] = defaultdict(set)
    n1_targets: dict[str, set[str]] = defaultdict(set)
    observed: set[str] = set()
    all_mismatch_ids: set[str] = set()
    trace_path = n1_run_dir / "normalization_trace.parquet"
    for row in _rows(trace_path):
        rid = row["record_id"]
        if row["source_content_hash"] != row["source_content_sha256"]:
            all_mismatch_ids.add(rid)
        if rid in listed:
            if (row["source_content_hash"] != listed[rid]["declared_hash"]
                    or row["source_content_sha256"] != listed[rid]["exact_content_sha256"]):
                raise ValueError(f"Diagnostic/N1 hash disagreement: {rid}")
            observed.add(rid)
            source_targets[row["source_content_sha256"]].add(rid)
            n1_targets[row["normalized_content_sha256"]].add(rid)
    if observed != set(listed) or all_mismatch_ids != set(listed):
        raise ValueError("Mismatch IDs differ between N1 trace and diagnostic CSV")

    pairs = _pairs(source_targets, _rows(trace_path), "source_content_sha256", "source")
    pairs |= _pairs(n1_targets, _rows(trace_path), "normalized_content_sha256", "n1")
    involved = set(listed)
    for _, a, b in pairs:
        involved.update((a, b))
    source_content = {row["record_id"]: row["content"] for row in _rows(source)
                      if row["record_id"] in involved}
    n1_content = {}
    for row in _rows(n1_run_dir / "normalized_corpus.parquet"):
        if row["record_id"] in involved:
            n1_content[row["record_id"]] = row["content"]
            if row["record_id"] in listed:
                if row["content_hash"] != hashlib.sha256(row["content"].encode("utf-8")).hexdigest():
                    raise ValueError(f"N1 derived hash is stale: {row['record_id']}")
    if set(source_content) != involved or set(n1_content) != involved:
        raise ValueError("A referenced record ID is absent from source or N1")

    output_dir.mkdir(parents=True, exist_ok=False)
    duplicate_rows = []
    collision_rows = []
    for phase, a, b in sorted(pairs):
        values = source_content if phase == "source" else n1_content
        row = {"phase": phase, "record_id_a": a, "record_id_b": b,
               "same_content": values[a] == values[b]}
        (duplicate_rows if row["same_content"] else collision_rows).append(row)
    with (output_dir / "matching_hash_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "phase", "record_id_a", "record_id_b", "same_content",
        ])
        writer.writeheader()
        writer.writerows(duplicate_rows + collision_rows)
    summary = {
        "source_sha256": source_sha,
        "n1_run_dir": str(n1_run_dir),
        "mismatched_source_hashes": len(listed),
        "distinct_recomputed_source_hashes": len(source_targets),
        "distinct_n1_hashes_for_targets": len(n1_targets),
        "exact_duplicate_pairs_involving_targets": len(duplicate_rows),
        "different_text_same_sha256_pairs": len(collision_rows),
        "safe_to_rely_on_recomputed_hashes": not collision_rows,
        "note": "No source hashes were edited. Exact duplicates, if present, still need leakage controls.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--n1-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.diagnostics_dir, args.n1_run_dir, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
