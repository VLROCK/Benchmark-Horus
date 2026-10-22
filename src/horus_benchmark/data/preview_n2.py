"""Preview N2 cuts against a verified N1 view, without writing a new corpus.

The researcher runs this on the cluster. Candidate text is retained only in
review examples; every proposed span has a metadata/offset row in CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import zip_longest
from pathlib import Path

from horus_benchmark.data.normalization import (
    apply_n2_edits, n2_loss_guard, normalize_text, propose_n2_edits,
)
from horus_benchmark.data.normalize_corpus import _config, sha256_file
from horus_benchmark.data.verify_normalization import _rows


def _priority(seed: int, kind: str, rid: str) -> str:
    return hashlib.sha256(f"{seed}|{kind}|{rid}".encode("utf-8")).hexdigest()


def run(config_path: Path, n1_run_dir: Path, output_dir: Path, *,
        examples_per_kind: int = 80, seed: int = 20261022) -> Path:
    _, catalog, rules, _ = _config(config_path)
    if rules.stage != "n2":
        raise ValueError("Preview requires the N2 experiment configuration")
    if examples_per_kind < 1:
        raise ValueError("examples_per_kind must be positive")
    source = catalog.resource("corpus").path
    n1_run_dir = n1_run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_relative_to(source.parent.resolve()):
        raise ValueError("Use a fresh output directory outside the source directory")
    n1_summary = json.loads((n1_run_dir / "summary.json").read_text(encoding="utf-8"))
    source_hash = sha256_file(source)
    if n1_summary["stage"] != "n1" or n1_summary["source_sha256"] != source_hash:
        raise ValueError("N1 view was not produced from this immutable source")
    if not (n1_run_dir / "verification.json").is_file():
        raise ValueError("N1 view lacks its verification report")
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = output_dir / "n2_candidate_spans.csv"
    columns = ["record_id", "kind", "start", "end", "span_chars", "span_words",
               "proposed_loss_words", "n2_loss_guard_retained", "label_name",
               "broad_area", "source_name", "generator", "n1_sha256", "proposal_sha256"]
    totals: Counter[str] = Counter()
    groups: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[tuple[str, dict, dict]]] = defaultdict(list)
    n1_rules = replace(rules, stage="n1")
    with candidate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for source_row, n1_row in zip_longest(
            _rows(source), _rows(n1_run_dir / "normalized_corpus.parquet")
        ):
            if source_row is None or n1_row is None:
                raise ValueError("N1 and source row counts differ")
            rid = source_row["record_id"]
            if rid != n1_row["record_id"]:
                raise ValueError(f"N1/source record_id alignment failed at {rid}")
            baseline = n1_row["content"]
            if normalize_text(source_row["content"], n1_rules).text != baseline:
                raise ValueError(f"N1 text does not match current common rules: {rid}")
            if n1_row["word_count"] != len(baseline.split()):
                raise ValueError(f"N1 word_count is stale for {rid}")
            if n1_row["char_count"] != len(baseline):
                raise ValueError(f"N1 char_count is stale for {rid}")
            if n1_row["content_hash"] != hashlib.sha256(baseline.encode("utf-8")).hexdigest():
                raise ValueError(f"N1 content_hash does not match its text for {rid}")
            edits = propose_n2_edits(baseline, rules)
            proposed = apply_n2_edits(baseline, edits) if edits else baseline
            guard = bool(edits) and n2_loss_guard(baseline, proposed, rules)
            result = normalize_text(source_row["content"], rules)
            if result.text != (baseline if guard else proposed):
                raise ValueError(f"N2 preview/normalizer disagreement at {rid}")
            totals["rows"] += 1
            label, area = source_row["label_name"], source_row["broad_area"]
            source_name = str(source_row.get("source_name") or "<missing>")
            generator = str(source_row.get("generator") or "<missing>")
            if edits:
                totals["candidate_rows"] += 1
                if guard:
                    totals["guard_retained_rows"] += 1
            if not edits and "ENGLISH_ABSTRACT_CANDIDATE" in result.flags:
                negative_kind = "ABSTRACT_FLAG_NO_CUT"
            elif not edits and "REFERENCES" in result.flags:
                negative_kind = "REFERENCES_FLAG_NO_CUT"
            else:
                negative_kind = None
            if negative_kind:
                examples[negative_kind].append((
                    _priority(seed, negative_kind, rid),
                    {"record_id": rid, "kind": negative_kind, "content_n1": baseline,
                     "content_proposed": baseline, "guard_retained": False, "spans": []},
                    {"record_id": rid, "kind": negative_kind, "label_name": label,
                     "broad_area": area, "source_name": source_name, "generator": generator},
                ))
            for edit in edits:
                totals[f"candidate:{edit.kind}"] += 1
                if guard:
                    totals[f"guard_retained:{edit.kind}"] += 1
                groups[(label, area, source_name)][f"candidate:{edit.kind}"] += 1
                if guard:
                    groups[(label, area, source_name)][f"guard_retained:{edit.kind}"] += 1
                writer.writerow({
                    "record_id": rid, "kind": edit.kind, "start": edit.start, "end": edit.end,
                    "span_chars": edit.end - edit.start,
                    "span_words": len(baseline[edit.start:edit.end].split()),
                    "proposed_loss_words": len(baseline.split()) - len(proposed.split()),
                    "n2_loss_guard_retained": guard, "label_name": label,
                    "broad_area": area, "source_name": source_name, "generator": generator,
                    "n1_sha256": hashlib.sha256(baseline.encode("utf-8")).hexdigest(),
                    "proposal_sha256": hashlib.sha256(proposed.encode("utf-8")).hexdigest(),
                })
                examples[edit.kind].append((
                    _priority(seed, edit.kind, rid),
                    {"record_id": rid, "kind": edit.kind, "content_n1": baseline,
                     "content_proposed": proposed, "guard_retained": guard,
                     "spans": [{"kind": item.kind, "start": item.start, "end": item.end,
                                "removed_text": baseline[item.start:item.end]}
                               for item in edits]},
                    {"record_id": rid, "kind": edit.kind, "label_name": label,
                     "broad_area": area, "source_name": source_name, "generator": generator,
                     "guard_retained": guard},
                ))
            groups[(label, area, source_name)]["rows"] += 1
    if totals["rows"] != n1_summary["source_rows"]:
        raise ValueError("Preview row count differs from N1")
    with (output_dir / "n2_candidates_by_source.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        keys = sorted({key for counts in groups.values() for key in counts if key != "rows"})
        writer = csv.DictWriter(stream, fieldnames=[
            "label_name", "broad_area", "source_name", "rows", *keys,
        ])
        writer.writeheader()
        for (label, area, source_name), counts in sorted(groups.items()):
            writer.writerow({"label_name": label, "broad_area": area,
                             "source_name": source_name, "rows": counts["rows"],
                             **{key: counts[key] for key in keys}})
    selected: dict[str, tuple[dict, dict]] = {}
    # Sampling is post hoc only. Keep rare class/source footer cells visible
    # and export every abstract cut so its boundary can be checked.
    for kind, rows in sorted(examples.items()):
        buckets: dict[tuple[str, str], list[tuple[str, dict, dict]]] = defaultdict(list)
        for row in rows:
            manifest = row[2]
            source_cell = manifest["source_name"] if kind == "TERMINAL_FOOTER" else ""
            buckets[(manifest["label_name"], source_cell)].append(row)
        for bucket in sorted(buckets):
            limit = len(buckets[bucket]) if kind == "PARALLEL_ABSTRACT_REMOVE" else examples_per_kind
            for _, blind, manifest in sorted(buckets[bucket])[:limit]:
                selected.setdefault(blind["record_id"], (blind, manifest))
    with (output_dir / "n2_examples_blind.jsonl").open("w", encoding="utf-8") as stream:
        for blind, _ in selected.values():
            stream.write(json.dumps(blind, ensure_ascii=False) + "\n")
    with (output_dir / "n2_examples_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "record_id", "kind", "label_name", "broad_area", "source_name",
            "generator", "guard_retained",
        ])
        writer.writeheader()
        for _, manifest in selected.values():
            writer.writerow(manifest)
    (output_dir / "summary.json").write_text(json.dumps({
        "source_sha256": source_hash, "n1_run_dir": str(n1_run_dir),
        "n2_config_sha256": sha256_file(config_path.resolve()),
        "counts": dict(sorted(totals.items())), "sample_seed": seed,
        "examples_per_kind": examples_per_kind,
        "note": "Preview only; no N2 corpus was written. Labels/sources appear only in post hoc tables."
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n1-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-kind", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args(argv)
    print(run(args.config, args.n1_run_dir, args.output_dir,
              examples_per_kind=args.examples_per_kind, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
