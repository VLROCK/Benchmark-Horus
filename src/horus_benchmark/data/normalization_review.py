"""Export reproducible, class-blind examples for assistant adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from horus_benchmark.data.normalize_corpus import _config
from horus_benchmark.data.verify_normalization import _rows


def _priority(seed: int, code: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}|{code}|{record_id}".encode("utf-8")).hexdigest()


def export_review(
    config_path: Path, run_dir: Path, *, per_cell: int = 3, maximum: int = 480,
    seed: int = 20261020,
) -> Path:
    if per_cell < 1 or maximum < 1:
        raise ValueError("per_cell and maximum must be positive")
    _, catalog, _, _ = _config(config_path)
    run_dir = run_dir.expanduser().resolve()
    cells: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in _rows(run_dir / "normalization_trace.parquet"):
        codes = row["normalization_flags"].split(";") if row["normalization_flags"] else ["UNFLAGGED_CONTROL"]
        for code in codes:
            if not code:
                continue
            key = (row["broad_area"], code)
            cells[key].append((_priority(seed, code, row["record_id"]), row["record_id"]))
    chosen: dict[str, set[str]] = defaultdict(set)
    for (_, code), candidates in sorted(cells.items()):
        for _, record_id in sorted(candidates)[:per_cell]:
            chosen[record_id].add(code)
    if len(chosen) > maximum:
        keep = sorted(chosen, key=lambda rid: _priority(seed, "CAP", rid))[:maximum]
        chosen = {rid: chosen[rid] for rid in keep}
    blind_path = run_dir / "review_examples_blind.jsonl"
    manifest_path = run_dir / "review_examples_manifest.csv"
    if blind_path.exists() or manifest_path.exists():
        raise FileExistsError("Review examples already exist; refuse to overwrite")
    source = catalog.resource("corpus").path
    with blind_path.open("w", encoding="utf-8") as blind, manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as manifest:
        fieldnames = ["record_id", "label_name", "broad_area", "source_name", "generator",
                      "model_display_name", "genre", "split", "selected_for_flags"]
        writer = csv.DictWriter(manifest, fieldnames=fieldnames)
        writer.writeheader()
        found = 0
        for original, derived in zip_longest(
            _rows(source), _rows(run_dir / "normalized_corpus.parquet")
        ):
            if original is None or derived is None or original["record_id"] != derived["record_id"]:
                raise ValueError("Source and normalized corpus are not row-aligned")
            rid = original["record_id"]
            if rid not in chosen:
                continue
            found += 1
            codes = ";".join(sorted(chosen[rid]))
            blind.write(json.dumps({
                "record_id": rid, "broad_area": original["broad_area"],
                "selected_for_flags": codes, "content_original": original["content"],
                "content_normalized": derived["content"],
            }, ensure_ascii=False) + "\n")
            writer.writerow({
                "record_id": rid, "label_name": original.get("label_name"),
                "broad_area": original.get("broad_area"),
                "source_name": original.get("source_name"),
                "generator": original.get("generator"),
                "model_display_name": original.get("model_display_name"),
                "genre": original.get("genre"), "split": original.get("split"),
                "selected_for_flags": codes,
            })
    if found != len(chosen):
        raise ValueError(f"Only found {found} / {len(chosen)} selected IDs")
    return blind_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=3)
    parser.add_argument("--maximum", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    print(export_review(args.config, args.run_dir, per_cell=args.per_cell,
                        maximum=args.maximum, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
