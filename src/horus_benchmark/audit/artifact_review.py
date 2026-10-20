"""Reproducible, non-mutating artifact-review samples from the full corpus.

The core sample is balanced for discovery, not a prevalence estimate of the
whole corpus. The extension deliberately seeks under-covered strata and must
never be pooled with the core when estimating artifact frequencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from horus_benchmark.data import DatasetCatalog, DatasetLoader


AREAS = (
    "academic", "informational", "journalistic", "juridical", "literary", "social",
)
META_COLUMNS = (
    "record_id", "label_name", "broad_area", "source_name", "source_record_id",
    "source_url", "generator", "model_display_name", "provider", "theme",
    "genre", "split", "split_group_id", "duplicate_group_id", "content_hash",
    "content",
)
MANIFEST_COLUMNS = (
    "record_id", "sample_role", "label_name", "broad_area", "source_name",
    "source_record_id", "source_url", "generator", "model_display_name", "provider",
    "theme", "genre", "split", "split_group_id", "duplicate_group_id",
    "content_hash", "content_sha256_review", "char_count_review", "word_count_review",
    "length_bin", "selection_priority",
)
REVIEW_COLUMNS = (
    "record_id", "broad_area", "content", "artifact_codes", "severity",
    "proposed_action", "span_or_excerpt", "review_note",
)


def stable_priority(seed: int, record_id: str, purpose: str = "sample") -> str:
    """A corpus-order-independent random-looking priority for a fixed seed."""

    return hashlib.sha256(f"{seed}|{purpose}|{record_id}".encode("utf-8")).hexdigest()


def length_bin(word_count: int) -> str:
    if word_count < 150:
        return "0-149"
    if word_count < 400:
        return "150-399"
    if word_count < 800:
        return "400-799"
    return "800+"


def read_pilot(path: Path) -> dict[str, dict[str, str]]:
    """Read the fixed human pilot without changing its annotations or contents."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"record_id", "broad_area", "content"}.issubset(reader.fieldnames):
            raise ValueError("Pilot needs record_id, broad_area and content")
        rows = {row["record_id"]: row for row in reader}
    if len(rows) != 120:
        raise ValueError("Pilot must contain 120 distinct human record IDs")
    counts = Counter(row["broad_area"] for row in rows.values())
    if counts != Counter({area: 20 for area in AREAS}):
        raise ValueError(f"Pilot must contain 20 texts per area: {dict(counts)}")
    return rows


def scan_metadata(loader: DatasetLoader, pilot: dict[str, dict[str, str]], seed: int) -> tuple[
    dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]],
]:
    """Scan once; retain metadata and lengths, but not the full corpus text."""

    missing = set(META_COLUMNS) - set(loader.schema.names)
    if missing:
        raise ValueError(f"Corpus lacks required columns: {sorted(missing)}")
    pilot_found: dict[str, dict[str, Any]] = {}
    humans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    synthetics: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for batch in loader.iter_batches(columns=META_COLUMNS, batch_size=8192, use_threads=False):
        for row in batch.to_pylist():
            rid = str(row["record_id"])
            if rid in seen_ids:
                raise ValueError(f"Duplicate record_id in corpus: {rid}")
            seen_ids.add(rid)
            area = row["broad_area"]
            if area not in AREAS:
                raise ValueError(f"Unexpected broad_area for {rid}: {area}")
            content = row.pop("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Empty content for {rid}")
            row["record_id"] = rid
            row["word_count_review"] = len(content.split())
            row["char_count_review"] = len(content)
            row["length_bin"] = length_bin(row["word_count_review"])
            row["selection_priority"] = stable_priority(seed, rid)
            if rid in pilot:
                if row["label_name"] != "human" or pilot[rid]["broad_area"] != area:
                    raise ValueError(f"Pilot label or area changed: {rid}")
                if pilot[rid]["content"] != content:
                    raise ValueError(f"Pilot text differs from configured corpus: {rid}")
                pilot_found[rid] = row
            if row["label_name"] == "human":
                humans[area].append(row)
            elif row["label_name"] == "synthetic":
                generator = row["generator"] or "<missing>"
                synthetics[(area, generator)].append(row)
            else:
                raise ValueError(f"Unexpected label_name for {rid}: {row['label_name']}")
    if set(pilot_found) != set(pilot):
        raise ValueError(f"Pilot IDs absent from corpus: {sorted(set(pilot) - set(pilot_found))[:5]}")
    return pilot_found, humans, synthetics


def select_diagnostic(
    candidates: list[dict[str, Any]], previous: list[dict[str, Any]],
    number: int, dimensions: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Greedily cover under-represented strata; not a probability sample."""

    used = {row["record_id"] for row in previous}
    available = [row for row in candidates if row["record_id"] not in used]
    if len(available) < number:
        raise ValueError(f"Only {len(available)} candidates for {number} diagnostic rows")
    counts = {field: Counter((row.get(field) or "<missing>") for row in previous)
              for field in dimensions}
    selected: list[dict[str, Any]] = []
    for _ in range(number):
        choice = min(
            available,
            key=lambda row: (
                sum(counts[field][row.get(field) or "<missing>"] for field in dimensions),
                tuple(counts[field][row.get(field) or "<missing>"] for field in dimensions),
                row["selection_priority"],
            ),
        )
        selected.append(choice)
        available.remove(choice)
        for field in dimensions:
            counts[field][choice.get(field) or "<missing>"] += 1
    return selected


def choose_sample(
    pilot_found: dict[str, dict[str, Any]],
    humans: dict[str, list[dict[str, Any]]],
    synthetics: dict[tuple[str, str], list[dict[str, Any]]],
    human_extension_per_area: int = 20,
    synthetic_core_per_cell: int = 2,
    synthetic_extension_per_cell: int = 2,
) -> list[dict[str, Any]]:
    """Keep the fixed pilot, then sample every generator × area cell."""

    selected: list[dict[str, Any]] = []
    for area in AREAS:
        core = sorted(
            (row for row in pilot_found.values() if row["broad_area"] == area),
            key=lambda row: row["selection_priority"],
        )
        selected.extend({**row, "sample_role": "core_human_pilot"} for row in core)
        extension = select_diagnostic(
            humans[area], core, human_extension_per_area,
            ("source_name", "length_bin", "split"),
        )
        selected.extend({**row, "sample_role": "diagnostic_human"} for row in extension)
    for cell in sorted(synthetics):
        candidates = sorted(synthetics[cell], key=lambda row: row["selection_priority"])
        required = synthetic_core_per_cell + synthetic_extension_per_cell
        if len(candidates) < required:
            raise ValueError(f"Synthetic cell {cell} has {len(candidates)} rows, needs {required}")
        core = candidates[:synthetic_core_per_cell]
        selected.extend({**row, "sample_role": "core_synthetic"} for row in core)
        extension = select_diagnostic(
            candidates, core, synthetic_extension_per_cell,
            ("length_bin", "theme", "genre", "split"),
        )
        selected.extend({**row, "sample_role": "diagnostic_synthetic"} for row in extension)
    ids = [row["record_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Sample contains repeated record IDs")
    return selected


def export_sample(
    catalog_path: Path, pilot_path: Path, output_dir: Path, seed: int,
) -> Path:
    """Write only derived review artifacts; never modify the corpus or pilot."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    catalog = DatasetCatalog.from_yaml(catalog_path)
    resource = catalog.resource("corpus")
    if resource.format != "parquet":
        raise ValueError("Artifact review expects a Parquet corpus")
    loader = DatasetLoader(resource)
    pilot = read_pilot(pilot_path)
    pilot_found, humans, synthetics = scan_metadata(loader, pilot, seed)
    selected = choose_sample(pilot_found, humans, synthetics)
    selected_by_id = {row["record_id"]: row for row in selected}
    content_by_id: dict[str, str] = {}
    for batch in loader.iter_batches(columns=("record_id", "content"), batch_size=8192, use_threads=False):
        for row in batch.to_pylist():
            rid = str(row["record_id"])
            if rid in selected_by_id:
                content_by_id[rid] = row["content"]
    if set(content_by_id) != set(selected_by_id):
        raise ValueError("Could not retrieve all sampled texts")
    for rid, row in selected_by_id.items():
        content = content_by_id[rid]
        if len(content) != row["char_count_review"] or len(content.split()) != row["word_count_review"]:
            raise ValueError(f"Content changed during sampling: {rid}")
        row["content_sha256_review"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    review_path = output_dir / "blind_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in sorted(selected, key=lambda item: stable_priority(seed, item["record_id"], "review")):
            writer.writerow({"record_id": row["record_id"], "broad_area": row["broad_area"],
                             "content": content_by_id[row["record_id"]]})
    summary = {
        "dataset_id": catalog.dataset_id,
        "dataset_version": catalog.dataset_version,
        "corpus_path": str(resource.path),
        "corpus_sha256": sha256_file(resource.path),
        "pilot_path": str(pilot_path.resolve()),
        "pilot_sha256": sha256_file(pilot_path),
        "seed": seed,
        "design": "120 fixed human pilot + 120 human diagnostic; 2 core + 2 diagnostic per area-generator cell",
        "counts_by_role": dict(sorted(Counter(row["sample_role"] for row in selected).items())),
        "synthetic_cells": len(synthetics),
        "counts_by_area_label": {
            f"{area}|{label}": count
            for (area, label), count in sorted(Counter(
                (row["broad_area"], row["label_name"]) for row in selected
            ).items())
        },
        "interpretation_warning": (
            "Core rows are balanced by design; diagnostic rows seek diversity. "
            "Do not pool them or treat raw percentages as corpus prevalence."
        ),
    }
    (output_dir / "sampling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--human-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    print(export_sample(args.dataset_config, args.human_pilot, args.output, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
