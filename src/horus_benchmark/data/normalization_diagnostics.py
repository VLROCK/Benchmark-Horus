"""Read-only provenance and academic-bibliography diagnostics for the full corpus.

The researcher runs this on the cluster. It never repairs source metadata or
changes eligibility; all groups are post hoc, not normalization decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from horus_benchmark.data.normalization import REFERENCES, _terminal_references_start
from horus_benchmark.data.normalize_corpus import _config, sha256_file


WORD_RE = re.compile(r"\w+", re.UNICODE)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
SOURCE_HEADING = re.compile(
    r"(?im)^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:\*\*|__)?"
    r"(?:referências(?: bibliográficas)?|bibliografia|references|"
    r"fontes(?: consultadas)?|obras citadas)(?:\*\*|__)?"
    r"(?:[ \t]+\*?\(selecionadas\)\*?)?[ \t]*:?[ \t]*$"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_explanation(content: str, declared: str) -> str:
    if not isinstance(declared, str) or not HEX_SHA256.fullmatch(declared):
        return "invalid_sha256_format"
    if declared.lower() == _sha(content):
        return "sha256_hex_case_only"
    variants = {
        "nfc": unicodedata.normalize("NFC", content),
        "newline_lf": content.replace("\r\n", "\n").replace("\r", "\n"),
        "trimmed": content.strip(),
        "nfc_lf_trimmed": unicodedata.normalize("NFC", content).replace("\r\n", "\n").replace("\r", "\n").strip(),
        "nfkc_casefold_space": re.sub(r"\s+", " ", unicodedata.normalize("NFKC", content).casefold()).strip(),
    }
    for name, candidate in variants.items():
        if candidate != content and _sha(candidate) == declared.lower():
            return f"matches_{name}"
    return "unexplained_by_tested_variants"


def _priority(seed: int, key: str, rid: str) -> str:
    return hashlib.sha256(f"{seed}|{key}|{rid}".encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _choose_academic_sample(cells: dict[str, list[tuple[str, dict]]], maximum: int) -> list[dict]:
    """Round-robin by generator, then deterministic within-generator priority."""

    ordered = {key: [record for _, record in sorted(items)] for key, items in cells.items()}
    chosen: list[dict] = []
    while len(chosen) < maximum and any(ordered.values()):
        for key in sorted(ordered):
            if ordered[key] and len(chosen) < maximum:
                chosen.append(ordered[key].pop(0))
    return chosen


def run(config_path: Path, output_dir: Path, *, academic_sample: int = 120,
        hash_sample: int = 36, seed: int = 20261022) -> Path:
    _, catalog, _, batch_size = _config(config_path)
    source = catalog.resource("corpus").path
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_relative_to(source.parent.resolve()):
        raise ValueError("Use a fresh output directory outside the immutable source directory")
    if academic_sample < 1 or hash_sample < 1:
        raise ValueError("Sample sizes must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    columns = ["record_id", "content", "content_hash", "normalized_hash", "char_count",
               "word_count", "label_name", "broad_area", "source_name", "generator",
               "model_display_name", "genre", "source_file"]
    available = set(ds.dataset(str(source), format="parquet").schema.names)
    required = set(columns) - {"normalized_hash", "model_display_name", "genre", "source_file"}
    if required - available:
        raise ValueError(f"Missing source fields: {sorted(required - available)}")
    columns = [name for name in columns if name in available]
    scanner = ds.dataset(str(source), format="parquet").scanner(
        columns=columns, batch_size=batch_size, use_threads=False,
    )
    totals: Counter[str] = Counter()
    groups: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    hash_groups: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    word_deltas: Counter[int] = Counter()
    hash_rows: list[dict] = []
    hash_examples: list[tuple[str, dict]] = []
    academic_cells: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    academic_positive_cells: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    academic_generators: dict[str, Counter[str]] = defaultdict(Counter)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            totals["rows"] += 1
            rid, content = row["record_id"], row["content"]
            label, area = row["label_name"], row["broad_area"]
            source_name = str(row.get("source_name") or "<missing>")
            source_file = str(row.get("source_file") or "<missing>")
            group = groups[(label, area, source_name)]
            group["rows"] += 1
            measured_words = len(content.split())
            lexical_words = len(WORD_RE.findall(content))
            declared_words = row["word_count"]
            if declared_words != measured_words:
                totals["word_count_mismatch"] += 1
                group["word_count_mismatch"] += 1
                if isinstance(declared_words, int):
                    word_deltas[declared_words - measured_words] += 1
                if declared_words == lexical_words:
                    group["split_mismatch_matches_lexical"] += 1
                    totals["split_mismatch_matches_lexical"] += 1
            if declared_words == lexical_words:
                group["declared_matches_lexical"] += 1
                totals["declared_matches_lexical"] += 1
            if row["char_count"] != len(content):
                group["char_count_mismatch"] += 1
                totals["char_count_mismatch"] += 1
            declared_hash = row["content_hash"]
            exact_hash = _sha(content)
            if declared_hash != exact_hash:
                cause = _hash_explanation(content, declared_hash)
                if cause == "unexplained_by_tested_variants" and (
                    declared_hash == row.get("normalized_hash")
                ):
                    cause = "equals_declared_normalized_hash"
                totals["content_hash_mismatch"] += 1
                totals[f"hash_cause:{cause}"] += 1
                group["content_hash_mismatch"] += 1
                hash_groups[(label, area, source_name, source_file, cause)]["rows"] += 1
                metadata = {
                    "record_id": rid, "label_name": label, "broad_area": area,
                    "source_name": source_name, "source_file": source_file,
                    "declared_hash": declared_hash,
                    "declared_normalized_hash": row.get("normalized_hash"),
                    "exact_content_sha256": exact_hash, "tested_variant_result": cause,
                    "declared_word_count": declared_words,
                    "measured_whitespace_words": measured_words,
                    "measured_lexical_tokens": lexical_words,
                    "char_count": len(content),
                    "has_crlf": "\r\n" in content,
                    "has_non_nfc": content != unicodedata.normalize("NFC", content),
                    "has_edge_whitespace": content != content.strip(),
                }
                hash_rows.append(metadata)
                hash_examples.append((_priority(seed, cause, rid), {
                    **metadata, "content_original": content,
                }))
            if area == "academic":
                broad_heading = bool(REFERENCES.search(content))
                source_heading = bool(SOURCE_HEADING.search(content))
                terminal = _terminal_references_start(content) is not None
                dated_lines = sum(bool(YEAR_RE.search(line)) for line in content.splitlines())
                group["academic_reference_heading"] += int(broad_heading)
                group["academic_source_heading"] += int(source_heading)
                group["academic_terminal_reference_candidate"] += int(terminal)
                totals[f"{label}_academic_rows"] += 1
                totals[f"{label}_academic_reference_heading"] += int(broad_heading)
                totals[f"{label}_academic_source_heading"] += int(source_heading)
                totals[f"{label}_academic_terminal_candidate"] += int(terminal)
            if label == "synthetic" and area == "academic":
                generator = str(row.get("generator") or source_name)
                academic_generators[generator]["rows"] += 1
                academic_generators[generator]["source_heading"] += int(source_heading)
                academic_generators[generator]["terminal_candidate"] += int(terminal)
                sample_record = {
                    "record_id": rid, "label_name": label, "broad_area": area,
                    "source_name": source_name, "generator": generator,
                    "model_display_name": row.get("model_display_name"),
                    "genre": row.get("genre"), "reference_heading": broad_heading,
                    "source_heading": source_heading,
                    "terminal_reference_candidate": terminal,
                    "dated_lines": dated_lines, "content_original": content,
                }
                academic_cells[generator].append((_priority(seed, generator, rid), sample_record))
                if source_heading:
                    academic_positive_cells[generator].append(
                        (_priority(seed, "positive:" + generator, rid), sample_record)
                    )
    if catalog.resource("corpus").expected_rows != totals["rows"]:
        raise ValueError("Observed corpus row count differs from dataset catalog")
    _write_csv(output_dir / "metadata_by_source.csv", [
        {"label_name": label, "broad_area": area, "source_name": source_name, **counts}
        for (label, area, source_name), counts in sorted(groups.items())
    ], ["label_name", "broad_area", "source_name", "rows", "word_count_mismatch",
        "declared_matches_lexical", "split_mismatch_matches_lexical",
        "char_count_mismatch", "content_hash_mismatch",
        "academic_reference_heading", "academic_source_heading",
        "academic_terminal_reference_candidate"])
    _write_csv(output_dir / "hash_mismatch_by_source.csv", [
        {"label_name": label, "broad_area": area, "source_name": source_name,
         "source_file": source_file,
         "tested_variant_result": cause, "rows": counts["rows"]}
        for (label, area, source_name, source_file, cause), counts in sorted(hash_groups.items())
    ], ["label_name", "broad_area", "source_name", "source_file",
        "tested_variant_result", "rows"])
    _write_csv(output_dir / "hash_mismatch_records.csv", hash_rows,
               list(hash_rows[0]) if hash_rows else ["record_id"])
    _write_csv(output_dir / "word_count_delta_histogram.csv", [
        {"declared_minus_measured": delta, "rows": count}
        for delta, count in sorted(word_deltas.items())
    ], ["declared_minus_measured", "rows"])
    _write_csv(output_dir / "synthetic_academic_references_by_generator.csv", [
        {"generator": generator, **counts} for generator, counts in sorted(academic_generators.items())
    ], ["generator", "rows", "source_heading", "terminal_candidate"])
    with (output_dir / "hash_examples.jsonl").open("w", encoding="utf-8") as stream:
        for _, record in sorted(hash_examples)[:hash_sample]:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    positive = _choose_academic_sample(academic_positive_cells, academic_sample // 2)
    positive_ids = {row["record_id"] for row in positive}
    chosen = positive + [
        row for row in _choose_academic_sample(academic_cells, academic_sample + len(positive))
        if row["record_id"] not in positive_ids
    ][:academic_sample - len(positive)]
    with (output_dir / "synthetic_academic_examples_blind.jsonl").open("w", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(json.dumps({
                "record_id": row["record_id"], "reference_heading": row["reference_heading"],
                "source_heading": row["source_heading"],
                "terminal_reference_candidate": row["terminal_reference_candidate"],
                "dated_lines": row["dated_lines"], "content_original": row["content_original"],
            }, ensure_ascii=False) + "\n")
    _write_csv(output_dir / "synthetic_academic_examples_manifest.csv", [
        {key: value for key, value in row.items() if key != "content_original"}
        for row in chosen
    ], ["record_id", "label_name", "broad_area", "source_name", "generator",
        "model_display_name", "genre", "reference_heading",
        "source_heading", "terminal_reference_candidate", "dated_lines"])
    (output_dir / "summary.json").write_text(json.dumps({
        "source": str(source), "source_sha256": sha256_file(source),
        "config_sha256": sha256_file(config_path.resolve()),
        "counts": dict(sorted(totals.items())), "hash_examples": min(hash_sample, len(hash_examples)),
        "synthetic_academic_examples": len(chosen), "sample_seed": seed,
        "note": "Read-only diagnostics. Tested hash variants are hypotheses, not repairs."
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--academic-sample", type=int, default=120)
    parser.add_argument("--hash-sample", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args(argv)
    print(run(args.config, args.output_dir, academic_sample=args.academic_sample,
              hash_sample=args.hash_sample, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
