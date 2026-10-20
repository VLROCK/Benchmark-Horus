"""Stream the complete immutable corpus into a traceable normalized view.

No split-specific input, fitting, balancing, or benchmark selection occurs.
Run this program only on the researcher's machine/cluster (AGENTS.md rule 19).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from horus_benchmark.audit.text_features import extract_text_features
from horus_benchmark.audit import text_features
from horus_benchmark.config import load_yaml, resolve_from
from horus_benchmark.data import DatasetCatalog, DatasetLoader
from horus_benchmark.data import normalization
from horus_benchmark.data.normalization import RuleConfig, normalize_text


TRACE_SCHEMA = pa.schema([
    pa.field("record_id", pa.large_string()),
    pa.field("label_name", pa.large_string()),
    pa.field("broad_area", pa.large_string()),
    pa.field("source_content_sha256", pa.large_string()),
    pa.field("normalized_content_sha256", pa.large_string()),
    pa.field("source_content_hash", pa.large_string()),
    pa.field("source_normalized_hash", pa.large_string()),
    pa.field("source_char_count_declared", pa.int64()),
    pa.field("source_word_count_declared", pa.int64()),
    pa.field("original_chars_measured", pa.int64()),
    pa.field("original_words_measured", pa.int64()),
    pa.field("normalized_chars_measured", pa.int64()),
    pa.field("normalized_words_measured", pa.int64()),
    pa.field("normalization_status", pa.large_string()),
    pa.field("normalization_flags", pa.large_string()),
    pa.field("normalization_operations", pa.large_string()),
])

FEATURES = (
    "char_count_audit", "word_count_audit", "punctuation_per_100_words",
    "newline_per_100_words", "markdown_bullet", "markdown_heading",
    "markdown_code_fence", "html_tag", "url_in_content",
    "replacement_character", "tab_character",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_hash(value: str) -> str:
    """Explicit NFKC/casefold/space comparison signature for the derived view."""

    comparison = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()
    return hashlib.sha256(comparison.encode("utf-8")).hexdigest()


def _safe_output(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source.parent):
        raise ValueError("Output must not contain, overwrite, or sit beside the immutable source Parquet")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")


def _config(path: Path) -> tuple[dict[str, Any], DatasetCatalog, RuleConfig, int]:
    raw = load_yaml(path.resolve())
    if raw.get("schema_version") != 1:
        raise ValueError("Expected normalization config schema_version: 1")
    root = resolve_from(path.resolve().parent, raw.get("workspace_root", "."))
    catalog = DatasetCatalog.from_yaml(resolve_from(root, raw["dataset_config"]))
    if raw.get("corpus_resource") != "corpus":
        raise ValueError("This pipeline only accepts the complete 'corpus' resource")
    source = catalog.resource("corpus")
    if source.role != "processed_canonical_corpus" or source.format != "parquet":
        raise ValueError("Corpus resource must be the immutable canonical Parquet")
    if not source.immutable:
        raise ValueError("Source corpus must be declared immutable")
    rules = RuleConfig(**raw.get("rules", {}))
    batch_size = int(raw.get("batch_size", 512))
    if not 1 <= batch_size <= 8192:
        raise ValueError("batch_size must be 1..8192")
    return raw, catalog, rules, batch_size


def _metric_update(target: dict[str, float], prefix: str, values: dict[str, Any]) -> None:
    for name in FEATURES:
        target[f"{prefix}_{name}"] += float(values[name])


def run(config_path: Path, output_dir: Path) -> Path:
    """Create a new run. Original source is read-only; output paths never reuse it."""

    raw, catalog, rules, batch_size = _config(config_path)
    source = catalog.resource("corpus").path
    output_dir = output_dir.expanduser().resolve()
    _safe_output(source, output_dir)
    loader = DatasetLoader(catalog.resource("corpus"))
    if source != loader.resource.path:
        raise ValueError("Resolved source changed unexpectedly")
    required = {"record_id", "content", "char_count", "word_count", "content_hash",
                "normalized_hash", "label_name", "broad_area"}
    missing = required - set(loader.schema.names)
    if missing:
        raise ValueError(f"Corpus missing required columns: {sorted(missing)}")
    expected = catalog.resource("corpus").expected_rows
    physical = pq.ParquetFile(source).metadata.num_rows
    if expected is not None and physical != expected:
        raise ValueError(f"Physical rows {physical} differ from catalog {expected}")
    source_hash = sha256_file(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    normalized_tmp = output_dir / "normalized_corpus.parquet.part"
    trace_tmp = output_dir / "normalization_trace.parquet.part"
    review_tmp = output_dir / "review_queue.csv.part"
    seen: set[str] = set()
    overall: Counter[str] = Counter()
    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    review_columns = ["record_id", "broad_area", "flags", "status", "original_words",
                      "normalized_words", "source_sha256", "normalized_sha256"]
    writer = pq.ParquetWriter(normalized_tmp, loader.schema, compression="zstd")
    trace_writer = pq.ParquetWriter(trace_tmp, TRACE_SCHEMA, compression="zstd")
    completed = False
    try:
        with review_tmp.open("w", encoding="utf-8", newline="") as review_stream:
            review_writer = csv.DictWriter(review_stream, fieldnames=review_columns)
            review_writer.writeheader()
            for batch in loader.iter_batches(batch_size=batch_size, use_threads=False):
                records = batch.to_pylist()
                derived: list[dict[str, Any]] = []
                traces: list[dict[str, Any]] = []
                for record in records:
                    rid = record["record_id"]
                    if not isinstance(rid, str) or not rid or rid in seen:
                        raise ValueError(f"Empty or duplicated record_id: {rid!r}")
                    seen.add(rid)
                    content = record["content"]
                    result = normalize_text(content, rules)
                    flags = set(result.flags)
                    if record.get("language_valid") is False:
                        flags.add("LANGUAGE_METADATA_INVALID")
                    if record.get("artifact_probable_truncation") is True:
                        flags.add("SOURCE_TRUNCATION_FLAG")
                    if record.get("artifact_reasoning_leak") is True:
                        flags.add("SOURCE_REASONING_FLAG")
                    if record["content_hash"] != result.original_sha256:
                        flags.add("SOURCE_HASH_MISMATCH")
                    status = "review" if flags - {"MARKUP", "CODE_EXAMPLE", "ENCODING_CANDIDATE"} else result.status
                    label = str(record["label_name"])
                    area = str(record["broad_area"])
                    if label not in {"human", "synthetic"}:
                        raise ValueError(f"Unexpected label_name for {rid}: {label}")
                    # Protected metadata are used only for *post hoc* reporting.
                    group = groups[(label, area)]
                    group["rows"] += 1
                    if record["char_count"] != result.original_chars:
                        overall["source_declared_char_count_mismatch"] += 1
                        group["source_declared_char_count_mismatch"] += 1
                    if record["word_count"] != result.original_words:
                        overall["source_declared_word_count_mismatch"] += 1
                        group["source_declared_word_count_mismatch"] += 1
                    original_features = extract_text_features(content)
                    normalized_features = extract_text_features(result.text)
                    _metric_update(group, "before", original_features)
                    _metric_update(group, "after", normalized_features)
                    for flag in flags:
                        overall[f"flag:{flag}"] += 1
                        group[f"flag:{flag}"] += 1
                    for operation, count in result.operations.items():
                        overall[f"operation:{operation}"] += count
                        group[f"operation:{operation}"] += count
                    overall[f"status:{status}"] += 1
                    group[f"status:{status}"] += 1
                    if status == "review":
                        review_writer.writerow({
                            "record_id": rid, "broad_area": area,
                            "flags": ";".join(sorted(flags)), "status": status,
                            "original_words": result.original_words,
                            "normalized_words": result.normalized_words,
                            "source_sha256": result.original_sha256,
                            "normalized_sha256": result.normalized_sha256,
                        })
                    traces.append({
                        "record_id": rid, "label_name": label, "broad_area": area,
                        "source_content_sha256": result.original_sha256,
                        "normalized_content_sha256": result.normalized_sha256,
                        "source_content_hash": record["content_hash"],
                        "source_normalized_hash": record["normalized_hash"],
                        "source_char_count_declared": record["char_count"],
                        "source_word_count_declared": record["word_count"],
                        "original_chars_measured": result.original_chars,
                        "original_words_measured": result.original_words,
                        "normalized_chars_measured": result.normalized_chars,
                        "normalized_words_measured": result.normalized_words,
                        "normalization_status": status,
                        "normalization_flags": ";".join(sorted(flags)),
                        "normalization_operations": json.dumps(result.operations, ensure_ascii=False, sort_keys=True),
                    })
                    # Preserve all other source fields, including split and
                    # original group IDs. They do not influence normalization.
                    record["content"] = result.text
                    record["char_count"] = result.normalized_chars
                    record["word_count"] = result.normalized_words
                    record["content_hash"] = result.normalized_sha256
                    record["normalized_hash"] = comparison_hash(result.text)
                    derived.append(record)
                writer.write_table(pa.Table.from_pylist(derived, schema=loader.schema))
                trace_writer.write_table(pa.Table.from_pylist(traces, schema=TRACE_SCHEMA))
                overall["rows"] += len(records)
                if overall["rows"] % 20_000 < batch_size:
                    print(f"Normalized {overall['rows']} / {physical} rows", flush=True)
        if overall["rows"] != physical or len(seen) != physical:
            raise ValueError("Output row count differs from immutable corpus")
        completed = True
    finally:
        writer.close()
        trace_writer.close()
    if not completed:
        raise RuntimeError("Normalization stopped before completion; partial .part files remain")
    normalized_tmp.rename(output_dir / "normalized_corpus.parquet")
    trace_tmp.rename(output_dir / "normalization_trace.parquet")
    review_tmp.rename(output_dir / "review_queue.csv")
    with (output_dir / "paired_surface_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        metric_columns = [f"{phase}_{feature}" for phase in ("before", "after") for feature in FEATURES]
        fieldnames = ["label_name", "broad_area", "rows", "review_rows", *metric_columns]
        csv_writer = csv.DictWriter(stream, fieldnames=fieldnames)
        csv_writer.writeheader()
        for (label, area), values in sorted(groups.items()):
            count = int(values["rows"])
            csv_writer.writerow({
                "label_name": label, "broad_area": area, "rows": count,
                "review_rows": int(values["status:review"]),
                **{name: values[name] / count for name in metric_columns},
            })
    summary = {
        "dataset_id": catalog.dataset_id, "dataset_version": catalog.dataset_version,
        "dataset_status": catalog.status, "source": str(source),
        "source_sha256": source_hash, "source_rows": physical,
        "output_rows": overall["rows"], "stage": rules.stage,
        "config_sha256": sha256_file(config_path.expanduser().resolve()),
        "code_sha256": {
            "normalize_corpus.py": sha256_file(Path(__file__)),
            "normalization.py": sha256_file(Path(normalization.__file__)),
            "text_features.py": sha256_file(Path(text_features.__file__)),
        },
        "python": sys.version, "pyarrow": pa.__version__,
        "rules": vars(rules) if hasattr(rules, "__dict__") else {
            name: getattr(rules, name) for name in rules.__dataclass_fields__
        },
        "counts": dict(sorted(overall.items())),
        "note": "No selection or split-specific processing. Review flags are not exclusions.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
