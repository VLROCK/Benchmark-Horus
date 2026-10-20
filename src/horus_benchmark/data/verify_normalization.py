"""Independent streaming verification of a completed derived corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import pyarrow.dataset as ds

from horus_benchmark.audit import text_features
from horus_benchmark.data import normalization
from horus_benchmark.data.normalize_corpus import _config, comparison_hash, sha256_file
from horus_benchmark.data import normalize_corpus
from horus_benchmark.data.normalization import normalize_text


REWRITTEN = {"content", "char_count", "word_count", "content_hash", "normalized_hash"}


def _rows(path: Path):
    scanner = ds.dataset(str(path), format="parquet").scanner(batch_size=512, use_threads=False)
    for batch in scanner.to_batches():
        yield from batch.to_pylist()


def verify(config_path: Path, run_dir: Path) -> Path:
    _, catalog, rules, _ = _config(config_path)
    run_dir = run_dir.expanduser().resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if sha256_file(config_path.expanduser().resolve()) != summary["config_sha256"]:
        raise ValueError("Normalization config changed since the run")
    current_code = {
        "normalize_corpus.py": sha256_file(Path(normalize_corpus.__file__)),
        "normalization.py": sha256_file(Path(normalization.__file__)),
        "text_features.py": sha256_file(Path(text_features.__file__)),
    }
    if current_code != summary["code_sha256"]:
        raise ValueError("Normalization code changed since the run")
    source = catalog.resource("corpus").path
    if sha256_file(source) != summary["source_sha256"]:
        raise ValueError("Immutable source hash differs from normalization run")
    if summary["dataset_version"] != catalog.dataset_version:
        raise ValueError("Dataset version differs from normalization run")
    normalized = run_dir / "normalized_corpus.parquet"
    trace = run_dir / "normalization_trace.parquet"
    if not normalized.is_file() or not trace.is_file():
        raise FileNotFoundError("Run lacks normalized corpus or trace")
    count = 0
    for triple in zip_longest(_rows(source), _rows(normalized), _rows(trace)):
        original, derived, decision = triple
        if original is None or derived is None or decision is None:
            raise ValueError("Source, derived corpus, and trace have different row counts")
        count += 1
        rid = original["record_id"]
        if rid != derived["record_id"] or rid != decision["record_id"]:
            raise ValueError(f"Row order or record ID mismatch at row {count}")
        for field, value in original.items():
            if field not in REWRITTEN and derived[field] != value:
                raise ValueError(f"Protected field {field} changed for {rid}")
        before = original["content"]
        after = derived["content"]
        if normalize_text(before, rules).text != after:
            raise ValueError(f"Derived text does not match configured rules for {rid}")
        before_hash = hashlib.sha256(before.encode("utf-8")).hexdigest()
        after_hash = hashlib.sha256(after.encode("utf-8")).hexdigest()
        if before_hash != decision["source_content_sha256"]:
            raise ValueError(f"Source SHA-256 mismatch for {rid}")
        if after_hash != decision["normalized_content_sha256"] or after_hash != derived["content_hash"]:
            raise ValueError(f"Derived SHA-256 mismatch for {rid}")
        if comparison_hash(after) != derived["normalized_hash"]:
            raise ValueError(f"Comparison SHA-256 mismatch for {rid}")
        if len(before) != decision["original_chars_measured"] or len(after) != derived["char_count"]:
            raise ValueError(f"Character count mismatch for {rid}")
        if len(before.split()) != decision["original_words_measured"] or len(after.split()) != derived["word_count"]:
            raise ValueError(f"Word count mismatch for {rid}")
        if original["content_hash"] != decision["source_content_hash"]:
            raise ValueError(f"Source metadata hash not preserved in trace for {rid}")
    if count != summary["source_rows"] or count != summary["output_rows"]:
        raise ValueError("Verified row count differs from summary")
    result = {"verified": True, "rows": count, "source_sha256": summary["source_sha256"],
              "stage": summary["stage"], "note": "Metadata and row identities preserved; source unchanged."}
    target = run_dir / "verification.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(verify(args.config, args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
