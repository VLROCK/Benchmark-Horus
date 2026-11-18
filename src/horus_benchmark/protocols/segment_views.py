"""Prepare document-linked middle paragraph and sentence views of frozen N2.

The researcher runs this module on the cluster. It reads the already frozen
release and selected N2 cache, writes only derived files, and never constructs
new train/test splits. A segment has its parent's record_id so clean document
and segment predictions can be compared on exactly the same IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from horus_benchmark.audit.middle_segments import WORD, segments, select_middle
from horus_benchmark.data.normalize_corpus import sha256_file


UNITS = ("paragraph", "sentence")
REQUIRED_PARENT = {
    "record_id", "content", "content_hash", "probe_group_id", "label_name",
    "broad_area", "source_name", "generator", "word_count_audit",
}
REQUIRED_SPLIT = {"record_id", "probe_group_id", "split", "fold"}
IDENTITY = ("record_id", "content_hash", "probe_group_id", "label_name",
            "broad_area", "source_name", "generator", "word_count_audit")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1 or \
            cfg.get("experiment_id") != "horus-n2-middle-units-v1":
        raise ValueError("Unexpected middle-unit configuration")
    if cfg.get("seeds") != [20261105, 20261106, 20261107]:
        raise ValueError("Middle units require the three frozen IID seeds")
    if not 0 <= float(cfg["middle_start"]) < float(cfg["middle_end"]) <= 1:
        raise ValueError("Invalid middle window")
    if not 0 < float(cfg["length_bin_width"]) <= 1:
        raise ValueError("Invalid length-bin width")
    if set(cfg["units"]) != set(UNITS):
        raise ValueError("Paragraph and sentence configurations are required")
    for unit in UNITS:
        rules = cfg["units"][unit]
        if not 0 < int(rules["min_words"]) <= int(rules["max_words"]):
            raise ValueError(f"Invalid {unit} word limits")
        if not 0 < float(rules["max_fraction_of_document"]) <= 1:
            raise ValueError(f"Invalid {unit} document fraction")
    return cfg


def _pinned(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Missing or changed pinned input: {path}")


def _same(left: Any, right: Any) -> bool:
    missing_left, missing_right = bool(pd.isna(left)), bool(pd.isna(right))
    if missing_left or missing_right:
        return missing_left and missing_right
    return bool(left == right)


def _load_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any],
                                                   dict[str, Any], dict[int, pd.DataFrame]]:
    release_dir = Path(cfg["release_dir"])
    prepared_dir = Path(cfg["prepared_dir"])
    release = _json(release_dir / "release_manifest.json")
    prepared = _json(prepared_dir / "prepared_manifest.json")
    if (release.get("verified") is not True or
            release.get("release_id") != "horus-n2-iid-v1" or
            release.get("text_view") != "n2" or
            release.get("split_seeds") != cfg["seeds"] or
            release.get("development_folds") != 5 or
            release.get("rows") != 107566):
        raise ValueError("Expected the verified frozen N2 IID release")
    if (prepared.get("kind") != "selected_n2_text_cache" or
            prepared.get("release_id") != release["release_id"] or
            prepared.get("release_manifest_sha256") !=
            sha256_file(release_dir / "release_manifest.json") or
            prepared.get("n2_corpus_sha256") !=
            release["inputs"]["n2_corpus"]["sha256"] or
            prepared.get("rows") != release["rows"]):
        raise ValueError("Prepared text cache disagrees with the frozen release")

    index_path = release_dir / "selected_index.parquet"
    text_path = Path(prepared["selected_n2_texts"]["path"])
    _pinned(index_path, release["artifacts"]["selected_index"]["sha256"])
    _pinned(text_path, prepared["selected_n2_texts"]["sha256"])
    index = pd.read_parquet(index_path)
    texts = pd.read_parquet(text_path)
    if (REQUIRED_PARENT - set(texts) or
            set(IDENTITY) - set(index) or
            texts.record_id.duplicated().any() or
            index.record_id.duplicated().any() or
            len(texts) != release["rows"] or len(index) != release["rows"] or
            set(texts.record_id) != set(index.record_id)):
        raise ValueError("Selected cache/index lacks the frozen document identities")
    parents = texts.set_index("record_id", verify_integrity=True)
    frozen = index.set_index("record_id", verify_integrity=True).loc[parents.index]
    for field in IDENTITY[1:]:
        if not all(_same(a, b) for a, b in zip(parents[field], frozen[field])):
            raise ValueError(f"Selected cache disagrees with frozen index: {field}")
    if (parents.content.isna().any() or
            not parents.content.map(lambda value: isinstance(value, str)).all() or
            not parents.content.map(lambda value: hashlib.sha256(
                value.encode("utf-8")).hexdigest()).eq(parents.content_hash).all()):
        raise ValueError("Selected N2 text has invalid content or content hash")
    if (parents.probe_group_id.isna().any() or
            parents.word_count_audit.isna().any() or
            parents.word_count_audit.lt(0).any()):
        raise ValueError("Invalid parent group or word count")

    split_frames: dict[int, pd.DataFrame] = {}
    for seed in cfg["seeds"]:
        name = f"split_seed{seed}"
        split_path = release_dir / f"{name}.parquet"
        _pinned(split_path, release["artifacts"][name]["sha256"])
        split = pd.read_parquet(split_path)
        if (REQUIRED_SPLIT - set(split) or split.record_id.duplicated().any() or
                set(split.record_id) != set(texts.record_id) or
                not split.set_index("record_id").loc[parents.index, "probe_group_id"].eq(
                    parents.probe_group_id).all()):
            raise ValueError(f"Invalid frozen split for seed {seed}")
        if (not set(split.split) <= {"development", "test"} or
                split.loc[split.split.eq("test"), "fold"].ne(-1).any() or
                not set(split.loc[split.split.eq("development"), "fold"]) <= set(range(5)) or
                split.groupby("probe_group_id").split.nunique().gt(1).any() or
                split.groupby("probe_group_id").fold.nunique().gt(1).any()):
            raise ValueError(f"Frozen split roles/groups are invalid for seed {seed}")
        split_frames[seed] = split
    return texts, release, prepared, split_frames


def _reason(text: str, unit: str, cfg: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Explain attrition without substituting a boundary or relaxing criteria."""
    rules = cfg["units"][unit]
    observed = segments(text, unit)
    middle = [part for part in observed if
              cfg["middle_start"] <= part.midpoint <= cfg["middle_end"]]
    fraction = [part for part in middle if
                (part.end - part.start) / max(len(text), 1) <=
                rules["max_fraction_of_document"]]
    word_counts = [len(WORD.findall(part.text)) for part in fraction]
    eligible = [part for part, words in zip(fraction, word_counts) if
                rules["min_words"] <= words <= rules["max_words"]]
    counts = {"units_total": len(observed), "units_middle": len(middle),
              "units_fraction_ok": len(fraction), "units_eligible": len(eligible),
              "units_middle_over_fraction": len(middle) - len(fraction),
              "units_too_short": sum(words < rules["min_words"] for words in word_counts),
              "units_too_long": sum(words > rules["max_words"] for words in word_counts)}
    if not observed:
        return "no_unit", counts
    if not middle:
        return "no_middle_unit", counts
    if not fraction:
        return "middle_unit_too_large_fraction", counts
    if not eligible:
        return "middle_unit_outside_word_bounds", counts
    return "eligible", counts


def build_units(parents: pd.DataFrame, cfg: dict[str, Any], unit: str
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure extraction used by prepare and tests; one unit per eligible parent."""
    if unit not in UNITS or REQUIRED_PARENT - set(parents):
        raise ValueError("Unknown unit or missing parent columns")
    rules = cfg["units"][unit]
    texts: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    for row in parents.sort_values("record_id").itertuples(index=False):
        content = str(row.content)
        reason, counts = _reason(content, unit, cfg)
        chosen = select_middle(
            content, unit, str(row.record_id),
            min_words=int(rules["min_words"]), max_words=int(rules["max_words"]),
            middle_start=float(cfg["middle_start"]),
            middle_end=float(cfg["middle_end"]),
            max_fraction_of_document=float(rules["max_fraction_of_document"]),
            seed=int(cfg["selection_seed"]),
        )
        if (reason == "eligible") != (chosen is not None):
            raise ValueError("Eligibility diagnosis and deterministic selection differ")
        parent_words = int(row.word_count_audit)
        common = {
            "record_id": str(row.record_id), "parent_record_id": str(row.record_id),
            "unit": unit, "probe_group_id": str(row.probe_group_id),
            "label_name": str(row.label_name), "broad_area": str(row.broad_area),
            "source_name": row.source_name, "generator": row.generator,
            "parent_content_hash": str(row.content_hash),
            "parent_word_count_audit": parent_words,
            "parent_length_bin": int(np.floor(np.log2(1 + parent_words) /
                                               float(cfg["length_bin_width"]))),
        }
        item = {**common, "eligible": chosen is not None, "reason": reason,
                **counts,
                "segment_start": chosen.start if chosen is not None else -1,
                "segment_end": chosen.end if chosen is not None else -1,
                "segment_index": chosen.index if chosen is not None else -1,
                "segment_words": len(WORD.findall(chosen.text)) if chosen else 0,
                "segment_content_hash": hashlib.sha256(
                    chosen.text.encode("utf-8")).hexdigest() if chosen else ""}
        eligibility.append(item)
        if chosen is None:
            continue
        segment_id = hashlib.sha256(
            f"{unit}|{row.record_id}|{chosen.start}|{chosen.end}".encode("utf-8")
        ).hexdigest()
        texts.append({**common, "segment_id": segment_id,
                      "segment_start": chosen.start, "segment_end": chosen.end,
                      "segment_index": chosen.index, "segment_midpoint": chosen.midpoint,
                      "segment_words": item["segment_words"],
                      "word_count_audit": len(chosen.text.split()),
                      "content": chosen.text,
                      "content_hash": item["segment_content_hash"]})
    text_columns = [
        "record_id", "parent_record_id", "unit", "probe_group_id", "label_name",
        "broad_area", "source_name", "generator", "parent_content_hash",
        "parent_word_count_audit", "parent_length_bin", "segment_id",
        "segment_start", "segment_end", "segment_index", "segment_midpoint",
        "segment_words", "word_count_audit", "content", "content_hash",
    ]
    return pd.DataFrame(texts, columns=text_columns), pd.DataFrame(eligibility)


def _derive_split(split: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """Subset frozen roles by parent ID, retaining every original column."""
    if (REQUIRED_SPLIT - set(split) or texts.record_id.duplicated().any() or
            not set(texts.record_id) <= set(split.record_id)):
        raise ValueError("Cannot derive segment split from unrelated parents")
    subset = split.loc[split.record_id.isin(set(texts.record_id))].sort_values("record_id")
    if (len(subset) != len(texts) or subset.record_id.duplicated().any() or
            subset.groupby("probe_group_id").split.nunique().gt(1).any() or
            subset.groupby("probe_group_id").fold.nunique().gt(1).any()):
        raise ValueError("Derived segment split is not group-safe and one-to-one")
    return subset


def _coverage(frame: pd.DataFrame, unit: str, splits: dict[int, pd.DataFrame]
              ) -> pd.DataFrame:
    """Give explicit denominators, including missing source/generator values."""
    result: list[dict[str, Any]] = []
    dimensions = {
        "class": ["label_name"],
        "class_area": ["label_name", "broad_area"],
        "class_area_source": ["label_name", "broad_area", "source_name"],
        "class_area_generator": ["label_name", "broad_area", "generator"],
        "class_area_length": ["label_name", "broad_area", "parent_length_bin"],
        "class_reason": ["label_name", "reason"],
    }
    for dimension, columns in dimensions.items():
        counts = frame.groupby(columns, dropna=False).eligible.agg(
            parent_rows="size", eligible_rows="sum").reset_index()
        for row in counts.to_dict("records"):
            result.append({"unit": unit, "dimension": dimension, "seed": -1,
                           "split": "all", **{col: row[col] for col in columns},
                           "parent_rows": int(row["parent_rows"]),
                           "eligible_rows": int(row["eligible_rows"]),
                           "retained_fraction": float(row["eligible_rows"] /
                                                      row["parent_rows"])})
    for seed, split in splits.items():
        assigned = frame[["record_id", "label_name", "broad_area", "eligible"]].merge(
            split[["record_id", "split", "fold"]], on="record_id", validate="one_to_one")
        counts = assigned.groupby(["split", "label_name", "broad_area"],
                                  dropna=False).eligible.agg(
            parent_rows="size", eligible_rows="sum").reset_index()
        for row in counts.to_dict("records"):
            result.append({"unit": unit, "dimension": "seed_split_class_area",
                           "seed": int(seed), "split": str(row["split"]),
                           "label_name": row["label_name"],
                           "broad_area": row["broad_area"],
                           "parent_rows": int(row["parent_rows"]),
                           "eligible_rows": int(row["eligible_rows"]),
                           "retained_fraction": float(row["eligible_rows"] /
                                                      row["parent_rows"])})
    return pd.DataFrame(result)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path,
                   compression="zstd")


def prepare(config_path: Path, output: Path) -> Path:
    """Build derived views after verifying every selected text and split hash."""
    cfg = _config(config_path)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    parents, release, prepared, splits = _load_inputs(cfg)
    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    coverage: list[pd.DataFrame] = []
    for unit in UNITS:
        texts, eligibility = build_units(parents, cfg, unit)
        if (not texts.empty and (texts.record_id.duplicated().any() or
                                 not texts.parent_record_id.eq(texts.record_id).all())):
            raise ValueError(f"Invalid {unit} identities")
        if len(eligibility) != release["rows"] or eligibility.record_id.duplicated().any():
            raise ValueError(f"Invalid {unit} coverage")
        for name, frame in ((f"{unit}_texts", texts),
                            (f"{unit}_eligibility", eligibility)):
            path = output / f"{name}.parquet"
            _write_parquet(frame, path)
            artifacts[name] = {"path": str(path), "sha256": sha256_file(path),
                               "rows": len(frame)}
        for seed, split in splits.items():
            name = f"split_seed{seed}_{unit}"
            subset = _derive_split(split, texts)
            path = output / f"{name}.parquet"
            _write_parquet(subset, path)
            artifacts[name] = {"path": str(path), "sha256": sha256_file(path),
                               "rows": len(subset)}
        summary[unit] = {
            "parents": len(eligibility), "eligible": len(texts),
            "retained_fraction": len(texts) / len(eligibility),
            "by_reason": {str(key): int(value) for key, value in
                          eligibility.reason.value_counts().items()},
            "by_label": {str(key): int(value) for key, value in
                         texts.label_name.value_counts().items()},
        }
        coverage.append(_coverage(eligibility, unit, splits))
    coverage_frame = pd.concat(coverage, ignore_index=True)
    coverage_path = output / "coverage.csv"
    coverage_frame.to_csv(coverage_path, index=False)
    artifacts["coverage"] = {"path": str(coverage_path),
                             "sha256": sha256_file(coverage_path),
                             "rows": len(coverage_frame)}
    release_dir = Path(cfg["release_dir"])
    prepared_dir = Path(cfg["prepared_dir"])
    manifest = {
        "schema_version": 1, "kind": "frozen_n2_middle_unit_views",
        "release_id": release["release_id"],
        "source_rows": release["rows"],
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "release_manifest_path": str(release_dir / "release_manifest.json"),
        "release_manifest_sha256": sha256_file(release_dir / "release_manifest.json"),
        "selected_index_sha256": release["artifacts"]["selected_index"]["sha256"],
        "prepared_manifest_path": str(prepared_dir / "prepared_manifest.json"),
        "prepared_manifest_sha256": sha256_file(prepared_dir / "prepared_manifest.json"),
        "selected_texts_sha256": prepared["selected_n2_texts"]["sha256"],
        "selection_seed": int(cfg["selection_seed"]),
        "split_seeds": cfg["seeds"],
        "summary": summary,
        "artifacts": artifacts,
        "note": "Secondary unit views; parents retain their frozen IID roles and probe groups.",
    }
    path = output / "segment_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def check(config_path: Path, output: Path) -> dict[str, Any]:
    """Verify derived bytes, parent spans and exact inherited split roles."""
    cfg = _config(config_path)
    parents, release, prepared, frozen_splits = _load_inputs(cfg)
    manifest = _json(output / "segment_manifest.json")
    if (manifest.get("kind") != "frozen_n2_middle_unit_views" or
            manifest.get("release_id") != release["release_id"] or
            manifest.get("source_rows") != release["rows"] or
            manifest.get("config_sha256") != sha256_file(config_path) or
            manifest.get("selected_texts_sha256") !=
            prepared["selected_n2_texts"]["sha256"] or
            manifest.get("release_manifest_sha256") !=
            sha256_file(Path(cfg["release_dir"]) / "release_manifest.json") or
            manifest.get("prepared_manifest_sha256") !=
            sha256_file(Path(cfg["prepared_dir"]) / "prepared_manifest.json") or
            manifest.get("split_seeds") != cfg["seeds"]):
        raise ValueError("Segment manifest does not match the frozen inputs/config")
    artifacts = manifest["artifacts"]
    for name, item in artifacts.items():
        path = output / (f"{name}.csv" if name == "coverage" else f"{name}.parquet")
        _pinned(path, item["sha256"])
        if Path(item["path"]).name != path.name:
            raise ValueError(f"Artifact path mismatch: {name}")
    source = parents.set_index("record_id", verify_integrity=True)
    for unit in UNITS:
        texts = pd.read_parquet(output / f"{unit}_texts.parquet")
        eligibility = pd.read_parquet(output / f"{unit}_eligibility.parquet")
        if (len(texts) != artifacts[f"{unit}_texts"]["rows"] or
                len(eligibility) != release["rows"] or
                set(eligibility.record_id) != set(source.index) or
                texts.record_id.duplicated().any() or
                eligibility.record_id.duplicated().any() or
                not texts.record_id.eq(texts.parent_record_id).all() or
                set(texts.record_id) != set(eligibility.loc[eligibility.eligible,
                                                           "record_id"])):
            raise ValueError(f"Invalid {unit} row identities")
        if (manifest["summary"][unit]["parents"] != len(eligibility) or
                manifest["summary"][unit]["eligible"] != len(texts) or
                manifest["summary"][unit]["by_reason"] != {
                    str(key): int(value) for key, value in
                    eligibility.reason.value_counts().items()}):
            raise ValueError(f"Invalid {unit} coverage summary")
        for row in eligibility.itertuples(index=False):
            parent = source.loc[row.record_id]
            reason, counts = _reason(parent.content, unit, cfg)
            if (row.parent_record_id != row.record_id or
                    row.reason != reason or row.eligible != (reason == "eligible") or
                    row.parent_content_hash != parent.content_hash or
                    any(getattr(row, key) != value for key, value in counts.items())):
                raise ValueError(f"Invalid {unit} eligibility lineage: {row.record_id}")
        for row in texts.itertuples(index=False):
            parent = source.loc[row.record_id]
            rules = cfg["units"][unit]
            expected = select_middle(
                parent.content, unit, row.record_id,
                min_words=int(rules["min_words"]),
                max_words=int(rules["max_words"]),
                middle_start=float(cfg["middle_start"]),
                middle_end=float(cfg["middle_end"]),
                max_fraction_of_document=float(rules["max_fraction_of_document"]),
                seed=int(cfg["selection_seed"]),
            )
            if (not _same(row.probe_group_id, parent.probe_group_id) or
                    not _same(row.label_name, parent.label_name) or
                    not _same(row.broad_area, parent.broad_area) or
                    not _same(row.source_name, parent.source_name) or
                    not _same(row.generator, parent.generator) or
                    row.parent_content_hash != parent.content_hash or
                    parent.content[row.segment_start:row.segment_end] != row.content or
                    hashlib.sha256(row.content.encode("utf-8")).hexdigest() !=
                    row.content_hash or
                    len(row.content.split()) != int(row.word_count_audit) or
                    len(WORD.findall(row.content)) != int(row.segment_words) or
                    expected is None or expected.text != row.content or
                    expected.start != row.segment_start or
                    expected.end != row.segment_end or
                    expected.index != row.segment_index):
                raise ValueError(f"Invalid {unit} segment lineage: {row.record_id}")
        for seed, frozen in frozen_splits.items():
            derived = pd.read_parquet(output / f"split_seed{seed}_{unit}.parquet")
            expected = frozen.loc[frozen.record_id.isin(set(texts.record_id))]
            columns = list(frozen.columns)
            if (len(derived) != len(texts) or
                    not derived[columns].sort_values("record_id").reset_index(drop=True).equals(
                        expected[columns].sort_values("record_id").reset_index(drop=True))):
                raise ValueError(f"Derived split changed parent roles: {seed}/{unit}")
    return {"verified": True, "release_id": release["release_id"],
            "summary": manifest["summary"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.config, args.output))
    else:
        print(json.dumps(check(args.config, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
