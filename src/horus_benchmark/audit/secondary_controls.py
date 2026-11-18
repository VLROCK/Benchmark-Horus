"""Derived N2 layout ablations and test-only nuisance common-support sets.

The reviewed horus-n2-iid-v1 corpus and partitions are never rewritten. Layout
views retain the whitespace-delimited token sequence. Nuisance matching uses
only observed text features and frozen test membership; no detector score or
outcome influences membership. The researcher runs this CLI on the cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from horus_benchmark.audit.prepare_selection_protocol import layout_control_text
from horus_benchmark.audit.text_features import extract_text_features
from horus_benchmark.protocols.grouped_iid import stable_key


EXPECTED_SEEDS = (20261105, 20261106, 20261107)
RELEASE_ID = "horus-n2-iid-v1"
PARAGRAPH_GAP = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)(?:[ \t]*(?:\r?\n))*")
NUMERIC_BALANCE = (
    "log_unit_words", "log_parent_words", "paragraph_count",
    "newline_per_100_words", "punctuation_per_100_words", "markdown_bullet",
)
REQUIRED_TEXT_COLUMNS = {
    "record_id", "content", "content_hash", "label_name", "broad_area",
    "probe_group_id", "source_name", "generator", "word_count_audit",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Invalid secondary-controls configuration")
    if config.get("experiment_id") != "horus-n2-secondary-controls-v1":
        raise ValueError("Secondary controls require the reviewed experiment ID")
    if tuple(config.get("split_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Secondary controls require the three frozen IID seeds")
    if not 0 < float(config.get("length_bin_width", 0)) <= 1:
        raise ValueError("length_bin_width must be in (0, 1]")
    if config.get("paragraph_bins") != [1, 5]:
        raise ValueError("Unreviewed paragraph bins")
    if config.get("presentation_marker") != "markdown_bullet":
        raise ValueError("Unreviewed presentation marker")
    if int(config.get("minimum_subgroup_per_label", 0)) < 1:
        raise ValueError("minimum_subgroup_per_label must be positive")
    return config


def _pinned_release(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    release_dir = Path(config["release_dir"])
    manifest_path = release_dir / "release_manifest.json"
    manifest = _json(manifest_path)
    if (manifest.get("verified") is not True or
            manifest.get("release_id") != RELEASE_ID or
            manifest.get("text_view") != "n2" or
            tuple(manifest.get("split_seeds", ())) != EXPECTED_SEEDS):
        raise ValueError("Expected the verified N2 release and frozen seeds")
    return release_dir, manifest


def _selected_texts(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    release_dir, release = _pinned_release(config)
    prepared_dir = Path(config["prepared_dir"])
    prepared_path = prepared_dir / "prepared_manifest.json"
    prepared = _json(prepared_path)
    if (prepared.get("kind") != "selected_n2_text_cache" or
            prepared.get("release_id") != RELEASE_ID or
            prepared.get("release_manifest_sha256") !=
            sha256_file(release_dir / "release_manifest.json")):
        raise ValueError("Selected text cache belongs to another release")
    text_info = prepared["selected_n2_texts"]
    text_path = Path(text_info["path"])
    if not text_path.is_file() or sha256_file(text_path) != text_info["sha256"]:
        raise ValueError("Selected N2 text cache is missing or changed")
    selected = pd.read_parquet(text_path)
    if (REQUIRED_TEXT_COLUMNS - set(selected) or len(selected) != release["rows"] or
            selected.record_id.duplicated().any() or selected.content.isna().any()):
        raise ValueError("Selected N2 text cache has invalid rows or columns")
    if (selected.content.map(_sha256_text) != selected.content_hash).any():
        raise ValueError("Selected N2 text no longer matches its content hash")
    return selected, {
        "release_manifest": {
            "path": str(release_dir / "release_manifest.json"),
            "sha256": sha256_file(release_dir / "release_manifest.json"),
        },
        "prepared_manifest": {"path": str(prepared_path), "sha256": sha256_file(prepared_path)},
        "selected_n2_texts": {"path": str(text_path), "sha256": text_info["sha256"]},
    }


def layout_blank_text(content: str) -> str:
    """Remove only explicit blank-line paragraph gaps, preserving tokens."""
    transformed = PARAGRAPH_GAP.sub(" ", content)
    if transformed.split() != content.split():
        raise ValueError("Blank-line ablation changed the word sequence")
    return transformed


def layout_all_text(content: str) -> str:
    """Reuse the reviewed all-line control, preserving whitespace tokens."""
    return layout_control_text(content)


def _paragraph_count(content: str) -> int:
    value = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    return len(PARAGRAPH_GAP.split(value)) if value else 0


def prepare_layout(config_path: Path, output: Path) -> Path:
    """Write two symmetric diagnostic views on exactly the selected N2 IDs."""
    if output.exists():
        raise FileExistsError(f"Layout output already exists: {output}")
    config = load_config(config_path)
    selected, inputs = _selected_texts(config)
    selected = selected.sort_values("record_id", kind="stable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for name, transform in (("blank", layout_blank_text), ("all", layout_all_text)):
        view = selected[["record_id", "content"]].copy()
        view["content"] = view.content.map(transform)
        if not view.record_id.equals(selected.record_id):
            raise ValueError("Layout ablation changed record order")
        path = output / f"layout_{name}.parquet"
        pq.write_table(pa.Table.from_pandas(view, preserve_index=False), path,
                       compression="zstd")
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path),
                           "rows": len(view)}
        diagnostic = selected[["record_id", "label_name", "broad_area"]].copy()
        diagnostic["changed"] = selected.content.ne(view.content)
        diagnostic["before_paragraphs"] = selected.content.map(_paragraph_count)
        diagnostic["after_paragraphs"] = view.content.map(_paragraph_count)
        diagnostic["before_newlines"] = selected.content.str.count("\n")
        diagnostic["after_newlines"] = view.content.str.count("\n")
        for (area, label), cell in diagnostic.groupby(
                ["broad_area", "label_name"], sort=True):
            summaries.append({
                "variant": name, "broad_area": area, "label_name": label,
                "rows": len(cell), "changed_rows": int(cell.changed.sum()),
                "changed_fraction": float(cell.changed.mean()),
                "mean_paragraphs_before": float(cell.before_paragraphs.mean()),
                "mean_paragraphs_after": float(cell.after_paragraphs.mean()),
                "mean_newlines_before": float(cell.before_newlines.mean()),
                "mean_newlines_after": float(cell.after_newlines.mean()),
            })
    summary_path = output / "layout_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    artifacts["summary"] = {"path": str(summary_path), "sha256": sha256_file(summary_path)}
    manifest = {
        "schema_version": 1, "kind": "n2_selected_layout_ablations",
        "release_id": RELEASE_ID, "rows": len(selected),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "inputs": inputs, "artifacts": artifacts,
        "note": "Diagnostic only; N2 and frozen splits unchanged. Both classes use identical rules.",
    }
    manifest_path = output / "layout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest_path


def _length_bins(words: pd.Series, width: float) -> pd.Series:
    numeric = pd.to_numeric(words, errors="raise")
    if numeric.isna().any() or numeric.lt(0).any():
        raise ValueError("Nuisance length contains missing or negative word counts")
    return pd.Series(np.floor(np.log2(1 + numeric.to_numpy(dtype=float)) / width).astype(int),
                     index=words.index)


def _paragraph_bin(paragraphs: int) -> str:
    if paragraphs <= 0:
        return "0"
    if paragraphs == 1:
        return "1"
    if paragraphs < 5:
        return "2-4"
    return "5+"


def _features(frame: pd.DataFrame, unit: str, width: float) -> pd.DataFrame:
    """Compute nuisance variables from each unit's text, never from labels."""
    if unit not in {"document", "paragraph", "sentence"}:
        raise ValueError("Unknown nuisance unit")
    required = REQUIRED_TEXT_COLUMNS | {"parent_record_id", "parent_word_count_audit"}
    if required - set(frame) or frame.record_id.duplicated().any():
        raise ValueError(f"Invalid {unit} unit frame")
    if frame.content.isna().any() or frame.probe_group_id.isna().any():
        raise ValueError(f"Missing {unit} text or group")
    measured = pd.DataFrame(frame.content.map(extract_text_features).tolist(), index=frame.index)
    if not measured.word_count_audit.eq(frame.word_count_audit).all():
        raise ValueError(f"{unit} word counts disagree with unit text")
    if frame.content.map(_sha256_text).ne(frame.content_hash).any():
        raise ValueError(f"{unit} text hash mismatch")
    out = frame[["record_id", "parent_record_id", "probe_group_id", "label_name",
                 "broad_area", "source_name", "generator", "word_count_audit",
                 "parent_word_count_audit"]].copy()
    out["unit"] = unit
    out["unit_length_bin"] = _length_bins(out.word_count_audit, width)
    out["parent_length_bin"] = _length_bins(out.parent_word_count_audit, width)
    out["paragraph_count"] = measured.paragraph_count.astype(int)
    out["paragraph_bin"] = out.paragraph_count.map(_paragraph_bin)
    out["markdown_bullet"] = measured.markdown_bullet.astype(bool)
    out["newline_per_100_words"] = measured.newline_per_100_words.astype(float)
    out["punctuation_per_100_words"] = measured.punctuation_per_100_words.astype(float)
    out["log_unit_words"] = np.log2(1 + out.word_count_audit.astype(float))
    out["log_parent_words"] = np.log2(1 + out.parent_word_count_audit.astype(float))
    return out


def _cell_columns(unit: str, tier: str) -> list[str]:
    if tier not in {"length", "presentation"}:
        raise ValueError("Unknown nuisance tier")
    if unit == "document":
        base = ["broad_area", "unit_length_bin"]
        return base + (["paragraph_bin", "markdown_bullet"] if tier == "presentation" else [])
    if unit in {"paragraph", "sentence"} and tier == "length":
        return ["broad_area", "unit_length_bin", "parent_length_bin"]
    raise ValueError("Presentation tier is defined only for documents")


def match_nuisance(frame: pd.DataFrame, *, unit: str, tier: str,
                   seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select deterministic 1:1 test rows within predeclared common-support cells.

    The caller supplies only the frozen test rows for one seed and unit. The
    model fit and calibration populations remain untouched. No score column is
    required or inspected. Returned pairs are an audit device, not paired
    semantic equivalents of individual human and synthetic documents.
    """
    if frame.record_id.duplicated().any() or set(frame.label_name) - {"human", "synthetic"}:
        raise ValueError("Nuisance test frame has duplicate IDs or unknown labels")
    forbidden = {"score", "score_n2", "prediction", "correct"}
    if forbidden & set(frame):
        raise ValueError("Nuisance matching frame must not contain model outcomes")
    columns = _cell_columns(unit, tier)
    missing = (set(columns) | {"record_id", "parent_record_id", "probe_group_id",
                               "label_name"}) - set(frame)
    if missing:
        raise ValueError(f"Nuisance frame lacks {sorted(missing)}")
    selected: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for key, subset in frame.groupby(columns, sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        cell = dict(zip(columns, values))
        human = subset.loc[subset.label_name.eq("human")].copy()
        synthetic = subset.loc[subset.label_name.eq("synthetic")].copy()
        count = min(len(human), len(synthetic))
        for label_frame in (human, synthetic):
            label_frame["sample_key"] = label_frame.record_id.map(
                lambda value: stable_key(seed, f"{unit}|{tier}|{value}"))
            label_frame.sort_values(["sample_key", "record_id"], kind="stable",
                                    inplace=True)
        cells.append({"unit": unit, "tier": tier, **cell,
                      "human_test_rows": len(human),
                      "synthetic_test_rows": len(synthetic),
                      "matched_per_class": count,
                      "human_retention": count / len(human) if len(human) else 0.0,
                      "synthetic_retention": count / len(synthetic) if len(synthetic) else 0.0})
        by_id = subset.set_index("record_id")
        if not by_id.index.is_unique:
            raise ValueError("Nuisance cell contains duplicate record IDs")
        for human_id, synthetic_id in zip(human.head(count).record_id,
                                          synthetic.head(count).record_id):
            pair_id = _sha256_text(f"{seed}|{unit}|{tier}|{human_id}|{synthetic_id}")[:20]
            for record_id in (human_id, synthetic_id):
                row = by_id.loc[record_id]
                selected.append({
                    "seed": seed, "unit": unit, "tier": tier,
                    "record_id": str(record_id),
                    "parent_record_id": str(row.parent_record_id),
                    "probe_group_id": str(row.probe_group_id),
                    "label_name": str(row.label_name),
                    "broad_area": str(row.broad_area),
                    "pair_id": pair_id,
                })
    membership = pd.DataFrame(selected, columns=[
        "seed", "unit", "tier", "record_id", "parent_record_id",
        "probe_group_id", "label_name", "broad_area", "pair_id",
    ])
    if not membership.empty:
        if (membership.record_id.duplicated().any() or
                membership.groupby("pair_id").label_name.nunique().ne(2).any() or
                membership.groupby("pair_id").size().ne(2).any()):
            raise ValueError("Nuisance pairs are not disjoint binary pairs")
    return membership, pd.DataFrame(cells)


def _smd(human: np.ndarray, synthetic: np.ndarray) -> float:
    if not len(human) or not len(synthetic):
        return float("nan")
    difference = float(np.mean(synthetic) - np.mean(human))
    spread = math.sqrt((float(np.var(human)) + float(np.var(synthetic))) / 2)
    if spread == 0:
        return 0.0 if difference == 0 else float("inf")
    return difference / spread


def _ks(human: np.ndarray, synthetic: np.ndarray) -> float:
    if not len(human) or not len(synthetic):
        return float("nan")
    support = np.sort(np.unique(np.concatenate((human, synthetic))))
    human_cdf = np.searchsorted(np.sort(human), support, side="right") / len(human)
    synthetic_cdf = np.searchsorted(np.sort(synthetic), support, side="right") / len(synthetic)
    return float(np.max(np.abs(human_cdf - synthetic_cdf)))


def _balance(frame: pd.DataFrame, membership: pd.DataFrame, *, seed: int,
             unit: str, tier: str, minimum: int) -> list[dict[str, Any]]:
    matched = frame.loc[frame.record_id.isin(set(membership.record_id))]
    rows: list[dict[str, Any]] = []
    for population, part in (("test", frame), ("matched", matched)):
        for area, cell in [("ALL", part), *list(part.groupby("broad_area", sort=True))]:
            human = cell.loc[cell.label_name.eq("human")]
            synthetic = cell.loc[cell.label_name.eq("synthetic")]
            enough = min(len(human), len(synthetic)) >= minimum
            for feature in NUMERIC_BALANCE:
                h = human[feature].to_numpy(dtype=float)
                s = synthetic[feature].to_numpy(dtype=float)
                rows.append({
                    "seed": seed, "unit": unit, "tier": tier,
                    "population": population, "broad_area": area,
                    "feature": feature, "human_n": len(h), "synthetic_n": len(s),
                    "subgroup_supported": enough,
                    "human_mean": float(h.mean()) if len(h) else float("nan"),
                    "synthetic_mean": float(s.mean()) if len(s) else float("nan"),
                    "smd_synthetic_minus_human": _smd(h, s) if enough else float("nan"),
                    "ks": _ks(h, s) if enough else float("nan"),
                })
    return rows


def _frozen_splits(config: dict[str, Any], release_dir: Path,
                   release: dict[str, Any], selected_ids: set[str]
                   ) -> tuple[dict[int, pd.DataFrame], dict[str, dict[str, Any]]]:
    splits: dict[int, pd.DataFrame] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        name = f"split_seed{seed}"
        path = release_dir / f"{name}.parquet"
        expected = release["artifacts"][name]["sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen IID split is missing or changed: {seed}")
        split = pd.read_parquet(path)
        required = {"record_id", "split", "fold", "probe_group_id"}
        if (required - set(split) or split.record_id.duplicated().any() or
                set(split.record_id) != selected_ids or
                set(split.split) != {"development", "test"} or
                split.groupby("probe_group_id").split.nunique().gt(1).any() or
                split.groupby("probe_group_id").fold.nunique().gt(1).any()):
            raise ValueError(f"Invalid frozen IID split: {seed}")
        splits[seed] = split
        inputs[name] = {"path": str(path), "sha256": expected}
    return splits, inputs


def _segment_frames(segments_run: Path, selected: pd.DataFrame
                    ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest_path = segments_run / "segment_manifest.json"
    manifest = _json(manifest_path)
    if (manifest.get("release_id") != RELEASE_ID or
            manifest.get("kind") != "frozen_n2_middle_unit_views"):
        raise ValueError("Segment run does not describe the frozen N2 release")
    reference = selected.set_index("record_id")
    frames: dict[str, pd.DataFrame] = {}
    inputs: dict[str, Any] = {
        "segment_manifest": {"path": str(manifest_path),
                             "sha256": sha256_file(manifest_path)}}
    for unit in ("paragraph", "sentence"):
        info = manifest["artifacts"][f"{unit}_texts"]
        path = Path(info["path"])
        if not path.is_file() or sha256_file(path) != info["sha256"]:
            raise ValueError(f"Segment text artifact missing or changed: {unit}")
        frame = pd.read_parquet(path)
        required = REQUIRED_TEXT_COLUMNS | {"unit", "parent_record_id",
                                            "parent_word_count_audit"}
        if (required - set(frame) or frame.record_id.duplicated().any() or
                set(frame.record_id) - set(reference.index) or
                not frame.record_id.eq(frame.parent_record_id).all() or
                not frame.unit.eq(unit).all()):
            raise ValueError(f"Invalid {unit} segment frame")
        joined = frame[["record_id", "label_name", "broad_area", "probe_group_id",
                        "parent_word_count_audit"]].merge(
            selected[["record_id", "label_name", "broad_area", "probe_group_id",
                      "word_count_audit"]], on="record_id", validate="one_to_one",
            suffixes=("_unit", "_parent"))
        if (joined.label_name_unit.ne(joined.label_name_parent).any() or
                joined.broad_area_unit.ne(joined.broad_area_parent).any() or
                joined.probe_group_id_unit.ne(joined.probe_group_id_parent).any() or
                joined.parent_word_count_audit.ne(joined.word_count_audit).any()):
            raise ValueError(f"{unit} metadata differs from its N2 parent")
        frames[unit] = frame
        inputs[f"{unit}_texts"] = {"path": str(path), "sha256": info["sha256"]}
    return frames, inputs


def _coverage(test: pd.DataFrame, membership: pd.DataFrame, *, seed: int,
              unit: str, tier: str) -> list[dict[str, Any]]:
    matched_ids = set(membership.record_id)
    rows: list[dict[str, Any]] = []
    for area, cell in [("ALL", test), *list(test.groupby("broad_area", sort=True))]:
        for label in ("human", "synthetic"):
            part = cell.loc[cell.label_name.eq(label)]
            matched = int(part.record_id.isin(matched_ids).sum())
            rows.append({
                "seed": seed, "unit": unit, "tier": tier,
                "broad_area": area, "label_name": label,
                "test_rows": len(part), "matched_rows": matched,
                "retention": matched / len(part) if len(part) else None,
            })
    return rows


def _provenance(test: pd.DataFrame, membership: pd.DataFrame, *, seed: int,
                unit: str, tier: str) -> list[dict[str, Any]]:
    matched_ids = set(membership.record_id)
    frame = test.copy()
    frame["provenance"] = np.where(
        frame.label_name.eq("human"), frame.source_name, frame.generator)
    frame["provenance"] = frame.provenance.fillna("<missing>").astype(str)
    rows: list[dict[str, Any]] = []
    for (area, label, value), cell in frame.groupby(
            ["broad_area", "label_name", "provenance"], sort=True):
        count = int(cell.record_id.isin(matched_ids).sum())
        rows.append({
            "seed": seed, "unit": unit, "tier": tier,
            "broad_area": area, "label_name": label,
            "provenance_field": "source_name" if label == "human" else "generator",
            "provenance": value, "test_rows": len(cell), "matched_rows": count,
            "retention": count / len(cell),
        })
    return rows


def prepare_nuisance(config_path: Path, output: Path, *, segments_run: Path | None
                     ) -> Path:
    """Freeze test-only matching once per seed, for reuse across all detectors."""
    if output.exists():
        raise FileExistsError(f"Nuisance output already exists: {output}")
    config = load_config(config_path)
    release_dir, release = _pinned_release(config)
    selected, inputs = _selected_texts(config)
    selected = selected.copy()
    selected["parent_record_id"] = selected.record_id
    selected["parent_word_count_audit"] = selected.word_count_audit
    splits, split_inputs = _frozen_splits(config, release_dir, release,
                                         set(selected.record_id))
    inputs.update(split_inputs)
    units: dict[str, pd.DataFrame] = {"document": selected}
    if segments_run is not None:
        segments, segment_inputs = _segment_frames(segments_run, selected)
        units.update(segments)
        inputs.update(segment_inputs)
    width = float(config["length_bin_width"])
    feature_units = {unit: _features(frame, unit, width)
                     for unit, frame in units.items()}
    membership_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    cell_parts: list[pd.DataFrame] = []
    for seed in EXPECTED_SEEDS:
        frozen_test_ids = set(splits[seed].loc[splits[seed].split.eq("test"), "record_id"])
        for unit, features in feature_units.items():
            test = features.loc[features.parent_record_id.isin(frozen_test_ids)].copy()
            tiers = ("length", "presentation") if unit == "document" else ("length",)
            for tier in tiers:
                membership, cells = match_nuisance(test, unit=unit, tier=tier,
                                                    seed=seed)
                membership_parts.append(membership)
                cells.insert(0, "seed", seed)
                cell_parts.append(cells)
                coverage_rows.extend(_coverage(test, membership, seed=seed,
                                               unit=unit, tier=tier))
                balance_rows.extend(_balance(
                    test, membership, seed=seed, unit=unit, tier=tier,
                    minimum=int(config["minimum_subgroup_per_label"])))
                provenance_rows.extend(_provenance(test, membership, seed=seed,
                                                   unit=unit, tier=tier))
    membership = pd.concat(membership_parts, ignore_index=True)
    if membership.duplicated(["seed", "unit", "tier", "record_id"]).any():
        raise ValueError("Duplicate nuisance membership")
    membership.sort_values(["seed", "unit", "tier", "record_id"],
                           kind="stable", inplace=True)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "membership": output / "nuisance_membership.parquet",
        "retention": output / "nuisance_retention.csv",
        "balance": output / "nuisance_balance.csv",
        "provenance": output / "nuisance_provenance.csv",
        "cells": output / "nuisance_cells.csv",
        "summary": output / "summary.json",
    }
    pq.write_table(pa.Table.from_pandas(membership, preserve_index=False),
                   paths["membership"], compression="zstd")
    pd.DataFrame(coverage_rows).to_csv(paths["retention"], index=False)
    pd.DataFrame(balance_rows).to_csv(paths["balance"], index=False)
    pd.DataFrame(provenance_rows).to_csv(paths["provenance"], index=False)
    pd.concat(cell_parts, ignore_index=True).to_csv(paths["cells"], index=False)
    overall = [row for row in coverage_rows if row["broad_area"] == "ALL"]
    paths["summary"].write_text(json.dumps({
        "schema_version": 1, "kind": "n2_test_only_nuisance_summary",
        "rows": len(membership), "pairs": len(membership) // 2,
        "coverage": overall,
        "note": "The presentation tier can retain a narrow support population; inspect retention before metrics.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts = {name: {"path": str(path), "sha256": sha256_file(path),
                        **({"rows": len(membership)} if name == "membership" else {})}
                 for name, path in paths.items()}
    manifest = {
        "schema_version": 1, "kind": "n2_test_only_nuisance_common_support",
        "release_id": RELEASE_ID, "seeds": list(EXPECTED_SEEDS),
        "units": sorted(units), "tiers": {"document": ["length", "presentation"],
                                           "paragraph": ["length"] if segments_run else [],
                                           "sentence": ["length"] if segments_run else []},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "inputs": inputs, "artifacts": artifacts,
        "note": "Evaluation-only matched test IDs. Training, validation, N2 and frozen splits unchanged.",
    }
    manifest_path = output / "nuisance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest_path


def _verify_manifest_files(manifest: dict[str, Any], config_path: Path) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("release_id") != RELEASE_ID:
        raise ValueError("Unrecognized controls manifest")
    if manifest["config"]["sha256"] != sha256_file(config_path):
        raise ValueError("Controls configuration changed")
    for group in ("inputs", "artifacts"):
        for name, item in manifest[group].items():
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Changed {group} file: {name}")


def check_layout(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = _json(run_dir / "layout_manifest.json")
    if manifest.get("kind") != "n2_selected_layout_ablations":
        raise ValueError("Expected a layout ablation run")
    _verify_manifest_files(manifest, config_path)
    selected, _ = _selected_texts(config)
    selected = selected.sort_values("record_id", kind="stable").reset_index(drop=True)
    if manifest["rows"] != len(selected):
        raise ValueError("Layout row count differs from selected release")
    for name, transform in (("blank", layout_blank_text), ("all", layout_all_text)):
        view = pd.read_parquet(Path(manifest["artifacts"][name]["path"]))
        if (list(view.columns) != ["record_id", "content"] or
                view.record_id.duplicated().any() or
                not view.record_id.equals(selected.record_id) or
                not view.content.equals(selected.content.map(transform))):
            raise ValueError(f"Invalid layout view: {name}")
    return {"verified": True, "kind": manifest["kind"], "rows": len(selected)}


def check_nuisance(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = _json(run_dir / "nuisance_manifest.json")
    if manifest.get("kind") != "n2_test_only_nuisance_common_support":
        raise ValueError("Expected a nuisance common-support run")
    _verify_manifest_files(manifest, config_path)
    selected, _ = _selected_texts(config)
    release_dir, release = _pinned_release(config)
    splits, _ = _frozen_splits(config, release_dir, release, set(selected.record_id))
    membership = pd.read_parquet(Path(manifest["artifacts"]["membership"]["path"]))
    required = {"seed", "unit", "tier", "record_id", "parent_record_id",
                "probe_group_id", "label_name", "broad_area", "pair_id"}
    if (required - set(membership) or membership.empty or
            membership.duplicated(["seed", "unit", "tier", "record_id"]).any() or
            not set(membership.seed) <= set(EXPECTED_SEEDS) or
            membership.groupby(["seed", "unit", "tier", "pair_id"]).size().ne(2).any() or
            membership.groupby(["seed", "unit", "tier", "pair_id"])
            .label_name.nunique().ne(2).any() or
            membership.record_id.ne(membership.parent_record_id).any()):
        raise ValueError("Invalid nuisance membership or pairs")
    reference = selected.set_index("record_id")
    if set(membership.parent_record_id) - set(reference.index):
        raise ValueError("Nuisance membership includes unselected parent IDs")
    for seed, subset in membership.groupby("seed", sort=True):
        test_ids = set(splits[int(seed)].loc[splits[int(seed)].split.eq("test"),
                                             "record_id"])
        if set(subset.parent_record_id) - test_ids:
            raise ValueError("Nuisance membership includes a non-test document")
    lookup = membership[["record_id", "probe_group_id", "label_name",
                         "broad_area"]].drop_duplicates()
    lookup = lookup.merge(selected[["record_id", "probe_group_id", "label_name",
                                    "broad_area"]], on="record_id", validate="one_to_one",
                          suffixes=("_matched", "_source"))
    if any(lookup[f"{name}_matched"].ne(lookup[f"{name}_source"]).any()
           for name in ("probe_group_id", "label_name", "broad_area")):
        raise ValueError("Nuisance membership metadata differs from N2")
    return {"verified": True, "kind": manifest["kind"],
            "membership_rows": len(membership),
            "pairs": len(membership) // 2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    layout = sub.add_parser("prepare-layout")
    layout.add_argument("--config", type=Path, required=True)
    layout.add_argument("--output", type=Path, required=True)
    nuisance = sub.add_parser("prepare-nuisance")
    nuisance.add_argument("--config", type=Path, required=True)
    nuisance.add_argument("--output", type=Path, required=True)
    nuisance.add_argument("--segments-run", type=Path)
    check = sub.add_parser("check")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-layout":
        print(prepare_layout(args.config, args.output))
    elif args.command == "prepare-nuisance":
        print(prepare_nuisance(args.config, args.output,
                               segments_run=args.segments_run))
    else:
        path = args.run / "layout_manifest.json"
        result = (check_layout(args.config, args.run) if path.is_file() else
                  check_nuisance(args.config, args.run))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
