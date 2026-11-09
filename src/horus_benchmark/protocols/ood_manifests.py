"""Build group-safe LOGO, LODO and LOSO manifests from the frozen N2 release.

The OOD target is completely withheld, including target rows that were in the
IID development partition. All other IID test rows remain unavailable for
training. Matched opposite-class controls may be drawn from either IID split,
but their entire probe groups are withheld. Thus the common IID test subset
remains available for paired comparisons without using it for OOD fitting.

Only IDs and metadata are written. No corpus content or detector features are
read here. Run on the researcher's cluster, per AGENTS.md section 7.19.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from horus_benchmark.protocols.grouped_iid import stable_key


INDEX_COLUMNS = {
    "record_id", "label_name", "broad_area", "source_name", "generator",
    "word_count_audit", "probe_group_id",
}
SPLIT_COLUMNS = {"record_id", "probe_group_id", "split", "fold"}
PROTOCOLS = ("logo", "lodo", "loso")
TEST_ROLES = {"test_target", "test_unmatched_target", "test_control"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported OOD configuration")
    if data.get("release_id") != "horus-n2-iid-v1":
        raise ValueError("OOD configuration must use the reviewed N2 IID release")
    if not isinstance(data.get("release_dir"), str):
        raise ValueError("release_dir is required")
    if data.get("seeds") != [20261105, 20261106, 20261107]:
        raise ValueError("OOD configuration must cover the three frozen IID seeds")
    if not 0 < float(data.get("length_bin_width", 0)) <= 1:
        raise ValueError("length_bin_width must be in (0, 1]")
    if int(data.get("min_binary_test_per_class", 0)) < 1:
        raise ValueError("min_binary_test_per_class must be positive")
    if int(data.get("min_other_human_development", 0)) < 1:
        raise ValueError("min_other_human_development must be positive")
    aliases = data.get("source_aliases", {})
    if not isinstance(aliases, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or not k.strip() or not v.strip()
            for k, v in aliases.items()):
        raise ValueError("source_aliases must map nonempty exact names to canonical names")
    data["source_aliases"] = aliases
    return data


def _load_release(release_dir: Path, seeds: list[int]) -> tuple[dict, pd.DataFrame,
                                                                  dict[int, pd.DataFrame]]:
    release_path = release_dir / "release_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (release.get("verified") is not True or
            release.get("kind") != "frozen_n2_source_balanced_iid_release" or
            release.get("release_id") != "horus-n2-iid-v1" or
            release.get("text_view") != "n2" or
            release.get("development_folds") != 5 or
            release.get("split_seeds") != seeds):
        raise ValueError("Expected the verified N2 IID v1 release and its three seeds")
    index_path = release_dir / "selected_index.parquet"
    if _sha256(index_path) != release["artifacts"]["selected_index"]["sha256"]:
        raise ValueError("Frozen selected index SHA-256 mismatch")
    index = pd.read_parquet(index_path)
    missing = INDEX_COLUMNS - set(index)
    if missing or index.record_id.duplicated().any() or len(index) != release["rows"]:
        raise ValueError(f"Invalid selected index; missing={sorted(missing)}")
    if (index.probe_group_id.isna().any() or index.word_count_audit.isna().any() or
            index.word_count_audit.lt(0).any() or
            set(index.label_name) != {"human", "synthetic"}):
        raise ValueError("Invalid selected index groups, lengths or labels")
    if index.groupby("probe_group_id").label_name.nunique().gt(1).any():
        raise ValueError("Cross-label probe group requires review before OOD construction")
    splits = {}
    for seed in seeds:
        name = f"split_seed{seed}"
        split_path = release_dir / f"{name}.parquet"
        if _sha256(split_path) != release["artifacts"][name]["sha256"]:
            raise ValueError(f"Frozen IID split SHA-256 mismatch for {seed}")
        split = pd.read_parquet(split_path)
        missing = SPLIT_COLUMNS - set(split)
        if missing or split.record_id.duplicated().any() or set(split.record_id) != set(index.record_id):
            raise ValueError(f"Invalid IID split {seed}; missing={sorted(missing)}")
        same_groups = index[["record_id", "probe_group_id"]].merge(
            split[["record_id", "probe_group_id"]], on="record_id",
            validate="one_to_one", suffixes=("_index", "_split"))
        if same_groups.probe_group_id_index.ne(same_groups.probe_group_id_split).any():
            raise ValueError(f"Frozen IID split group IDs differ from selected index: {seed}")
        if (set(split.split) != {"development", "test"} or
                split.loc[split.split.eq("test"), "fold"].ne(-1).any() or
                not set(split.loc[split.split.eq("development"), "fold"]) <= set(range(5))):
            raise ValueError(f"Invalid frozen IID split/fold roles: {seed}")
        if split.groupby("probe_group_id").agg(
                split_count=("split", "nunique"), fold_count=("fold", "nunique")
        ).gt(1).any().any():
            raise ValueError(f"Group crosses frozen IID split/fold for {seed}")
        splits[seed] = split[["record_id", "split", "fold"]].copy()
    return release, index, splits


def _source_family(names: pd.Series, aliases: dict[str, str]) -> pd.Series:
    # Exact aliases only. No fuzzy publisher inference or automatic host merging.
    clean = names.fillna("<missing>").astype(str).str.strip()
    return clean.replace(aliases)


def _slug(protocol: str, area: str | None, target: str) -> str:
    value = f"{protocol}|{area or ''}|{target}"
    readable = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")[:28]
    return f"{protocol}-{readable}-{hashlib.sha256(value.encode()).hexdigest()[:10]}"


def _tasks(index: pd.DataFrame) -> list[tuple[str, str, str | None]]:
    tasks: list[tuple[str, str, str | None]] = []
    tasks.extend(("logo", name, None) for name in sorted(
        index.loc[index.label_name.eq("synthetic"), "generator"].dropna().astype(str).unique()))
    tasks.extend(("lodo", name, None) for name in sorted(index.broad_area.astype(str).unique()))
    human = index.loc[index.label_name.eq("human")]
    tasks.extend(("loso", source, area) for area, source in sorted(set(zip(
        human.broad_area.astype(str), human.source_family.astype(str)))))
    return tasks


def _select_controls(base: pd.DataFrame, target: pd.DataFrame, target_groups: set[str],
                     protocol: str) -> tuple[set[str], set[str], set[str]]:
    """Match opposite-class controls within area × original log2-length cell.

    Priority is given to controls and targets in the frozen IID test, so that
    the paired ID/OOD comparison has support. Every target still enters the
    OOD test: any target lacking a control is tagged test_unmatched_target.
    """
    opposite = "human" if protocol == "logo" else "synthetic"
    candidates = base.loc[
        base.label_name.eq(opposite) & ~base.probe_group_id.isin(target_groups)]
    control_ids: set[str] = set()
    matched_ids: set[str] = set()
    for cell, targets in target.groupby(["broad_area", "length_bin"], sort=True):
        pool = candidates.loc[
            candidates.broad_area.eq(cell[0]) & candidates.length_bin.eq(cell[1])]
        if pool.empty:
            continue
        # First reserve controls for target IDs that have a valid paired IID
        # baseline prediction. The rest is selected by stable SHA-256 order.
        target_sorted = targets.sort_values(
            ["iid_test_priority", "sample_key", "record_id"],
            ascending=[False, True, True], kind="stable")
        pool_sorted = pool.sort_values(
            ["iid_test_priority", "sample_key", "record_id"],
            ascending=[False, True, True], kind="stable")
        paired_needed = int(target_sorted.iid_test_priority.sum())
        paired_controls = pool_sorted.loc[pool_sorted.iid_test_priority].head(paired_needed)
        remaining = pool_sorted.loc[~pool_sorted.record_id.isin(paired_controls.record_id)]
        # After reserving paired IID controls, prefer development controls.
        # This avoids needlessly overrepresenting controls in the paired subset.
        remaining = remaining.sort_values(
            ["iid_test_priority", "sample_key", "record_id"],
            ascending=[True, True, True], kind="stable")
        chosen = pd.concat((paired_controls, remaining.head(len(targets) - len(paired_controls))))
        # If there are fewer frozen IID controls than target IID examples,
        # additional controls are sampled by ID hash, never with replacement.
        if len(chosen) < min(len(targets), len(pool_sorted)):
            deficit = min(len(targets), len(pool_sorted)) - len(chosen)
            extra = pool_sorted.loc[~pool_sorted.record_id.isin(chosen.record_id)].head(deficit)
            chosen = pd.concat((chosen, extra))
        count = min(len(chosen), len(targets))
        control_ids.update(chosen.head(count).record_id.astype(str))
        matched_ids.update(target_sorted.head(count).record_id.astype(str))
    unmatched_ids = set(target.record_id.astype(str)) - matched_ids
    return control_ids, matched_ids, unmatched_ids


def build_one(base: pd.DataFrame, protocol: str, target: str, area: str | None,
              config: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """Construct one full OOD manifest; returns it even when support fails."""
    if protocol not in PROTOCOLS:
        raise ValueError("Unsupported OOD protocol")
    if base.record_id.duplicated().any():
        raise ValueError("Duplicate record IDs")
    if protocol == "logo":
        mask = base.label_name.eq("synthetic") & base.generator.astype(str).eq(target)
    elif protocol == "lodo":
        mask = base.broad_area.astype(str).eq(target)
    else:
        if area is None:
            raise ValueError("LOSO requires a source area")
        mask = (base.label_name.eq("human") & base.broad_area.astype(str).eq(area) &
                base.source_family.astype(str).eq(target))
    targets = base.loc[mask].copy()
    if targets.empty:
        raise ValueError(f"No target rows for {protocol}/{target}")
    target_groups = set(targets.probe_group_id.astype(str))
    roles = pd.Series("embargo", index=base.index, dtype="object")
    if protocol == "lodo":
        roles.loc[mask] = "test_target"
        controls: set[str] = set()
        matched = set(targets.record_id.astype(str))
        unmatched: set[str] = set()
    else:
        controls, matched, unmatched = _select_controls(base, targets, target_groups, protocol)
        roles.loc[base.record_id.isin(matched)] = "test_target"
        roles.loc[base.record_id.isin(unmatched)] = "test_unmatched_target"
        roles.loc[base.record_id.isin(controls)] = "test_control"
    touched_groups = target_groups | set(base.loc[base.record_id.isin(controls),
                                           "probe_group_id"].astype(str))
    development = base.iid_split.eq("development") & ~base.probe_group_id.isin(touched_groups)
    roles.loc[development] = "development"
    out = base[["record_id", "probe_group_id", "label_name", "broad_area",
                "word_count_audit"]].copy()
    out.insert(0, "seed", seed)
    out.insert(1, "protocol", protocol)
    out.insert(2, "target", target)
    out.insert(3, "source_area", area or "")
    out["role"] = roles.to_numpy()
    out["fold"] = np.where(out.role.eq("development"), base.iid_fold, -1).astype(int)
    out["paired_iid_test"] = base.iid_split.eq("test") & out.role.isin(TEST_ROLES)
    # Group-level exclusion is mandatory; other members of touched groups
    # may be embargoed, never silently returned to development.
    if out.assign(is_dev=out.role.eq("development")).groupby("probe_group_id").is_dev.nunique(
    ).gt(1).any():
        raise AssertionError("Probe group crosses OOD development and test/embargo")
    if out.loc[out.role.eq("development")].groupby("probe_group_id").fold.nunique().gt(1).any():
        raise AssertionError("Probe group crosses OOD development folds")
    if out.loc[out.role.eq("development"), "fold"].lt(0).any():
        raise AssertionError("OOD development row lacks a validation fold")
    if len(out.loc[out.role.isin({"test_target", "test_unmatched_target"})]) != len(targets):
        raise AssertionError("OOD target rows were lost")
    if protocol == "logo" and base.loc[out.role.eq("development"), "generator"].astype(
            str).eq(target).any():
        raise AssertionError("Held-out generator leaked into OOD development")
    if protocol == "lodo" and base.loc[out.role.eq("development"), "broad_area"].astype(
            str).eq(target).any():
        raise AssertionError("Held-out area leaked into OOD development")
    if protocol == "loso" and base.loc[out.role.eq("development") &
                                        base.broad_area.eq(area) &
                                        base.label_name.eq("human"), "source_family"].astype(
            str).eq(target).any():
        raise AssertionError("Held-out human source leaked into OOD development")
    binary = out.loc[out.role.isin({"test_target", "test_control"})]
    binary_counts = binary.label_name.value_counts().to_dict()
    dev = out.loc[out.role.eq("development")]
    fold_labels = dev.groupby("fold").label_name.nunique()
    other_human = int(base.loc[
        out.role.eq("development") & base.label_name.eq("human") &
        base.broad_area.eq(area) & base.source_family.ne(target)].shape[0]
    ) if protocol == "loso" else None
    reasons = []
    minimum = int(config["min_binary_test_per_class"])
    if any(int(binary_counts.get(label, 0)) < minimum for label in ("human", "synthetic")):
        reasons.append("binary_test_below_minimum")
    if len(fold_labels) != 5 or fold_labels.ne(2).any():
        reasons.append("development_fold_missing_class")
    if protocol == "loso" and (other_human or 0) < int(config["min_other_human_development"]):
        reasons.append("other_human_source_development_below_minimum")
    paired = binary.loc[binary.paired_iid_test]
    stats = {
        "seed": seed, "protocol": protocol, "target": target, "source_area": area or "",
        "status": "eligible" if not reasons else "unsupported",
        "eligible": not reasons, "reason": ";".join(reasons),
        "target_rows": len(targets), "matched_target_rows": len(matched),
        "unmatched_target_rows": len(unmatched), "control_rows": len(controls),
        "binary_human_rows": int(binary_counts.get("human", 0)),
        "binary_synthetic_rows": int(binary_counts.get("synthetic", 0)),
        "paired_iid_human_rows": int(paired.label_name.eq("human").sum()),
        "paired_iid_synthetic_rows": int(paired.label_name.eq("synthetic").sum()),
        "development_rows": len(dev), "embargo_rows": int(out.role.eq("embargo").sum()),
        "other_human_development": other_human,
    }
    return out, stats


def _prepare_base(index: pd.DataFrame, split: pd.DataFrame, aliases: dict[str, str],
                  width: float, seed: int) -> pd.DataFrame:
    base = index.merge(split, on="record_id", validate="one_to_one", suffixes=("", "_iid"))
    if "split" not in base or "fold" not in base:
        raise ValueError("Frozen IID split columns missing")
    base = base.rename(columns={"split": "iid_split", "fold": "iid_fold"})
    base["source_family"] = _source_family(base.source_name, aliases)
    base["length_bin"] = np.floor(np.log2(1 + base.word_count_audit) / width).astype(int)
    base["sample_key"] = base.record_id.map(lambda rid: stable_key(seed, str(rid)))
    base["iid_test_priority"] = base.iid_split.eq("test")
    return base.sort_values("record_id", kind="stable").reset_index(drop=True)


def build(config_path: Path, output: Path) -> Path:
    config = _load_config(config_path)
    if output.exists():
        raise FileExistsError(f"OOD output already exists: {output}")
    release_dir = Path(config["release_dir"])
    release, index, splits = _load_release(release_dir, config["seeds"])
    # Explicit source aliases are part of the protocol, never inferred from
    # outcomes or text. Unknown aliases retain their literal source_name.
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in config["seeds"]:
        base = _prepare_base(index, splits[seed], config["source_aliases"],
                             float(config["length_bin_width"]), seed)
        for protocol, target, area in _tasks(base):
            manifest, stats = build_one(base, protocol, target, area, config, seed)
            if stats["eligible"]:
                name = f"seed{seed}/{_slug(protocol, area, target)}.parquet"
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(manifest, preserve_index=False), path,
                               compression="zstd")
                stats["manifest_path"] = str(path)
                stats["manifest_sha256"] = _sha256(path)
            else:
                stats["manifest_path"] = ""
                stats["manifest_sha256"] = ""
            rows.append(stats)
    tasks = pd.DataFrame(rows).sort_values(
        ["seed", "protocol", "source_area", "target"], kind="stable")
    tasks_path = output / "tasks.csv"
    tasks.to_csv(tasks_path, index=False)
    summary = {
        "schema_version": 1,
        "kind": "n2_independent_ood_manifests",
        "release_id": release["release_id"],
        "release_manifest_sha256": _sha256(release_dir / "release_manifest.json"),
        "selected_index_sha256": release["artifacts"]["selected_index"]["sha256"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "seeds": config["seeds"],
        "counts": {f"{protocol}:{status}": int(count) for (protocol, status), count in
                   tasks.groupby(["protocol", "status"], dropna=False).size().items()},
        "source_aliases": config["source_aliases"],
        "tasks_sha256": _sha256(tasks_path),
        "limitations": [
            "OOD test targets are exhaustive only within the frozen selected N2 release.",
            "Matching is area and N2 word-length-bin based, not source/topic/genre equivalence.",
            "Unmatched targets are retained for target-only rates, excluded from binary AUC.",
            "Only observed lineage groups are excluded; unknown relationships remain possible.",
            "Frozen IID test rows not used as controls are embargoed from OOD fitting.",
        ],
    }
    target = output / "summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def check(output: Path) -> Path:
    target = output / "summary.json"
    summary = json.loads(target.read_text(encoding="utf-8"))
    config_path = Path(summary["config_path"])
    config = _load_config(config_path)
    if _sha256(config_path) != summary["config_sha256"]:
        raise ValueError("OOD configuration changed")
    release_dir = Path(config["release_dir"])
    if _sha256(release_dir / "release_manifest.json") != summary["release_manifest_sha256"]:
        raise ValueError("Frozen IID release manifest changed")
    _load_release(release_dir, config["seeds"])
    tasks_path = output / "tasks.csv"
    if _sha256(tasks_path) != summary["tasks_sha256"]:
        raise ValueError("OOD task index changed")
    tasks = pd.read_csv(tasks_path, keep_default_na=False)
    for row in tasks.loc[tasks.status.eq("eligible")].itertuples(index=False):
        path = Path(row.manifest_path)
        if _sha256(path) != row.manifest_sha256:
            raise ValueError(f"OOD manifest changed: {path}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_cmd = sub.add_parser("build")
    build_cmd.add_argument("--config", type=Path, required=True)
    build_cmd.add_argument("--output", type=Path, required=True)
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.config, args.output) if args.command == "build" else check(args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
