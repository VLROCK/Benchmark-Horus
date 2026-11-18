"""Build a generator-and-domain held-out protocol from the frozen N2 release.

Each test is the area-restricted slice of an existing LOGO test. This makes its
test IDs identical to the corresponding LOGO and LODO IDs for paired analysis.
All rows of the held-out generator and area, and their observed lineage groups,
are unavailable to the new development folds. The IID test is still embargoed.

Only IDs and metadata are read or written here. Run the builder on the
researcher's cluster, as required by AGENTS.md section 7.19.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from horus_benchmark.protocols.ood_manifests import (
    TEST_ROLES, _load_release, _prepare_base, _sha256, _slug,
    check as check_independent_ood,
)


PROTOCOL = "logo_lodo"
TEST_OUTPUT_COLUMNS = (
    "seed", "protocol", "target", "source_area", "heldout_generator",
    "heldout_area", "record_id", "probe_group_id", "label_name", "broad_area",
    "word_count_audit", "role", "fold", "paired_iid_test",
)
PARENT_ROLE_COLUMNS = {"record_id", "role", "label_name", "broad_area",
                       "probe_group_id", "protocol", "target", "seed"}


def _load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("Unsupported compound OOD configuration")
    if cfg.get("experiment_id") != "horus-n2-compound-generator-domain-v1":
        raise ValueError("Unexpected compound OOD experiment_id")
    if cfg.get("release_id") != "horus-n2-iid-v1":
        raise ValueError("Compound OOD requires the frozen N2 IID release")
    if cfg.get("seeds") != [20261105, 20261106, 20261107]:
        raise ValueError("Compound OOD must use all three frozen split seeds")
    for name in ("release_dir", "parent_ood_dir"):
        if not isinstance(cfg.get(name), str) or not cfg[name].strip():
            raise ValueError(f"{name} is required")
    if float(cfg.get("length_bin_width", 0)) != 0.25:
        raise ValueError("Length cells must match the frozen LOGO protocol")
    if int(cfg.get("min_binary_test_per_class", 0)) < 1:
        raise ValueError("min_binary_test_per_class must be positive")
    if int(cfg.get("development_folds", 0)) != 5:
        raise ValueError("The frozen release has five development folds")
    if "source_aliases" in cfg and not isinstance(cfg["source_aliases"], dict):
        raise ValueError("source_aliases must be a mapping")
    return cfg


def _target_name(generator: str, area: str) -> str:
    if "::" in generator or "::" in area:
        raise ValueError("Target names cannot contain '::'")
    return f"{generator}::{area}"


def _parent_tasks(parent_dir: Path, release_manifest_sha256: str,
                  seeds: list[int]) -> tuple[dict, pd.DataFrame]:
    # This checks every existing eligible manifest and its pinned config.
    check_independent_ood(parent_dir)
    summary = json.loads((parent_dir / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("kind") != "n2_independent_ood_manifests" or
            summary.get("release_id") != "horus-n2-iid-v1" or
            summary.get("release_manifest_sha256") != release_manifest_sha256 or
            summary.get("seeds") != seeds):
        raise ValueError("Parent OOD manifests do not match the frozen release")
    tasks = pd.read_csv(parent_dir / "tasks.csv", keep_default_na=False)
    required = {"seed", "protocol", "target", "source_area", "status",
                "manifest_path", "manifest_sha256"}
    if required - set(tasks):
        raise ValueError("Parent OOD task index lacks required columns")
    # LOSO legitimately repeats a source family in different areas. Its area
    # is part of the parent protocol identity; LOGO still has one row per
    # generator and seed, checked again when each LOGO manifest is loaded.
    if tasks.duplicated(["seed", "protocol", "source_area", "target"]).any():
        raise ValueError("Parent OOD task index has duplicate identities")
    return summary, tasks


def _load_logo(path: Path, sha256: str, base: pd.DataFrame,
               generator: str, seed: int) -> pd.DataFrame:
    if _sha256(path) != sha256:
        raise ValueError(f"Parent LOGO manifest changed: {path}")
    parent = pd.read_parquet(path)
    if (PARENT_ROLE_COLUMNS - set(parent) or
            len(parent) != len(base) or parent.record_id.duplicated().any() or
            set(parent.record_id) != set(base.record_id)):
        raise ValueError("Parent LOGO manifest does not cover the frozen release")
    if (not parent.protocol.eq("logo").all() or
            not parent.target.eq(generator).all() or
            not parent.seed.eq(seed).all()):
        raise ValueError("Wrong parent LOGO target or seed")
    aligned = base[["record_id", "probe_group_id", "label_name", "broad_area"]].merge(
        parent[["record_id", "probe_group_id", "label_name", "broad_area", "role"]],
        on="record_id", validate="one_to_one", suffixes=("_base", "_logo"))
    for field in ("probe_group_id", "label_name", "broad_area"):
        if not aligned[f"{field}_base"].eq(aligned[f"{field}_logo"]).all():
            raise ValueError(f"Parent LOGO {field} differs from frozen release")
    if not aligned.record_id.eq(base.record_id).all():
        raise ValueError("Parent LOGO rows are not aligned to the frozen release")
    return aligned[["record_id", "role"]]


def build_one(base: pd.DataFrame, logo: pd.DataFrame, generator: str,
              area: str, config: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """Make a full-ID manifest; eligibility only controls whether it is saved."""
    required = {"record_id", "probe_group_id", "label_name", "broad_area",
                "generator", "source_name", "word_count_audit", "length_bin",
                "iid_split", "iid_fold"}
    if required - set(base) or {"record_id", "role"} - set(logo):
        raise ValueError("Frozen base or LOGO roles lack required fields")
    if (base.record_id.duplicated().any() or logo.record_id.duplicated().any() or
            len(base) != len(logo) or not base.record_id.eq(logo.record_id).all()):
        raise ValueError("LOGO and frozen release IDs must align one-to-one")
    target = _target_name(generator, area)
    is_target = (base.label_name.eq("synthetic") &
                 base.generator.astype(str).eq(generator) &
                 base.broad_area.astype(str).eq(area))
    is_area = base.broad_area.astype(str).eq(area)
    parent_test = logo.role.isin(TEST_ROLES)
    selected = parent_test & is_area
    allowed_target = logo.role.isin({"test_target", "test_unmatched_target"}) & is_target
    if (selected & logo.role.eq("test_control") & ~base.label_name.eq("human")).any():
        raise ValueError("LOGO controls are not human")
    if (selected & logo.role.isin({"test_target", "test_unmatched_target"}) &
            ~is_target).any():
        raise ValueError("LOGO target is not the requested generator and area")
    if int(allowed_target.sum()) != int(is_target.sum()):
        raise ValueError("Parent LOGO did not withhold every generator-area target")
    roles = pd.Series("embargo", index=base.index, dtype="object")
    roles.loc[selected] = logo.loc[selected, "role"].to_numpy()

    # Excluding whole groups handles observed near-pairs spanning a generator,
    # source, or area; no member of a touched group may enter development.
    excluded = base.generator.astype(str).eq(generator) | is_area
    excluded_groups = set(base.loc[excluded, "probe_group_id"].astype(str))
    development = base.iid_split.eq("development") & ~base.probe_group_id.isin(
        excluded_groups)
    if (development & selected).any():
        raise AssertionError("Test record entered compound development")
    roles.loc[development] = "development"

    out = base[["record_id", "probe_group_id", "label_name", "broad_area",
                "word_count_audit"]].copy()
    out.insert(0, "seed", int(seed))
    out.insert(1, "protocol", PROTOCOL)
    out.insert(2, "target", target)
    out.insert(3, "source_area", area)
    out.insert(4, "heldout_generator", generator)
    out.insert(5, "heldout_area", area)
    out["role"] = roles.to_numpy()
    out["fold"] = np.where(out.role.eq("development"), base.iid_fold, -1).astype(int)
    out["paired_iid_test"] = base.iid_split.eq("test") & out.role.isin(TEST_ROLES)
    out = out.loc[:, list(TEST_OUTPUT_COLUMNS)]

    if out.loc[out.role.eq("development"), "fold"].lt(0).any():
        raise AssertionError("Compound development row lacks a frozen fold")
    if out.assign(is_development=out.role.eq("development")).groupby(
            "probe_group_id").is_development.nunique().gt(1).any():
        raise AssertionError("Probe group crosses compound development and embargo/test")
    if out.loc[out.role.eq("development")].groupby(
            "probe_group_id").fold.nunique().gt(1).any():
        raise AssertionError("Probe group crosses development folds")
    if base.loc[out.role.eq("development"), "generator"].astype(str).eq(generator).any():
        raise AssertionError("Held-out generator leaked into development")
    if base.loc[out.role.eq("development"), "broad_area"].astype(str).eq(area).any():
        raise AssertionError("Held-out area leaked into development")
    if out.loc[out.role.isin(TEST_ROLES), "broad_area"].ne(area).any():
        raise AssertionError("Compound test includes another area")
    if (set(out.loc[out.role.isin(TEST_ROLES), "record_id"]) !=
            set(base.loc[selected, "record_id"])):
        raise AssertionError("Compound test differs from area-restricted LOGO test")
    binary_cells = base.loc[roles.isin({"test_target", "test_control"}),
                            ["length_bin", "label_name"]]
    if not binary_cells.empty:
        cell_counts = binary_cells.groupby(["length_bin", "label_name"]).size().unstack(
            fill_value=0)
        if not cell_counts.get("human", pd.Series(0, index=cell_counts.index)).eq(
                cell_counts.get("synthetic", pd.Series(0, index=cell_counts.index))).all():
            raise AssertionError("Area-restricted LOGO length cells lost matched controls")

    binary = out.loc[out.role.isin({"test_target", "test_control"})]
    counts = binary.label_name.value_counts().to_dict()
    dev = out.loc[out.role.eq("development")]
    by_fold = dev.groupby("fold").label_name.nunique()
    minimum = int(config["min_binary_test_per_class"])
    reasons = []
    if not is_target.any():
        reasons.append("no_generator_area_target")
    if any(int(counts.get(label, 0)) < minimum for label in ("human", "synthetic")):
        reasons.append("binary_test_below_minimum")
    if len(by_fold) != int(config["development_folds"]) or by_fold.ne(2).any():
        reasons.append("development_fold_missing_class")
    matched = int(out.role.eq("test_target").sum())
    paired = binary.loc[binary.paired_iid_test]
    dev_class = dev.label_name.value_counts().to_dict()
    stats = {
        "seed": int(seed), "protocol": PROTOCOL, "target": target,
        "heldout_generator": generator, "heldout_area": area, "source_area": area,
        "status": "eligible" if not reasons else "unsupported",
        "eligible": not reasons, "reason": ";".join(reasons),
        "target_rows": int(is_target.sum()), "matched_target_rows": matched,
        "unmatched_target_rows": int(out.role.eq("test_unmatched_target").sum()),
        "matched_target_fraction": matched / max(int(is_target.sum()), 1),
        "control_rows": int(out.role.eq("test_control").sum()),
        "binary_human_rows": int(counts.get("human", 0)),
        "binary_synthetic_rows": int(counts.get("synthetic", 0)),
        "paired_iid_human_rows": int(paired.label_name.eq("human").sum()),
        "paired_iid_synthetic_rows": int(paired.label_name.eq("synthetic").sum()),
        "development_rows": len(dev),
        "development_human_rows": int(dev_class.get("human", 0)),
        "development_synthetic_rows": int(dev_class.get("synthetic", 0)),
        "embargo_rows": int(out.role.eq("embargo").sum()),
        "heldout_generator_rows": int(base.generator.astype(str).eq(generator).sum()),
        "heldout_area_rows": int(is_area.sum()),
        "development_fold_label_counts": json.dumps({
            str(int(fold)): {name: int(count) for name, count in
                             group.label_name.value_counts().to_dict().items()}
            for fold, group in dev.groupby("fold", sort=True)
        }, sort_keys=True),
    }
    return out, stats


def build(config_path: Path, output: Path) -> Path:
    cfg = _load_config(config_path)
    if output.exists():
        raise FileExistsError(f"Compound OOD output already exists: {output}")
    release_dir = Path(cfg["release_dir"])
    release, index, splits = _load_release(release_dir, cfg["seeds"])
    parent_dir = Path(cfg["parent_ood_dir"])
    parent_summary, parent_tasks = _parent_tasks(
        parent_dir, _sha256(release_dir / "release_manifest.json"), cfg["seeds"])
    parent_aliases = parent_summary["source_aliases"]
    if (("source_aliases" in cfg and cfg["source_aliases"] != parent_aliases) or
            float(yaml.safe_load(Path(parent_summary["config_path"]).read_text(
                encoding="utf-8")).get("length_bin_width", 0)) !=
            float(cfg["length_bin_width"])):
        raise ValueError("Compound matching settings differ from parent LOGO")
    generators = sorted(index.loc[index.label_name.eq("synthetic"),
                                  "generator"].dropna().astype(str).unique())
    areas = sorted(index.broad_area.dropna().astype(str).unique())
    parent_logos = parent_tasks.loc[parent_tasks.protocol.eq("logo")]
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    provenance_rows: list[dict] = []
    for seed in cfg["seeds"]:
        base = _prepare_base(index, splits[seed], parent_aliases,
                             float(cfg["length_bin_width"]), seed)
        for generator in generators:
            parent_match = parent_logos.loc[
                parent_logos.seed.eq(seed) & parent_logos.target.eq(generator)]
            if len(parent_match) != 1:
                raise ValueError(f"Missing or duplicate parent LOGO task: {seed}/{generator}")
            parent_task = parent_match.iloc[0]
            logo = None
            if parent_task.status == "eligible":
                logo = _load_logo(Path(parent_task.manifest_path),
                                  str(parent_task.manifest_sha256), base,
                                  generator, seed)
            for area in areas:
                if logo is None:
                    target_rows = int((base.label_name.eq("synthetic") &
                                       base.generator.astype(str).eq(generator) &
                                       base.broad_area.astype(str).eq(area)).sum())
                    stats = {
                        "seed": seed, "protocol": PROTOCOL,
                        "target": _target_name(generator, area),
                        "heldout_generator": generator, "heldout_area": area,
                        "source_area": area, "status": "unsupported",
                        "eligible": False, "reason": "parent_logo_unsupported",
                        "target_rows": target_rows,
                        "parent_logo_status": str(parent_task.status),
                        "parent_logo_reason": str(parent_task.reason),
                        "manifest_path": "", "manifest_sha256": "",
                    }
                else:
                    manifest, stats = build_one(base, logo, generator, area, cfg, seed)
                    stats["parent_logo_status"] = "eligible"
                    stats["parent_logo_reason"] = ""
                    if stats["eligible"]:
                        path = output / f"seed{seed}" / _slug(PROTOCOL, area, generator)
                        path = path.with_suffix(".parquet")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        pq.write_table(pa.Table.from_pandas(manifest, preserve_index=False),
                                       path, compression="zstd")
                        stats["manifest_path"] = str(path)
                        stats["manifest_sha256"] = _sha256(path)
                    else:
                        stats["manifest_path"] = ""
                        stats["manifest_sha256"] = ""
                    sample = base.loc[manifest.role.isin(TEST_ROLES),
                                      ["record_id", "source_name", "generator",
                                       "label_name", "broad_area"]].copy()
                    sample["role"] = manifest.loc[manifest.role.isin(TEST_ROLES),
                                                  "role"].to_numpy()
                    sample["provenance"] = np.where(sample.label_name.eq("human"),
                                                    sample.source_name, sample.generator)
                    for (role, label, provenance), group in sample.groupby(
                            ["role", "label_name", "provenance"], dropna=False):
                        provenance_rows.append({
                            "seed": seed, "target": stats["target"],
                            "heldout_generator": generator, "heldout_area": area,
                            "role": role, "label_name": label,
                            "provenance": provenance, "rows": len(group),
                        })
                rows.append(stats)
    tasks = pd.DataFrame(rows).sort_values(
        ["seed", "heldout_generator", "heldout_area"], kind="stable")
    tasks_path = output / "tasks.csv"
    tasks.to_csv(tasks_path, index=False)
    provenance_path = output / "test_provenance.csv"
    pd.DataFrame(provenance_rows, columns=[
        "seed", "target", "heldout_generator", "heldout_area", "role",
        "label_name", "provenance", "rows",
    ]).to_csv(provenance_path, index=False)
    summary = {
        "schema_version": 1, "kind": "n2_compound_generator_domain_ood_manifests",
        "release_id": release["release_id"],
        "release_manifest_sha256": _sha256(release_dir / "release_manifest.json"),
        "selected_index_sha256": release["artifacts"]["selected_index"]["sha256"],
        "parent_ood_dir": str(parent_dir),
        "parent_summary_sha256": _sha256(parent_dir / "summary.json"),
        "parent_tasks_sha256": _sha256(parent_dir / "tasks.csv"),
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
        "seeds": cfg["seeds"], "generator_count": len(generators),
        "area_count": len(areas),
        "counts": {status: int(count) for status, count in
                   tasks.status.value_counts().to_dict().items()},
        "tasks_sha256": _sha256(tasks_path),
        "test_provenance_sha256": _sha256(provenance_path),
        "limitations": [
            "Only generator-area cells observed in the selected N2 release can be evaluated.",
            "Matched binary tests reuse LOGO controls; unmatched targets are retained separately.",
            "Thresholds are recalibrated on each protocol's validation population.",
            "Paired protocol deltas are full-protocol comparisons, not causal interactions.",
            "Human sources within an area may remain concentrated or single-source.",
        ],
    }
    target_path = output / "summary.json"
    target_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return target_path


def check(output: Path) -> Path:
    target_path = output / "summary.json"
    summary = json.loads(target_path.read_text(encoding="utf-8"))
    if (summary.get("kind") != "n2_compound_generator_domain_ood_manifests" or
            summary.get("release_id") != "horus-n2-iid-v1"):
        raise ValueError("Wrong compound OOD summary kind or release")
    cfg_path = Path(summary["config_path"])
    if _sha256(cfg_path) != summary["config_sha256"]:
        raise ValueError("Compound OOD config changed")
    cfg = _load_config(cfg_path)
    release_dir = Path(cfg["release_dir"])
    if _sha256(release_dir / "release_manifest.json") != summary["release_manifest_sha256"]:
        raise ValueError("Frozen release manifest changed")
    _, index, splits = _load_release(release_dir, cfg["seeds"])
    parent_dir = Path(summary["parent_ood_dir"])
    if (_sha256(parent_dir / "summary.json") != summary["parent_summary_sha256"] or
            _sha256(parent_dir / "tasks.csv") != summary["parent_tasks_sha256"]):
        raise ValueError("Parent OOD manifests changed")
    parent_summary, parent_tasks = _parent_tasks(
        parent_dir, summary["release_manifest_sha256"], cfg["seeds"])
    tasks_path = output / "tasks.csv"
    if _sha256(tasks_path) != summary["tasks_sha256"]:
        raise ValueError("Compound task index changed")
    provenance_path = output / "test_provenance.csv"
    if _sha256(provenance_path) != summary["test_provenance_sha256"]:
        raise ValueError("Compound provenance table changed")
    tasks = pd.read_csv(tasks_path, keep_default_na=False)
    if (tasks.duplicated(["seed", "protocol", "target"]).any() or
            not tasks.protocol.eq(PROTOCOL).all() or
            set(tasks.seed) != set(cfg["seeds"])):
        raise ValueError("Compound task index identity is invalid")
    for seed in cfg["seeds"]:
        base = _prepare_base(index, splits[seed], parent_summary["source_aliases"],
                             float(cfg["length_bin_width"]), seed)
        for generator in sorted(tasks.loc[tasks.seed.eq(seed),
                                           "heldout_generator"].unique()):
            parent_task = parent_tasks.loc[
                parent_tasks.seed.eq(seed) & parent_tasks.protocol.eq("logo") &
                parent_tasks.target.eq(generator)]
            if len(parent_task) != 1:
                raise ValueError("Missing parent LOGO during compound check")
            logo = None
            if parent_task.iloc[0].status == "eligible":
                logo = _load_logo(Path(parent_task.iloc[0].manifest_path),
                                  str(parent_task.iloc[0].manifest_sha256), base,
                                  generator, seed)
            for row in tasks.loc[tasks.seed.eq(seed) &
                                 tasks.heldout_generator.eq(generator)].itertuples(index=False):
                if row.status != "eligible":
                    if row.manifest_path or row.manifest_sha256:
                        raise ValueError("Unsupported compound target has a manifest")
                    continue
                if logo is None:
                    raise ValueError("Eligible compound target lacks parent LOGO")
                path = Path(row.manifest_path)
                if _sha256(path) != row.manifest_sha256:
                    raise ValueError(f"Compound manifest changed: {path}")
                frame = pd.read_parquet(path)
                expected, expected_stats = build_one(
                    base, logo, generator, row.heldout_area, cfg, seed)
                if not expected_stats["eligible"]:
                    raise ValueError(f"Compound manifest differs from protocol: {path}")
                pd.testing.assert_frame_equal(
                    frame.loc[:, list(TEST_OUTPUT_COLUMNS)].reset_index(drop=True),
                    expected.loc[:, list(TEST_OUTPUT_COLUMNS)].reset_index(drop=True),
                    check_dtype=False, check_exact=True,
                    obj=f"compound manifest {path}")
    return target_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.config, args.output) if args.command == "prepare" else check(args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
