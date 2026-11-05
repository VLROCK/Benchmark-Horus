"""Freeze the reviewed N2 selection and source-balanced IID splits by hash.

The release index contains identifiers and metadata, never corpus text. The N2
view and three split Parquets remain in their original, hash-pinned run folders.
Run this module on the researcher's cluster, per AGENTS.md section 7.19.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from horus_benchmark.audit.prepare_selection_protocol import _load_inputs
from horus_benchmark.audit.refine_grouped_iid import _provenance_table, provenance_shift
from horus_benchmark.data.freeze_normalization import check as check_n2_freeze
from horus_benchmark.data.normalize_corpus import sha256_file
from horus_benchmark.protocols.grouped_iid import (
    add_strata, check_observed_pairs, split_diagnostics,
)


MANIFEST = "release_manifest.json"
INDEX = "selected_index.parquet"
README = "README.md"
PATH_KEYS = {
    "raw", "n2_run", "selection_run", "v3_run", "baseline_run",
    "v3_config", "baseline_config", "v3_code", "grouping_code",
}
INDEX_COLUMNS = [
    "record_id", "label_name", "broad_area", "source_name", "generator",
    "genre", "word_count_audit", "content_hash", "selection_group_id",
    "probe_group_id", "length_stratum", "stratum",
]
SPLIT_COLUMNS = {
    "record_id", "label_name", "broad_area", "probe_group_id",
    "length_stratum", "stratum", "split", "fold",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(path: Path) -> tuple[dict, dict[str, Path]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("release_id") != "horus-n2-iid-v1":
        raise ValueError("Expected the reviewed horus-n2-iid-v1 release config")
    seeds = config.get("split_seeds")
    if seeds != [20261105, 20261106, 20261107]:
        raise ValueError("The reviewed release requires all three v3 split seeds")
    if set(config.get("expected_split_sha256", {})) != {str(seed) for seed in seeds}:
        raise ValueError("Exactly one pinned split SHA-256 is required per seed")
    raw_paths = config.get("paths")
    if not isinstance(raw_paths, dict) or set(raw_paths) != PATH_KEYS:
        raise ValueError("Release input paths are incomplete")
    paths = {name: Path(value) for name, value in raw_paths.items()}
    if (path.is_absolute() or ".." in path.parts or
            any(p.is_absolute() or ".." in p.parts for p in paths.values())):
        raise ValueError("Use repository-relative paths for a portable freeze")
    return config, paths


def _files(config_path: Path, paths: dict[str, Path], seeds: list[int]) -> dict[str, Path]:
    selection, n2, v3, baseline = (paths[key] for key in
                                    ("selection_run", "n2_run", "v3_run", "baseline_run"))
    files = {
        "release_config": config_path,
        "raw_corpus": paths["raw"],
        "n2_corpus": n2 / "normalized_corpus.parquet",
        "n2_freeze": n2 / "freeze_manifest.json",
        "selection_summary": selection / "summary.json",
        "candidate_membership": selection / "candidate_membership.parquet",
        "lineage_manifest": selection / "lineage_manifest.parquet",
        "near_pair_decisions": selection / "near_pair_decisions.csv",
        "v3_summary": v3 / "summary.json",
        "v3_config": paths["v3_config"],
        "v3_code": paths["v3_code"],
        "grouping_code": paths["grouping_code"],
        "baseline_summary": baseline / "summary.json",
        "baseline_config": paths["baseline_config"],
        "freeze_code": Path("src/horus_benchmark/protocols/freeze_iid.py"),
    }
    files.update({f"split_seed{seed}": v3 / f"iid_holdout_5fold_seed{seed}.parquet"
                  for seed in seeds})
    return files


def _inventory(files: dict[str, Path]) -> dict[str, dict[str, str | int]]:
    result = {}
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty freeze input: {path}")
        result[name] = {"path": str(path), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)}
    return result


def _selected_n2(path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    columns = ["record_id", "label_name", "broad_area", "source_name",
               "generator", "genre", "word_count", "content_hash"]
    n2 = pd.read_parquet(path, columns=columns)
    if n2.record_id.duplicated().any():
        raise ValueError("Duplicate N2 record IDs")
    selected = n2.loc[n2.record_id.isin(set(candidates.record_id))].merge(
        candidates, on="record_id", validate="one_to_one")
    if len(selected) != len(candidates):
        raise ValueError("Selected IDs missing from the N2 corpus")
    if (selected[["label_name", "broad_area", "content_hash", "selection_group_id",
                  "probe_group_id"]].isna().any().any() or
            selected.content_hash.duplicated().any()):
        raise ValueError("Selected N2 metadata is incomplete or has exact text duplicates")
    selected = selected.rename(columns={"word_count": "word_count_audit"})
    return selected


def _same_number(actual: float, expected: float, name: str) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-9):
        raise ValueError(f"Split metric disagrees with v3 summary: {name}")


def validate_split_manifest(split: pd.DataFrame, selected: pd.DataFrame,
                            pairs: pd.DataFrame, expected: dict,
                            folds: int, test_fraction: float,
                            material_min_rows: int) -> dict:
    """Recompute ID, metadata, group, pair and balance checks for one seed."""
    if SPLIT_COLUMNS - set(split):
        raise ValueError(f"Split columns missing: {sorted(SPLIT_COLUMNS - set(split))}")
    if (split.record_id.isna().any() or split.record_id.duplicated().any() or
            len(split) != len(selected) or set(split.record_id) != set(selected.record_id)):
        raise ValueError("Split IDs differ from selected N2 IDs")
    if set(split.split) != {"development", "test"} or not pd.api.types.is_integer_dtype(split.fold):
        raise ValueError("Invalid holdout or fold values")
    dev = split.loc[split.split.eq("development")]
    test = split.loc[split.split.eq("test")]
    if (not test.fold.eq(-1).all() or set(dev.fold) != set(range(folds)) or
            len(test) != expected["test_rows"] or len(split) != expected["rows"]):
        raise ValueError("Holdout or development fold sizes disagree with v3")
    fold_sizes = {str(key): int(value) for key, value in dev.fold.value_counts().items()}
    if fold_sizes != expected["development_fold_sizes"]:
        raise ValueError("Development fold sizes disagree with v3")
    joined = split.merge(selected, on="record_id", validate="one_to_one", suffixes=("", "_n2"))
    for field in ("label_name", "broad_area", "probe_group_id", "length_stratum", "stratum"):
        if not joined[field].eq(joined[f"{field}_n2"]).all():
            raise ValueError(f"Split {field} disagrees with selected N2/lineage")
    for group_field in ("probe_group_id", "selection_group_id"):
        if joined.groupby(group_field)[["split", "fold"]].nunique().gt(1).any().any():
            raise ValueError(f"{group_field} crosses holdout or folds")
    cells = set(zip(selected.broad_area, selected.label_name))
    for partition in [test, *(dev.loc[dev.fold.eq(fold)] for fold in range(folds))]:
        if set(zip(partition.broad_area, partition.label_name)) != cells:
            raise ValueError("An area/class cell is absent from the holdout or a fold")
    pair_check = check_observed_pairs(split, pairs)
    if pair_check != {key: expected[key] for key in pair_check}:
        raise ValueError("Observed near-pair counts disagree with v3")

    outer_strata, _ = split_diagnostics(
        split, {"development": 1 - test_fraction, "test": test_fraction})
    outer_provenance = provenance_shift(_provenance_table(split, selected), material_min_rows)
    fold_frame = dev.copy()
    fold_frame["split"] = fold_frame.fold.map(lambda value: f"fold_{value}")
    fold_strata, _ = split_diagnostics(
        fold_frame, {f"fold_{fold}": 1 / folds for fold in range(folds)})
    fold_provenance = provenance_shift(_provenance_table(fold_frame, selected),
                                       material_min_rows)
    actual = {
        "max_test_stratum_n100_pp": float(100 * outer_strata.loc[
            outer_strata.total_rows.ge(100), "share_minus_target"].abs().max()),
        "max_fold_stratum_n100_pp": float(100 * fold_strata.loc[
            fold_strata.total_rows.ge(100), "share_minus_target"].abs().max()),
        "max_test_material_shift_pp": float(outer_provenance.loc[
            outer_provenance.material_diverse & outer_provenance.split.eq("test"),
            "shift_pp"].abs().max()),
        "max_fold_material_shift_pp": float(fold_provenance.loc[
            fold_provenance.material_diverse, "shift_pp"].abs().max()),
        "material_provenances_missing_test": int(outer_provenance.loc[
            outer_provenance.material_diverse & outer_provenance.split.eq("test"),
            "partition_rows"].eq(0).sum()),
        "material_provenances_missing_folds": int(fold_provenance.loc[
            fold_provenance.material_diverse, "partition_rows"].eq(0).sum()),
        **pair_check,
    }
    for key in ("max_test_stratum_n100_pp", "max_fold_stratum_n100_pp",
                "max_test_material_shift_pp", "max_fold_material_shift_pp"):
        _same_number(actual[key], expected[key], key)
    for key in ("material_provenances_missing_test", "material_provenances_missing_folds"):
        if actual[key] != expected[key]:
            raise ValueError(f"Split metric disagrees with v3 summary: {key}")
    return {"seed": int(expected["seed"]), "rows": len(split), "test_rows": len(test),
            "fold_sizes": fold_sizes, **actual}


def _build(config_path: Path) -> tuple[dict, pd.DataFrame]:
    config, paths = _config(config_path)
    files = _files(config_path, paths, config["split_seeds"])
    inventory = _inventory(files)
    n2_freeze = check_n2_freeze(paths["n2_run"])
    if n2_freeze.resolve() != files["n2_freeze"].resolve():
        raise ValueError("Wrong N2 freeze manifest")
    if (inventory["raw_corpus"]["sha256"] != config["expected_source_sha256"] or
            inventory["n2_corpus"]["sha256"] != config["expected_n2_corpus_sha256"] or
            inventory["selection_summary"]["sha256"] != config["expected_selection_summary_sha256"] or
            inventory["v3_config"]["sha256"] != config["expected_v3_config_sha256"] or
            inventory["v3_code"]["sha256"] != config["expected_v3_code_sha256"] or
            inventory["grouping_code"]["sha256"] != config["expected_grouping_code_sha256"]):
        raise ValueError("An input differs from the reviewed N2/selection/v3 release")
    v3 = _json(files["v3_summary"])
    selection_summary = _json(files["selection_summary"])
    baseline = _json(files["baseline_summary"])
    v3_config = yaml.safe_load(files["v3_config"].read_text(encoding="utf-8"))
    if (v3.get("status") != "provisional_source_balanced_iid_candidates" or
            v3["selection_scenario"] != config["selection_scenario"] or
            v3["rows"] != config["expected_rows"] or
            v3["source_sha256"] != config["expected_source_sha256"] or
            v3["selection_summary_sha256"] != inventory["selection_summary"]["sha256"] or
            v3["baseline_summary_sha256"] != inventory["baseline_summary"]["sha256"] or
            v3["baseline_config_sha256"] != inventory["baseline_config"]["sha256"] or
            v3["config_sha256"] != inventory["v3_config"]["sha256"] or
            v3["code_sha256"] != inventory["v3_code"]["sha256"] or
            v3["grouping_code_sha256"] != inventory["grouping_code"]["sha256"] or
            v3["development_folds"] != 5 or
            v3["test_fraction"] != v3_config["test_fraction"] or
            v3["material_min_rows"] != v3_config["material_min_rows"] or
            baseline.get("status") != "provisional_iid_holdout_cv" or
            baseline["selected_rows"] != config["expected_rows"] or
            baseline["candidate_membership_sha256"] != inventory["candidate_membership"]["sha256"] or
            baseline["lineage_sha256"] != inventory["lineage_manifest"]["sha256"] or
            baseline["n2_corpus_sha256"] != inventory["n2_corpus"]["sha256"] or
            selection_summary["source_sha256"] != config["expected_source_sha256"] or
            selection_summary["n2_corpus_sha256"] != config["expected_n2_corpus_sha256"]):
        raise ValueError("Release summaries, configs or source versions disagree")
    by_seed = {int(item["seed"]): item for item in v3["split_candidates"]}
    if (len(by_seed) != len(v3["split_candidates"]) or
            set(by_seed) != set(config["split_seeds"]) or
            v3_config["split_seeds"] != config["split_seeds"]):
        raise ValueError("V3 seeds differ from the reviewed release")
    scenario = next((item for item in selection_summary["scenarios"]
                     if item["scenario"] == config["selection_scenario"]), None)
    if (scenario is None or scenario["rows"] != config["expected_rows"] or
            scenario["per_class"] != config["expected_per_class"] or
            scenario["coverage_gate_met"] is not True):
        raise ValueError("Selected scenario differs from the reviewed candidate")
    candidates, pairs, _ = _load_inputs(paths["selection_run"],
                                        config["selection_scenario"], files["n2_corpus"])
    selected = _selected_n2(files["n2_corpus"], candidates)
    if (len(selected) != config["expected_rows"] or
            selected.label_name.value_counts().to_dict() !=
            {"human": config["expected_per_class"],
             "synthetic": config["expected_per_class"]} or
            selected.broad_area.nunique() != 6 or
            selected.groupby("broad_area").label_name.nunique().ne(2).any()):
        raise ValueError("Selected N2 population differs from the reviewed candidate")
    area_class = selected.groupby(["broad_area", "label_name"]).size().unstack()
    if (area_class.human.ne(area_class.synthetic).any() or
            selected.groupby("probe_group_id").label_name.nunique().gt(1).any()):
        raise ValueError("Area/class balance or group label integrity changed")
    selected = add_strata(selected, list(v3_config["coarse_word_edges"]))
    checks = []
    for seed in config["split_seeds"]:
        expected = by_seed[seed]
        name = f"split_seed{seed}"
        digest = inventory[name]["sha256"]
        if (digest != config["expected_split_sha256"][str(seed)] or
                digest != expected["split_manifest_sha256"] or
                expected["exploratory_gates_met"] is not True):
            raise ValueError(f"Seed {seed} is not the approved v3 split candidate")
        split = pd.read_parquet(files[name])
        checks.append(validate_split_manifest(
            split, selected, pairs, expected, int(v3["development_folds"]),
            float(v3["test_fraction"]), int(v3["material_min_rows"])))
    index = selected[INDEX_COLUMNS].sort_values("record_id").reset_index(drop=True)
    ids_digest = hashlib.sha256("".join(f"{rid}\n" for rid in index.record_id).encode(
        "utf-8")).hexdigest()
    payload = {
        "schema_version": 1,
        "kind": "frozen_n2_source_balanced_iid_release",
        "verified": True,
        "release_id": config["release_id"],
        "config_path": str(config_path),
        "selection_scenario": config["selection_scenario"],
        "text_view": "n2",
        "rows": len(index),
        "per_class": config["expected_per_class"],
        "area_class_rows": {area: {label: int(count) for label, count in row.items()}
                            for area, row in area_class.iterrows()},
        "selected_ids_sha256": ids_digest,
        "split_seeds": config["split_seeds"],
        "test_fraction": v3["test_fraction"],
        "development_folds": v3["development_folds"],
        "validation": checks,
        "inputs": inventory,
        "limitations": [
            "This is an IID data/protocol freeze, not a detector or OOD result.",
            "Observed near-pair checks do not prove exhaustive duplicate linkage.",
            "Balancing provenance within area and class does not remove class/source confounding.",
            "N2 source text remains in its pinned run folder; split copies contain no text.",
        ],
    }
    return payload, index


def freeze(config_path: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Release directory already exists: {output}")
    payload, index = _build(config_path)
    if output.name != payload["release_id"]:
        raise ValueError("Output directory must match release_id")
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pandas(index, preserve_index=False), output / INDEX,
                   compression="zstd")
    copied_splits = {}
    for seed in payload["split_seeds"]:
        name = f"split_seed{seed}"
        destination = output / f"{name}.parquet"
        shutil.copyfile(payload["inputs"][name]["path"], destination)
        copied_splits[name] = destination
    (output / README).write_text(
        "# Hórus N2 IID v1\n\n"
        "Seleção `w0.25_s20261103` sobre N2, com teste reservado de 15% e "
        "cinco folds de desenvolvimento por seed. Cópias dos três splits v3 "
        "estão neste diretório e são verificadas por SHA-256; os textos "
        "permanecem no corpus N2. `selected_index.parquet` contém IDs e "
        "metadados principais, sem conteúdo textual. Execute "
        "`python -m horus_benchmark.protocols.freeze_iid check "
        "--release-dir results/releases/horus-n2-iid-v1` antes de usar o release. "
        "Esta versão não inclui avaliações de detectores "
        "ou protocolos OOD.\n", encoding="utf-8")
    payload["artifacts"] = _inventory({
        "selected_index": output / INDEX, "readme": output / README,
        **copied_splits})
    for name in copied_splits:
        if payload["artifacts"][name]["sha256"] != payload["inputs"][name]["sha256"]:
            raise ValueError(f"Copied split differs from source: {name}")
    target = output / MANIFEST
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return target


def check(output: Path) -> Path:
    target = output / MANIFEST
    frozen = _json(target)
    if (frozen.get("schema_version") != 1 or
            frozen.get("kind") != "frozen_n2_source_balanced_iid_release" or
            frozen.get("verified") is not True):
        raise ValueError("Unsupported IID release manifest")
    fresh, _ = _build(Path(frozen["config_path"]))
    if output.name != fresh["release_id"]:
        raise ValueError("Release directory differs from the frozen release_id")
    fresh["artifacts"] = _inventory({
        "selected_index": output / INDEX, "readme": output / README,
        **{f"split_seed{seed}": output / f"split_seed{seed}.parquet"
           for seed in fresh["split_seeds"]}})
    for seed in fresh["split_seeds"]:
        name = f"split_seed{seed}"
        if fresh["artifacts"][name]["sha256"] != fresh["inputs"][name]["sha256"]:
            raise ValueError(f"Frozen split differs from source: {name}")
    if frozen != fresh:
        raise ValueError("Frozen IID release inputs or artifacts changed")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_cmd = sub.add_parser("freeze")
    freeze_cmd.add_argument("--config", type=Path, required=True)
    freeze_cmd.add_argument("--output", type=Path, required=True)
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    target = (freeze(args.config, args.output) if args.command == "freeze" else
              check(args.release_dir))
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
