"""Build and verify clean-linked perturbations of frozen IID test documents.

The researcher executes this module on the cluster. All three frozen IID test
sets are represented by their union, while `test_membership.parquet` retains
which seed actually reserves a parent for testing. Neither train nor validation
texts enter these derived attack files. Each attack is applied to clean N2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from horus_benchmark.evaluation.benchmark_runner import sha256_file
from horus_benchmark.robustness.attacks import ATTACKS, perturb, sha256_text


SEEDS = [20261105, 20261106, 20261107]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pin(path: Path, digest: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"Missing or changed pinned input: {path}")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {name: ([1.0] if name == "insert_paragraphs" else [.01, .05])
                for name in ATTACKS}
    if (not isinstance(config, dict) or config.get("schema_version") != 1 or
            config.get("experiment_id") != "horus-n2-query-free-robustness-v1" or
            config.get("release_id") != "horus-n2-iid-v1" or
            config.get("split_seeds") != SEEDS or config.get("attacks") != expected or
            config.get("fpr_targets") != [.01, .05] or
            int(config.get("bootstrap_replicates", 0)) != 2000 or
            float(config.get("bootstrap_confidence", 0)) != .95):
        raise ValueError("Unreviewed robustness configuration")
    return config


def _inputs(release_dir: Path, texts_path: Path) -> tuple[pd.DataFrame, pd.DataFrame,
                                                        dict[str, Any]]:
    path = release_dir / "release_manifest.json"
    release = _json(path)
    if (release.get("verified") is not True or
            release.get("release_id") != "horus-n2-iid-v1" or
            release.get("text_view") != "n2" or release.get("split_seeds") != SEEDS):
        raise ValueError("Robustness requires the frozen N2 IID release")
    index_path = release_dir / "selected_index.parquet"
    _pin(index_path, release["artifacts"]["selected_index"]["sha256"])
    index = pd.read_parquet(index_path)
    texts = pd.read_parquet(texts_path)
    required = {"record_id", "content", "content_hash", "label_name", "probe_group_id",
                "broad_area", "source_name", "generator"}
    if (required - set(texts) or texts.record_id.duplicated().any() or
            set(texts.record_id) != set(index.record_id) or len(texts) != release["rows"] or
            texts.content.isna().any() or
            not texts.content.map(lambda value: isinstance(value, str)).all()):
        raise ValueError("Selected N2 cache IDs/schema differ from the release")
    aligned = texts[["record_id", "content_hash", "label_name", "probe_group_id"]].merge(
        index[["record_id", "content_hash", "label_name", "probe_group_id"]],
        on="record_id", validate="one_to_one", suffixes=("_text", "_index"))
    if any(not aligned[f"{field}_text"].eq(aligned[f"{field}_index"]).all()
           for field in ("content_hash", "label_name", "probe_group_id")):
        raise ValueError("Selected N2 cache provenance or hashes changed")
    if not texts.content.map(sha256_text).eq(texts.content_hash).all():
        raise ValueError("Selected N2 cache content hash mismatch")
    parts = []
    inputs: dict[str, Any] = {
        "release_manifest": {"path": str(path), "sha256": sha256_file(path)},
        "selected_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "texts": {"path": str(texts_path), "sha256": sha256_file(texts_path)},
    }
    for seed in SEEDS:
        name = f"split_seed{seed}"
        split_path = release_dir / f"{name}.parquet"
        _pin(split_path, release["artifacts"][name]["sha256"])
        split = pd.read_parquet(split_path)
        if ({"record_id", "split", "fold", "probe_group_id"} - set(split) or
                split.record_id.duplicated().any() or
                set(split.record_id) != set(texts.record_id) or
                split.groupby("probe_group_id").split.nunique().gt(1).any() or
                split.groupby("probe_group_id").fold.nunique().gt(1).any()):
            raise ValueError(f"Invalid frozen split: {seed}")
        chosen = split.loc[split.split.eq("test"), ["record_id", "probe_group_id"]].copy()
        if not chosen.record_id.map(texts.set_index("record_id").probe_group_id).eq(
                chosen.probe_group_id).all():
            raise ValueError(f"Frozen test group identity changed: {seed}")
        chosen["seed"] = seed
        parts.append(chosen)
        inputs[name] = {"path": str(split_path), "sha256": sha256_file(split_path)}
    membership = pd.concat(parts, ignore_index=True).sort_values(["seed", "record_id"])
    selected = texts.loc[texts.record_id.isin(set(membership.record_id))].sort_values(
        "record_id", kind="stable").reset_index(drop=True)
    return selected, membership, inputs


def condition_name(attack: str, intensity: float) -> str:
    return f"{attack}__{intensity:.8g}".replace(".", "p")


def prepare(config_path: Path, release_dir: Path, texts_path: Path, output: Path) -> Path:
    config = load_config(config_path)
    if output.exists():
        raise FileExistsError(f"Attack output already exists: {output}")
    selected, membership, inputs = _inputs(release_dir, texts_path)
    output.mkdir(parents=True, exist_ok=False)
    membership_path = output / "test_membership.parquet"
    membership.to_parquet(membership_path, index=False, compression="zstd")
    artifacts: dict[str, Any] = {
        "test_membership": {"path": str(membership_path),
                             "sha256": sha256_file(membership_path),
                             "rows": len(membership)}}
    coverage: list[dict[str, Any]] = []
    metadata = selected[["record_id", "label_name", "probe_group_id", "broad_area",
                         "source_name", "generator"]]
    for attack, intensities in config["attacks"].items():
        for intensity in intensities:
            name = condition_name(attack, float(intensity))
            transformed = pd.DataFrame([
                perturb(str(row.content), str(row.record_id), attack, float(intensity),
                        seed=int(config["attack_seed"]))
                for row in selected.itertuples(index=False)])
            frame = metadata.merge(transformed, on="record_id", validate="one_to_one")
            path = output / f"{name}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            artifacts[name] = {"path": str(path), "sha256": sha256_file(path),
                               "rows": len(frame), "attack": attack,
                               "intensity": float(intensity), "kind": "attack_texts"}
            for dimension, columns in (("class", ["label_name"]),
                                        ("class_area", ["label_name", "broad_area"]),
                                        ("class_source", ["label_name", "source_name"]),
                                        ("class_generator", ["label_name", "generator"])):
                for key, part in frame.groupby(columns, sort=True, dropna=False):
                    values = key if isinstance(key, tuple) else (key,)
                    coverage.append({
                        "condition": name, "attack": attack, "intensity": intensity,
                        "dimension": dimension, **dict(zip(columns, values)),
                        "rows": len(part), "changed_rows": int(part.status.eq("changed").sum()),
                        "no_op_rows": int(part.status.eq("no_op").sum()),
                        "invalid_rows": int(part.status.eq("invalid").sum()),
                        "mean_eligible_positions": float(part.eligible_positions.mean()),
                        "mean_edits": float(part.num_edits.mean()),
                        "mean_effective_fraction": float(part.effective_fraction.mean()),
                    })
    coverage_path = output / "attack_coverage.csv"
    pd.DataFrame(coverage).to_csv(coverage_path, index=False)
    artifacts["coverage"] = {"path": str(coverage_path),
                              "sha256": sha256_file(coverage_path), "rows": len(coverage)}
    manifest = {
        "schema_version": 1, "kind": "n2_iid_query_free_attack_views",
        "release_id": config["release_id"], "config_path": str(config_path),
        "config_sha256": sha256_file(config_path), "inputs": inputs,
        "split_seeds": SEEDS, "test_union_rows": len(selected),
        "membership_rows": len(membership), "artifacts": artifacts,
        "note": "Each attack starts from clean N2. Preserve no-ops and failures in denominators. "
                "Sparse PT-BR variants; not a literal reproduction of RAID intensities.",
    }
    manifest_path = output / "attack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest_path


def check(output: Path) -> dict[str, Any]:
    manifest = _json(output / "attack_manifest.json")
    if (manifest.get("kind") != "n2_iid_query_free_attack_views" or
            manifest.get("release_id") != "horus-n2-iid-v1"):
        raise ValueError("Unexpected attack manifest")
    config_path = Path(manifest["config_path"])
    _pin(config_path, manifest["config_sha256"])
    config = load_config(config_path)
    for entry in manifest["inputs"].values():
        _pin(Path(entry["path"]), entry["sha256"])
    selected, membership, _ = _inputs(
        Path(manifest["inputs"]["release_manifest"]["path"]).parent,
        Path(manifest["inputs"]["texts"]["path"]))
    source = selected.set_index("record_id")
    for name, entry in manifest["artifacts"].items():
        _pin(Path(entry["path"]), entry["sha256"])
        if entry.get("kind") != "attack_texts":
            continue
        frame = pd.read_parquet(entry["path"])
        if (len(frame) != len(source) or frame.record_id.duplicated().any() or
                set(frame.record_id) != set(source.index) or
                not frame.parent_record_id.eq(frame.record_id).all()):
            raise ValueError(f"Invalid attack parent identities: {name}")
        for row in frame.itertuples(index=False):
            parent = source.loc[row.record_id]
            expected = perturb(parent.content, str(row.record_id), entry["attack"],
                               float(entry["intensity"]), seed=int(config["attack_seed"]))
            if (any(getattr(row, key) != value for key, value in expected.items()) or
                    row.label_name != parent.label_name or
                    row.probe_group_id != parent.probe_group_id or
                    row.broad_area != parent.broad_area):
                raise ValueError(f"Attack differs from fixed transformation: {name}/{row.record_id}")
    saved_membership = pd.read_parquet(manifest["artifacts"]["test_membership"]["path"])
    pd.testing.assert_frame_equal(
        saved_membership.reset_index(drop=True), membership.reset_index(drop=True))
    return {"verified": True, "test_union_rows": len(source),
            "conditions": sum(entry.get("kind") == "attack_texts"
                              for entry in manifest["artifacts"].values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--config", type=Path, required=True)
    prepare_cmd.add_argument("--release-dir", type=Path, required=True)
    prepare_cmd.add_argument("--texts", type=Path, required=True)
    prepare_cmd.add_argument("--output", type=Path, required=True)
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.config, args.release_dir, args.texts, args.output))
    else:
        print(json.dumps(check(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
