"""Prepare and run frozen N2 IID/OOD text-only detector tasks on Slurm.

This CLI is executed by the researcher on the cluster. It never edits raw/N2
inputs, and it never uses provenance metadata as a detector feature. One task
fits one detector on one seed/fold/protocol target and writes scores by ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from horus_benchmark.metrics.binary_detection import evaluate_binary, metric_values


TEXT_MODELS = (
    "surface_length", "surface_presentation", "surface_punctuation",
    "surface_all", "word_lr", "char4_lr", "char4_svm", "human_iforest",
)
ALL_MODELS = (*TEXT_MODELS, "bertimbau")
PROTOCOLS = ("iid", "logo", "lodo", "loso", "logo_lodo")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(cfg, dict) or cfg.get("schema_version") != 1 or
            cfg.get("experiment_id") not in {
                "horus-n2-iid-ood-text-evaluation-v1",
                "horus-n2-compound-ood-text-evaluation-v1",
            }):
        raise ValueError("Unexpected text evaluation config")
    if cfg["seeds"] != [20261105, 20261106, 20261107] or cfg["folds"] != list(range(5)):
        raise ValueError("Configuration must use all frozen seeds and five folds")
    if cfg["bertimbau_ood_folds"] != [0] or cfg["fpr_targets"] != [0.01, 0.05]:
        raise ValueError("Unreviewed neural OOD fold or FPR targets")
    if int(cfg["bootstrap_replicates"]) < 1 or float(cfg["bootstrap_confidence"]) != 0.95:
        raise ValueError("Bootstrap must use positive repetitions and reviewed 95% intervals")
    return cfg


def _release(cfg: dict[str, Any]) -> tuple[Path, dict]:
    release = Path(cfg["release_dir"])
    manifest = _json(release / "release_manifest.json")
    if (manifest.get("verified") is not True or
            manifest.get("release_id") != "horus-n2-iid-v1" or
            manifest.get("rows") != 107566 or
            manifest.get("split_seeds") != cfg["seeds"]):
        raise ValueError("Evaluation requires the frozen horus-n2-iid-v1 release")
    return release, manifest


def _check_pinned_file(path: Path, expected_sha: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"Missing or changed pinned input: {path}")


def prepare_texts(config_path: Path, output: Path, *, with_n3s: bool) -> Path:
    """Materialize selected text once; verify IDs, N2 hashes and source bytes."""
    cfg = _config(config_path)
    release, frozen = _release(cfg)
    if output.exists():
        raise FileExistsError(f"Prepared text directory already exists: {output}")
    index_path = release / "selected_index.parquet"
    n2_path = Path(cfg["n2_corpus"])
    _check_pinned_file(index_path, frozen["artifacts"]["selected_index"]["sha256"])
    _check_pinned_file(n2_path, frozen["inputs"]["n2_corpus"]["sha256"])
    index = pd.read_parquet(index_path)
    if index.record_id.duplicated().any() or len(index) != frozen["rows"]:
        raise ValueError("Selected index is not the reviewed release")
    content = pd.read_parquet(
        n2_path, columns=["record_id", "content", "content_hash", "label_name", "broad_area"])
    content = content.loc[content.record_id.isin(set(index.record_id))]
    selected = index.merge(content, on="record_id", validate="one_to_one",
                           suffixes=("", "_n2"))
    if (len(selected) != len(index) or selected.record_id.duplicated().any() or
            not selected.label_name.eq(selected.label_name_n2).all() or
            not selected.broad_area.eq(selected.broad_area_n2).all() or
            not selected.content_hash.eq(selected.content_hash_n2).all()):
        raise ValueError("N2 selected content disagrees with the release index")
    computed = selected.content.map(lambda value: hashlib.sha256(
        value.encode("utf-8")).hexdigest())
    if not computed.eq(selected.content_hash).all():
        raise ValueError("A selected N2 text disagrees with its content hash")
    selected = selected.drop(columns=["content_hash_n2", "label_name_n2", "broad_area_n2"])
    n3s = None
    selection_summary = None
    if with_n3s:
        n3s_path = Path(cfg["n3s_corpus"])
        selection_summary = _json(Path(cfg["selection_summary"]))
        _check_pinned_file(n3s_path, selection_summary["n3s_corpus_sha256"])
        n3s = pd.read_parquet(n3s_path, columns=["record_id", "content"])
        n3s = n3s.loc[n3s.record_id.isin(set(index.record_id))]
        n3s = index[["record_id", "label_name"]].merge(n3s, on="record_id",
                                                       validate="one_to_one")
        if len(n3s) != len(index) or n3s.content.isna().any():
            raise ValueError("N3s is missing selected texts")
        human_check = n3s.loc[n3s.label_name.eq("human"), ["record_id", "content"]].merge(
            selected.loc[selected.label_name.eq("human"), ["record_id", "content"]],
            on="record_id", validate="one_to_one", suffixes=("_n3s", "_n2"))
        if not human_check.content_n3s.eq(human_check.content_n2).all():
            raise ValueError("N3s changed a selected human text")
    output.mkdir(parents=True, exist_ok=False)
    text_path = output / "selected_n2_texts.parquet"
    pq.write_table(pa.Table.from_pandas(selected.sort_values("record_id"),
                                        preserve_index=False), text_path,
                   compression="zstd")
    prepared: dict[str, Any] = {
        "schema_version": 1, "kind": "selected_n2_text_cache",
        "release_id": frozen["release_id"],
        "release_manifest_sha256": sha256_file(release / "release_manifest.json"),
        "n2_corpus_sha256": frozen["inputs"]["n2_corpus"]["sha256"],
        "rows": len(selected),
        "selected_n2_texts": {"path": str(text_path), "sha256": sha256_file(text_path)},
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
    }
    if with_n3s:
        assert n3s is not None and selection_summary is not None
        n3s_output = output / "selected_n3s_texts.parquet"
        pq.write_table(pa.Table.from_pandas(n3s[["record_id", "content"]]
                                            .sort_values("record_id"),
                                            preserve_index=False), n3s_output,
                       compression="zstd")
        prepared["selected_n3s_texts"] = {
            "path": str(n3s_output), "sha256": sha256_file(n3s_output),
            "source_sha256": selection_summary["n3s_corpus_sha256"]}
    manifest_path = output / "prepared_manifest.json"
    manifest_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest_path


def _task_id(row: dict[str, Any]) -> str:
    identity = {key: str(row[key]) for key in
                ("protocol", "target", "model", "manifest_path", "manifest_sha256")}
    identity["seed"] = int(row["seed"])
    identity["fold"] = int(row["fold"])
    identity["counterfactual_n3s"] = str(row["counterfactual_n3s"]).lower() == "true"
    material = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def create_plan(config_path: Path, output: Path, *, models: list[str],
                protocols: list[str], counterfactual: bool = False,
                seeds: list[int] | None = None,
                folds: list[int] | None = None) -> Path:
    cfg = _config(config_path)
    release, frozen = _release(cfg)
    if output.exists():
        raise FileExistsError(f"Plan already exists: {output}")
    if not models or set(models) - set(ALL_MODELS) or not protocols or set(protocols) - set(PROTOCOLS):
        raise ValueError("Unknown or empty model/protocol selection")
    if counterfactual and "bertimbau" in models:
        raise ValueError("N3s inference-only counterfactual is implemented for text baselines")
    selected_seeds = cfg["seeds"] if seeds is None else seeds
    selected_folds = cfg["folds"] if folds is None else folds
    if (not selected_seeds or not selected_folds or
            set(selected_seeds) - set(cfg["seeds"]) or
            set(selected_folds) - set(cfg["folds"])):
        raise ValueError("Plan filters must be subsets of frozen seeds/folds")
    rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        if "iid" in protocols:
            path = release / f"split_seed{seed}.parquet"
            _check_pinned_file(path, frozen["artifacts"][f"split_seed{seed}"]["sha256"])
            digest = sha256_file(path)
            for model in models:
                for fold in selected_folds:
                    rows.append({"protocol": "iid", "target": "all", "seed": seed,
                                 "fold": fold, "model": model, "manifest_path": str(path),
                                 "manifest_sha256": digest,
                                 "counterfactual_n3s": bool(counterfactual)})
    if set(protocols) - {"iid"}:
        index_path = Path(cfg["ood_index"])
        ood = pd.read_csv(index_path, keep_default_na=False)
        required = {"seed", "protocol", "target", "status", "manifest_path", "manifest_sha256"}
        if required - set(ood):
            raise ValueError(f"OOD index is missing {sorted(required - set(ood))}")
        if ood.duplicated(["seed", "protocol", "target"]).any():
            raise ValueError("OOD target index has duplicate tasks")
        for item in ood.itertuples(index=False):
            if item.status != "eligible" or item.protocol not in protocols:
                continue
            if int(item.seed) not in cfg["seeds"]:
                raise ValueError("OOD index has an unreviewed seed")
            if int(item.seed) not in selected_seeds:
                continue
            path = Path(item.manifest_path)
            _check_pinned_file(path, str(item.manifest_sha256))
            for model in models:
                allowed_folds = (cfg["bertimbau_ood_folds"] if model == "bertimbau"
                                 else cfg["folds"])
                for fold in sorted(set(allowed_folds) & set(selected_folds)):
                    rows.append({"protocol": item.protocol, "target": item.target,
                                 "seed": int(item.seed), "fold": fold,
                                 "model": model, "manifest_path": str(path),
                                 "manifest_sha256": str(item.manifest_sha256),
                                 "counterfactual_n3s": bool(counterfactual)})
    if not rows:
        raise ValueError("No eligible tasks for the requested plan")
    plan = pd.DataFrame(rows)
    plan.insert(0, "task_id", [_task_id(row) for row in rows])
    if plan.task_id.duplicated().any():
        raise ValueError("Duplicate tasks in plan")
    output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output, index=False)
    return output


def partition_task(manifest: pd.DataFrame, fold: int, protocol: str) -> pd.DataFrame:
    """Assign roles without resplitting or inspecting text/model scores."""
    if not 0 <= fold < 5 or protocol not in PROTOCOLS:
        raise ValueError("Invalid fold or protocol")
    frame = manifest.copy()
    if protocol == "iid":
        if set(frame.split) != {"development", "test"}:
            raise ValueError("Invalid frozen IID split")
        frame["role"] = "train"
        frame.loc[frame.split.eq("development") & frame.fold.eq(fold), "role"] = "validation"
        frame.loc[frame.split.eq("test"), "role"] = "test_target"
        frame["paired_iid_test"] = frame.split.eq("test")
    else:
        required = {"role", "fold", "paired_iid_test", "record_id", "probe_group_id"}
        if required - set(frame):
            raise ValueError("OOD manifest lacks partition columns")
        frame.loc[frame.role.eq("development") & frame.fold.ne(fold), "role"] = "train"
        frame.loc[frame.role.eq("development") & frame.fold.eq(fold), "role"] = "validation"
    model_rows = frame.loc[frame.role.isin(
        ["train", "validation", "test_target", "test_control", "test_unmatched_target"])]
    if model_rows.record_id.duplicated().any():
        raise ValueError("Duplicate IDs in experiment partition")
    partitions = model_rows.role.where(
        model_rows.role.isin(["train", "validation"]), "test")
    by_group = model_rows.assign(partition=partitions).groupby(
        "probe_group_id").partition.nunique()
    if by_group.gt(1).any():
        raise ValueError("Probe group crosses train, validation or test")
    return frame


def _labels(frame: pd.DataFrame) -> np.ndarray:
    if set(frame.label_name) - {"human", "synthetic"}:
        raise ValueError("Unknown label")
    return frame.label_name.eq("synthetic").to_numpy(dtype=np.int8)


def _digest_ids(frame: pd.DataFrame) -> str:
    return hashlib.sha256("".join(f"{rid}\n" for rid in sorted(frame.record_id))
                          .encode("utf-8")).hexdigest()


def run_task(config_path: Path, prepared_dir: Path, plan_path: Path,
             task_index: int, output_root: Path) -> Path:
    cfg = _config(config_path)
    prepared = _json(prepared_dir / "prepared_manifest.json")
    if (prepared.get("kind") != "selected_n2_text_cache" or
            prepared.get("config_sha256") != sha256_file(config_path)):
        raise ValueError("Prepared texts/config mismatch")
    release, frozen = _release(cfg)
    if prepared["release_manifest_sha256"] != sha256_file(release / "release_manifest.json"):
        raise ValueError("Prepared texts reference a different IID release")
    plan = pd.read_csv(plan_path, keep_default_na=False)
    if task_index < 0 or task_index >= len(plan):
        raise IndexError("Task index outside plan")
    task = plan.iloc[task_index].to_dict()
    if _task_id(task) != task["task_id"] or task["model"] not in ALL_MODELS:
        raise ValueError("Plan task was edited")
    manifest_path = Path(task["manifest_path"])
    _check_pinned_file(manifest_path, task["manifest_sha256"])
    text_path = Path(prepared["selected_n2_texts"]["path"])
    _check_pinned_file(text_path, prepared["selected_n2_texts"]["sha256"])
    selected = pd.read_parquet(text_path)
    source_manifest = pd.read_parquet(manifest_path)
    source_manifest = partition_task(source_manifest, int(task["fold"]), task["protocol"])
    if (len(source_manifest) != len(selected) or
            set(source_manifest.record_id) != set(selected.record_id)):
        raise ValueError("Task manifest and selected text IDs differ")
    frame = selected.merge(source_manifest[["record_id", "probe_group_id", "label_name",
                                            "role", "fold", "paired_iid_test"]],
                           on="record_id", validate="one_to_one", suffixes=("", "_manifest"))
    if (not frame.probe_group_id.eq(frame.probe_group_id_manifest).all() or
            not frame.label_name.eq(frame.label_name_manifest).all()):
        raise ValueError("Partition metadata disagrees with frozen release")
    train = frame.loc[frame.role.eq("train")].sort_values("record_id")
    validation = frame.loc[frame.role.eq("validation")].sort_values("record_id")
    test = frame.loc[frame.role.isin(
        ["test_target", "test_control", "test_unmatched_target"])].sort_values("record_id")
    binary_test = test.loc[test.role.isin(["test_target", "test_control"])]
    if any(set(_labels(part)) != {0, 1} for part in (train, validation, binary_test)):
        raise ValueError("Train, validation and binary test must contain both classes")
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / task["task_id"]
    if task_dir.exists():
        if (task_dir / "metrics.json").exists():
            raise FileExistsError(f"Task result already exists: {task_dir}")
        # Only this runner's own, incomplete task directory can be discarded
        # for a clean Slurm retry. Never remove a completed result or an input.
        if task_dir.resolve().parent != output_root.resolve():
            raise ValueError("Incomplete task directory escapes the output root")
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=False)
    # The same model/seed/fold starts IID and OOD with the same random state;
    # target-specific data changes, not an avoidable initialization draw.
    seed_material = f"{task['model']}|{task['seed']}|{task['fold']}"
    model_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
    x_train = train.content.tolist()
    x_validation = validation.content.tolist()
    x_test = test.content.tolist()
    y_train, y_validation = _labels(train), _labels(validation)
    if task["model"] == "bertimbau":
        from horus_benchmark.detectors.bertimbau_baseline import train_and_score
        model_cfg = yaml.safe_load(Path(cfg["bertimbau_config"]).read_text(encoding="utf-8"))
        fitted = train_and_score(x_train, y_train.tolist(), x_validation,
                                 y_validation.tolist(), x_test, model_cfg,
                                 task_dir, model_seed)
        model_info = fitted["metadata"]
    else:
        from horus_benchmark.detectors.text_baselines import (
            fit_score, linear_coefficients, load_config, score_texts)
        model_cfg = load_config(cfg["text_model_config"])
        fitted = fit_score(task["model"], x_train, y_train, x_validation, x_test,
                           seed=model_seed, inner_folds=train.fold.to_numpy(),
                           config=model_cfg)
        model_info = {"selected_c": fitted["selected_c"],
                      "inner_auc_by_c": fitted["inner_auc_by_c"]}
        coefficients = linear_coefficients(fitted["estimator"])
        if coefficients:
            coefficients.sort(key=lambda item: item[1])
            pd.DataFrame(coefficients[:50] + coefficients[-50:],
                         columns=["feature", "signed_coefficient"]).to_csv(
                             task_dir / "extreme_coefficients.csv", index=False)
    validation_scores = np.asarray(fitted["validation_scores"], dtype=float)
    test_scores = np.asarray(fitted["test_scores"], dtype=float)
    if len(validation_scores) != len(validation) or len(test_scores) != len(test):
        raise ValueError("Fitted model returned unaligned scores")
    test = test.copy()
    test["score_n2"] = test_scores
    matched = test.role.isin(["test_target", "test_control"]).to_numpy()
    evaluation = evaluate_binary(
        y_validation, validation_scores, _labels(test.loc[matched]),
        test_scores[matched], test.loc[matched, "probe_group_id"].to_numpy(),
        fpr_targets=tuple(float(item) for item in cfg["fpr_targets"]),
        bootstrap_replicates=int(cfg["bootstrap_replicates"]),
        bootstrap_seed=model_seed + 1)
    paired = test.paired_iid_test.astype(bool).to_numpy() & matched
    paired_eval = None
    if paired.any() and set(_labels(test.loc[paired])) == {0, 1}:
        paired_eval = metric_values(
            _labels(test.loc[paired]), test_scores[paired],
            evaluation["validation_thresholds"])
    unmatched = test.role.eq("test_unmatched_target").to_numpy()
    unmatched_rates = {}
    if unmatched.any():
        yy = _labels(test.loc[unmatched])
        for key, threshold in evaluation["validation_thresholds"].items():
            unmatched_rates[f"positive_rate_{key}"] = float(
                (test_scores[unmatched] >= threshold).mean())
        unmatched_rates["target_label"] = "synthetic" if yy[0] else "human"
        if len(set(yy)) != 1:
            raise ValueError("Unmatched OOD targets span labels")
    counterfactual = None
    if str(task["counterfactual_n3s"]).lower() == "true":
        if task["model"] == "bertimbau" or "selected_n3s_texts" not in prepared:
            raise ValueError("N3s text-only counterfactual is unavailable")
        sidecar = Path(prepared["selected_n3s_texts"]["path"])
        _check_pinned_file(sidecar, prepared["selected_n3s_texts"]["sha256"])
        n3s = pd.read_parquet(sidecar)
        test_n3s = test[["record_id"]].merge(n3s, on="record_id", validate="one_to_one")
        if not test_n3s.record_id.eq(test.record_id.reset_index(drop=True)).all():
            raise ValueError("N2/N3s test IDs misaligned")
        n3s_scores = score_texts(fitted["estimator"], task["model"],
                                 test_n3s.content.tolist())
        test["score_n3s"] = n3s_scores
        counterfactual = metric_values(
            _labels(test.loc[matched]), n3s_scores[matched],
            evaluation["validation_thresholds"])
    columns = ["record_id", "label_name", "broad_area", "source_name", "generator",
               "word_count_audit", "probe_group_id", "role", "paired_iid_test", "score_n2"]
    if "score_n3s" in test:
        columns.append("score_n3s")
    prediction_path = task_dir / "predictions.parquet"
    pq.write_table(pa.Table.from_pandas(test[columns], preserve_index=False),
                   prediction_path, compression="zstd")
    payload = {
        "schema_version": 1, "task": task, "model_seed": model_seed,
        "input_hashes": {"release_manifest": prepared["release_manifest_sha256"],
                         "selected_texts": prepared["selected_n2_texts"]["sha256"],
                         "task_manifest": task["manifest_sha256"],
                         "evaluation_config": sha256_file(config_path),
                         "model_config": sha256_file(Path(
                             cfg["bertimbau_config"] if task["model"] == "bertimbau"
                             else cfg["text_model_config"]))},
        "counts": {"train": len(train), "validation": len(validation),
                   "test_scored": len(test), "binary_test": len(binary_test),
                   "paired_binary_test": int(paired.sum()),
                   "unmatched_target": int(unmatched.sum())},
        "id_sha256": {"train": _digest_ids(train),
                      "validation": _digest_ids(validation),
                      "test": _digest_ids(test)},
        "model_info": model_info, "evaluation": evaluation,
        "paired_iid_test_metrics": paired_eval,
        "unmatched_target_rates": unmatched_rates,
        "n3s_inference_only_metrics": counterfactual,
        "predictions_sha256": sha256_file(prediction_path),
        "warning": "Per-task test groups recur across folds and seeds; these intervals "
                   "do not make rounds independent. OOD matched controls define "
                   "the reported binary population."}
    (task_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return task_dir / "metrics.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--with-n3s", action="store_true")
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--models", nargs="+", required=True)
    plan.add_argument("--protocols", nargs="+", required=True)
    plan.add_argument("--seeds", nargs="+", type=int)
    plan.add_argument("--folds", nargs="+", type=int)
    plan.add_argument("--counterfactual-n3s", action="store_true")
    task = sub.add_parser("run-task")
    task.add_argument("--config", type=Path, required=True)
    task.add_argument("--prepared-dir", type=Path, required=True)
    task.add_argument("--plan", type=Path, required=True)
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_texts(args.config, args.output, with_n3s=args.with_n3s)
    elif args.command == "plan":
        result = create_plan(args.config, args.output, models=args.models,
                             protocols=args.protocols,
                             counterfactual=args.counterfactual_n3s,
                             seeds=args.seeds, folds=args.folds)
    else:
        result = run_task(args.config, args.prepared_dir, args.plan,
                          args.task_index, args.output_root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
