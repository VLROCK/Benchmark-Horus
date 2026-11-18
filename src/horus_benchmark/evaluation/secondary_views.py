"""Evaluate frozen N2 detectors on middle units and diagnostic layout views.

The researcher runs this module on the cluster. The frozen release and original
IID scores are immutable inputs. Every task is one model/seed/fold; inference
tasks reuse a verified document fit and training tasks fit only their assigned
training unit. Provenance columns are carried for auditing, never model input.
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

from horus_benchmark.evaluation import benchmark_runner as base
from horus_benchmark.metrics.binary_detection import evaluate_binary, metric_values


UNITS = ("paragraph", "sentence")
LAYOUTS = ("layout_blank", "layout_all")
MODES = ("document_to_views", "unit_trained", "layout_retrained")
TEXT_FIELDS = ("record_id", "content")
PREDICTION_FIELDS = (
    "record_id", "label_name", "probe_group_id", "broad_area",
    "source_name", "generator", "mode", "view", "score_view", "score_document",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"baseline_config", "baseline_prepared", "baseline_cpu_plan",
                "baseline_bert_plan", "baseline_runs", "segments_dir", "layout_dir",
                "seeds", "folds", "fpr_targets", "bootstrap_replicates"}
    if (not isinstance(cfg, dict) or cfg.get("schema_version") != 1 or
            cfg.get("experiment_id") != "horus-n2-secondary-views-v1" or
            required - set(cfg)):
        raise ValueError("Unexpected secondary-view configuration")
    if (cfg["seeds"] != [20261105, 20261106, 20261107] or
            cfg["folds"] != list(range(5)) or cfg["fpr_targets"] != [0.01, 0.05] or
            int(cfg["bootstrap_replicates"]) != 2000):
        raise ValueError("Secondary views must retain reviewed seeds, folds and metrics")
    return cfg


def _inputs(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any],
                                         dict[str, Any], dict[str, Any]]:
    baseline_cfg = base._config(Path(cfg["baseline_config"]))
    _, release = base._release(baseline_cfg)
    prepared = _read_json(Path(cfg["baseline_prepared"]) / "prepared_manifest.json")
    if (prepared.get("kind") != "selected_n2_text_cache" or
            prepared.get("release_manifest_sha256") != base.sha256_file(
                Path(baseline_cfg["release_dir"]) / "release_manifest.json") or
            prepared.get("config_sha256") != base.sha256_file(Path(cfg["baseline_config"]))):
        raise ValueError("The baseline text cache is not from the frozen release")
    base._check_pinned_file(Path(prepared["selected_n2_texts"]["path"]),
                            prepared["selected_n2_texts"]["sha256"])
    segment = _read_json(Path(cfg["segments_dir"]) / "segment_manifest.json")
    layout = _read_json(Path(cfg["layout_dir"]) / "layout_manifest.json")
    release_sha = base.sha256_file(Path(baseline_cfg["release_dir"]) /
                                   "release_manifest.json")
    if (segment.get("kind") != "frozen_n2_middle_unit_views" or
            segment.get("release_id") != release["release_id"] or
            segment.get("source_rows") != release["rows"] or
            segment.get("release_manifest_sha256") != release_sha or
            segment.get("selected_texts_sha256") !=
            prepared["selected_n2_texts"]["sha256"]):
        raise ValueError("Segment manifest targets a different release")
    if (layout.get("kind") != "n2_selected_layout_ablations" or
            layout.get("release_id") != release["release_id"] or
            layout.get("rows") != release["rows"] or
            layout.get("inputs", {}).get("release_manifest", {}).get("sha256") !=
            release_sha or
            layout.get("inputs", {}).get("selected_n2_texts", {}).get("sha256") !=
            prepared["selected_n2_texts"]["sha256"]):
        raise ValueError("Layout manifest targets a different release")
    return baseline_cfg, prepared, segment, layout


def _artifact(manifest: dict[str, Any], name: str) -> tuple[Path, str]:
    """Accept the common manifest artifact schema used by derived-view prep."""
    artifacts = manifest.get("artifacts", {})
    entry = artifacts.get(name, manifest.get(name))
    if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
        raise ValueError(f"Missing pinned artifact {name}")
    path = Path(entry["path"])
    base._check_pinned_file(path, entry["sha256"])
    return path, entry["sha256"]


def _view_paths(cfg: dict[str, Any], segment: dict[str, Any],
                layout: dict[str, Any]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for name in UNITS:
        output[name] = _artifact(segment, f"{name}_texts")[0]
    for name, artifact in (("layout_blank", "blank"),
                           ("layout_all", "all")):
        output[name] = _artifact(layout, artifact)[0]
    return output


def _baseline_plan(cfg: dict[str, Any]) -> pd.DataFrame:
    parts = [pd.read_csv(cfg[key], keep_default_na=False)
             for key in ("baseline_cpu_plan", "baseline_bert_plan")]
    plan = pd.concat(parts, ignore_index=True)
    plan = plan.loc[plan.protocol.eq("iid") & plan.target.eq("all")].copy()
    if (len(plan) != 135 or plan.duplicated(["model", "seed", "fold"]).any() or
            set(plan.model) != set(base.ALL_MODELS)):
        raise ValueError("Expected the complete frozen nine-model IID plan")
    for row in plan.itertuples(index=False):
        if base._task_id(row._asdict()) != row.task_id:
            raise ValueError("A baseline task ID was edited")
    return plan


def _task_id(row: dict[str, Any], config_sha: str, view_sha: str) -> str:
    fields = {key: row[key] for key in ("mode", "view", "model", "seed", "fold",
                                      "baseline_task_id")}
    fields["config_sha256"] = config_sha
    fields["view_sha256"] = view_sha
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:20]


def create_plan(config_path: Path, output_dir: Path) -> list[Path]:
    """Create six CPU/GPU plans without fitting or scoring a detector."""
    cfg = _load_config(config_path)
    _, _, segment, layout = _inputs(cfg)
    views = _view_paths(cfg, segment, layout)
    prior = _baseline_plan(cfg)
    if output_dir.exists():
        raise FileExistsError(f"Plan directory already exists: {output_dir}")
    config_sha = base.sha256_file(config_path)
    segment_sha = base.sha256_file(Path(cfg["segments_dir"]) / "segment_manifest.json")
    layout_sha = base.sha256_file(Path(cfg["layout_dir"]) / "layout_manifest.json")
    rows: list[dict[str, Any]] = []
    for old in prior.itertuples(index=False):
        common = {"model": old.model, "seed": int(old.seed), "fold": int(old.fold),
                  "baseline_task_id": old.task_id}
        rows.append({**common, "mode": "document_to_views", "view": "all",
                     "view_sha256": hashlib.sha256(
                         f"{segment_sha}|{layout_sha}".encode()).hexdigest()})
        rows.extend({**common, "mode": "unit_trained", "view": name,
                     "view_sha256": base.sha256_file(views[name])} for name in UNITS)
        rows.extend({**common, "mode": "layout_retrained", "view": name,
                     "view_sha256": base.sha256_file(views[name])} for name in LAYOUTS)
    for row in rows:
        row["task_id"] = _task_id(row, config_sha, row["view_sha256"])
    plan = pd.DataFrame(rows)
    if plan.task_id.duplicated().any():
        raise AssertionError("Secondary task IDs collided")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for mode in MODES:
        for kind, is_bert in (("cpu", False), ("bert", True)):
            subset = plan.loc[plan["mode"].eq(mode) & plan["model"].eq("bertimbau").eq(is_bert)]
            path = output_dir / f"{mode}_{kind}.csv"
            subset.to_csv(path, index=False)
            paths.append(path)
    manifest = {"schema_version": 1, "kind": "secondary_view_task_plan",
                "config_sha256": config_sha,
                "segment_manifest_sha256": segment_sha,
                "layout_manifest_sha256": layout_sha,
                "plans": {p.name: {"rows": len(pd.read_csv(p)),
                                    "sha256": base.sha256_file(p)} for p in paths}}
    (output_dir / "plan_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def _model_seed(task: dict[str, Any]) -> int:
    value = f"{task['model']}|{task['seed']}|{task['fold']}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _load_text(path: Path, selected: pd.DataFrame, *, subset: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if set(TEXT_FIELDS) - set(frame) or frame.record_id.duplicated().any():
        raise ValueError(f"Invalid view schema or duplicate IDs: {path}")
    if (not subset and set(frame.record_id) != set(selected.record_id)) or (
            subset and not set(frame.record_id) <= set(selected.record_id)):
        raise ValueError(f"View IDs do not match the frozen release: {path}")
    if frame.content.isna().any() or not frame.content.map(
            lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"Empty view text: {path}")
    if "content_hash" in frame and not frame.content.map(
            lambda x: hashlib.sha256(x.encode()).hexdigest()).eq(frame.content_hash).all():
        raise ValueError(f"View content hash mismatch: {path}")
    return frame[["record_id", "content"]].copy()


def _baseline_task(cfg: dict[str, Any], task: dict[str, Any]) -> tuple[dict[str, Any],
                                                                      pd.DataFrame]:
    path = Path(cfg["baseline_runs"]) / task["baseline_task_id"]
    record = _read_json(path / "metrics.json")
    if (record.get("task", {}).get("task_id") != task["baseline_task_id"] or
            record.get("task", {}).get("model") != task["model"] or
            int(record.get("task", {}).get("seed", -1)) != int(task["seed"]) or
            int(record.get("task", {}).get("fold", -1)) != int(task["fold"])):
        raise ValueError("Baseline metric identity does not match the planned task")
    base._check_pinned_file(path / "predictions.parquet", record["predictions_sha256"])
    predictions = pd.read_parquet(path / "predictions.parquet")
    if (predictions.record_id.duplicated().any() or
            set(predictions.role) != {"test_target"} or
            not np.isfinite(predictions.score_n2.to_numpy(dtype=float)).all()):
        raise ValueError("Invalid baseline IID predictions")
    return record, predictions


def _partition(cfg: dict[str, Any], selected: pd.DataFrame,
               seed: int, fold: int) -> pd.DataFrame:
    baseline_cfg = base._config(Path(cfg["baseline_config"]))
    release_dir = Path(baseline_cfg["release_dir"])
    frozen = _read_json(release_dir / "release_manifest.json")
    path = release_dir / f"split_seed{seed}.parquet"
    base._check_pinned_file(path, frozen["artifacts"][f"split_seed{seed}"]["sha256"])
    split = base.partition_task(pd.read_parquet(path), fold, "iid")
    frame = selected.merge(split[["record_id", "probe_group_id", "role", "fold"]],
                           on="record_id", validate="one_to_one", suffixes=("", "_split"))
    if (len(frame) != len(selected) or
            not frame.probe_group_id.eq(frame.probe_group_id_split).all()):
        raise ValueError("Frozen split and selected text group IDs disagree")
    return frame.drop(columns="probe_group_id_split")


def _join_view(partition: pd.DataFrame, view: pd.DataFrame) -> pd.DataFrame:
    frame = partition.drop(columns="content").merge(view, on="record_id",
                                                    how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("View has no eligible documents")
    return frame.sort_values("record_id").reset_index(drop=True)


def _reference(test: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    merged = test[["record_id", "label_name", "probe_group_id"]].merge(
        prior[["record_id", "label_name", "probe_group_id", "score_n2"]],
        on="record_id", validate="one_to_one", suffixes=("", "_prior"))
    if (len(merged) != len(test) or
            not merged.label_name.eq(merged.label_name_prior).all() or
            not merged.probe_group_id.eq(merged.probe_group_id_prior).all()):
        raise ValueError("Baseline scores are missing or mismatched for view test IDs")
    return merged[["record_id", "score_n2"]].rename(columns={"score_n2": "score_document"})


def _predictions(test: pd.DataFrame, reference: pd.DataFrame, scores: np.ndarray,
                 mode: str, view: str) -> pd.DataFrame:
    frame = test[["record_id", "label_name", "probe_group_id", "broad_area",
                  "source_name", "generator"]].merge(reference, on="record_id",
                                                    validate="one_to_one")
    frame["score_view"] = np.asarray(scores, dtype=float)
    frame["mode"], frame["view"] = mode, view
    if len(frame) != len(scores) or not np.isfinite(frame.score_view).all():
        raise ValueError("Invalid or misaligned view scores")
    return frame[list(PREDICTION_FIELDS)]


def _metrics(frame: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, float]:
    return metric_values(frame.label_name.eq("synthetic").to_numpy(dtype=int),
                         frame.score_view.to_numpy(dtype=float), thresholds)


def _fit_variant(task: dict[str, Any], cfg: dict[str, Any], frame: pd.DataFrame,
                 task_dir: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    train = frame.loc[frame.role.eq("train")]
    validation = frame.loc[frame.role.eq("validation")]
    test = frame.loc[frame.role.eq("test_target")]
    if any(set(part.label_name) != {"human", "synthetic"}
           for part in (train, validation, test)):
        raise ValueError("The derived unit lacks a class in train/validation/test")
    model_seed = _model_seed(task)
    base_cfg = base._config(Path(cfg["baseline_config"]))
    if task["model"] == "bertimbau":
        from horus_benchmark.detectors.bertimbau_baseline import train_and_score
        model_cfg = yaml.safe_load(Path(base_cfg["bertimbau_config"]).read_text(
            encoding="utf-8"))
        fitted = train_and_score(train.content.tolist(), base._labels(train).tolist(),
                                 validation.content.tolist(),
                                 base._labels(validation).tolist(), test.content.tolist(),
                                 model_cfg, task_dir, model_seed)
        info = fitted["metadata"]
    else:
        from horus_benchmark.detectors.text_baselines import fit_score, load_config
        model_cfg = load_config(base_cfg["text_model_config"])
        fitted = fit_score(task["model"], train.content.tolist(), base._labels(train),
                           validation.content.tolist(), test.content.tolist(),
                           seed=model_seed, inner_folds=train.fold.to_numpy(),
                           config=model_cfg)
        info = {"selected_c": fitted["selected_c"],
                "inner_auc_by_c": fitted["inner_auc_by_c"]}
    evaluation = evaluate_binary(
        base._labels(validation), np.asarray(fitted["validation_scores"]),
        base._labels(test), np.asarray(fitted["test_scores"]),
        test.probe_group_id.to_numpy(), fpr_targets=tuple(cfg["fpr_targets"]),
        bootstrap_replicates=int(cfg["bootstrap_replicates"]),
        bootstrap_seed=model_seed + 1)
    return np.asarray(fitted["test_scores"], dtype=float), evaluation, info


def _score_document_fit(task: dict[str, Any], cfg: dict[str, Any],
                        partition: pd.DataFrame, prior: dict[str, Any],
                        prior_scores: pd.DataFrame,
                        view_texts: dict[str, list[str]]) -> tuple[dict[str, np.ndarray],
                                                                  dict[str, Any]]:
    base_cfg = base._config(Path(cfg["baseline_config"]))
    test = partition.loc[partition.role.eq("test_target")].sort_values("record_id")
    old = prior_scores.sort_values("record_id")
    if not test.record_id.reset_index(drop=True).equals(old.record_id.reset_index(drop=True)):
        raise ValueError("Original and replayed IID test IDs differ")
    model_seed = _model_seed(task)
    if int(prior["model_seed"]) != model_seed:
        raise ValueError("Original model seed differs from frozen protocol")
    if task["model"] == "bertimbau":
        from horus_benchmark.detectors.bertimbau_baseline import score_saved_checkpoint
        checkpoint = Path(cfg["baseline_runs"]) / task["baseline_task_id"] / "model"
        sample = pd.concat((test.loc[test.label_name.eq("human")].head(128),
                            test.loc[test.label_name.eq("synthetic")].head(128)))
        sample_prior = sample[["record_id"]].merge(old[["record_id", "score_n2"]],
                                                    on="record_id", validate="one_to_one")
        model_cfg = yaml.safe_load(Path(base_cfg["bertimbau_config"]).read_text(
            encoding="utf-8"))
        scored = score_saved_checkpoint(
            checkpoint, {"clean_check": sample.content.tolist(), **view_texts}, model_cfg,
            expected_sha256=prior["model_info"]["checkpoint_sha256"])
        clean = scored["scores"].pop("clean_check")
        expected = sample_prior.score_n2.to_numpy(dtype=float)
        if not np.allclose(clean, expected, rtol=1e-4, atol=1e-4):
            raise ValueError("Frozen BERTimbau checkpoint does not reproduce clean scores")
        return scored["scores"], {"clean_check_rows": len(clean),
                                   "clean_check_max_abs_delta": float(np.max(abs(clean - expected))),
                                   "token_coverage": scored["token_coverage"],
                                   "checkpoint_sha256": scored["checkpoint_sha256"]}
    from horus_benchmark.detectors.text_baselines import (
        build_estimator, load_config, synthetic_scores)
    model_cfg = load_config(base_cfg["text_model_config"])
    chosen = prior["model_info"].get("selected_c") or 1.0
    fitted = build_estimator(task["model"], c=float(chosen), seed=model_seed,
                             config=model_cfg)
    train = partition.loc[partition.role.eq("train")].sort_values("record_id")
    fitted.fit(train.content.tolist(), base._labels(train))
    clean = synthetic_scores(fitted, test.content.tolist())
    expected = old.score_n2.to_numpy(dtype=float)
    if not np.allclose(clean, expected, rtol=1e-6, atol=1e-6):
        raise ValueError("Reconstructed CPU model does not reproduce clean scores")
    output = {name: synthetic_scores(fitted, texts) for name, texts in view_texts.items()}
    return output, {"clean_check_rows": len(clean),
                    "clean_check_max_abs_delta": float(np.max(abs(clean - expected))),
                    "selected_c": chosen}


def run_task(config_path: Path, plan_path: Path, index: int, output_root: Path) -> Path:
    cfg = _load_config(config_path)
    _, prepared, segment, layout = _inputs(cfg)
    paths = _view_paths(cfg, segment, layout)
    plan = pd.read_csv(plan_path, keep_default_na=False)
    if index < 0 or index >= len(plan):
        raise IndexError("Task index is outside the plan")
    task = plan.iloc[index].to_dict()
    if (task["mode"] not in MODES or task["model"] not in base.ALL_MODELS or
            int(task["seed"]) not in cfg["seeds"] or int(task["fold"]) not in cfg["folds"] or
            task["view"] not in (*UNITS, *LAYOUTS, "all") or
            _task_id(task, base.sha256_file(config_path), task["view_sha256"]) !=
            task["task_id"]):
        raise ValueError("Unreviewed secondary task")
    expected_view = (UNITS if task["mode"] == "unit_trained" else LAYOUTS)
    if task["mode"] == "document_to_views" and task["view"] != "all":
        raise ValueError("Document inference task must cover all views")
    if task["mode"] != "document_to_views" and task["view"] not in expected_view:
        raise ValueError("Task view does not match its mode")
    if task["mode"] == "document_to_views":
        combined = hashlib.sha256(
            (base.sha256_file(Path(cfg["segments_dir"]) / "segment_manifest.json") +
             "|" + base.sha256_file(Path(cfg["layout_dir"]) /
                                   "layout_manifest.json")).encode()).hexdigest()
        if task["view_sha256"] != combined:
            raise ValueError("Combined view manifest hash changed after planning")
    elif task["view_sha256"] != base.sha256_file(paths[task["view"]]):
        raise ValueError("Derived view bytes changed after planning")
    baseline_plan = _baseline_plan(cfg)
    old_row = baseline_plan.loc[
        baseline_plan.task_id.eq(task["baseline_task_id"])]
    if len(old_row) != 1 or any(str(old_row.iloc[0][name]) != str(task[name])
                                 for name in ("model", "seed", "fold")):
        raise ValueError("Secondary task does not reference its original IID fit")
    prior, prior_scores = _baseline_task(cfg, task)
    selected = pd.read_parquet(prepared["selected_n2_texts"]["path"])
    if len(selected) != 107566 or selected.record_id.duplicated().any():
        raise ValueError("Frozen selected text cache has changed")
    partition = _partition(cfg, selected, int(task["seed"]), int(task["fold"]))
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / task["task_id"]
    if task_dir.exists():
        if (task_dir / "metrics.json").exists():
            raise FileExistsError(f"Secondary task is already complete: {task_dir}")
        if task_dir.resolve().parent != output_root.resolve():
            raise ValueError("Incomplete task directory escapes output root")
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=False)
    scored_frames: list[pd.DataFrame] = []
    results: dict[str, Any] = {}
    if task["mode"] == "document_to_views":
        chosen: dict[str, pd.DataFrame] = {}
        text_views: dict[str, list[str]] = {}
        for name in (*UNITS, *LAYOUTS):
            view = _load_text(paths[name], selected, subset=name in UNITS)
            joined = _join_view(partition, view)
            test = joined.loc[joined.role.eq("test_target")].sort_values("record_id")
            if set(test.label_name) != {"human", "synthetic"}:
                raise ValueError(f"No binary test support for {name}")
            chosen[name] = test
            text_views[name] = test.content.tolist()
        scores_by_view, model_info = _score_document_fit(
            task, cfg, partition, prior, prior_scores, text_views)
        thresholds = prior["evaluation"]["validation_thresholds"]
        for name, test in chosen.items():
            reference = _reference(test, prior_scores)
            prediction = _predictions(test, reference, scores_by_view[name],
                                      task["mode"], name)
            scored_frames.append(prediction)
            results[name] = {"rows": len(prediction), "thresholds": thresholds,
                             "view_metrics": _metrics(prediction, thresholds),
                             "document_metrics_same_ids": metric_values(
                                 prediction.label_name.eq("synthetic").to_numpy(dtype=int),
                                 prediction.score_document.to_numpy(dtype=float), thresholds)}
    else:
        name = task["view"]
        view = _load_text(paths[name], selected, subset=name in UNITS)
        joined = _join_view(partition, view)
        test = joined.loc[joined.role.eq("test_target")].sort_values("record_id")
        scores, evaluation, model_info = _fit_variant(task, cfg, joined, task_dir)
        reference = _reference(test, prior_scores)
        prediction = _predictions(test, reference, scores, task["mode"], name)
        scored_frames.append(prediction)
        results[name] = {"rows": len(prediction),
                         "thresholds": evaluation["validation_thresholds"],
                         "view_metrics": evaluation["metrics"],
                         "document_thresholds": prior["evaluation"]["validation_thresholds"],
                         "document_metrics_same_ids": metric_values(
                             prediction.label_name.eq("synthetic").to_numpy(dtype=int),
                             prediction.score_document.to_numpy(dtype=float),
                             prior["evaluation"]["validation_thresholds"])}
    prediction_path = task_dir / "predictions.parquet"
    pq.write_table(pa.Table.from_pandas(pd.concat(scored_frames, ignore_index=True),
                                        preserve_index=False), prediction_path,
                   compression="zstd")
    record = {"schema_version": 1, "kind": "n2_secondary_view_task",
              "task": task, "config_sha256": base.sha256_file(config_path),
              "baseline_predictions_sha256": prior["predictions_sha256"],
              "segment_manifest_sha256": base.sha256_file(
                  Path(cfg["segments_dir"]) / "segment_manifest.json"),
              "layout_manifest_sha256": base.sha256_file(
                  Path(cfg["layout_dir"]) / "layout_manifest.json"),
              "results": results, "model_info": model_info,
              "predictions_sha256": base.sha256_file(prediction_path)}
    (task_dir / "metrics.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return task_dir / "metrics.json"


def check_plan(plan_dir: Path, runs: Path) -> dict[str, Any]:
    manifest = _read_json(plan_dir / "plan_manifest.json")
    counts: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    for filename, entry in manifest["plans"].items():
        path = plan_dir / filename
        base._check_pinned_file(path, entry["sha256"])
        plan = pd.read_csv(path, keep_default_na=False)
        complete = 0
        for index, row in plan.iterrows():
            task_dir = runs / row.task_id
            metrics = task_dir / "metrics.json"
            predictions = task_dir / "predictions.parquet"
            if metrics.is_file() and predictions.is_file():
                record = _read_json(metrics)
                if (record.get("task", {}).get("task_id") != row.task_id or
                        base.sha256_file(predictions) != record.get("predictions_sha256")):
                    raise ValueError(f"Corrupt secondary task: {row.task_id}")
                complete += 1
            else:
                missing.append({"plan": filename, "index": int(index),
                                "task_id": row.task_id})
        counts[filename] = {"complete": complete, "planned": len(plan)}
    return {"complete": not missing, "counts": counts, "missing": missing}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run-task")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--plan-dir", type=Path, required=True)
    check.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        for path in create_plan(args.config, args.output_dir):
            print(path)
    elif args.command == "run-task":
        print(run_task(args.config, args.plan, args.task_index, args.output_root))
    else:
        result = check_plan(args.plan_dir, args.runs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["complete"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
