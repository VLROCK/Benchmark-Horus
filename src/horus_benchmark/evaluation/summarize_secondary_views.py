"""Summarize secondary N2 view experiments on the same frozen test IDs.

The researcher runs this on the cluster after every planned task is complete.
It reads score files, never corpus text. Fold-level tests reuse documents, so
we align IDs and draw linked probe groups once for each paired comparison.
The three release seeds describe sensitivity of one corpus, not independent
studies; no pooled cross-seed confidence interval is computed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from horus_benchmark.evaluation import secondary_views
from horus_benchmark.metrics.binary_detection import metric_values


METRICS = (
    "roc_auc", "balanced_accuracy", "tpr_at_fpr_0.01",
    "observed_fpr_at_fpr_0.01", "balanced_accuracy_at_fpr_0.01",
    "tpr_at_fpr_0.05", "observed_fpr_at_fpr_0.05",
    "balanced_accuracy_at_fpr_0.05",
)
THRESHOLDS = {"youden", "fpr_0.01", "fpr_0.05"}
PREDICTION_COLUMNS = {"record_id", "label_name", "probe_group_id",
                      "mode", "view", "score_view", "score_document"}
EXPECTED_PLANS = {
    f"{mode}_{hardware}.csv"
    for mode in secondary_views.MODES for hardware in ("cpu", "bert")
}


@dataclass(frozen=True)
class Fold:
    model: str
    seed: int
    fold: int
    mode: str
    view: str
    task_id: str
    prediction: pd.DataFrame
    view_thresholds: dict[str, float]
    document_thresholds: dict[str, float]


@dataclass(frozen=True)
class ScorePlan:
    scores: np.ndarray
    order: np.ndarray
    tie_starts: np.ndarray
    positive_at_threshold: dict[str, np.ndarray]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _as_thresholds(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != THRESHOLDS:
        raise ValueError("Unexpected validation threshold schema")
    result = {key: float(item) for key, item in value.items()}
    if any(np.isnan(item) or item == float("-inf") for item in result.values()):
        raise ValueError("Invalid validation threshold")
    return result


def _record_fold(task: dict[str, Any], record: dict[str, Any],
                 predictions: pd.DataFrame, view: str) -> Fold:
    result = record.get("results", {}).get(view)
    if not isinstance(result, dict):
        raise ValueError(f"Missing task result for {task['task_id']}/{view}")
    frame = predictions.loc[predictions.view.eq(view)].sort_values("record_id")
    frame = frame.reset_index(drop=True)
    if (len(frame) != int(result.get("rows", -1)) or
            frame.record_id.duplicated().any() or
            frame.empty or set(frame.label_name) != {"human", "synthetic"} or
            frame.probe_group_id.isna().any() or
            frame["score_view"].isna().any() or
            frame["score_document"].isna().any() or
            not np.isfinite(frame.score_view.to_numpy(dtype=float)).all() or
            not np.isfinite(frame.score_document.to_numpy(dtype=float)).all()):
        raise ValueError(f"Invalid aligned prediction rows for {task['task_id']}/{view}")
    if frame.groupby("probe_group_id").label_name.nunique().gt(1).any():
        raise ValueError("A linked test group spans labels")
    view_thresholds = _as_thresholds(result["thresholds"])
    document_thresholds = _as_thresholds(result.get("document_thresholds",
                                                   result["thresholds"]))
    y = frame.label_name.eq("synthetic").to_numpy(dtype=np.int8)
    observed_view = metric_values(y, frame.score_view.to_numpy(dtype=float),
                                  view_thresholds)
    observed_document = metric_values(
        y, frame.score_document.to_numpy(dtype=float), document_thresholds)
    for name, observed, expected in (
            ("view", observed_view, result["view_metrics"]),
            ("document", observed_document, result["document_metrics_same_ids"])):
        if set(expected) != set(METRICS) or any(
                not np.isclose(observed[key], float(expected[key]),
                               rtol=1e-10, atol=1e-12) for key in METRICS):
            raise ValueError(f"Saved {name} metrics differ from scores: {task['task_id']}/{view}")
    return Fold(str(task["model"]), int(task["seed"]), int(task["fold"]),
                str(task["mode"]), view, str(task["task_id"]), frame,
                view_thresholds, document_thresholds)


def _read_rounds(config_path: Path, plan_dir: Path, runs: Path,
                 models: set[str] | None = None
                 ) -> tuple[list[Fold], dict[str, Any]]:
    cfg = secondary_views._load_config(config_path)
    selected_models = set(secondary_views.base.ALL_MODELS) if models is None else set(models)
    if not selected_models or selected_models - set(secondary_views.base.ALL_MODELS):
        raise ValueError("Unknown or empty secondary-summary model selection")
    plan_manifest = _json(plan_dir / "plan_manifest.json")
    if (plan_manifest.get("kind") != "secondary_view_task_plan" or
            plan_manifest.get("config_sha256") !=
            secondary_views.base.sha256_file(config_path) or
            set(plan_manifest.get("plans", {})) != EXPECTED_PLANS):
        raise ValueError("Unexpected or changed secondary-view plan")
    rounds: list[Fold] = []
    missing: list[str] = []
    task_ids: set[str] = set()
    selected_task_ids: set[str] = set()
    for filename, entry in sorted(plan_manifest["plans"].items()):
        plan_path = plan_dir / filename
        secondary_views.base._check_pinned_file(plan_path, entry["sha256"])
        plan = pd.read_csv(plan_path, keep_default_na=False)
        if len(plan) != int(entry["rows"]):
            raise ValueError(f"Plan row count differs from manifest: {filename}")
        for task in plan.to_dict("records"):
            task_id = str(task["task_id"])
            if (task_id in task_ids or
                    secondary_views._task_id(
                        task, plan_manifest["config_sha256"], task["view_sha256"])
                    != task_id):
                raise ValueError(f"Duplicate or changed secondary task: {task_id}")
            task_ids.add(task_id)
            if task["model"] not in selected_models:
                continue
            selected_task_ids.add(task_id)
            task_dir = runs / task_id
            metric_path, score_path = task_dir / "metrics.json", task_dir / "predictions.parquet"
            if not metric_path.is_file() or not score_path.is_file():
                missing.append(task_id)
                continue
            record = _json(metric_path)
            if (record.get("kind") != "n2_secondary_view_task" or
                    record.get("config_sha256") != plan_manifest["config_sha256"] or
                    secondary_views.base.sha256_file(score_path) !=
                    record.get("predictions_sha256") or
                    any(str(record.get("task", {}).get(key)) != str(task[key])
                        for key in ("task_id", "model", "seed", "fold", "mode",
                                    "view", "baseline_task_id", "view_sha256"))):
                raise ValueError(f"Corrupt or mismatched secondary task: {task_id}")
            prediction = pd.read_parquet(score_path)
            if (PREDICTION_COLUMNS - set(prediction) or
                    prediction.duplicated(["view", "record_id"]).any() or
                    set(prediction["mode"]) != {task["mode"]} or
                    not set(prediction["view"]) <= set((*secondary_views.UNITS,
                                                        *secondary_views.LAYOUTS))):
                raise ValueError(f"Invalid secondary score schema: {task_id}")
            expected_views = (set((*secondary_views.UNITS, *secondary_views.LAYOUTS))
                              if task["mode"] == "document_to_views" else {task["view"]})
            if set(prediction["view"]) != expected_views or \
                    set(record.get("results", {})) != expected_views:
                raise ValueError(f"Unexpected views in secondary task: {task_id}")
            rounds.extend(_record_fold(task, record, prediction, view)
                          for view in sorted(expected_views))
    if missing:
        raise ValueError(f"{len(missing)} secondary tasks incomplete; first IDs: {missing[:12]}")
    if not rounds:
        raise ValueError("No completed secondary tasks")
    return rounds, {"tasks": len(selected_task_ids), "fold_results": len(rounds),
                    "models": sorted(selected_models),
                    "plan_manifest_sha256": secondary_views.base.sha256_file(
                        plan_dir / "plan_manifest.json"),
                    "config_sha256": plan_manifest["config_sha256"],
                    "bootstrap_replicates": int(cfg["bootstrap_replicates"])}


def _aligned(folds: list[Fold]) -> tuple[pd.DataFrame, list[Fold]]:
    """Require identical selected test IDs and labels across all five folds."""
    ordered = sorted(folds, key=lambda item: item.fold)
    if len(ordered) != 5 or [item.fold for item in ordered] != list(range(5)):
        raise ValueError("Secondary summary requires exactly five folds")
    first = ordered[0].prediction
    for item in ordered[1:]:
        frame = item.prediction
        if (not frame.record_id.equals(first.record_id) or
                not frame.label_name.equals(first.label_name) or
                not frame.probe_group_id.equals(first.probe_group_id)):
            raise ValueError("Fold test IDs, labels or linked groups differ")
    return first, ordered


def _score_plan(scores: np.ndarray, thresholds: dict[str, float]) -> ScorePlan:
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    return ScorePlan(scores, order, starts,
                     {key: scores >= value for key, value in thresholds.items()})


def _auc_batch(labels: np.ndarray, plan: ScorePlan,
               row_group_index: np.ndarray, group_weights: np.ndarray) -> np.ndarray:
    """Exact weighted ROC-AUC under whole-group bootstrap, including score ties."""
    sorted_labels = labels[plan.order]
    weights = group_weights[:, row_group_index[plan.order]]
    positives = np.add.reduceat(weights * sorted_labels, plan.tie_starts, axis=1)
    negatives = np.add.reduceat(weights * (1 - sorted_labels), plan.tie_starts, axis=1)
    negatives_below = np.cumsum(negatives, axis=1) - negatives
    denominator = positives.sum(axis=1) * negatives.sum(axis=1)
    if np.any(denominator == 0):
        raise ValueError("Bootstrap draw lost a label")
    return np.sum(positives * (negatives_below + 0.5 * negatives),
                  axis=1) / denominator


def _metric_batch(labels: np.ndarray, plan: ScorePlan,
                  row_group_index: np.ndarray, group_weights: np.ndarray,
                  thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    """Vectorized weighted metrics for one fold and a batch of shared draws."""
    row_weights = group_weights[:, row_group_index]
    human = labels == 0
    synthetic = ~human
    nh = row_weights[:, human].sum(axis=1)
    ns = row_weights[:, synthetic].sum(axis=1)
    if np.any(nh == 0) or np.any(ns == 0):
        raise ValueError("Bootstrap draw lost a class")
    output = {"roc_auc": _auc_batch(labels, plan, row_group_index, group_weights)}
    for key in thresholds:
        predicted = plan.positive_at_threshold[key]
        tpr = (row_weights[:, synthetic] * predicted[synthetic]).sum(axis=1) / ns
        fpr = (row_weights[:, human] * predicted[human]).sum(axis=1) / nh
        balanced = (tpr + 1 - fpr) / 2
        if key == "youden":
            output["balanced_accuracy"] = balanced
        else:
            output[f"tpr_at_{key}"] = tpr
            output[f"observed_fpr_at_{key}"] = fpr
            output[f"balanced_accuracy_at_{key}"] = balanced
    if set(output) != set(METRICS):
        raise ValueError("Unexpected weighted metric schema")
    return output


def _group_index(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[int, int]]:
    groups = frame[["probe_group_id", "label_name"]].drop_duplicates()
    if groups.probe_group_id.duplicated().any():
        raise ValueError("A linked group spans labels")
    keys: list[str] = []
    sizes: list[int] = []
    for label in ("human", "synthetic"):
        chosen = sorted(groups.loc[groups.label_name.eq(label), "probe_group_id"].astype(str))
        if not chosen:
            raise ValueError("Both classes need bootstrap groups")
        sizes.append(len(chosen))
        keys.extend(chosen)
    lookup = {key: index for index, key in enumerate(keys)}
    return frame.probe_group_id.astype(str).map(lookup).to_numpy(dtype=np.int32), \
        (sizes[0], sizes[1])


def _draws(rng: np.random.Generator, sizes: tuple[int, int],
           batch: int) -> np.ndarray:
    parts = [rng.multinomial(size, np.full(size, 1 / size), size=batch)
             for size in sizes]
    return np.concatenate(parts, axis=1)


def _seed_key(seed: int, mode: str, view: str, population: str) -> int:
    value = f"secondary|{seed}|{mode}|{view}|{population}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def paired_summary(folds: list[Fold], *, replicates: int, population: str = "full",
                   membership_ids: set[str] | None = None,
                   batch_size: int = 32) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fold mean and shared group-bootstrap paired view-minus-document deltas."""
    if replicates < 1 or batch_size < 1:
        raise ValueError("Positive bootstrap repetitions/batch size required")
    first, ordered = _aligned(folds)
    if membership_ids is not None:
        if not membership_ids <= set(first.record_id):
            raise ValueError("Nuisance membership contains IDs outside this view")
        positions = first.record_id.isin(membership_ids).to_numpy()
        if not positions.any():
            raise ValueError("Nuisance subset is empty")
        ordered = [replace(item, prediction=item.prediction.loc[positions].reset_index(
            drop=True)) for item in ordered]
        first = ordered[0].prediction
    if set(first.label_name) != {"human", "synthetic"}:
        raise ValueError("Paired comparison requires both classes")
    labels = first.label_name.eq("synthetic").to_numpy(dtype=np.int8)
    row_group_index, sizes = _group_index(first)
    point_view: dict[str, list[float]] = defaultdict(list)
    point_document: dict[str, list[float]] = defaultdict(list)
    for item in ordered:
        values_view = metric_values(labels, item.prediction.score_view.to_numpy(dtype=float),
                                    item.view_thresholds)
        values_document = metric_values(
            labels, item.prediction.score_document.to_numpy(dtype=float),
            item.document_thresholds)
        for key in METRICS:
            point_view[key].append(values_view[key])
            point_document[key].append(values_document[key])
    model, seed, mode, view = (ordered[0].model, ordered[0].seed,
                               ordered[0].mode, ordered[0].view)
    rng = np.random.default_rng(_seed_key(seed, mode, view, population))
    drawn_view: dict[str, list[np.ndarray]] = defaultdict(list)
    drawn_document: dict[str, list[np.ndarray]] = defaultdict(list)
    plans = [(_score_plan(item.prediction.score_view.to_numpy(dtype=float),
                          item.view_thresholds),
              _score_plan(item.prediction.score_document.to_numpy(dtype=float),
                          item.document_thresholds)) for item in ordered]
    for start in range(0, replicates, batch_size):
        weights = _draws(rng, sizes, min(batch_size, replicates - start))
        view_batch: dict[str, np.ndarray] = {}
        document_batch: dict[str, np.ndarray] = {}
        for item, (view_plan, document_plan) in zip(ordered, plans):
            scored_view = _metric_batch(
                labels, view_plan,
                row_group_index, weights, item.view_thresholds)
            scored_document = _metric_batch(
                labels, document_plan,
                row_group_index, weights, item.document_thresholds)
            for key in METRICS:
                view_batch[key] = view_batch.get(key, 0) + scored_view[key] / len(ordered)
                document_batch[key] = (document_batch.get(key, 0) +
                                       scored_document[key] / len(ordered))
        for key in METRICS:
            drawn_view[key].append(view_batch[key])
            drawn_document[key].append(document_batch[key])
    counts = first.label_name.value_counts()
    record = {
        "model": model, "seed": seed, "mode": mode, "view": view,
        "population": population, "folds": len(ordered),
        "test_rows": len(first), "human_rows": int(counts.get("human", 0)),
        "synthetic_rows": int(counts.get("synthetic", 0)),
        "human_probe_groups": sizes[0], "synthetic_probe_groups": sizes[1],
        "parent_ids_sha256": hashlib.sha256("".join(
            f"{value}\n" for value in first.record_id).encode()).hexdigest(),
    }
    delta_rows: list[dict[str, Any]] = []
    for key in METRICS:
        view_draw = np.concatenate(drawn_view[key])
        document_draw = np.concatenate(drawn_document[key])
        delta_draw = view_draw - document_draw
        view_point = float(np.mean(point_view[key]))
        document_point = float(np.mean(point_document[key]))
        record[f"view_{key}"] = view_point
        record[f"document_{key}_same_ids"] = document_point
        record[f"view_{key}_ci_low"] = float(np.quantile(view_draw, .025))
        record[f"view_{key}_ci_high"] = float(np.quantile(view_draw, .975))
        record[f"document_{key}_ci_low"] = float(np.quantile(document_draw, .025))
        record[f"document_{key}_ci_high"] = float(np.quantile(document_draw, .975))
        delta_rows.append({
            "model": model, "seed": seed, "mode": mode, "view": view,
            "population": population, "metric": key,
            "view_value": view_point,
            "document_value_same_ids": document_point,
            "delta_view_minus_document": view_point - document_point,
            "delta_ci_low": float(np.quantile(delta_draw, .025)),
            "delta_ci_high": float(np.quantile(delta_draw, .975)),
            "test_rows": len(first), "folds": len(ordered),
        })
    return record, delta_rows


def _membership(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_parquet(path)
    required = {"seed", "unit", "tier", "record_id", "parent_record_id",
                "probe_group_id", "label_name", "broad_area", "pair_id"}
    if required - set(frame) or frame.duplicated(["seed", "unit", "tier", "record_id"]).any():
        raise ValueError("Invalid nuisance membership schema or duplicate IDs")
    if (not frame.record_id.eq(frame.parent_record_id).all() or
            set(frame.label_name) - {"human", "synthetic"} or
            set(frame.unit) - {"document", "paragraph", "sentence"} or
            set(frame.tier) - {"length", "presentation"} or
            not set(frame.loc[frame.tier.eq("presentation"), "unit"]) <= {"document"}):
        raise ValueError("Invalid nuisance membership lineage/tier")
    for _, group in frame.groupby(["seed", "unit", "tier", "pair_id"]):
        if len(group) != 2 or set(group.label_name) != {"human", "synthetic"}:
            raise ValueError("Nuisance pair is not a binary one-to-one match")
    return frame


def _unit_for_view(view: str) -> str:
    if view in secondary_views.UNITS:
        return view
    if view in secondary_views.LAYOUTS:
        return "document"
    raise ValueError(f"Unknown secondary view {view}")


def summarize(config_path: Path, plan_dir: Path, runs: Path, output: Path,
              nuisance_membership: Path | None = None,
              models: set[str] | None = None) -> Path:
    if output.exists():
        raise FileExistsError(f"Summary directory already exists: {output}")
    rounds, inputs = _read_rounds(config_path, plan_dir, runs, models)
    nuisance = _membership(nuisance_membership)
    grouped: dict[tuple[str, int, str, str], list[Fold]] = defaultdict(list)
    for item in rounds:
        grouped[(item.model, item.seed, item.mode, item.view)].append(item)
    if len(grouped) != len(inputs["models"]) * 3 * 8:
        # Four document-inference views plus two unit and two layout fits.
        raise ValueError("Secondary matrix has missing model/seed/mode/view groups")
    metrics: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    nuisance_rows: list[dict[str, Any]] = []
    nuisance_deltas: list[dict[str, Any]] = []
    for (model, seed, mode, view), folds in sorted(grouped.items()):
        full, paired = paired_summary(folds, replicates=inputs["bootstrap_replicates"])
        metrics.append(full)
        deltas.extend(paired)
        if nuisance is None:
            continue
        unit = _unit_for_view(view)
        tiers = ("length", "presentation") if unit == "document" else ("length",)
        for tier in tiers:
            matching = nuisance.loc[
                nuisance.seed.eq(seed) & nuisance.unit.eq(unit) & nuisance.tier.eq(tier)]
            if matching.empty:
                continue
            ids = set(matching.record_id)
            matched, matched_deltas = paired_summary(
                folds, replicates=inputs["bootstrap_replicates"],
                population=f"nuisance_{tier}", membership_ids=ids)
            matched["tier"], matched["unit"] = tier, unit
            matched["human_retention"] = matched["human_rows"] / full["human_rows"]
            matched["synthetic_retention"] = (matched["synthetic_rows"] /
                                               full["synthetic_rows"])
            nuisance_rows.append(matched)
            for row in matched_deltas:
                row["tier"], row["unit"] = tier, unit
                row["human_retention"] = matched["human_retention"]
                row["synthetic_retention"] = matched["synthetic_retention"]
            nuisance_deltas.extend(matched_deltas)
    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, Any] = {}
    for filename, rows in (
            ("per_seed_view_metrics.csv", metrics),
            ("paired_document_deltas.csv", deltas),
            ("nuisance_metrics.csv", nuisance_rows),
            ("nuisance_deltas.csv", nuisance_deltas)):
        path = output / filename
        pd.DataFrame(rows).to_csv(path, index=False)
        artifacts[filename] = {"rows": len(rows), "sha256":
                               secondary_views.base.sha256_file(path)}
    summary = {
        "schema_version": 1, "kind": "n2_secondary_view_summary",
        "config_sha256": inputs["config_sha256"],
        "plan_manifest_sha256": inputs["plan_manifest_sha256"],
        "plan_dir": str(plan_dir), "runs_dir": str(runs),
        "models": inputs["models"],
        "planned_tasks": inputs["tasks"], "completed_tasks": inputs["tasks"],
        "fold_results": inputs["fold_results"],
        "model_seed_view_groups": len(grouped),
        "bootstrap_replicates": inputs["bootstrap_replicates"],
        "scope_complete": True,
        "complete": len(inputs["models"]) == len(secondary_views.base.ALL_MODELS),
        "intervals": "95% percentile; linked probe_group_id bootstrap within label; "
                     "same draws for all folds and paired document/view metrics; "
                     "conditional on fitted model and validation thresholds",
        "nuisance_membership_sha256": (
            secondary_views.base.sha256_file(nuisance_membership)
            if nuisance_membership is not None else None),
        "nuisance_rows": len(nuisance_rows),
        "note": "Per-seed results only; seeds reuse one corpus and are not independent studies. "
                "Unit eligibility and nuisance retention define secondary populations.",
        "artifacts": artifacts,
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def _read_csv_verified(directory: Path, summary: dict[str, Any],
                       filename: str) -> pd.DataFrame:
    path = directory / filename
    item = summary["artifacts"][filename]
    if not path.is_file() or secondary_views.base.sha256_file(path) != item["sha256"]:
        raise ValueError(f"Missing or changed summary artifact: {path}")
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    if len(frame) != item["rows"]:
        raise ValueError(f"Summary artifact row count differs: {path}")
    return frame


def _check_expected_summary_rows(metrics: pd.DataFrame,
                                 deltas: pd.DataFrame,
                                 models: set[str]) -> None:
    expected_views = (
        {("document_to_views", name) for name in
         (*secondary_views.UNITS, *secondary_views.LAYOUTS)} |
        {("unit_trained", name) for name in secondary_views.UNITS} |
        {("layout_retrained", name) for name in secondary_views.LAYOUTS}
    )
    expected = {(model, seed, mode, view)
                for model in models
                for seed in (20261105, 20261106, 20261107)
                for mode, view in expected_views}
    required = {"model", "seed", "mode", "view", "population"}
    if required - set(metrics) or len(metrics) != len(expected):
        raise ValueError("Summary is missing seed/model/view metrics")
    keys = set(zip(metrics.model, metrics.seed.astype(int), metrics["mode"],
                   metrics["view"]))
    if keys != expected or metrics.duplicated(list(required)).any() or \
            set(metrics.population) != {"full"}:
        raise ValueError("Unexpected or duplicate full-population metric rows")
    if ({"metric"} | required) - set(deltas) or \
            len(deltas) != len(expected) * len(METRICS) or \
            deltas.duplicated([*required, "metric"]).any():
        raise ValueError("Missing or duplicate paired-delta rows")
    delta_keys = set(zip(deltas.model, deltas.seed.astype(int), deltas["mode"],
                         deltas["view"], deltas.metric))
    if delta_keys != {(model, seed, mode, view, metric)
                      for model, seed, mode, view in expected for metric in METRICS}:
        raise ValueError("Paired deltas do not cover all expected metrics")


def _write_figures(deltas: pd.DataFrame, nuisance: pd.DataFrame,
                   output: Path) -> dict[str, dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    names = {
        "roc_auc": "ROC-AUC",
        "tpr_at_fpr_0.01": "TPR no limiar de FPR 1%",
    }
    order = ["surface_length", "surface_presentation", "surface_punctuation",
             "surface_all", "word_lr", "char4_lr", "char4_svm",
             "human_iforest", "bertimbau"]
    columns = [
        ("document_to_views", "paragraph"),
        ("document_to_views", "sentence"),
        ("document_to_views", "layout_blank"),
        ("document_to_views", "layout_all"),
        ("unit_trained", "paragraph"),
        ("unit_trained", "sentence"),
        ("layout_retrained", "layout_blank"),
        ("layout_retrained", "layout_all"),
    ]
    for metric, title in names.items():
        values = deltas.loc[deltas.metric.eq(metric)].groupby(
            ["model", "mode", "view"], sort=True).delta_view_minus_document.mean()
        matrix = np.asarray([[float(values.loc[(model, mode, view)])
                              for mode, view in columns] for model in order])
        limit = max(float(np.abs(matrix).max()), 0.01)
        fig, ax = plt.subplots(figsize=(14, 7))
        plotted = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit,
                            aspect="auto")
        ax.set_yticks(range(len(order)), order)
        ax.set_xticks(range(len(columns)),
                      [f"{mode}\n{view}" for mode, view in columns], rotation=35,
                      ha="right", fontsize=8)
        ax.set_title(f"{title}: vista menos documento nos mesmos IDs")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center",
                        fontsize=7, color="black")
        fig.colorbar(plotted, ax=ax, label="Delta; média descritiva de 3 sementes")
        fig.text(.01, .01, "As sementes reutilizam o mesmo corpus; médias não são ICs independentes.",
                 fontsize=8)
        fig.tight_layout(rect=(0, .03, 1, 1))
        filename = f"paired_{metric}_delta.png"
        path = figures / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        artifacts[f"figures/{filename}"] = {"sha256":
                                              secondary_views.base.sha256_file(path)}
    if not nuisance.empty:
        population = nuisance[["seed", "unit", "tier", "human_retention",
                               "synthetic_retention"]].drop_duplicates()
        grouped = population.groupby(["unit", "tier"], sort=True)[
            ["human_retention", "synthetic_retention"]].mean()
        indexes = [f"{unit}/{tier}" for unit, tier in grouped.index]
        x = np.arange(len(indexes))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - .18, grouped.human_retention, width=.36, label="Humano")
        ax.bar(x + .18, grouped.synthetic_retention, width=.36, label="Sintético")
        ax.set_xticks(x, indexes, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fração retida do teste elegível")
        ax.set_title("Retenção nos controles nuisance")
        ax.legend()
        fig.tight_layout()
        path = figures / "nuisance_retention.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        artifacts["figures/nuisance_retention.png"] = {
            "sha256": secondary_views.base.sha256_file(path)}
    return artifacts


def merge_shards(shards_root: Path, output: Path) -> Path:
    """Merge nine model shards only after all plans and runs are complete."""
    if output.exists():
        raise FileExistsError(f"Merged summary already exists: {output}")
    models = set(secondary_views.base.ALL_MODELS)
    summaries: dict[str, dict[str, Any]] = {}
    tables: dict[str, list[pd.DataFrame]] = defaultdict(list)
    expected_filenames = ("per_seed_view_metrics.csv", "paired_document_deltas.csv",
                          "nuisance_metrics.csv", "nuisance_deltas.csv")
    for model in sorted(models):
        directory = shards_root / model
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"Missing model shard: {model}")
        summary = _json(summary_path)
        if (summary.get("kind") != "n2_secondary_view_summary" or
                summary.get("models") != [model] or
                summary.get("scope_complete") is not True or
                summary.get("bootstrap_replicates") != 2000):
            raise ValueError(f"Invalid model shard: {model}")
        summaries[model] = summary
        for filename in expected_filenames:
            tables[filename].append(_read_csv_verified(directory, summary, filename))
    comparable = ("config_sha256", "plan_manifest_sha256", "plan_dir", "runs_dir",
                  "nuisance_membership_sha256", "bootstrap_replicates")
    reference = summaries[sorted(models)[0]]
    if any(any(item.get(key) != reference.get(key) for key in comparable)
           for item in summaries.values()):
        raise ValueError("Model shards used different inputs or bootstrap settings")
    plan_dir, runs = Path(reference["plan_dir"]), Path(reference["runs_dir"])
    if secondary_views.base.sha256_file(plan_dir / "plan_manifest.json") != \
            reference["plan_manifest_sha256"]:
        raise ValueError("Frozen secondary plan changed after shard summaries")
    checked = secondary_views.check_plan(plan_dir, runs)
    if not checked["complete"]:
        raise ValueError("Cannot merge while secondary tasks remain incomplete")
    combined = {filename: pd.concat(frames, ignore_index=True)
                for filename, frames in tables.items()}
    _check_expected_summary_rows(combined["per_seed_view_metrics.csv"],
                                 combined["paired_document_deltas.csv"], models)
    for filename in ("nuisance_metrics.csv", "nuisance_deltas.csv"):
        frame = combined[filename]
        if not frame.empty and frame.duplicated(
                ["model", "seed", "mode", "view", "population"] +
                (["metric"] if filename == "nuisance_deltas.csv" else [])).any():
            raise ValueError(f"Duplicate nuisance rows across shards: {filename}")
    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, Any] = {}
    for filename, frame in combined.items():
        path = output / filename
        frame.to_csv(path, index=False)
        artifacts[filename] = {"rows": len(frame), "sha256":
                               secondary_views.base.sha256_file(path)}
    artifacts.update(_write_figures(combined["paired_document_deltas.csv"],
                                    combined["nuisance_metrics.csv"], output))
    result = {
        "schema_version": 1, "kind": "n2_secondary_view_summary",
        "complete": True, "models": sorted(models),
        "config_sha256": reference["config_sha256"],
        "plan_manifest_sha256": reference["plan_manifest_sha256"],
        "nuisance_membership_sha256": reference["nuisance_membership_sha256"],
        "planned_tasks": sum(item["planned_tasks"] for item in summaries.values()),
        "completed_tasks": sum(item["completed_tasks"] for item in summaries.values()),
        "model_seed_view_groups": len(combined["per_seed_view_metrics.csv"]),
        "bootstrap_replicates": 2000,
        "intervals": reference["intervals"],
        "interpretation": "Deltas compare document and derived views on exactly the same "
                          "source IDs with shared linked-group draws. Seeds are descriptive "
                          "reuses of one corpus; nuisance populations are selected controls.",
        "shards": {model: {"sha256": secondary_views.base.sha256_file(
            shards_root / model / "summary.json")} for model in sorted(models)},
        "artifacts": artifacts,
    }
    path = output / "summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "merge":
        parser = argparse.ArgumentParser(description="Merge verified model shards")
        parser.add_argument("--shards-root", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        print(merge_shards(args.shards_root, args.output))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nuisance-membership", type=Path)
    parser.add_argument("--models", nargs="+", choices=secondary_views.base.ALL_MODELS)
    args = parser.parse_args(argv)
    print(summarize(args.config, args.plan_dir, args.runs, args.output,
                    args.nuisance_membership,
                    set(args.models) if args.models else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
