"""Aggregate verified IID/OOD score files without rereading corpus text.

Fold predictions share the same test documents, so confidence intervals draw
linked probe groups once and apply that draw to every fold. Seed means/ranges
are descriptive; repeated splits of one corpus are not independent datasets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from horus_benchmark.metrics.binary_detection import metric_values


METRICS = (
    "roc_auc", "balanced_accuracy", "tpr_at_fpr_0.01",
    "observed_fpr_at_fpr_0.01", "balanced_accuracy_at_fpr_0.01",
    "tpr_at_fpr_0.05", "observed_fpr_at_fpr_0.05",
    "balanced_accuracy_at_fpr_0.05",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Round:
    task_id: str
    model: str
    protocol: str
    target: str
    seed: int
    fold: int
    thresholds: dict[str, float]
    predictions: pd.DataFrame | None
    task_metrics: dict[str, float]
    n3s_metrics: dict[str, float] | None = None


def _read_completed(plan: pd.DataFrame, runs: Path, *, allow_incomplete: bool
                    ) -> tuple[list[Round], list[str]]:
    required = {"task_id", "model", "protocol", "target", "seed", "fold"}
    if required - set(plan) or plan.task_id.duplicated().any():
        raise ValueError("Invalid or duplicate task IDs in the execution plan")
    completed: list[Round] = []
    missing: list[str] = []
    for task in plan.itertuples(index=False):
        directory = runs / task.task_id
        metrics_path = directory / "metrics.json"
        predictions_path = directory / "predictions.parquet"
        if not metrics_path.is_file() or not predictions_path.is_file():
            missing.append(task.task_id)
            continue
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        identity = record.get("task", {})
        for name in ("task_id", "model", "protocol", "target", "seed", "fold",
                     "manifest_sha256", "counterfactual_n3s"):
            if name not in plan:
                continue
            if str(identity.get(name)) != str(getattr(task, name)):
                raise ValueError(f"Task identity mismatch: {task.task_id}/{name}")
        if _sha256(predictions_path) != record.get("predictions_sha256"):
            raise ValueError(f"Score file hash mismatch: {task.task_id}")
        thresholds = record["evaluation"]["validation_thresholds"]
        values = record["evaluation"]["metrics"]
        if set(values) != set(METRICS) or set(thresholds) != {"youden", "fpr_0.01", "fpr_0.05"}:
            raise ValueError(f"Metric schema mismatch: {task.task_id}")
        n3s_values = record.get("n3s_inference_only_metrics")
        expected_n3s = str(getattr(task, "counterfactual_n3s", False)).lower() == "true"
        if (n3s_values is None) == expected_n3s:
            raise ValueError(f"N3s counterfactual flag/metrics mismatch: {task.task_id}")
        if n3s_values is not None and set(n3s_values) != set(METRICS):
            raise ValueError(f"N3s metric schema mismatch: {task.task_id}")
        completed.append(Round(task.task_id, task.model, task.protocol,
                               str(task.target), int(task.seed), int(task.fold),
                               thresholds, None, values, n3s_values))
    if missing and not allow_incomplete:
        raise ValueError(f"{len(missing)} planned tasks lack metrics/predictions; "
                         "rerun failed jobs or use --allow-incomplete")
    return completed, missing


def _materialize(round_: Round, runs: Path) -> Round:
    """Load one score file only while its model/protocol/target/seed is reduced."""
    directory = runs / round_.task_id
    frame = pd.read_parquet(directory / "predictions.parquet")
    fields = {"record_id", "label_name", "probe_group_id", "role",
              "paired_iid_test", "score_n2", "broad_area", "word_count_audit"}
    if (fields - set(frame) or frame.record_id.duplicated().any() or
            frame.probe_group_id.isna().any() or frame.paired_iid_test.isna().any()):
        raise ValueError(f"Invalid score rows: {round_.task_id}")
    if (frame.score_n2.isna().any() or not np.isfinite(frame.score_n2).all() or
            set(frame.label_name) - {"human", "synthetic"}):
        raise ValueError(f"Invalid scores or labels: {round_.task_id}")
    if frame.groupby("probe_group_id").label_name.nunique().gt(1).any():
        raise ValueError(f"A score group spans labels: {round_.task_id}")
    test = frame.loc[frame.role.isin(["test_target", "test_control"])]
    record = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if len(test) != record["counts"]["binary_test"]:
        raise ValueError(f"Binary test count mismatch: {round_.task_id}")
    recomputed = metric_values(test.label_name.eq("synthetic").to_numpy(dtype=int),
                               test.score_n2.to_numpy(dtype=float), round_.thresholds)
    if any(not np.isclose(recomputed[key], round_.task_metrics[key],
                          rtol=1e-10, atol=1e-12) for key in METRICS):
        raise ValueError(f"Metrics do not match scored predictions: {round_.task_id}")
    if round_.n3s_metrics is not None:
        if ("score_n3s" not in frame or frame.score_n3s.isna().any() or
                not np.isfinite(frame.score_n3s).all()):
            raise ValueError(f"N3s scores are missing or invalid: {round_.task_id}")
        n3s = metric_values(test.label_name.eq("synthetic").to_numpy(dtype=int),
                            test.score_n3s.to_numpy(dtype=float), round_.thresholds)
        if any(not np.isclose(n3s[key], round_.n3s_metrics[key],
                              rtol=1e-10, atol=1e-12) for key in METRICS):
            raise ValueError(f"N3s metrics do not match scored predictions: {round_.task_id}")
    elif "score_n3s" in frame:
        raise ValueError(f"Unexpected N3s scores: {round_.task_id}")
    return Round(round_.task_id, round_.model, round_.protocol, round_.target,
                 round_.seed, round_.fold, round_.thresholds, frame,
                 round_.task_metrics, round_.n3s_metrics)


def _aligned(rounds: list[Round], *, paired_only: bool = False
             ) -> tuple[pd.DataFrame, list[np.ndarray], list[dict[str, float]]]:
    """Align the same test IDs across folds; never stack folds as new people."""
    if not rounds:
        raise ValueError("No model rounds")
    ordered = sorted(rounds, key=lambda item: item.fold)
    reference: pd.DataFrame | None = None
    scores: list[np.ndarray] = []
    thresholds: list[dict[str, float]] = []
    for round_ in ordered:
        if round_.predictions is None:
            raise ValueError("Score file was not materialized")
        frame = round_.predictions.loc[round_.predictions.role.isin(
            ["test_target", "test_control"])].copy()
        if paired_only:
            frame = frame.loc[frame.paired_iid_test.astype(bool)]
        frame = frame.sort_values("record_id").reset_index(drop=True)
        if reference is None:
            reference = frame
            if not len(frame) or set(frame.label_name) != {"human", "synthetic"}:
                raise ValueError("Binary comparison lacks a class")
        elif (not frame.record_id.equals(reference.record_id) or
              not frame.label_name.equals(reference.label_name) or
              not frame.probe_group_id.equals(reference.probe_group_id)):
            raise ValueError("Fold test IDs, labels or groups differ")
        scores.append(frame.score_n2.to_numpy(dtype=float))
        thresholds.append(round_.thresholds)
    assert reference is not None
    return reference, scores, thresholds


def _group_positions(frame: pd.DataFrame) -> dict[int, list[np.ndarray]]:
    groups: dict[int, list[np.ndarray]] = {0: [], 1: []}
    labels = frame.label_name.eq("synthetic").to_numpy(dtype=int)
    for _, subset in frame.reset_index().groupby("probe_group_id", sort=False):
        positions = subset["index"].to_numpy(dtype=int)
        class_values = np.unique(labels[positions])
        if len(class_values) != 1:
            raise ValueError("Bootstrap group spans labels")
        groups[int(class_values[0])].append(positions)
    if not groups[0] or not groups[1]:
        raise ValueError("Bootstrap requires both labels")
    return groups


def _sample_group_positions(groups: dict[int, list[np.ndarray]],
                            rng: np.random.Generator) -> np.ndarray:
    picked = []
    for label in (0, 1):
        members = groups[label]
        picked.extend(members[int(index)] for index in rng.integers(
            0, len(members), size=len(members)))
    return np.concatenate(picked)


def _mean_fold_metrics(labels: np.ndarray, scores: list[np.ndarray],
                       thresholds: list[dict[str, float]],
                       positions: np.ndarray | None = None) -> dict[str, float]:
    if positions is None:
        positions = np.arange(len(labels))
    measures = [metric_values(labels[positions], score[positions], threshold)
                for score, threshold in zip(scores, thresholds)]
    return {key: float(np.mean([item[key] for item in measures])) for key in METRICS}


def _fold_bootstrap(frame: pd.DataFrame, scores: list[np.ndarray],
                    thresholds: list[dict[str, float]], *, replicates: int,
                    seed: int) -> dict[str, tuple[float, float]]:
    labels = frame.label_name.eq("synthetic").to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    groups = _group_positions(frame)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        positions = _sample_group_positions(groups, rng)
        result = _mean_fold_metrics(labels, scores, thresholds, positions)
        for key, value in result.items():
            values[key].append(value)
    return {key: (float(np.quantile(samples, .025)),
                  float(np.quantile(samples, .975)))
            for key, samples in values.items()}


def _seed_summary(rounds: list[Round], *, replicates: int, seed: int) -> dict[str, Any]:
    frame, scores, thresholds = _aligned(rounds)
    labels = frame.label_name.eq("synthetic").to_numpy(dtype=int)
    point = _mean_fold_metrics(labels, scores, thresholds)
    interval = _fold_bootstrap(frame, scores, thresholds,
                               replicates=replicates, seed=seed)
    return {"model": rounds[0].model, "protocol": rounds[0].protocol,
            "target": rounds[0].target, "seed": rounds[0].seed,
            "folds": len(rounds), "test_rows": len(frame),
            "human_rows": int((labels == 0).sum()),
            "synthetic_rows": int((labels == 1).sum()),
            **point,
            **{f"{key}_ci_low": bounds[0] for key, bounds in interval.items()},
            **{f"{key}_ci_high": bounds[1] for key, bounds in interval.items()}}


def _n3s_summary(rounds: list[Round], *, replicates: int, seed: int) -> dict[str, Any]:
    """Inference-only N3s minus N2 on the same IID IDs and N2 thresholds."""
    if (rounds[0].protocol != "iid" or len(rounds) != 5 or
            any(item.n3s_metrics is None for item in rounds)):
        raise ValueError("N3s aggregate requires five complete IID counterfactual folds")
    frame, n2_scores, thresholds = _aligned(rounds)
    ordered = sorted(rounds, key=lambda item: item.fold)
    n3s_scores = [item.predictions.loc[item.predictions.role.isin(
        ["test_target", "test_control"])].sort_values("record_id")
                  .score_n3s.to_numpy(dtype=float) for item in ordered]
    labels = frame.label_name.eq("synthetic").to_numpy(dtype=int)
    n2_point = _mean_fold_metrics(labels, n2_scores, thresholds)
    n3s_point = _mean_fold_metrics(labels, n3s_scores, thresholds)
    rng = np.random.default_rng(seed)
    groups = _group_positions(frame)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        positions = _sample_group_positions(groups, rng)
        before = _mean_fold_metrics(labels, n2_scores, thresholds, positions)
        after = _mean_fold_metrics(labels, n3s_scores, thresholds, positions)
        for key in METRICS:
            draws[key].append(after[key] - before[key])
    return {"model": rounds[0].model, "seed": rounds[0].seed,
            "folds": len(rounds), "test_rows": len(frame),
            "interpretation": "N3s inference only; model and N2 thresholds unchanged",
            **{f"n2_{key}": n2_point[key] for key in METRICS},
            **{f"n3s_{key}": n3s_point[key] for key in METRICS},
            **{f"delta_{key}": n3s_point[key] - n2_point[key] for key in METRICS},
            **{f"delta_{key}_ci_low": float(np.quantile(draws[key], .025))
               for key in METRICS},
            **{f"delta_{key}_ci_high": float(np.quantile(draws[key], .975))
               for key in METRICS}}


def _paired_summary(ood_rounds: list[Round], iid_rounds: list[Round],
                    *, replicates: int, seed: int) -> dict[str, Any] | None:
    iid_by_fold = {round_.fold: round_ for round_ in iid_rounds}
    ood_ordered = sorted(ood_rounds, key=lambda item: item.fold)
    if any(round_.fold not in iid_by_fold for round_ in ood_ordered):
        return None
    try:
        target_frame, ood_scores, ood_thresholds = _aligned(ood_ordered, paired_only=True)
    except ValueError as error:
        if str(error) == "Binary comparison lacks a class":
            return None
        raise
    common = target_frame[["record_id", "label_name", "probe_group_id"]].copy()
    iid_scores: list[np.ndarray] = []
    iid_thresholds: list[dict[str, float]] = []
    for ood in ood_ordered:
        iid = iid_by_fold[ood.fold]
        if iid.predictions is None:
            raise ValueError("IID scores were not materialized")
        iid_frame = iid.predictions.loc[
            iid.predictions.role.eq("test_target") &
            iid.predictions.paired_iid_test.astype(bool),
            ["record_id", "label_name", "probe_group_id", "score_n2"]]
        joined = common.merge(iid_frame, on="record_id", how="left", validate="one_to_one",
                              suffixes=("", "_iid"))
        if (joined.score_n2.isna().any() or
                not joined.label_name.eq(joined.label_name_iid).all() or
                not joined.probe_group_id.eq(joined.probe_group_id_iid).all()):
            raise ValueError("Paired IID/OOD comparison IDs or metadata disagree")
        iid_scores.append(joined.score_n2.to_numpy(dtype=float))
        iid_thresholds.append(iid.thresholds)
    labels = target_frame.label_name.eq("synthetic").to_numpy(dtype=int)
    iid_point = _mean_fold_metrics(labels, iid_scores, iid_thresholds)
    ood_point = _mean_fold_metrics(labels, ood_scores, ood_thresholds)
    rng = np.random.default_rng(seed)
    groups = _group_positions(target_frame)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        positions = _sample_group_positions(groups, rng)
        iid_b = _mean_fold_metrics(labels, iid_scores, iid_thresholds, positions)
        ood_b = _mean_fold_metrics(labels, ood_scores, ood_thresholds, positions)
        for key in METRICS:
            draws[key].append(ood_b[key] - iid_b[key])
    return {"model": ood_rounds[0].model, "protocol": ood_rounds[0].protocol,
            "target": ood_rounds[0].target, "seed": ood_rounds[0].seed,
            "folds": len(ood_rounds), "paired_rows": len(target_frame),
            "paired_human_rows": int((labels == 0).sum()),
            "paired_synthetic_rows": int((labels == 1).sum()),
            **{f"iid_{key}": iid_point[key] for key in METRICS},
            **{f"ood_{key}": ood_point[key] for key in METRICS},
            **{f"delta_{key}": ood_point[key] - iid_point[key] for key in METRICS},
            **{f"delta_{key}_ci_low": float(np.quantile(draws[key], .025))
               for key in METRICS},
            **{f"delta_{key}_ci_high": float(np.quantile(draws[key], .975))
               for key in METRICS}}


def _iid_subgroups(rounds: list[Round]) -> list[dict[str, Any]]:
    frame, scores, thresholds = _aligned(rounds)
    edges = [0, 100, 250, 500, 1000, 2000, 5000, np.inf]
    frame = frame.copy()
    frame["length_bin"] = pd.cut(frame.word_count_audit, bins=edges,
                                  right=False, include_lowest=True).astype(str)
    rows: list[dict[str, Any]] = []
    for factor in ("broad_area", "length_bin"):
        for value, subset in frame.groupby(factor, sort=True):
            positions = subset.index.to_numpy(dtype=int)
            labels = subset.label_name.eq("synthetic").to_numpy(dtype=int)
            if min((labels == 0).sum(), (labels == 1).sum()) < 100:
                continue
            result = _mean_fold_metrics(
                frame.label_name.eq("synthetic").to_numpy(dtype=int),
                scores, thresholds, positions)
            rows.append({"model": rounds[0].model, "seed": rounds[0].seed,
                         "factor": factor, "value": value,
                         "human_rows": int((labels == 0).sum()),
                         "synthetic_rows": int((labels == 1).sum()), **result})
    return rows


def _plot_iid_ood(seed_table: pd.DataFrame, macro: pd.DataFrame,
                  output: Path) -> list[str]:
    """Plot descriptive seed points; no inferential error bars across reused data."""
    if macro.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iid = seed_table.loc[seed_table.protocol.eq("iid"),
                         ["model", "protocol", "seed", *METRICS]]
    plot_data = pd.concat((iid, macro[["model", "protocol", "seed", *METRICS]]),
                          ignore_index=True)
    order = ["iid", "logo", "lodo", "loso"]
    measures = [("roc_auc", "ROC-AUC"),
                ("balanced_accuracy", "Balanced accuracy"),
                ("tpr_at_fpr_0.01", "TPR @ FPR 1%")]
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    names: list[str] = []
    for model, model_rows in plot_data.groupby("model", sort=True):
        protocols = [name for name in order if name in set(model_rows.protocol)]
        if len(protocols) < 2:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
        for axis, (column, title) in zip(axes, measures):
            for seed, seed_rows in model_rows.groupby("seed", sort=True):
                values = seed_rows.set_index("protocol").reindex(protocols)[column]
                axis.plot(range(len(protocols)), values, marker="o", linewidth=1.3,
                          markersize=4, alpha=.75, label=f"seed {seed}")
            axis.set_title(title)
            axis.set_xticks(range(len(protocols)), [name.upper() for name in protocols])
            axis.set_ylim(0, 1.02)
            axis.grid(axis="y", alpha=.25)
        axes[0].set_ylabel("Métrica")
        axes[-1].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{model}: IID e OOD (macro de alvos; mesmas amostras entre folds)")
        fig.tight_layout()
        filename = f"iid_ood_{re.sub(r'[^a-zA-Z0-9_-]+', '_', model)}.png"
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)
        names.append(str(Path("figures") / filename))
    return names


def _expected_folds(model: str, protocol: str) -> set[int]:
    """Frozen v1 evaluation design: five folds, except one BERTimbau OOD fold."""
    return {0} if model == "bertimbau" and protocol != "iid" else set(range(5))


def summarize(plan_paths: Path | list[Path], runs: Path, output: Path, *, replicates: int = 2000,
              allow_incomplete: bool = False) -> Path:
    if replicates < 1 or output.exists():
        raise ValueError("Positive replicates and a new output directory are required")
    if isinstance(plan_paths, Path):
        plan_paths = [plan_paths]
    if not plan_paths:
        raise ValueError("At least one plan is required")
    plan = pd.concat((pd.read_csv(path, keep_default_na=False) for path in plan_paths),
                     ignore_index=True)
    completed, missing = _read_completed(plan, runs, allow_incomplete=allow_incomplete)
    if not completed:
        raise ValueError("No completed tasks to summarize")
    by_key: dict[tuple[str, str, str, int], list[Round]] = defaultdict(list)
    for item in completed:
        by_key[(item.model, item.protocol, item.target, item.seed)].append(item)
    for key, group in by_key.items():
        if len({item.fold for item in group}) != len(group):
            raise ValueError(f"Duplicate fold in {key}")
    planned = plan.groupby(["model", "protocol", "target", "seed"]).fold.agg(
        lambda values: set(int(value) for value in values))
    complete_keys = [key for key, group in by_key.items()
                     if ({item.fold for item in group} == planned.loc[key] ==
                         _expected_folds(key[0], key[1]) and
                         len(group) == len(planned.loc[key]))]
    if not complete_keys:
        raise ValueError("No complete seed/protocol/model target remains")
    iid = {(model, seed): by_key[(model, "iid", "all", seed)]
           for model, protocol, target, seed in complete_keys
           if protocol == "iid" and target == "all"}
    seed_rows = []
    paired = []
    paired_skipped = []
    subgroup = []
    n3s_seed_rows = []
    for model, protocol, target, seed in sorted(complete_keys):
        group = [_materialize(item, runs) for item in by_key[(model, protocol, target, seed)]]
        seed_rows.append(_seed_summary(group, replicates=replicates,
                         seed=int(hashlib.sha256(str((model, protocol, target, seed)).encode()
                         ).hexdigest()[:8], 16)))
        if protocol == "iid":
            subgroup.extend(_iid_subgroups(group))
            if all(item.n3s_metrics is not None for item in group):
                n3s_seed_rows.append(_n3s_summary(
                    group, replicates=replicates,
                    seed=int(hashlib.sha256(f"n3s:{model}:{seed}".encode()
                    ).hexdigest()[:8], 16)))
            continue
        if (model, seed) not in iid:
            paired_skipped.append({"model": model, "protocol": protocol,
                                   "target": target, "seed": seed,
                                   "reason": "iid_rounds_missing"})
            continue
        iid_group = [_materialize(item, runs) for item in iid[(model, seed)]
                     if item.fold in {ood.fold for ood in group}]
        result = _paired_summary(group, iid_group, replicates=replicates,
                                 seed=int(hashlib.sha256(
                                     f"paired:{model}:{protocol}:{target}:{seed}".encode()
                                 ).hexdigest()[:8], 16))
        if result is not None:
            paired.append(result)
        else:
            paired_skipped.append({"model": model, "protocol": protocol,
                                   "target": target, "seed": seed,
                                   "reason": "paired_binary_support_missing"})
    seed_table = pd.DataFrame(seed_rows)
    macro = seed_table.loc[seed_table.protocol.ne("iid")].groupby(
        ["model", "protocol", "seed"], as_index=False)[list(METRICS)].mean()
    if len(macro):
        counts = seed_table.loc[seed_table.protocol.ne("iid")].groupby(
            ["model", "protocol", "seed"]).target.nunique().rename("targets").reset_index()
        macro = macro.merge(counts, on=["model", "protocol", "seed"], validate="one_to_one")
    per_round = []
    n3s_per_round = []
    for item in completed:
        per_round.append({"model": item.model, "protocol": item.protocol,
                          "target": item.target, "seed": item.seed, "fold": item.fold,
                          "task_id": item.task_id, **item.task_metrics})
        if item.n3s_metrics is not None:
            n3s_per_round.append({
                "model": item.model, "protocol": item.protocol, "target": item.target,
                "seed": item.seed, "fold": item.fold, "task_id": item.task_id,
                **{f"n2_{key}": item.task_metrics[key] for key in METRICS},
                **{f"n3s_{key}": item.n3s_metrics[key] for key in METRICS},
                **{f"delta_{key}": item.n3s_metrics[key] - item.task_metrics[key]
                   for key in METRICS}})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(per_round).to_csv(output / "round_metrics.csv", index=False)
    seed_table.to_csv(output / "seed_metrics.csv", index=False)
    pd.DataFrame([
        {"model": model, "protocol": protocol, "target": target, "seed": seed,
         "planned_folds": len(folds),
         "completed_folds": len(by_key.get((model, protocol, target, seed), [])),
         "expected_folds": len(_expected_folds(model, protocol)),
         "complete": (model, protocol, target, seed) in complete_keys}
        for (model, protocol, target, seed), folds in planned.items()
    ]).to_csv(output / "fold_coverage.csv", index=False)
    macro.to_csv(output / "ood_macro_by_seed.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_iid_ood_deltas.csv", index=False)
    pd.DataFrame(paired_skipped).to_csv(output / "paired_comparison_skips.csv", index=False)
    pd.DataFrame(subgroup).to_csv(output / "iid_subgroup_metrics.csv", index=False)
    n3s_prefix = ["model", "protocol", "target", "seed", "fold", "task_id"]
    n3s_values = [f"{prefix}_{key}" for prefix in ("n2", "n3s", "delta")
                  for key in METRICS]
    pd.DataFrame(n3s_per_round, columns=n3s_prefix + n3s_values).to_csv(
        output / "n3s_inference_only_rounds.csv", index=False)
    n3s_seed_prefix = ["model", "seed", "folds", "test_rows", "interpretation"]
    n3s_intervals = [f"delta_{key}_ci_{bound}" for key in METRICS
                     for bound in ("low", "high")]
    pd.DataFrame(n3s_seed_rows,
                 columns=n3s_seed_prefix + n3s_values + n3s_intervals).to_csv(
                     output / "n3s_inference_only_iid_by_seed.csv", index=False)
    figures = _plot_iid_ood(seed_table, macro, output)
    summary = {"schema_version": 1, "kind": "n2_text_baseline_summary",
               "plans": [{"path": str(path), "sha256": _sha256(path)}
                         for path in plan_paths],
               "planned_tasks": len(plan), "completed_tasks": len(completed),
               "complete": not missing and len(planned) == len(complete_keys),
               "complete_seed_targets": len(seed_rows), "missing_task_ids": missing,
               "incomplete_seed_targets": int(len(planned) - len(complete_keys)),
               "paired_seed_targets": len(paired),
               "paired_seed_targets_skipped": len(paired_skipped),
               "n3s_iid_seed_models": len(n3s_seed_rows),
               "n3s_note": "N3s is inference-only with the N2-trained detector and frozen "
                           "N2 validation thresholds; deltas are paired on the same IDs "
                           "and shared group-bootstrap draws. No N3s retraining occurred.",
               "figures": figures,
               "bootstrap_replicates": replicates,
               "intervals": "95% percentile, probe-group bootstrap within label; same draw "
                            "across folds and paired IID/OOD scores; conditional on fit "
                            "and validation thresholds",
               "fold_note": "A group is complete when every fold listed in its plan has a "
                            "verified run. BERTimbau OOD may have one planned fold; its "
                            "estimate is single-fold, not a five-fold mean.",
               "interpretation": "OOD macro gives each eligible held-out target equal weight. "
                                 "Three seeds reuse one corpus and are not independent studies. "
                                 "Paired deltas use only the common frozen IID test IDs."}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--plan", type=Path)
    choices.add_argument("--plans", type=Path, nargs="+")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    print(summarize(args.plans if args.plans else args.plan, args.runs, args.output,
                    replicates=args.bootstrap_replicates,
                    allow_incomplete=args.allow_incomplete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
