"""Verified, sharded summaries and exact-ID error agreements for N2 detectors.

Score validity is part of the result: no invalid text silently disappears from
coverage. Fold means use their common valid binary test IDs, and bootstrap
draws resample linked probe groups once across all folds and paired scores.
Seeds reuse one corpus; descriptive seed/target means are not independent CIs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from horus_benchmark.evaluation.advanced_data import config
from horus_benchmark.evaluation.advanced_runner import task_id
from horus_benchmark.evaluation.research_io import read_json, sha256, write_json
from horus_benchmark.evaluation.summarize_benchmarks import (
    METRICS, Round, _group_positions, _mean_fold_metrics,
    _sample_group_positions, _seed_summary,
)
from horus_benchmark.metrics.binary_detection import metric_values


GROUP = ["model", "protocol", "target", "source_area", "seed", "text_view"]
BINARY = {"test_target", "test_control"}
THRESHOLDS = {"youden", "fpr_0.01", "fpr_0.05"}
IDENTITY = ["record_id", "label_name", "probe_group_id"]


def _same_identity(one: pd.DataFrame, two: pd.DataFrame) -> bool:
    return (len(one) == len(two) and all(one[key].astype(str).reset_index(drop=True).equals(
        two[key].astype(str).reset_index(drop=True)) for key in IDENTITY))


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _plans(paths: list[Path], config_sha: str | None = None) -> tuple[pd.DataFrame, dict]:
    parts, hashes = [], {}
    for path in paths:
        marker = read_json(path.with_suffix(".json"))
        if (marker.get("kind") != "horus_advanced_plan" or
                marker.get("sha256") != sha256(path) or
                (config_sha is not None and marker.get("config_sha256") != config_sha)):
            raise ValueError(f"Advanced plan/config changed: {path}")
        frame = pd.read_csv(path, keep_default_na=False)
        if ({*GROUP, "task_id", "fold", "config_sha256"} - set(frame) or
                len(frame) != marker["rows"] or
                any(row["task_id"] != task_id(row) for row in frame.to_dict("records"))):
            raise ValueError(f"Invalid advanced task plan: {path}")
        parts.append(frame)
        hashes[str(path)] = {"sha256": sha256(path),
                             "marker_sha256": sha256(path.with_suffix(".json"))}
    result = pd.concat(parts, ignore_index=True)
    if result.task_id.duplicated().any():
        raise ValueError("Duplicate task IDs across summary input plans")
    return result, hashes


def plan(config_path: Path, paths: list[Path], output: Path) -> Path:
    cfg = config(config_path)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Summary shard plan exists: {output}")
    tasks, inputs = _plans(paths, sha256(config_path))
    rows = []
    for key, group in tasks.groupby(GROUP, sort=True, dropna=False):
        identity = dict(zip(GROUP, key))
        expected = cfg["iid_folds"] if identity["protocol"] == "iid" else cfg["ood_folds"]
        if sorted(group.fold.tolist()) != expected:
            raise ValueError(f"Summary requires all approved folds: {identity}")
        references = {}
        if identity["protocol"] != "iid":
            selected = tasks.loc[
                tasks.model.eq(identity["model"]) & tasks.seed.eq(identity["seed"]) &
                tasks.text_view.eq(identity["text_view"]) & tasks.protocol.eq("iid") &
                tasks.fold.isin(expected)]
            if sorted(selected.fold.tolist()) != expected:
                raise ValueError(f"Missing same-model IID reference: {identity}")
            references["iid"] = selected.to_dict("records")
        if identity["protocol"] == "logo_lodo":
            generator, area = str(identity["target"]).split("::", 1)
            for protocol, target in (("logo", generator), ("lodo", area)):
                selected = tasks.loc[
                    tasks.model.eq(identity["model"]) & tasks.seed.eq(identity["seed"]) &
                    tasks.text_view.eq(identity["text_view"]) & tasks.protocol.eq(protocol) &
                    tasks.target.eq(target) & tasks.fold.isin(expected)]
                if sorted(selected.fold.tolist()) != expected:
                    raise ValueError(f"Missing compound reference {protocol}: {identity}")
                references[protocol] = selected.to_dict("records")
        rows.append({**identity, "shard_index": len(rows),
                     "tasks_json": json.dumps(group.sort_values("fold").to_dict("records"), sort_keys=True),
                     "references_json": json.dumps(references, sort_keys=True)})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    write_json(output.with_suffix(".json"), {
        "kind": "horus_advanced_summary_plan", "config_path": str(config_path),
        "config_sha256": sha256(config_path), "plan_sha256": sha256(output),
        "rows": len(rows), "planned_tasks": len(tasks), "inputs": inputs,
        "bootstrap_replicates": cfg["bootstrap_replicates"],
    })
    return output


def _shard_plan(path: Path) -> tuple[pd.DataFrame, dict]:
    marker = read_json(path.with_suffix(".json"))
    if (marker.get("kind") != "horus_advanced_summary_plan" or
            marker["plan_sha256"] != sha256(path) or
            marker["config_sha256"] != sha256(marker["config_path"])):
        raise ValueError("Advanced summary shard plan/config changed")
    for source, expected in marker["inputs"].items():
        if (sha256(source) != expected["sha256"] or
                sha256(Path(source).with_suffix(".json")) != expected["marker_sha256"]):
            raise ValueError("Advanced summary source plan changed")
    frame = pd.read_csv(path, keep_default_na=False)
    if (len(frame) != marker["rows"] or
            frame.shard_index.tolist() != list(range(len(frame)))):
        raise ValueError("Invalid advanced summary shard identities")
    return frame, marker


def _thresholds(value: dict) -> dict[str, float]:
    if set(value) != THRESHOLDS:
        raise ValueError("Unexpected validation threshold schema")
    result = {key: float(item) for key, item in value.items()}
    if any(np.isnan(item) or item == -np.inf for item in result.values()):
        raise ValueError("Invalid detector threshold")
    return result


def load_result(task: dict, runs: Path) -> tuple[Round | None, dict, dict]:
    directory = runs / str(task["task_id"])
    record_path = directory / "metrics.json"
    record = read_json(record_path)
    if (record.get("kind") != "horus_advanced_result" or any(
            str(record.get("task", {}).get(key)) != str(value) for key, value in task.items())):
        raise ValueError(f"Advanced task output identity mismatch: {task['task_id']}")
    hashes = {str(record_path): sha256(record_path)}
    if record.get("status") == "unsupported":
        if not record.get("reason"):
            raise ValueError("Unsupported result requires a reason")
        return None, record, hashes
    if record.get("status") != "complete" or record.get("evaluation") is None:
        raise ValueError(f"Incomplete advanced result: {task['task_id']}")
    score_path = directory / "predictions.parquet"
    if sha256(score_path) != record["predictions_sha256"]:
        raise ValueError("Advanced prediction artifact changed")
    hashes[str(score_path)] = sha256(score_path)
    frame = pd.read_parquet(score_path)
    needed = {*IDENTITY, "role", "paired_iid_test", "status", "score_n2"}
    if (needed - set(frame) or frame.record_id.duplicated().any() or
            frame.probe_group_id.isna().any() or frame.paired_iid_test.isna().any() or
            set(frame.label_name) - {"human", "synthetic"} or
            set(frame.status) - {"ok", "invalid"}):
        raise ValueError("Invalid advanced prediction schema")
    if frame.groupby("probe_group_id").label_name.nunique().gt(1).any():
        raise ValueError("Advanced prediction group spans labels")
    if not np.isfinite(frame.loc[frame.status.eq("ok"), "score_n2"].to_numpy(dtype=float)).all():
        raise ValueError("Valid advanced scores must be finite")
    thresholds = _thresholds(record["evaluation"]["validation_thresholds"])
    binary = frame.loc[frame.role.isin(BINARY) & frame.status.eq("ok")]
    if len(binary) != record["counts"]["binary_test"]:
        raise ValueError("Advanced binary count mismatch")
    metrics = metric_values(binary.label_name.eq("synthetic").to_numpy(dtype=np.int8),
                            binary.score_n2.to_numpy(dtype=float), thresholds)
    saved = record["evaluation"]["metrics"]
    if set(saved) != set(METRICS) or any(not np.isclose(
            metrics[key], float(saved[key]), rtol=1e-10, atol=1e-12) for key in METRICS):
        raise ValueError("Advanced metrics differ from valid scores")
    result = Round(str(task["task_id"]), str(task["model"]), str(task["protocol"]),
                   str(task["target"]), int(task["seed"]), int(task["fold"]),
                   thresholds, frame, metrics)
    return result, record, hashes


def common_valid(rounds: list[Round]) -> tuple[list[Round], dict]:
    """One binary population across folds, with explicit full denominators."""
    if not rounds:
        raise ValueError("No advanced rounds")
    ids, reference, coverage = None, None, []
    for item in sorted(rounds, key=lambda value: value.fold):
        assert item.predictions is not None
        full = item.predictions.loc[item.predictions.role.isin(BINARY)].sort_values(
            "record_id", kind="stable").reset_index(drop=True)
        if reference is None:
            reference = full[IDENTITY]
        elif not _same_identity(full, reference):
            raise ValueError("Frozen binary test identities differ across folds")
        valid = set(full.loc[full.status.eq("ok"), "record_id"])
        ids = valid if ids is None else ids & valid
        coverage.append({"fold": item.fold, "binary_total": len(full),
                         "binary_valid": len(valid)})
    assert reference is not None and ids is not None
    retained = reference.loc[reference.record_id.isin(ids)]
    counts = retained.label_name.value_counts()
    total_counts = reference.label_name.value_counts()
    result = [replace(item, predictions=item.predictions.loc[
        item.predictions.record_id.isin(ids) & item.predictions.role.isin(BINARY)].copy())
              for item in rounds]
    return result, {"binary_total": len(reference), "binary_common_valid": len(ids),
                    "human_total": int(total_counts.get("human", 0)),
                    "synthetic_total": int(total_counts.get("synthetic", 0)),
                    "human_common_valid": int(counts.get("human", 0)),
                    "synthetic_common_valid": int(counts.get("synthetic", 0)),
                    "retention": len(ids) / len(reference) if len(reference) else 0.0,
                    "fold_coverage_json": json.dumps(coverage, sort_keys=True)}


def paired_delta(new: list[Round], reference: list[Round], *, replicates: int,
                 seed: int) -> dict:
    refs = {item.fold: item for item in reference}
    ordered = sorted(new, key=lambda item: item.fold)
    if len(refs) != len(reference) or any(item.fold not in refs for item in ordered):
        return {"status": "unsupported", "reason": "reference_fold_missing", "paired_rows": 0}
    common = None
    frames = []
    for item in ordered:
        other = refs[item.fold]
        assert item.predictions is not None and other.predictions is not None
        one = item.predictions.loc[item.predictions.role.isin(BINARY) &
                                    item.predictions.status.eq("ok")]
        two = other.predictions.loc[other.predictions.role.isin(BINARY) &
                                     other.predictions.status.eq("ok")]
        ids = set(one.record_id) & set(two.record_id)
        common = ids if common is None else common & ids
        frames.append((one, two))
    common = common or set()
    metadata = None
    new_scores, old_scores, new_thresholds, old_thresholds = [], [], [], []
    for item, (one, two) in zip(ordered, frames):
        one = one.loc[one.record_id.isin(common)].sort_values("record_id").reset_index(drop=True)
        two = two.loc[two.record_id.isin(common)].sort_values("record_id").reset_index(drop=True)
        if not _same_identity(one, two):
            raise ValueError("Paired detector/protocol identities disagree")
        if metadata is None:
            metadata = one[IDENTITY]
        elif not _same_identity(one, metadata):
            raise ValueError("Paired valid identities differ across folds")
        new_scores.append(one.score_n2.to_numpy(dtype=float))
        old_scores.append(two.score_n2.to_numpy(dtype=float))
        new_thresholds.append(item.thresholds)
        old_thresholds.append(refs[item.fold].thresholds)
    assert metadata is not None
    counts = metadata.label_name.value_counts()
    result = {"status": "eligible", "reason": "", "paired_rows": len(metadata),
              "paired_human_rows": int(counts.get("human", 0)),
              "paired_synthetic_rows": int(counts.get("synthetic", 0))}
    if set(metadata.label_name) != {"human", "synthetic"}:
        return {**result, "status": "unsupported", "reason": "common_valid_binary_class_missing"}
    labels = metadata.label_name.eq("synthetic").to_numpy(dtype=np.int8)
    point_new = _mean_fold_metrics(labels, new_scores, new_thresholds)
    point_old = _mean_fold_metrics(labels, old_scores, old_thresholds)
    rng = np.random.default_rng(seed)
    groups = _group_positions(metadata)
    draws = defaultdict(list)
    for _ in range(replicates):
        positions = _sample_group_positions(groups, rng)
        before = _mean_fold_metrics(labels, old_scores, old_thresholds, positions)
        after = _mean_fold_metrics(labels, new_scores, new_thresholds, positions)
        for key in METRICS:
            draws[key].append(after[key] - before[key])
    for key in METRICS:
        result.update({f"reference_{key}": point_old[key], f"new_{key}": point_new[key],
                       f"delta_{key}": point_new[key] - point_old[key],
                       f"delta_{key}_ci_low": float(np.quantile(draws[key], .025)),
                       f"delta_{key}_ci_high": float(np.quantile(draws[key], .975))})
    return result


def subgroup_metrics(rounds: list[Round], identity: dict, *, minimum: int = 20
                     ) -> list[dict]:
    """Descriptive fold means; per-label counts remain visible in small cells.

    No subgroup confidence intervals are fabricated. AUC/BA require both
    classes and at least ``minimum`` per class. TPR/FPR and their numerator
    counts are reported for each available class, with small-cell flags.
    """
    if not rounds:
        return []
    ordered = sorted(rounds, key=lambda item: item.fold)
    first = ordered[0].predictions
    assert first is not None
    required = {"broad_area", "source_name", "generator", "word_count_audit"}
    if required - set(first):
        raise ValueError("Prediction metadata needed for IID subgroup summaries is missing")
    first = first.sort_values("record_id").reset_index(drop=True)
    populations = []
    for area in sorted(first.broad_area.astype(str).unique()):
        populations.append(("broad_area", area, first.broad_area.astype(str).eq(area)))
    lengths = pd.cut(first.word_count_audit.astype(float), [-1, 99, 299, 999, np.inf],
                     labels=["<100", "100-299", "300-999", "1000+"])
    for name in ("<100", "100-299", "300-999", "1000+"):
        populations.append(("word_count", name, lengths.eq(name)))
    for label, column, dimension in (("human", "source_name", "human_source"),
                                     ("synthetic", "generator", "synthetic_generator")):
        candidates = first.loc[first.label_name.eq(label)].copy()
        for (area, provenance), cell in candidates.groupby(
                ["broad_area", column], dropna=False, sort=True):
            ids = set(cell.record_id)
            value = f"{area}|{'<missing>' if pd.isna(provenance) else provenance}"
            populations.append((dimension, value, first.record_id.isin(ids)))
    rows = []
    for dimension, value, mask in populations:
        ids = set(first.loc[mask, "record_id"])
        if not ids:
            continue
        metadata = first.loc[mask]
        counts = metadata.label_name.value_counts()
        human, synthetic = int(counts.get("human", 0)), int(counts.get("synthetic", 0))
        binary_supported = min(human, synthetic) >= minimum
        row = {**identity, "dimension": dimension, "subgroup": value,
               "test_rows": len(ids), "human_rows": human, "synthetic_rows": synthetic,
               "folds": len(ordered), "minimum_per_class_for_binary": minimum,
               "binary_supported": binary_supported,
               "human_small_cell": human < minimum,
               "synthetic_small_cell": synthetic < minimum,
               "interval_note": "Descriptive fold means; subgroup CIs not computed"}
        gathered = defaultdict(list)
        raw_counts = []
        for item in ordered:
            assert item.predictions is not None
            frame = item.predictions.loc[item.predictions.record_id.isin(ids)].sort_values(
                "record_id").reset_index(drop=True)
            if not _same_identity(frame, metadata.sort_values("record_id").reset_index(drop=True)):
                raise ValueError("Subgroup test IDs differ across folds")
            labels = frame.label_name.eq("synthetic").to_numpy(dtype=np.int8)
            scores = frame.score_n2.to_numpy(dtype=float)
            if binary_supported:
                for metric, measure in metric_values(labels, scores, item.thresholds).items():
                    gathered[metric].append(measure)
            for name, threshold in item.thresholds.items():
                positive = scores >= threshold
                fp = int(positive[labels == 0].sum())
                tp = int(positive[labels == 1].sum())
                gathered[f"mean_fp_{name}"].append(fp)
                gathered[f"mean_tp_{name}"].append(tp)
                if human:
                    gathered[f"fpr_{name}"].append(fp / human)
                if synthetic:
                    gathered[f"tpr_{name}"].append(tp / synthetic)
                raw_counts.append({"fold": item.fold, "threshold": name, "fp": fp, "tp": tp,
                                   "human_rows": human, "synthetic_rows": synthetic})
        row.update({name: float(np.mean(values)) for name, values in gathered.items()})
        row["fold_error_counts_json"] = json.dumps(raw_counts, sort_keys=True)
        rows.append(row)
    return rows


def shard(plan_path: Path, index: int, runs: Path, output_root: Path) -> Path:
    plans, marker = _shard_plan(plan_path)
    if not 0 <= index < len(plans):
        raise IndexError("Summary shard index outside plan")
    row = plans.iloc[index].to_dict()
    output = output_root / f"{index:05d}"
    if output.exists():
        raise FileExistsError(f"Summary shard exists: {output}")
    identity = {key: row[key] for key in GROUP}
    identity["seed"] = int(identity["seed"])
    rounds, unsupported, hashes = [], [], {}
    fold_eligibility = []
    fitted_inputs = None
    for task in json.loads(row["tasks_json"]):
        item, record, inputs = load_result(task, runs)
        hashes.update(inputs)
        fold_eligibility.append({"fold": int(task["fold"]),
                                 "status": record["status"],
                                 "coverage": record.get("coverage", {}),
                                 "counts": record.get("counts", {})})
        if fitted_inputs is None:
            fitted_inputs = record.get("inputs", {})
        elif record.get("inputs", {}) != fitted_inputs:
            raise ValueError("Advanced folds used different configurations/model or feature caches")
        if item is None:
            unsupported.append({"task_id": task["task_id"], "reason": record["reason"]})
        else:
            rounds.append(item)
    result = {**identity, "status": "unsupported", "reason": "", "folds": len(rounds)}
    coverage = {**identity, "fold_eligibility_json": json.dumps(fold_eligibility, sort_keys=True)}
    contrasts = []
    subgroups = []
    if unsupported:
        result["reason"] = "one_or_more_folds_unsupported"
        result["unsupported_folds_json"] = json.dumps(unsupported, sort_keys=True)
    else:
        aligned, counts = common_valid(rounds)
        coverage.update(counts)
        if identity["protocol"] == "iid":
            subgroups = subgroup_metrics(aligned, identity)
        if counts["human_common_valid"] and counts["synthetic_common_valid"]:
            result.update(_seed_summary(aligned, replicates=marker["bootstrap_replicates"],
                                       seed=_seed(json.dumps(identity, sort_keys=True))))
            result.update(status="eligible", reason="", source_area=identity["source_area"],
                          text_view=identity["text_view"])
        else:
            result["reason"] = "common_valid_binary_class_missing"
    for protocol, tasks in json.loads(row["references_json"]).items():
        reference, reference_unsupported = [], []
        for task in tasks:
            item, record, inputs = load_result(task, runs)
            hashes.update(inputs)
            if item is None:
                reference_unsupported.append(record["reason"])
            else:
                reference.append(item)
        if unsupported or reference_unsupported:
            comparison = {"status": "unsupported", "reason": "protocol_fold_unsupported",
                          "paired_rows": 0}
        else:
            comparison = paired_delta(rounds, reference,
                                      replicates=marker["bootstrap_replicates"],
                                      seed=_seed(json.dumps(identity, sort_keys=True) + protocol))
        contrasts.append({**identity, "reference_protocol": protocol, **comparison,
                          "new_binary_total": coverage.get("binary_total", 0)})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([result]).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame([coverage]).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "paired_deltas.csv", index=False)
    pd.DataFrame(subgroups).to_csv(output / "iid_subgroup_metrics.csv", index=False)
    target = output / "manifest.json"
    write_json(target, {"kind": "horus_advanced_summary_shard", "index": index,
                        "identity": identity, "plan_sha256": sha256(plan_path),
                        "inputs": hashes,
                        "artifacts": {name: sha256(output / name) for name in (
                            "metrics.csv", "coverage.csv", "paired_deltas.csv",
                            "iid_subgroup_metrics.csv")}})
    return target


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _figures(metrics: pd.DataFrame, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = metrics.loc[metrics.status.eq("eligible")].copy()
    if selected.empty:
        return []
    names = []
    folder = output / "figures"
    folder.mkdir()
    for view, part in selected.groupby("text_view", sort=True):
        for metric in ("roc_auc", "observed_fpr_at_fpr_0.01"):
            table = part.pivot_table(index="model", columns="protocol", values=metric,
                                    aggfunc="mean")
            fig, ax = plt.subplots(figsize=(max(6, len(table.columns) * 1.4),
                                             max(4, len(table) * .45)))
            matrix = table.to_numpy(dtype=float)
            image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(table.columns)), table.columns)
            ax.set_yticks(range(len(table)), table.index)
            ax.set_title(f"{metric}: média descritiva de alvos/sementes\n{view}")
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            name = f"figures/{_seed(str(view)):08x}_{metric}.png"
            fig.savefig(output / name, dpi=180)
            plt.close(fig)
            names.append(name)
    return names


def merge(plan_path: Path, shards: Path, output: Path) -> Path:
    plans, marker = _shard_plan(plan_path)
    if output.exists():
        raise FileExistsError(f"Advanced summary exists: {output}")
    tables = defaultdict(list)
    shard_hashes = {}
    for row in plans.to_dict("records"):
        index = int(row["shard_index"])
        directory = shards / f"{index:05d}"
        record = read_json(directory / "manifest.json")
        if (record.get("kind") != "horus_advanced_summary_shard" or
                record["index"] != index or record["plan_sha256"] != sha256(plan_path) or
                any(str(record["identity"].get(key)) != str(row[key]) for key in GROUP)):
            raise ValueError(f"Advanced summary shard identity mismatch: {index}")
        for path, expected in record["inputs"].items():
            if sha256(path) != expected:
                raise ValueError(f"Result changed after shard summary: {path}")
        for name, expected in record["artifacts"].items():
            if sha256(directory / name) != expected:
                raise ValueError("Advanced summary shard artifact changed")
            tables[name].append(_csv(directory / name))
        shard_hashes[str(index)] = sha256(directory / "manifest.json")
    combined = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    metrics = combined["metrics.csv"]
    if len(metrics) != len(plans) or metrics.duplicated(GROUP).any():
        raise ValueError("Advanced summary matrix has duplicate or missing groups")
    output.mkdir(parents=True, exist_ok=False)
    for name, table in combined.items():
        table.to_csv(output / name, index=False)
    selected = metrics.loc[metrics.status.eq("eligible")]
    if selected.empty:
        macro = pd.DataFrame(columns=["model", "protocol", "seed", "text_view", *METRICS])
    else:
        macro = selected.groupby(["model", "protocol", "seed", "text_view"], as_index=False)[
            list(METRICS)].mean()
        count = selected.groupby(["model", "protocol", "seed", "text_view"]).size().rename(
            "supported_targets").reset_index()
        macro = macro.merge(count, validate="one_to_one")
    macro.to_csv(output / "descriptive_macro_by_seed.csv", index=False)
    figures = _figures(metrics, output)
    target = output / "summary.json"
    write_json(target, {"kind": "horus_advanced_summary", "complete": True,
                        "plan_sha256": sha256(plan_path), "shards": len(plans),
                        "scope": "All tasks in the explicitly supplied input plans",
                        "input_plans": marker["inputs"],
                        "planned_tasks": marker["planned_tasks"],
                        "eligible_groups": int(metrics.status.eq("eligible").sum()),
                        "unsupported_groups": int(metrics.status.ne("eligible").sum()),
                        "shard_hashes": shard_hashes,
                        "artifacts": {name: sha256(output / name) for name in (
                            *combined, "descriptive_macro_by_seed.csv", *figures)},
                        "interpretation": [
                            "Common valid test IDs are intersected across folds and retention is explicit.",
                            "Paired protocol deltas use identical IDs and shared linked-group draws.",
                            "Three seeds reuse one corpus; macro means are descriptive, not independent studies.",
                            "Unsupported groups are retained with reasons and excluded from metric means."]})
    return target


def agreement(first: Round, second: Round) -> list[dict]:
    """Counts on the same valid test IDs; no ensemble or threshold fitting."""
    assert first.predictions is not None and second.predictions is not None
    one = first.predictions.loc[first.predictions.role.isin(BINARY) &
                                first.predictions.status.eq("ok")]
    two = second.predictions.loc[second.predictions.role.isin(BINARY) &
                                 second.predictions.status.eq("ok")]
    joined = one.merge(two, on="record_id", validate="one_to_one", suffixes=("_a", "_b"))
    if (joined.label_name_a.ne(joined.label_name_b).any() or
            joined.probe_group_id_a.ne(joined.probe_group_id_b).any()):
        raise ValueError("Detector agreement has inconsistent labels/groups")
    rows = []
    for population in ("all", "human", "synthetic"):
        part = joined if population == "all" else joined.loc[joined.label_name_a.eq(population)]
        if part.empty:
            continue
        y = part.label_name_a.eq("synthetic").to_numpy()
        a, b = part.score_n2_a.to_numpy(dtype=float), part.score_n2_b.to_numpy(dtype=float)
        rank_a, rank_b = pd.Series(a).rank(), pd.Series(b).rank()
        correlation = (float(rank_a.corr(rank_b)) if rank_a.nunique() > 1 and
                       rank_b.nunique() > 1 else None)
        for threshold in sorted(THRESHOLDS):
            correct_a = (a >= first.thresholds[threshold]) == y
            correct_b = (b >= second.thresholds[threshold]) == y
            either_wrong = (~correct_a | ~correct_b).sum()
            rows.append({"model_a": first.model, "model_b": second.model,
                         "population": population, "threshold": threshold,
                         "common_valid_rows": len(part),
                         "a_valid_binary_rows": len(one), "b_valid_binary_rows": len(two),
                         "both_correct": int((correct_a & correct_b).sum()),
                         "only_a_wrong": int((~correct_a & correct_b).sum()),
                         "only_b_wrong": int((correct_a & ~correct_b).sum()),
                         "both_wrong": int((~correct_a & ~correct_b).sum()),
                         "error_jaccard": float((~correct_a & ~correct_b).sum() / either_wrong)
                         if either_wrong else None, "spearman_scores": correlation})
    return rows


def errors(plans: list[Path], runs: Path, baseline_plans: list[Path],
            baseline_runs: Path | None, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Error agreement output exists: {output}")
    tasks, hashes = _plans(plans)
    groups = defaultdict(list)
    for task in tasks.loc[tasks.fold.eq(0)].to_dict("records"):
        groups[tuple(task[key] for key in GROUP[1:])].append((task, runs, False))
    for path in baseline_plans:
        if baseline_runs is None:
            raise ValueError("Baseline runs directory is required")
        table = pd.read_csv(path, keep_default_na=False)
        hashes[str(path)] = {"sha256": sha256(path)}
        for task in table.loc[table.fold.eq(0)].to_dict("records"):
            # Only LOSO uses source_area as part of identity in the old plan;
            # older CSVs store it in their frozen target manifest.
            area = task.get("source_area", "")
            if task["protocol"] == "loso" and not area:
                manifest = pd.read_parquet(task["manifest_path"], columns=["source_area"])
                values = manifest.source_area.unique()
                if len(values) != 1:
                    raise ValueError("Legacy LOSO source-area identity is ambiguous")
                area = str(values[0])
            key = (task["protocol"], task["target"], area, task["seed"], "native")
            if key in groups:
                groups[key].append((task, baseline_runs, True))
    rows, skips, input_results = [], [], {}
    for key, planned in sorted(groups.items()):
        loaded = []
        for task, directory, legacy in planned:
            if legacy:
                record_path = directory / task["task_id"] / "metrics.json"
                record = read_json(record_path)
                if any(str(record.get("task", {}).get(name)) != str(task[name])
                       for name in ("task_id", "model", "protocol", "target", "seed", "fold")):
                    raise ValueError("Legacy baseline identity mismatch")
                score_path = record_path.parent / "predictions.parquet"
                if sha256(score_path) != record["predictions_sha256"]:
                    raise ValueError("Legacy baseline prediction changed")
                frame = pd.read_parquet(score_path)
                if frame.record_id.duplicated().any() or not np.isfinite(frame.score_n2).all():
                    raise ValueError("Legacy baseline has invalid scores")
                frame["status"] = "ok"
                item = Round(task["task_id"], task["model"], task["protocol"], task["target"],
                             int(task["seed"]), 0,
                             _thresholds(record["evaluation"]["validation_thresholds"]),
                             frame, record["evaluation"]["metrics"])
                input_results.update({str(record_path): sha256(record_path),
                                      str(score_path): sha256(score_path)})
            else:
                item, record, digests = load_result(task, directory)
                input_results.update(digests)
                if item is None:
                    skips.append({"model": task["model"], "protocol": key[0],
                                  "target": key[1], "source_area": key[2], "seed": key[3],
                                  "text_view": key[4], "reason": record["reason"]})
                    continue
            loaded.append(item)
        if len({item.model for item in loaded}) != len(loaded):
            raise ValueError("Duplicate model in error agreement population")
        for index, first in enumerate(loaded):
            for second in loaded[index + 1:]:
                pair = agreement(first, second)
                if not pair:
                    skips.append({"model": f"{first.model}|{second.model}",
                                  "protocol": key[0], "target": key[1],
                                  "source_area": key[2], "seed": key[3],
                                  "text_view": key[4], "reason": "no_common_valid_test_ids"})
                rows.extend({"protocol": key[0], "target": key[1], "source_area": key[2],
                             "seed": key[3], "text_view": key[4], "fold": 0, **row}
                            for row in pair)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "error_agreement.csv", index=False)
    pd.DataFrame(skips).to_csv(output / "unsupported_comparisons.csv", index=False)
    target = output / "summary.json"
    write_json(target, {"kind": "horus_advanced_error_agreement", "fold": 0,
                        "comparison_rows": len(rows), "skips": len(skips),
                        "plans": hashes, "inputs": input_results,
                        "artifacts": {name: sha256(output / name) for name in (
                            "error_agreement.csv", "unsupported_comparisons.csv")},
                        "note": "Exact common valid IDs; score ranks also reported within each class. No ensemble fitted."})
    return target


def contrast(native_plan: Path, native_runs: Path, view_plan: Path,
              view_runs: Path, output: Path, models: list[str] | None = None
              ) -> Path:
    """Common-prefix minus native policy on identical valid IID document IDs.

    Fitted supervised models and validation thresholds may differ by view;
    this is a complete text-policy comparison, not an isolated causal effect.
    ``models`` allows bounded per-model jobs without altering the populations.
    """
    if output.exists():
        raise FileExistsError(f"Native/view contrast already exists: {output}")
    native, native_hashes = _plans([native_plan])
    view, view_hashes = _plans([view_plan])
    if (not native.protocol.eq("iid").all() or not view.protocol.eq("iid").all() or
            not native.text_view.eq("native").all() or view.text_view.eq("native").any() or
            view.text_view.nunique() != 1):
        raise ValueError("Contrast requires native IID and one derived IID text policy")
    if set(native.model) != set(view.model):
        raise ValueError("Native and derived IID plans must contain the same model set")
    chosen_models = set(native.model) if models is None else set(models)
    if not chosen_models or chosen_models - set(native.model):
        raise ValueError("Unknown or empty native/view contrast model selection")
    selected_native = native.loc[native.model.isin(chosen_models)]
    selected_view = view.loc[view.model.isin(chosen_models)]
    keys = ["model", "protocol", "target", "source_area", "seed"]
    grouped_native = {key: group for key, group in selected_native.groupby(keys, sort=True)}
    grouped_view = {key: group for key, group in selected_view.groupby(keys, sort=True)}
    if set(grouped_native) != set(grouped_view):
        raise ValueError("Native/derived model, seed or target coverage differs")
    rows, result_hashes = [], {}
    for key in sorted(grouped_native):
        one, two = grouped_native[key], grouped_view[key]
        if sorted(one.fold.tolist()) != list(range(5)) or \
                sorted(two.fold.tolist()) != list(range(5)):
            raise ValueError("Native/view contrast requires all five frozen IID folds")
        # Main detector definitions remain fixed while input text and its
        # feature caches change. Newer plan schemas pin these in every task.
        for field in ("config_sha256", "classical_config_sha256", "lm_lock_sha256"):
            if field in one and field in two and set(one[field]) != set(two[field]):
                raise ValueError(f"Native/view contrast changed detector definition: {field}")
        rounds = {"native": [], "view": []}
        unsupported, coverage = [], []
        for population, group, runs in (("native", one, native_runs),
                                         ("view", two, view_runs)):
            for task in group.sort_values("fold").to_dict("records"):
                item, record, hashes = load_result(task, runs)
                result_hashes.update(hashes)
                coverage.append({"population": population, "fold": int(task["fold"]),
                                 "status": record["status"],
                                 "coverage": record.get("coverage", {})})
                if item is None:
                    unsupported.append({"population": population,
                                        "fold": int(task["fold"]),
                                        "reason": record["reason"]})
                else:
                    rounds[population].append(item)
        identity = dict(zip(keys, key))
        identity["seed"] = int(identity["seed"])
        if unsupported:
            comparison = {"status": "unsupported", "reason": "native_or_view_fold_unsupported",
                          "paired_rows": 0,
                          "unsupported_folds_json": json.dumps(unsupported, sort_keys=True)}
        else:
            comparison = paired_delta(rounds["view"], rounds["native"], replicates=2000,
                                      seed=_seed("native-common-prefix:" + json.dumps(
                                          identity, sort_keys=True)))
        rows.append({**identity, "view": str(view.text_view.iloc[0]),
                     "direction": "common_prefix_minus_native", **comparison,
                     "fold_coverage_json": json.dumps(coverage, sort_keys=True),
                     "interpretation": "Text-policy comparison; supervised fits and validation thresholds differ"})
    output.mkdir(parents=True, exist_ok=False)
    table_path = output / "native_common_prefix_deltas.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    target = output / "summary.json"
    write_json(target, {"kind": "horus_native_common_prefix_contrast", "complete": True,
                        "models": sorted(chosen_models), "model_seed_groups": len(rows),
                        "eligible_groups": sum(row["status"] == "eligible" for row in rows),
                        "bootstrap_replicates": 2000,
                        "native_plan": native_hashes, "view_plan": view_hashes,
                        "inputs": result_hashes,
                        "artifacts": {table_path.name: sha256(table_path)},
                        "intervals": "95% percentile, shared linked-group draws across views and all five folds",
                        "interpretation": "Same common valid source IDs. Seeds reuse one corpus. No causal input-only claim."})
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("plan")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--plan", type=Path, nargs="+", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("shard")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--index", type=int, required=True)
    command.add_argument("--runs", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("merge")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--shards", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("errors")
    command.add_argument("--plans", type=Path, nargs="+", required=True)
    command.add_argument("--runs", type=Path, required=True)
    command.add_argument("--baseline-plans", type=Path, nargs="*", default=[])
    command.add_argument("--baseline-runs", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("contrast")
    command.add_argument("--native-plan", type=Path, required=True)
    command.add_argument("--native-runs", type=Path, required=True)
    command.add_argument("--view-plan", type=Path, required=True)
    command.add_argument("--view-runs", type=Path, required=True)
    command.add_argument("--models", nargs="+")
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan(args.config, args.plan, args.output)
    elif args.command == "shard":
        result = shard(args.plan, args.index, args.runs, args.output)
    elif args.command == "merge":
        result = merge(args.plan, args.shards, args.output)
    elif args.command == "errors":
        result = errors(args.plans, args.runs, args.baseline_plans, args.baseline_runs, args.output)
    else:
        result = contrast(args.native_plan, args.native_runs, args.view_plan,
                          args.view_runs, args.output, args.models)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
