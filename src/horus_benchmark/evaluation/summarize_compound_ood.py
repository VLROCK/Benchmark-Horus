"""Summarize joint generator/domain OOD with exact-ID protocol contrasts.

All comparisons use the same test record IDs, each fitted model's validation
thresholds, and shared group bootstrap draws. The three seeds reuse one corpus
and are descriptive replications, not independent studies. This module reads
verified score files; it never fits detectors or reads the text corpus.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from horus_benchmark.evaluation.summarize_benchmarks import (
    METRICS, Round, _group_positions, _materialize, _mean_fold_metrics,
    _read_completed, _sample_group_positions, _seed_summary,
)
from horus_benchmark.protocols.compound_ood import check as check_compound_ood
from horus_benchmark.protocols.ood_manifests import _sha256


BINARY_ROLES = {"test_target", "test_control"}
REFERENCES = ("logo", "lodo", "iid")


def _config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("Unsupported compound OOD summary config")
    if cfg.get("experiment_id") != "horus-n2-compound-ood-summary-v1":
        raise ValueError("Unexpected compound OOD summary experiment_id")
    for name in ("compound_tasks", "compound_runs", "reference_runs"):
        if not isinstance(cfg.get(name), str) or not cfg[name].strip():
            raise ValueError(f"Missing {name}")
    if not isinstance(cfg.get("shard_plan"), str) or not cfg["shard_plan"].strip():
        raise ValueError("Missing shard_plan")
    for name in ("compound_plans", "reference_plans"):
        if not isinstance(cfg.get(name), list) or not cfg[name] or not all(
                isinstance(item, str) and item for item in cfg[name]):
            raise ValueError(f"{name} must be a nonempty list of paths")
    if cfg.get("seeds") != [20261105, 20261106, 20261107]:
        raise ValueError("Summary must cover the three frozen IID split seeds")
    if int(cfg.get("bootstrap_replicates", 0)) < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if float(cfg.get("bootstrap_confidence", 0)) != 0.95:
        raise ValueError("This summary implements 95% percentile intervals")
    models_cpu = cfg.get("models_cpu")
    models_bert = cfg.get("models_bert")
    if (not isinstance(models_cpu, list) or not isinstance(models_bert, list) or
            set(models_cpu) & set(models_bert) or not models_cpu or not models_bert):
        raise ValueError("Distinct CPU and BERT model lists are required")
    if cfg.get("folds_cpu") != [0, 1, 2, 3, 4] or cfg.get("folds_bert") != [0]:
        raise ValueError("Summary folds must match the approved OOD design")
    return cfg


def _plan_index(paths: list[str]) -> tuple[dict[tuple[str, str, str, int, int], dict],
                                             dict[str, str]]:
    rows: dict[tuple[str, str, str, int, int], dict] = {}
    hashes = {}
    task_ids: set[str] = set()
    for path_text in paths:
        path = Path(path_text)
        hashes[str(path)] = _sha256(path)
        plan = pd.read_csv(path, keep_default_na=False)
        required = {"task_id", "model", "protocol", "target", "seed", "fold"}
        if required - set(plan) or plan.task_id.duplicated().any():
            raise ValueError(f"Invalid plan: {path}")
        for task in plan.to_dict("records"):
            task_id = str(task["task_id"])
            if task_id in task_ids:
                raise ValueError(f"Duplicate task ID across plans: {task_id}")
            task_ids.add(task_id)
            key = (str(task["model"]), str(task["protocol"]), str(task["target"]),
                   int(task["seed"]), int(task["fold"]))
            if key in rows:
                raise ValueError(f"Duplicate model/protocol/target/seed/fold: {key}")
            rows[key] = task
    return rows, hashes


def _task_loader(compound_index: dict, reference_index: dict,
                 compound_runs: Path, reference_runs: Path):
    """Verify files on demand, with a bounded cache for repeated comparators."""
    cache: dict[tuple[str, str], Round] = {}
    order: list[tuple[str, str]] = []

    def load(key: tuple[str, str, str, int, int], *, compound: bool) -> Round:
        index = compound_index if compound else reference_index
        runs = compound_runs if compound else reference_runs
        if key not in index:
            raise ValueError(f"Required task is absent from execution plan: {key}")
        task = index[key]
        cache_key = ("compound" if compound else "reference", str(task["task_id"]))
        if cache_key in cache:
            return cache[cache_key]
        plan = pd.DataFrame([task])
        complete, missing = _read_completed(plan, runs, allow_incomplete=False)
        if missing or len(complete) != 1:
            raise ValueError(f"Task is incomplete: {key}")
        result = _materialize(complete[0], runs)
        cache[cache_key] = result
        order.append(cache_key)
        if len(order) > 64:
            oldest = order.pop(0)
            cache.pop(oldest, None)
        return result

    return load


def _binary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.role.isin(BINARY_ROLES),
                     ["record_id", "label_name", "probe_group_id", "score_n2"]].sort_values(
                         "record_id", kind="stable").reset_index(drop=True)


def _paired_frames(compound: Round, reference: Round,
                   reference_protocol: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if compound.predictions is None or reference.predictions is None:
        raise ValueError("Verified score files were not materialized")
    first = _binary(compound.predictions)
    other = _binary(reference.predictions)
    if reference_protocol == "iid":
        # Only frozen IID test IDs are comparable with IID predictions.
        expected_paired = set(compound.predictions.loc[
            compound.predictions.role.isin(BINARY_ROLES) &
            compound.predictions.paired_iid_test.astype(bool), "record_id"])
        first = first.loc[first.record_id.isin(set(other.record_id))].reset_index(drop=True)
        if set(first.record_id) != expected_paired:
            raise ValueError("Frozen IID paired flags disagree with IID score IDs")
    elif not set(first.record_id) <= set(other.record_id):
        raise ValueError(f"Joint test IDs are absent from {reference_protocol} test")
    paired = first.merge(other, on="record_id", how="left", validate="one_to_one",
                         suffixes=("_compound", "_reference"))
    if (paired.score_n2_reference.isna().any() or
            not paired.label_name_compound.eq(paired.label_name_reference).all() or
            not paired.probe_group_id_compound.eq(paired.probe_group_id_reference).all()):
        raise ValueError("Exact-ID contrast has different labels, groups or scores")
    metadata = paired[["record_id", "label_name_compound",
                       "probe_group_id_compound"]].rename(columns={
                           "label_name_compound": "label_name",
                           "probe_group_id_compound": "probe_group_id",
                       })
    return (metadata,
            paired.score_n2_compound.to_numpy(dtype=float),
            paired.score_n2_reference.to_numpy(dtype=float))


def paired_delta(compound: list[Round], reference: list[Round],
                 reference_protocol: str, *, replicates: int,
                 confidence: float, seed: int) -> dict:
    """Pair fold-mean metrics on common IDs; never pool repeated fold rows."""
    if reference_protocol not in REFERENCES:
        raise ValueError("Unknown reference protocol")
    by_fold = {item.fold: item for item in reference}
    if len(by_fold) != len(reference) or any(item.fold not in by_fold for item in compound):
        raise ValueError("Reference fold is missing or duplicated")
    frame: pd.DataFrame | None = None
    compound_scores = []
    reference_scores = []
    compound_thresholds = []
    reference_thresholds = []
    for one in sorted(compound, key=lambda item: item.fold):
        other = by_fold[one.fold]
        common, one_score, other_score = _paired_frames(one, other, reference_protocol)
        if frame is None:
            frame = common
        elif not common.equals(frame):
            raise ValueError("Paired IDs, labels or groups differ across folds")
        compound_scores.append(one_score)
        reference_scores.append(other_score)
        compound_thresholds.append(one.thresholds)
        reference_thresholds.append(other.thresholds)
    assert frame is not None
    labels = frame.label_name.eq("synthetic").to_numpy(dtype=int)
    human = int((labels == 0).sum())
    synthetic = int((labels == 1).sum())
    base = {"model": compound[0].model, "target": compound[0].target,
            "seed": compound[0].seed, "folds": len(compound),
            "reference_protocol": reference_protocol,
            "paired_rows": len(frame), "paired_human_rows": human,
            "paired_synthetic_rows": synthetic,
            "status": "eligible" if human and synthetic else "unsupported",
            "reason": "" if human and synthetic else "paired_binary_class_missing"}
    if not human or not synthetic:
        return base
    reference_point = _mean_fold_metrics(labels, reference_scores,
                                         reference_thresholds)
    compound_point = _mean_fold_metrics(labels, compound_scores,
                                        compound_thresholds)
    rng = np.random.default_rng(seed)
    groups = _group_positions(frame)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        positions = _sample_group_positions(groups, rng)
        old = _mean_fold_metrics(labels, reference_scores,
                                 reference_thresholds, positions)
        new = _mean_fold_metrics(labels, compound_scores,
                                compound_thresholds, positions)
        for metric in METRICS:
            draws[metric].append(new[metric] - old[metric])
    tail = (1 - confidence) / 2
    base.update({f"reference_{metric}": reference_point[metric] for metric in METRICS})
    base.update({f"compound_{metric}": compound_point[metric] for metric in METRICS})
    base.update({f"delta_{metric}": compound_point[metric] - reference_point[metric]
                 for metric in METRICS})
    for metric in METRICS:
        base[f"delta_{metric}_ci_low"] = float(np.quantile(draws[metric], tail))
        base[f"delta_{metric}_ci_high"] = float(np.quantile(draws[metric], 1 - tail))
    return base


def _target_only_rates(rounds: list[Round]) -> dict:
    """Keep unmatched synthetic targets visible without using them for AUC."""
    results = []
    expected_ids = None
    for item in sorted(rounds, key=lambda round_: round_.fold):
        if item.predictions is None:
            raise ValueError("Unmatched target scores were not materialized")
        unmatched = item.predictions.loc[
            item.predictions.role.eq("test_unmatched_target")].sort_values("record_id")
        ids = tuple(unmatched.record_id.astype(str))
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("Unmatched target IDs differ across folds")
        if not unmatched.empty:
            if not unmatched.label_name.eq("synthetic").all():
                raise ValueError("Compound unmatched targets must be synthetic")
            score = unmatched.score_n2.to_numpy(dtype=float)
            results.append({name: float(np.mean(score >= threshold))
                            for name, threshold in item.thresholds.items()})
    if not results:
        return {"unmatched_synthetic_rows": 0,
                "unmatched_tpr_youden": float("nan"),
                "unmatched_tpr_fpr_0.01": float("nan"),
                "unmatched_tpr_fpr_0.05": float("nan")}
    return {"unmatched_synthetic_rows": len(expected_ids or ()),
            **{f"unmatched_tpr_{name}": float(np.mean([item[name]
                                                       for item in results]))
               for name in ("youden", "fpr_0.01", "fpr_0.05")}}


def _figure(table: pd.DataFrame, output: Path) -> list[str]:
    if table.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    generators = sorted(table.heldout_generator.unique())
    areas = sorted(table.heldout_area.unique())
    models = sorted(table.model.unique())
    measures = (
        ("roc_auc", "ROC-AUC", "compound_auc_by_target.png"),
        ("observed_fpr_at_fpr_0.01", "FPR observado no limiar de 1%",
         "compound_fpr1_by_target.png"),
    )
    names = []
    for metric, title, name in measures:
        fig, axes = plt.subplots(len(models), 1, figsize=(max(8, len(areas) * 1.3),
                                                       max(3, len(models) * 2.5)),
                                 squeeze=False)
        last_image = None
        for axis, model in zip(axes[:, 0], models):
            matrix = table.loc[table.model.eq(model)].groupby(
                ["heldout_generator", "heldout_area"])[metric].mean().unstack().reindex(
                    index=generators, columns=areas)
            values = matrix.to_numpy(dtype=float)
            last_image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis",
                                     aspect="auto")
            axis.set_title(model)
            axis.set_yticks(range(len(generators)), generators, fontsize=7)
            axis.set_xticks(range(len(areas)), areas, rotation=30, ha="right",
                            fontsize=8)
            for row in range(len(generators)):
                for col in range(len(areas)):
                    value = values[row, col]
                    if np.isfinite(value):
                        axis.text(col, row, f"{value:.2f}", ha="center", va="center",
                                  color="white" if value < .55 else "black", fontsize=6)
        fig.suptitle(f"{title}: média descritiva das três sementes")
        if last_image is not None:
            fig.colorbar(last_image, ax=axes[:, 0].tolist(), shrink=.65)
        fig.tight_layout()
        fig.savefig(folder / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        names.append(str(Path("figures") / name))
    return names


def _validated_grid(cfg: dict, *, full_check: bool) -> tuple[Path, pd.DataFrame]:
    task_path = Path(cfg["compound_tasks"])
    if full_check:
        check_compound_ood(task_path.parent)
    else:
        summary = json.loads((task_path.parent / "summary.json").read_text(
            encoding="utf-8"))
        if (summary.get("kind") != "n2_compound_generator_domain_ood_manifests" or
                summary.get("tasks_sha256") != _sha256(task_path)):
            raise ValueError("Compound target grid changed")
    tasks = pd.read_csv(task_path, keep_default_na=False)
    needed = {"seed", "protocol", "target", "heldout_generator",
              "heldout_area", "status", "manifest_path", "manifest_sha256"}
    if (needed - set(tasks) or tasks.duplicated(["seed", "target"]).any() or
            not tasks.protocol.eq("logo_lodo").all() or
            set(tasks.seed) != set(cfg["seeds"])):
        raise ValueError("Invalid compound OOD task grid")
    return task_path, tasks


def plan_shards(config_path: Path, output: Path) -> Path:
    cfg = _config(config_path)
    if output != Path(cfg["shard_plan"]):
        raise ValueError("Shard plan output must match its pinned config path")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Compound shard plan already exists: {output}")
    task_path, tasks = _validated_grid(cfg, full_check=True)
    eligible = tasks.loc[tasks.status.eq("eligible"),
                         ["seed", "target", "heldout_generator", "heldout_area",
                          "manifest_sha256"]].sort_values(
                              ["seed", "target"], kind="stable").reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No supported generator-area targets")
    eligible.insert(0, "shard_index", np.arange(len(eligible), dtype=int))
    output.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(output, index=False)
    manifest = {
        "schema_version": 1, "kind": "n2_compound_ood_shard_plan",
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
        "compound_tasks_sha256": _sha256(task_path),
        "shard_count": len(eligible), "plan_sha256": _sha256(output),
        "note": "One shard per eligible seed × generator × domain target.",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return output


def _read_shard_plan(cfg: dict, config_path: Path,
                     task_path: Path) -> pd.DataFrame:
    plan_path = Path(cfg["shard_plan"])
    manifest_path = plan_path.with_suffix(".manifest.json")
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (record.get("kind") != "n2_compound_ood_shard_plan" or
            record.get("config_sha256") != _sha256(config_path) or
            record.get("compound_tasks_sha256") != _sha256(task_path) or
            record.get("plan_sha256") != _sha256(plan_path)):
        raise ValueError("Shard plan/config/task grid changed")
    plan = pd.read_csv(plan_path, keep_default_na=False)
    if (len(plan) != record["shard_count"] or
            plan.shard_index.tolist() != list(range(len(plan))) or
            plan.duplicated(["seed", "target"]).any()):
        raise ValueError("Shard plan identities are invalid")
    return plan


def summarize_shard(config_path: Path, shard_index: int,
                    output_root: Path) -> Path:
    cfg = _config(config_path)
    task_path, tasks = _validated_grid(cfg, full_check=False)
    plan = _read_shard_plan(cfg, config_path, task_path)
    if not 0 <= shard_index < len(plan):
        raise IndexError("Shard index outside plan")
    planned = plan.iloc[shard_index]
    selected = tasks.loc[tasks.seed.eq(planned.seed) & tasks.target.eq(planned.target)]
    if (len(selected) != 1 or selected.iloc[0].status != "eligible" or
            selected.iloc[0].manifest_sha256 != planned.manifest_sha256):
        raise ValueError("Shard target differs from supported compound task")
    task = selected.iloc[0]
    if _sha256(Path(task.manifest_path)) != task.manifest_sha256:
        raise ValueError("Compound target manifest changed")
    output = output_root / f"{shard_index:04d}"
    if output.exists():
        raise FileExistsError(f"Compound OOD shard already exists: {output}")
    compound_index, compound_plan_hashes = _plan_index(cfg["compound_plans"])
    reference_index, reference_plan_hashes = _plan_index(cfg["reference_plans"])
    loader = _task_loader(compound_index, reference_index,
                          Path(cfg["compound_runs"]), Path(cfg["reference_runs"]))
    target_metrics = []
    paired_rows = []
    coverage_rows = []
    models = cfg["models_cpu"] + cfg["models_bert"]
    for model in models:
        folds = cfg["folds_bert"] if model in cfg["models_bert"] else cfg["folds_cpu"]
        for fold in folds:
            key = (model, "logo_lodo", task.target, int(task.seed), int(fold))
            if key not in compound_index:
                raise ValueError(f"Compound evaluation plan lacks {key}")
            planned_task = compound_index[key]
            if (str(planned_task.get("manifest_sha256")) !=
                    str(task.manifest_sha256) or
                    Path(str(planned_task.get("manifest_path"))) !=
                    Path(str(task.manifest_path))):
                raise ValueError("Compound evaluation plan references another manifest")
        compound = [loader((model, "logo_lodo", task.target,
                            int(task.seed), int(fold)), compound=True)
                    for fold in folds]
        if len({item.fold for item in compound}) != len(folds):
            raise ValueError("Compound target has missing or repeated folds")
        derived_seed = int(hashlib.sha256(
            f"joint:{model}:{task.target}:{task.seed}".encode()).hexdigest()[:8], 16)
        metrics = _seed_summary(compound, replicates=int(cfg["bootstrap_replicates"]),
                                seed=derived_seed)
        fold_counts = json.loads(task.development_fold_label_counts)
        minimum_human = min(fold_counts.get(str(fold), {}).get("human", 0)
                            for fold in folds)
        minimum_synthetic = min(fold_counts.get(str(fold), {}).get("synthetic", 0)
                                for fold in folds)
        metrics.update({"heldout_generator": task.heldout_generator,
                        "heldout_area": task.heldout_area,
                        "matched_target_fraction": float(task.matched_target_fraction),
                        "validation_human_min": minimum_human,
                        "validation_synthetic_min": minimum_synthetic,
                        "validation_fpr_resolution_max": (
                            1 / minimum_human if minimum_human else float("nan")),
                        **_target_only_rates(compound)})
        target_metrics.append(metrics)
        coverage_rows.append({"model": model, "target": task.target,
                              "seed": task.seed, "folds": len(folds),
                              "binary_human_rows": int(task.binary_human_rows),
                              "binary_synthetic_rows": int(task.binary_synthetic_rows),
                              "matched_target_fraction": float(task.matched_target_fraction)})
        for reference_protocol, reference_target in (
                ("logo", task.heldout_generator),
                ("lodo", task.heldout_area),
                ("iid", "all")):
            reference = [loader((model, reference_protocol, reference_target,
                                 int(task.seed), int(fold)), compound=False)
                         for fold in folds]
            result = paired_delta(
                compound, reference, reference_protocol,
                replicates=int(cfg["bootstrap_replicates"]),
                confidence=float(cfg["bootstrap_confidence"]),
                seed=int(hashlib.sha256(
                    f"paired-joint:{model}:{task.target}:{task.seed}:"
                    f"{reference_protocol}".encode()).hexdigest()[:8], 16))
            result["heldout_generator"] = task.heldout_generator
            result["heldout_area"] = task.heldout_area
            result["compound_binary_rows"] = int(task.binary_human_rows) + int(
                task.binary_synthetic_rows)
            result["paired_fraction_of_compound"] = result["paired_rows"] / max(
                result["compound_binary_rows"], 1)
            paired_rows.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(target_metrics)
    table.to_csv(output / "target_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output / "paired_protocol_deltas.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "model_target_coverage.csv", index=False)
    summary = {
        "schema_version": 1, "kind": "n2_compound_ood_target_shard",
        "shard_index": shard_index, "seed": int(task.seed),
        "target": str(task.target), "heldout_generator": task.heldout_generator,
        "heldout_area": task.heldout_area,
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
        "compound_tasks_sha256": _sha256(task_path),
        "shard_plan_sha256": _sha256(Path(cfg["shard_plan"])),
        "compound_manifest_sha256": task.manifest_sha256,
        "compound_plan_hashes": compound_plan_hashes,
        "reference_plan_hashes": reference_plan_hashes,
        "models": models, "model_rows": len(table),
        "paired_contrasts": len(paired_rows),
        "bootstrap_replicates": int(cfg["bootstrap_replicates"]),
        "bootstrap_confidence": float(cfg["bootstrap_confidence"]),
        "outputs": {name: _sha256(output / name) for name in (
            "target_metrics.csv", "paired_protocol_deltas.csv",
            "model_target_coverage.csv")},
    }
    target = output / "shard_manifest.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return target


def merge_shards(config_path: Path, shard_root: Path, output: Path) -> Path:
    cfg = _config(config_path)
    if output.exists():
        raise FileExistsError(f"Compound OOD summary already exists: {output}")
    task_path, tasks = _validated_grid(cfg, full_check=True)
    plan = _read_shard_plan(cfg, config_path, task_path)
    models = cfg["models_cpu"] + cfg["models_bert"]
    current_compound_plans = {str(path): _sha256(Path(path))
                              for path in cfg["compound_plans"]}
    current_reference_plans = {str(path): _sha256(Path(path))
                               for path in cfg["reference_plans"]}
    metrics_parts = []
    paired_parts = []
    coverage_parts = []
    shard_hashes = {}
    for item in plan.itertuples(index=False):
        folder = shard_root / f"{item.shard_index:04d}"
        manifest_path = folder / "shard_manifest.json"
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (record.get("kind") != "n2_compound_ood_target_shard" or
                record.get("config_sha256") != _sha256(config_path) or
                record.get("compound_tasks_sha256") != _sha256(task_path) or
                record.get("shard_plan_sha256") != _sha256(Path(cfg["shard_plan"])) or
                record.get("shard_index") != item.shard_index or
                record.get("seed") != item.seed or record.get("target") != item.target or
                record.get("compound_manifest_sha256") != item.manifest_sha256 or
                record.get("compound_plan_hashes") != current_compound_plans or
                record.get("reference_plan_hashes") != current_reference_plans or
                record.get("models") != models or
                record.get("model_rows") != len(models) or
                record.get("paired_contrasts") != len(models) * len(REFERENCES)):
            raise ValueError(f"Shard identity or coverage mismatch: {folder}")
        for name, digest in record["outputs"].items():
            if _sha256(folder / name) != digest:
                raise ValueError(f"Shard output changed: {folder / name}")
        metrics = pd.read_csv(folder / "target_metrics.csv", keep_default_na=False)
        paired = pd.read_csv(folder / "paired_protocol_deltas.csv", keep_default_na=False)
        coverage = pd.read_csv(folder / "model_target_coverage.csv", keep_default_na=False)
        for table, expected in ((metrics, len(models)),
                                (paired, len(models) * len(REFERENCES)),
                                (coverage, len(models))):
            if (len(table) != expected or set(table.model) != set(models) or
                    set(table.seed) != {item.seed} or
                    set(table.target) != {item.target}):
                raise ValueError(f"Shard table has wrong model/target coverage: {folder}")
        if paired.duplicated(["model", "reference_protocol"]).any() or set(
                paired.reference_protocol) != set(REFERENCES):
            raise ValueError(f"Shard lacks an exact-ID reference contrast: {folder}")
        metrics_parts.append(metrics)
        paired_parts.append(paired)
        coverage_parts.append(coverage)
        shard_hashes[f"{item.shard_index:04d}"] = _sha256(manifest_path)
    table = pd.concat(metrics_parts, ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    if (table.duplicated(["model", "seed", "target"]).any() or
            paired.duplicated(["model", "seed", "target", "reference_protocol"]).any()):
        raise ValueError("Merged compound results contain duplicate target rows")
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "target_metrics.csv", index=False)
    paired.to_csv(output / "paired_protocol_deltas.csv", index=False)
    coverage.to_csv(output / "model_target_coverage.csv", index=False)
    unsupported = tasks.loc[tasks.status.ne("eligible")].copy()
    unsupported.to_csv(output / "unsupported_grid.csv", index=False)
    macro = table.groupby(["model", "seed"], as_index=False)[list(METRICS)].mean()
    counts = table.groupby(["model", "seed"]).target.nunique().rename("targets").reset_index()
    macro = macro.merge(counts, on=["model", "seed"], validate="one_to_one")
    macro.to_csv(output / "descriptive_macro_by_seed.csv", index=False)
    figures = _figure(table, output)
    summary = {
        "schema_version": 1, "kind": "n2_compound_ood_score_summary",
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
        "compound_tasks_sha256": _sha256(task_path),
        "shard_plan_sha256": _sha256(Path(cfg["shard_plan"])),
        "shard_manifest_hashes": shard_hashes,
        "eligible_seed_targets": len(plan),
        "unsupported_seed_targets": len(unsupported),
        "model_seed_target_rows": len(table),
        "paired_contrasts": len(paired),
        "paired_contrasts_unsupported": int(paired.status.ne("eligible").sum()),
        "bootstrap_replicates": int(cfg["bootstrap_replicates"]),
        "bootstrap_confidence": float(cfg["bootstrap_confidence"]),
        "figures": figures,
        "outputs": {name: _sha256(output / name) for name in (
            "target_metrics.csv", "paired_protocol_deltas.csv",
            "model_target_coverage.csv", "unsupported_grid.csv",
            "descriptive_macro_by_seed.csv")},
        "interpretation": [
            "Deltas use exact common test IDs and model-specific frozen validation thresholds.",
            "LOGO and LODO must contain every compound binary test ID; IID uses its intersection.",
            "Fold means share documents; bootstrap samples probe groups once per replicate.",
            "The three seeds reuse the same corpus and are not independent studies.",
            "Macro means are descriptive and give each supported target equal weight.",
            "FPR thresholds were selected on validation; test FPR is reported separately.",
            "Comparisons quantify full protocol changes, not causal generator/domain interaction.",
        ],
    }
    target = output / "summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_cmd = sub.add_parser("plan")
    plan_cmd.add_argument("--config", type=Path, required=True)
    plan_cmd.add_argument("--output", type=Path, required=True)
    shard_cmd = sub.add_parser("shard")
    shard_cmd.add_argument("--config", type=Path, required=True)
    shard_cmd.add_argument("--index", type=int, required=True)
    shard_cmd.add_argument("--output-root", type=Path, required=True)
    merge_cmd = sub.add_parser("merge")
    merge_cmd.add_argument("--config", type=Path, required=True)
    merge_cmd.add_argument("--shard-root", type=Path, required=True)
    merge_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan_shards(args.config, args.output)
    elif args.command == "shard":
        result = summarize_shard(args.config, args.index, args.output_root)
    else:
        result = merge_shards(args.config, args.shard_root, args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
