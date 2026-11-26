"""Paired robustness metrics with clean thresholds and linked-group intervals.

Consumes joined score files, never fits detectors or tunes thresholds. Results
are conditional on the fitted models, clean-validation thresholds and valid
common test population. Seed results are never pooled. No-ops remain in the
primary population and also have explicit coverage counts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horus_benchmark.evaluation.benchmark_runner import sha256_file
from horus_benchmark.evaluation.summarize_secondary_views import (
    METRICS, _draws, _group_index, _metric_batch, _score_plan,
)
from horus_benchmark.metrics.binary_detection import metric_values


KEYS = ["model", "protocol", "target", "seed", "attack", "intensity"]
THRESHOLD_COLUMNS = {"youden": "threshold_youden",
                     "fpr_0.01": "threshold_fpr_0.01",
                     "fpr_0.05": "threshold_fpr_0.05"}
REQUIRED = set(KEYS) | {"parent_record_id", "probe_group_id", "label_name", "fold",
                         "score_clean", "score_attacked", "status"} | set(
                             THRESHOLD_COLUMNS.values())
CONDITIONS = ("synthetic_only", "human_only", "both")


def _thresholds(frame: pd.DataFrame) -> dict[str, float]:
    values = {}
    for key, column in THRESHOLD_COLUMNS.items():
        observed = frame[column].unique()
        if len(observed) != 1 or np.isnan(float(observed[0])):
            raise ValueError("Each fitted fold must have one frozen clean threshold per criterion")
        values[key] = float(observed[0])
    return values


def _quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def _extra(labels: np.ndarray, clean: np.ndarray, attacked: np.ndarray,
           thresholds: dict[str, float], weights: np.ndarray) -> dict[str, np.ndarray]:
    """Weighted paired score shifts, decision flips and conditional evasion."""
    human = labels == 0
    synthetic = ~human
    total = weights.sum(axis=1)
    nh, ns = weights[:, human].sum(axis=1), weights[:, synthetic].sum(axis=1)
    output = {
        "score_delta_mean": (weights * (attacked - clean)).sum(axis=1) / total,
        "score_delta_human": (weights[:, human] * (attacked[human] - clean[human])).sum(
            axis=1) / nh,
        "score_delta_synthetic": (weights[:, synthetic] * (
            attacked[synthetic] - clean[synthetic])).sum(axis=1) / ns,
    }
    for key, threshold in thresholds.items():
        before = clean >= threshold
        after = attacked >= threshold
        output[f"flip_rate_{key}"] = (weights * (before != after)).sum(axis=1) / total
        eligible = synthetic & before
        denominator = weights[:, eligible].sum(axis=1)
        numerator = weights[:, eligible & ~after].sum(axis=1)
        output[f"evasion_given_clean_detected_{key}"] = np.divide(
            numerator, denominator, out=np.full(weights.shape[0], np.nan),
            where=denominator > 0)
        output[f"new_false_positive_rate_{key}"] = (
            weights[:, human] * (~before[human] & after[human])).sum(axis=1) / nh
    return output


def summarize_group(frame: pd.DataFrame, *, replicates: int = 2000,
                    population: str = "all_valid", batch_size: int = 32
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Mean paired effects across aligned folds; whole groups share each draw."""
    if REQUIRED - set(frame) or replicates < 1 or batch_size < 1:
        raise ValueError("Missing robustness columns or invalid bootstrap parameters")
    if population not in {"all_valid", "changed_only"}:
        raise ValueError("Unknown robustness population")
    if any(frame[column].nunique(dropna=False) != 1 for column in KEYS):
        raise ValueError("A robustness group must have one model/protocol/target/seed/attack/dose")
    if (frame.duplicated(["fold", "parent_record_id"]).any() or
            frame.parent_record_id.isna().any() or frame.probe_group_id.isna().any() or
            set(frame.label_name) - {"human", "synthetic"} or
            frame.groupby("probe_group_id").label_name.nunique().gt(1).any()):
        raise ValueError("Duplicate IDs, missing identity or cross-label linked group")
    folds = [part.sort_values("parent_record_id", kind="stable").reset_index(drop=True)
             for _, part in frame.groupby("fold", sort=True)]
    first = folds[0]
    metadata = ["parent_record_id", "probe_group_id", "label_name", "status"]
    if any(not part[metadata].equals(first[metadata]) for part in folds[1:]):
        raise ValueError("Fold source IDs, labels, groups or attack status differ")
    finite = np.ones(len(first), dtype=bool)
    for part in folds:
        finite &= np.isfinite(pd.to_numeric(part.score_clean, errors="coerce"))
        finite &= np.isfinite(pd.to_numeric(part.score_attacked, errors="coerce"))
    valid = finite & first.status.isin(["changed", "no_op"]).to_numpy()
    if population == "changed_only":
        valid &= first.status.eq("changed").to_numpy()
    identity = {key: first.iloc[0][key] for key in KEYS}
    identity["seed"] = int(identity["seed"])
    identity["intensity"] = float(identity["intensity"])
    coverage = {
        **identity, "population": population, "folds": len(folds),
        "fold_ids": ",".join(str(int(part.iloc[0].fold)) for part in folds),
        "source_rows": len(first), "valid_common_rows": int(valid.sum()),
        "excluded_rows": int((~valid).sum()),
        "nonfinite_score_rows": int((~finite).sum()),
        "changed_rows": int(first.status.eq("changed").sum()),
        "no_op_rows": int(first.status.eq("no_op").sum()),
        "invalid_rows": int((~first.status.isin(["changed", "no_op"])).sum()),
        "valid_human_rows": int((valid & first.label_name.eq("human").to_numpy()).sum()),
        "valid_synthetic_rows": int((valid & first.label_name.eq("synthetic").to_numpy()).sum()),
    }
    if not coverage["valid_human_rows"] or not coverage["valid_synthetic_rows"]:
        coverage["status"] = "unsupported"
        coverage["reason"] = "valid_common_population_missing_class"
        return [], [], coverage
    coverage["status"], coverage["reason"] = "eligible", ""
    scored = [part.loc[valid].reset_index(drop=True) for part in folds]
    labels = scored[0].label_name.eq("synthetic").to_numpy(dtype=np.int8)
    indexed = scored[0][["parent_record_id", "label_name", "probe_group_id"]].rename(
        columns={"parent_record_id": "record_id"})
    row_group_index, sizes = _group_index(indexed)
    conditions: dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, float]]]] = {
        name: [] for name in CONDITIONS}
    for part in scored:
        clean = part.score_clean.to_numpy(dtype=float)
        attacked = part.score_attacked.to_numpy(dtype=float)
        for name in CONDITIONS:
            changed_scores = (np.where(labels == 1, attacked, clean)
                              if name == "synthetic_only" else
                              np.where(labels == 0, attacked, clean)
                              if name == "human_only" else attacked)
            conditions[name].append((clean, changed_scores, _thresholds(part)))
    seed_material = "|".join(str(identity[key]) for key in KEYS) + "|" + population
    bootstrap_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
    metric_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    # Reinitialize with the same seed per condition: corresponding effects use
    # exactly the same document-group bootstrap draws across the three setups.
    for condition, parts in conditions.items():
        rng = np.random.default_rng(bootstrap_seed)
        collected: dict[str, list[np.ndarray]] = defaultdict(list)
        clean_points: dict[str, list[float]] = defaultdict(list)
        attacked_points: dict[str, list[float]] = defaultdict(list)
        extra_points: dict[str, list[float]] = defaultdict(list)
        plans = []
        for clean, changed_scores, thresholds in parts:
            before = metric_values(labels, clean, thresholds)
            after = metric_values(labels, changed_scores, thresholds)
            for metric in METRICS:
                clean_points[metric].append(before[metric])
                attacked_points[metric].append(after[metric])
            for key, values in _extra(labels, clean, changed_scores, thresholds,
                                       np.ones((1, len(labels)))).items():
                extra_points[key].append(float(values[0]))
            plans.append((_score_plan(clean, thresholds),
                          _score_plan(changed_scores, thresholds)))
        for start in range(0, replicates, batch_size):
            weights = _draws(rng, sizes, min(batch_size, replicates - start))
            batch: dict[str, np.ndarray] = {}
            for (clean, changed_scores, thresholds), (old_plan, new_plan) in zip(parts, plans):
                before = _metric_batch(labels, old_plan, row_group_index, weights, thresholds)
                after = _metric_batch(labels, new_plan, row_group_index, weights, thresholds)
                for metric in METRICS:
                    for key, values in ((f"clean_{metric}", before[metric]),
                                        (f"attacked_{metric}", after[metric]),
                                        (f"delta_{metric}", after[metric] - before[metric])):
                        batch[key] = batch.get(key, 0) + values / len(parts)
                for key, values in _extra(labels, clean, changed_scores, thresholds,
                                           weights[:, row_group_index]).items():
                    batch[key] = batch.get(key, 0) + values / len(parts)
            for key, values in batch.items():
                collected[key].append(values)
        meta = {**identity, "condition": condition, "population": population,
                "folds": len(parts), "test_rows": len(labels),
                "human_rows": int((labels == 0).sum()),
                "synthetic_rows": int((labels == 1).sum())}
        for metric in METRICS:
            row = {**meta, "metric": metric,
                   "clean_value": float(np.mean(clean_points[metric])),
                   "attacked_value": float(np.mean(attacked_points[metric]))}
            row["delta_attacked_minus_clean"] = row["attacked_value"] - row["clean_value"]
            for prefix in ("clean", "attacked", "delta"):
                draws = np.concatenate(collected[f"{prefix}_{metric}"])
                row[f"{prefix}_ci_low"] = _quantile(draws, .025)
                row[f"{prefix}_ci_high"] = _quantile(draws, .975)
            metric_rows.append(row)
        for key, points in extra_points.items():
            draws = np.concatenate(collected[key])
            finite_points = np.asarray(points)[np.isfinite(points)]
            effect_rows.append({**meta, "effect": key,
                                "value": float(finite_points.mean()) if len(finite_points)
                                else float("nan"),
                                "ci_low": _quantile(draws, .025),
                                "ci_high": _quantile(draws, .975),
                                "bootstrap_defined_fraction": float(np.isfinite(draws).mean())})
    return metric_rows, effect_rows, coverage


def error_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    """Detector discordance on identical IDs at clean-validation Youden cutoffs."""
    rows = []
    keys = ["protocol", "target", "seed", "fold", "attack", "intensity"]
    for group_key, population in frame.groupby(keys, sort=True, dropna=False):
        by_model = {name: part for name, part in population.groupby("model", sort=True)}
        for first_name, second_name in itertools.combinations(sorted(by_model), 2):
            a = by_model[first_name]
            b = by_model[second_name]
            common = a.merge(b, on="parent_record_id", how="inner", validate="one_to_one",
                             suffixes=("_a", "_b"))
            if (not common.label_name_a.eq(common.label_name_b).all() or
                    not common.probe_group_id_a.eq(common.probe_group_id_b).all()):
                raise ValueError("Error agreement has inconsistent labels or groups")
            for score_column in ("score_clean", "score_attacked"):
                valid = np.isfinite(pd.to_numeric(common[f"{score_column}_a"], errors="coerce")) & \
                    np.isfinite(pd.to_numeric(common[f"{score_column}_b"], errors="coerce"))
                valid &= common.status_a.isin(["changed", "no_op"]) & \
                    common.status_b.isin(["changed", "no_op"])
                part = common.loc[valid]
                label = part.label_name_a.eq("synthetic").to_numpy()
                correct_a = (part[f"{score_column}_a"].to_numpy() >=
                             part.threshold_youden_a.to_numpy()) == label
                correct_b = (part[f"{score_column}_b"].to_numpy() >=
                             part.threshold_youden_b.to_numpy()) == label
                corr = (part[f"{score_column}_a"].corr(part[f"{score_column}_b"],
                                                      method="spearman")
                        if len(part) > 1 else float("nan"))
                rows.append({**dict(zip(keys, group_key)), "model_a": first_name,
                             "model_b": second_name, "view": score_column,
                             "common_rows": len(part), "rows_a": len(a), "rows_b": len(b),
                             "both_correct": int((correct_a & correct_b).sum()),
                             "only_a_wrong": int((~correct_a & correct_b).sum()),
                             "only_b_wrong": int((correct_a & ~correct_b).sum()),
                             "both_wrong": int((~correct_a & ~correct_b).sum()),
                             "score_spearman": float(corr)})
    return pd.DataFrame(rows)


def run(inputs: list[Path], output: Path, *, replicates: int = 2000) -> Path:
    if output.exists():
        raise FileExistsError(f"Robustness summary already exists: {output}")
    frame = pd.concat([pd.read_parquet(path) for path in inputs], ignore_index=True)
    if (REQUIRED - set(frame) or frame.empty or frame.duplicated(
            [*KEYS, "fold", "parent_record_id"]).any()):
        raise ValueError("Missing columns, empty scores or duplicate robustness tasks")
    metrics, effects, coverage = [], [], []
    for _, part in frame.groupby(KEYS, sort=True, dropna=False):
        for population in ("all_valid", "changed_only"):
            one, two, counts = summarize_group(part, replicates=replicates,
                                                population=population)
            metrics.extend(one)
            effects.extend(two)
            coverage.append(counts)
    output.mkdir(parents=True, exist_ok=False)
    tables = {"paired_metrics.csv": pd.DataFrame(metrics),
              "paired_effects.csv": pd.DataFrame(effects),
              "coverage.csv": pd.DataFrame(coverage),
              "detector_error_agreement.csv": error_agreement(frame)}
    artifacts = {}
    for name, table in tables.items():
        path = output / name
        table.to_csv(path, index=False)
        artifacts[name] = {"rows": len(table), "sha256": sha256_file(path)}
    if metrics:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        table = tables["paired_metrics.csv"]
        chosen = table.loc[table.population.eq("all_valid") & table.condition.eq("both") &
                           table.metric.eq("roc_auc")]
        matrix = chosen.groupby(["model", "attack", "intensity"])[
            "delta_attacked_minus_clean"].mean().unstack(["attack", "intensity"])
        fig, ax = plt.subplots(figsize=(max(9, len(matrix.columns)), max(3, len(matrix) * .5)))
        values = matrix.to_numpy(dtype=float)
        bound = max(float(np.nanmax(np.abs(values))), .01)
        plotted = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set_xticks(range(len(matrix.columns)),
                      [f"{attack}\n{dose:g}" for attack, dose in matrix.columns],
                      rotation=35, ha="right", fontsize=8)
        ax.set_title("ROC-AUC alterado menos limpo; média descritiva entre alvos/sementes")
        fig.colorbar(plotted, ax=ax)
        fig.tight_layout()
        figure = output / "robustness_auc_delta.png"
        fig.savefig(figure, dpi=180)
        plt.close(fig)
        artifacts[figure.name] = {"sha256": sha256_file(figure)}
    summary = {
        "schema_version": 1, "kind": "n2_paired_robustness_summary",
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in inputs],
        "bootstrap_replicates": replicates, "bootstrap_confidence": .95,
        "seeds": sorted(int(value) for value in frame.seed.unique()),
        "score_rows": len(frame), "artifacts": artifacts,
        "interpretation": [
            "All thresholds originate from clean validation; no attacked recalibration.",
            "All-valid includes no-ops; changed-only is a secondary restricted population.",
            "Compare paired common IDs and sample whole linked groups within each label.",
            "Same group draws are shared across folds and clean/attacked scores.",
            "Seeds reuse one corpus and are reported separately; figures are descriptive.",
            "Conditional evasion uses only synthetic texts detected before the attack.",
        ],
    }
    target = output / "summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    print(run(args.inputs, args.output, replicates=args.replicates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
