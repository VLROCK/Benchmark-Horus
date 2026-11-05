"""Deterministic group-disjoint IID split candidates for an audited corpus.

All grouping is inherited from the selection audit. This module does not infer
new document identities, train detectors, or create OOD folds.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def add_strata(frame: pd.DataFrame, word_edges: list[int]) -> pd.DataFrame:
    """Keep coarse length inside area and class; never use source or generator."""
    if len(word_edges) < 2 or word_edges[0] != 0 or sorted(set(word_edges)) != word_edges:
        raise ValueError("word_edges must be increasing and start at zero")
    if frame.word_count_audit.isna().any() or frame.word_count_audit.lt(0).any():
        raise ValueError("Invalid word counts")
    out = frame.copy()
    out["length_stratum"] = np.searchsorted(word_edges, out.word_count_audit, side="right") - 1
    out["stratum"] = (out.broad_area.astype(str) + "|" + out.label_name.astype(str) + "|" +
                      out.length_stratum.astype(str))
    return out


def assign_partitions(frame: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Greedy minimum incremental standardized imbalance over group vectors.

    Groups are indivisible. Largest groups are placed first; deterministic hash
    breaks ties. The objective controls area × class × coarse length and total
    size. It is a construction heuristic, not proof of ideal stratification.
    """
    parts = tuple(fractions)
    if len(parts) < 2 or any(fractions[x] <= 0 for x in parts):
        raise ValueError("At least two positive partition fractions required")
    if not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to one")
    if frame.record_id.duplicated().any() or frame.probe_group_id.isna().any():
        raise ValueError("Duplicate IDs or missing probe groups")
    counts = pd.crosstab(frame.probe_group_id, frame.stratum).astype(np.int64)
    groups = counts.index.astype(str).tolist()
    vectors = counts.to_numpy(dtype=np.float64)
    sizes = vectors.sum(axis=1)
    strata_totals = vectors.sum(axis=0)
    target = np.array([fractions[name] for name in parts])[:, None] * strata_totals
    size_target = np.array([fractions[name] for name in parts]) * len(frame)
    weights = 1.0 / np.maximum(strata_totals, 1.0)
    current = np.zeros_like(target)
    current_size = np.zeros(len(parts), dtype=np.float64)
    order = sorted(range(len(groups)), key=lambda idx: (-sizes[idx], stable_key(seed, groups[idx])))
    assignments: dict[str, str] = {}
    for idx in order:
        vector = vectors[idx]
        size = sizes[idx]
        scores = []
        for part in range(len(parts)):
            old = current[part] - target[part]
            new = old + vector
            delta = float(np.sum((new * new - old * old) * weights))
            old_size = current_size[part] - size_target[part]
            delta += float((2 * old_size * size + size * size) / len(frame))
            scores.append((delta, stable_key(seed, f"{groups[idx]}|{parts[part]}"), part))
        choice = min(scores)[2]
        current[choice] += vector
        current_size[choice] += size
        assignments[groups[idx]] = parts[choice]
    result = frame[["record_id", "probe_group_id", "stratum", "broad_area", "label_name",
                    "length_stratum"]].copy()
    result["split"] = result.probe_group_id.map(assignments)
    if result.split.isna().any() or result.groupby("probe_group_id").split.nunique().max() != 1:
        raise AssertionError("Group leakage in assigned split")
    return result


def assign_groups(frame: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Legacy three-way candidate splitter, retained for existing tests."""
    if set(fractions) != set(SPLITS):
        raise ValueError("Expected train/validation/test fractions")
    return assign_partitions(frame, fractions, seed)


def assign_heldout_cv(frame: pd.DataFrame, test_fraction: float, folds: int,
                      seed: int) -> pd.DataFrame:
    """Reserve grouped IID test, then allocate each development group one fold."""
    if not 0 < test_fraction < 1 or folds < 2:
        raise ValueError("Invalid test fraction or fold count")
    outer = assign_partitions(frame, {"development": 1 - test_fraction,
                                      "test": test_fraction}, seed)
    if set(outer.split) != {"development", "test"}:
        raise ValueError("Grouped holdout produced an empty partition")
    development = outer.loc[outer.split.eq("development")].copy()
    if development.probe_group_id.nunique() < folds:
        raise ValueError("Too few development groups for requested folds")
    names = tuple(f"fold_{index}" for index in range(folds))
    inner = assign_partitions(development, {name: 1 / folds for name in names}, seed + 1000003)
    if set(inner.split) != set(names):
        raise ValueError("Grouped cross-validation produced an empty fold")
    fold_by_id = inner.set_index("record_id").split.map(
        {name: index for index, name in enumerate(names)})
    outer["fold"] = outer.record_id.map(fold_by_id).fillna(-1).astype(int)
    if outer.loc[outer.split.eq("test"), "fold"].ne(-1).any():
        raise AssertionError("Test rows entered a development fold")
    if outer.loc[outer.split.eq("development"), "fold"].lt(0).any():
        raise AssertionError("Development rows missing a fold")
    if outer.groupby("probe_group_id").agg(
            partitions=("split", "nunique"), folds=("fold", "nunique")
    ).max().gt(1).any():
        raise AssertionError("Group crosses holdout or development folds")
    return outer


def check_observed_pairs(split: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, int]:
    columns = ["split", "fold"] if "fold" in split else ["split"]
    by_id = {rid: tuple(values) for rid, values in split.set_index("record_id")[columns].iterrows()}
    both = crossing = 0
    for pair in pairs.itertuples(index=False):
        left = by_id.get(pair.left_record_id)
        right = by_id.get(pair.right_record_id)
        if left is None or right is None:
            continue
        both += 1
        crossing += left != right
    if crossing:
        raise ValueError(f"{crossing} observed near-duplicate pairs cross splits")
    return {"observed_pairs_both_selected": both, "observed_pairs_cross_split": crossing}


def split_diagnostics(split: pd.DataFrame, fractions: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = split.groupby(["split", "broad_area", "label_name", "length_stratum"]).size(
    ).rename("rows").reset_index()
    overall = split.groupby(["broad_area", "label_name", "length_stratum"]).size(
    ).rename("total_rows").reset_index()
    grid = pd.MultiIndex.from_product(
        [tuple(fractions), sorted(split.broad_area.unique()), sorted(split.label_name.unique()),
         sorted(split.length_stratum.unique())],
        names=["split", "broad_area", "label_name", "length_stratum"]).to_frame(index=False)
    table = grid.merge(overall, on=["broad_area", "label_name", "length_stratum"], how="left")
    table = table.loc[table.total_rows.fillna(0).gt(0)].merge(
        counts, on=["split", "broad_area", "label_name", "length_stratum"], how="left")
    table["rows"] = table.rows.fillna(0).astype(int)
    table["target_rows"] = table.total_rows * table.split.map(fractions)
    table["share_of_stratum"] = table.rows / table.total_rows
    table["share_minus_target"] = table.share_of_stratum - table.split.map(fractions)
    by_area = split.groupby(["split", "broad_area", "label_name"]).size().rename("rows").reset_index()
    return table, by_area
