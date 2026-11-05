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


def assign_groups(frame: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Greedy minimum incremental standardized imbalance over group vectors.

    Groups are indivisible. Largest groups are placed first; deterministic hash
    breaks ties. The objective controls area × class × coarse length and total
    size. It is a construction heuristic, not proof of ideal stratification.
    """
    if set(fractions) != set(SPLITS) or any(fractions[x] <= 0 for x in SPLITS):
        raise ValueError("Positive train/validation/test fractions required")
    if not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to one")
    if frame.record_id.duplicated().any() or frame.probe_group_id.isna().any():
        raise ValueError("Duplicate IDs or missing probe groups")
    counts = pd.crosstab(frame.probe_group_id, frame.stratum).astype(np.int64)
    groups = counts.index.astype(str).tolist()
    vectors = counts.to_numpy(dtype=np.float64)
    sizes = vectors.sum(axis=1)
    strata_totals = vectors.sum(axis=0)
    target = np.array([fractions[name] for name in SPLITS])[:, None] * strata_totals
    size_target = np.array([fractions[name] for name in SPLITS]) * len(frame)
    weights = 1.0 / np.maximum(strata_totals, 1.0)
    current = np.zeros_like(target)
    current_size = np.zeros(3, dtype=np.float64)
    order = sorted(range(len(groups)), key=lambda idx: (-sizes[idx], stable_key(seed, groups[idx])))
    assignments: dict[str, str] = {}
    for idx in order:
        vector = vectors[idx]
        size = sizes[idx]
        scores = []
        for part in range(3):
            old = current[part] - target[part]
            new = old + vector
            delta = float(np.sum((new * new - old * old) * weights))
            old_size = current_size[part] - size_target[part]
            delta += float((2 * old_size * size + size * size) / len(frame))
            scores.append((delta, stable_key(seed, f"{groups[idx]}|{SPLITS[part]}"), part))
        choice = min(scores)[2]
        current[choice] += vector
        current_size[choice] += size
        assignments[groups[idx]] = SPLITS[choice]
    result = frame[["record_id", "probe_group_id", "stratum", "broad_area", "label_name",
                    "length_stratum"]].copy()
    result["split"] = result.probe_group_id.map(assignments)
    if result.split.isna().any() or result.groupby("probe_group_id").split.nunique().max() != 1:
        raise AssertionError("Group leakage in assigned split")
    return result


def check_observed_pairs(split: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, int]:
    by_id = split.set_index("record_id").split.to_dict()
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
        [SPLITS, sorted(split.broad_area.unique()), sorted(split.label_name.unique()),
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
