"""Read-only N2/N3s selection diagnostics and *candidate* simulations.

No row is deleted from a corpus and no final benchmark split is made. N3s
contains N2 human text and N3 presentation-cleaned synthetic text. Run the
analysis on the researcher's cluster, per AGENTS.md rule 19.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from horus_benchmark.audit.probes import validation_threshold
from horus_benchmark.audit.text_features import extract_text_features
from horus_benchmark.data.normalize_corpus import sha256_file


META = ("record_id", "label_name", "broad_area", "source_name", "generator", "split_group_id")
FEATURES = (
    "char_count_audit", "word_count_audit", "paragraph_count", "nonempty_line_count",
    "punctuation_per_100_words", "comma_per_100_words", "quote_per_100_words",
    "newline_per_100_words", "markdown_bullet", "unordered_list_marker",
    "numbered_list_marker", "markdown_heading",
    "markdown_code_fence", "html_tag", "url_in_content",
)
BALANCE_FEATURES = (
    "log_words", "log_chars", "paragraph_count", "nonempty_line_count",
    "punctuation_per_100_words", "comma_per_100_words", "quote_per_100_words",
    "newline_per_100_words", "markdown_bullet", "unordered_list_marker",
    "numbered_list_marker", "markdown_heading",
    "markdown_code_fence", "html_tag", "url_in_content",
)
PROBES = {
    "length": ("log_words", "log_chars"),
    "presentation": (
        "paragraph_count", "nonempty_line_count", "newline_per_100_words",
        "markdown_bullet", "unordered_list_marker", "numbered_list_marker",
        "markdown_heading", "markdown_code_fence", "html_tag",
    ),
    "punctuation": (
        "punctuation_per_100_words", "comma_per_100_words", "quote_per_100_words",
    ),
    "length_presentation": (
        "log_words", "log_chars", "paragraph_count", "nonempty_line_count",
        "newline_per_100_words", "markdown_bullet", "unordered_list_marker",
        "numbered_list_marker", "markdown_heading",
        "markdown_code_fence", "html_tag",
    ),
}
LABELS = ("human", "synthetic")
SCENARIOS = ("full", "common_support", "balanced_coarse", "balanced_fine")
VIEWS = ("n2", "n3")
UNORDERED_LIST = re.compile(r"(?m)^[ \t]*[-*+][ \t]+\S")
NUMBERED_LIST = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]+\S")


def _validated_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("experiment_id") != "n2-n3s-selection-diagnostics-v1":
        raise ValueError("Unexpected selection diagnostics config")
    if tuple(config.get("scenarios", ())) != SCENARIOS or tuple(config.get("views", ())) != VIEWS:
        raise ValueError("This version requires N2, N3s and four scenarios")
    edges = config.get("coarse_word_edges")
    if not isinstance(edges, list) or len(edges) < 2 or edges[0] != 0 or sorted(set(edges)) != edges:
        raise ValueError("coarse_word_edges must be unique, increasing and start at zero")
    if float(config.get("fine_log2_width", 0)) <= 0:
        raise ValueError("fine_log2_width must be positive")
    fractions = [float(config[f"probe_{name}_fraction"]) for name in ("train", "validation", "test")]
    if any(x <= 0 for x in fractions) or not math.isclose(sum(fractions), 1.0):
        raise ValueError("Probe split fractions must be positive and total one")
    return config


def _check_view(run_dir: Path, expected_stage: str) -> tuple[Path, dict]:
    verification = json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if verification.get("verified") is not True or verification.get("stage") != expected_stage:
        raise ValueError(f"Unverified {expected_stage} view: {run_dir}")
    if summary.get("output_rows") != verification.get("rows") or summary.get("stage") != expected_stage:
        raise ValueError(f"Summary disagrees with verification: {run_dir}")
    if summary.get("source_sha256") != verification.get("source_sha256"):
        raise ValueError(f"Source SHA-256 disagrees: {run_dir}")
    if expected_stage == "n3" and (
        summary.get("presentation_target_labels") != ["synthetic"]
        or summary.get("experiment_id") != "corpus-normalization-n3-synthetic-presentation-v1"
    ):
        raise ValueError("Selection diagnostics require the NEW synthetic-only N3 run")
    if expected_stage == "n3":
        cross_check = json.loads((run_dir / "class_specific_verification.json").read_text(encoding="utf-8"))
        if cross_check.get("verified") is not True or cross_check["counts"]["human_changed"] != 0:
            raise ValueError("N3s must pass the direct N2 human-preservation check")
    corpus = run_dir / "normalized_corpus.parquet"
    if pq.ParquetFile(corpus).metadata.num_rows != verification["rows"]:
        raise ValueError(f"Physical row count disagrees: {corpus}")
    return corpus, verification


def _load_view(corpus: Path, trace: Path) -> pd.DataFrame:
    wanted = (*META, "content")
    schema = ds.dataset(str(corpus), format="parquet").schema
    if set(wanted) - set(schema.names):
        raise ValueError(f"Missing required corpus fields: {sorted(set(wanted) - set(schema.names))}")
    rows = []
    for batch in ds.dataset(str(corpus), format="parquet").to_batches(columns=list(wanted), batch_size=512):
        for row in batch.to_pylist():
            if row["label_name"] not in LABELS or not isinstance(row["content"], str):
                raise ValueError("Invalid label or content in normalized corpus")
            content = row.pop("content")
            feature = extract_text_features(content)
            feature["unordered_list_marker"] = bool(UNORDERED_LIST.search(content))
            feature["numbered_list_marker"] = bool(NUMBERED_LIST.search(content))
            rows.append({**row, **{key: feature[key] for key in FEATURES},
                         "content_sha256_audit": feature["content_sha256_audit"]})
    frame = pd.DataFrame.from_records(rows)
    if frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise ValueError("Record IDs must be non-null and unique")
    statuses = pd.read_parquet(trace, columns=["record_id", "normalization_status"])
    if statuses.record_id.duplicated().any() or len(statuses) != len(frame):
        raise ValueError("Trace IDs are duplicated or incomplete")
    frame = frame.merge(statuses, on="record_id", how="left", validate="one_to_one")
    if frame.normalization_status.isna().any():
        raise ValueError("Trace is missing corpus IDs")
    frame["log_words"] = np.log2(frame.word_count_audit.astype(float) + 1)
    frame["log_chars"] = np.log2(frame.char_count_audit.astype(float) + 1)
    return frame.sort_values("record_id", kind="stable").reset_index(drop=True)


def _check_paired_views(n2: pd.DataFrame, n3: pd.DataFrame) -> None:
    if len(n2) != len(n3) or not n2.record_id.equals(n3.record_id):
        raise ValueError("N2 and N3 must have precisely the same sorted IDs")
    for col in META[1:]:
        if not n2[col].fillna("<NULL>").equals(n3[col].fillna("<NULL>")):
            raise ValueError(f"Protected metadata changed between N2 and N3: {col}")
    human = n2.label_name.eq("human")
    if not n2.loc[human, "content_sha256_audit"].equals(
        n3.loc[human, "content_sha256_audit"]
    ):
        raise ValueError("Human N3s texts must match N2 byte for byte")


def _coarse_bins(words: pd.Series, edges: list[int]) -> pd.Series:
    boundaries = [*edges, np.inf]
    labels = [f"{left}-{right - 1}" for left, right in zip(edges[:-1], edges[1:])]
    labels.append(f"{edges[-1]}+")
    return pd.cut(words, bins=boundaries, labels=labels, right=False, include_lowest=True).astype(str)


def _add_bins(frame: pd.DataFrame, config: dict, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["coarse_bin"] = _coarse_bins(frame.word_count_audit, config["coarse_word_edges"])
    frame["fine_bin"] = np.floor(frame.log_words / float(config["fine_log2_width"])).astype(int)
    frame["sample_key"] = frame.record_id.map(
        lambda rid: hashlib.sha256(f"{seed}|{rid}".encode("utf-8")).hexdigest()
    )
    return frame


def _support_table(frame: pd.DataFrame, bin_column: str, view: str) -> pd.DataFrame:
    counts = frame.groupby(["broad_area", bin_column, "label_name"], observed=True).size()
    table = counts.unstack("label_name", fill_value=0).reset_index()
    for label in LABELS:
        if label not in table:
            table[label] = 0
    table["potential_pairs"] = table[["human", "synthetic"]].min(axis=1)
    table["has_common_support"] = table.potential_pairs.gt(0)
    table["view"] = view
    return table.rename(columns={bin_column: "length_bin"})[
        ["view", "broad_area", "length_bin", "human", "synthetic", "potential_pairs", "has_common_support"]
    ]


def _candidate_ids(frame: pd.DataFrame, scenario: str) -> set[str]:
    if scenario == "full":
        return set(frame.record_id)
    key = "fine_bin" if scenario == "balanced_fine" else "coarse_bin"
    selected: set[str] = set()
    for _, cell in frame.groupby(["broad_area", key], sort=True, observed=True):
        human = cell.loc[cell.label_name.eq("human")]
        synthetic = cell.loc[cell.label_name.eq("synthetic")]
        if human.empty or synthetic.empty:
            continue
        if scenario == "common_support":
            selected.update(cell.record_id)
            continue
        n = min(len(human), len(synthetic))
        for group in (human, synthetic):
            # SHA-256 keys are strings; pandas.nsmallest only accepts numbers.
            selected.update(group.sort_values(["sample_key", "record_id"], kind="stable")
                            .head(n).record_id)
    return selected


def _smd(human: np.ndarray, synthetic: np.ndarray) -> float:
    if not len(human) or not len(synthetic):
        return float("nan")
    difference = float(np.mean(synthetic) - np.mean(human))
    spread = math.sqrt((float(np.var(human)) + float(np.var(synthetic))) / 2)
    if spread == 0:
        return 0.0 if difference == 0 else float("inf")
    return difference / spread


def _ks(human: np.ndarray, synthetic: np.ndarray) -> float:
    if not len(human) or not len(synthetic):
        return float("nan")
    combined = np.sort(np.unique(np.concatenate([human, synthetic])))
    h = np.searchsorted(np.sort(human), combined, side="right") / len(human)
    s = np.searchsorted(np.sort(synthetic), combined, side="right") / len(synthetic)
    return float(np.max(np.abs(h - s)))


def _balance_rows(frame: pd.DataFrame, view: str, scenario: str) -> list[dict]:
    output = []
    for area, group in [("ALL", frame), *list(frame.groupby("broad_area", sort=True))]:
        human = group.loc[group.label_name.eq("human")]
        synthetic = group.loc[group.label_name.eq("synthetic")]
        for feature in BALANCE_FEATURES:
            h = human[feature].to_numpy(dtype=float)
            s = synthetic[feature].to_numpy(dtype=float)
            output.append({
                "view": view, "scenario": scenario, "broad_area": area,
                "feature": feature, "human_n": len(h), "synthetic_n": len(s),
                "human_mean": float(np.mean(h)) if len(h) else np.nan,
                "synthetic_mean": float(np.mean(s)) if len(s) else np.nan,
                "smd_synthetic_minus_human": _smd(h, s), "ks": _ks(h, s),
            })
    return output


def _composition_rows(frame: pd.DataFrame, full: pd.DataFrame, view: str, scenario: str) -> list[dict]:
    output = []
    full_counts = full.groupby(["broad_area", "label_name"]).size().to_dict()
    for (area, label), group in frame.groupby(["broad_area", "label_name"], sort=True):
        provenance = "source_name" if label == "human" else "generator"
        counts = group[provenance].fillna("<missing>").value_counts()
        shares = counts / len(group)
        words = group.word_count_audit
        output.append({
            "view": view, "scenario": scenario, "broad_area": area, "label_name": label,
            "rows": len(group), "retention": len(group) / full_counts[(area, label)],
            "words_q05": float(words.quantile(.05)), "words_q25": float(words.quantile(.25)),
            "words_median": float(words.median()), "words_q75": float(words.quantile(.75)),
            "words_q95": float(words.quantile(.95)),
            "review_rate": float(group.normalization_status.eq("review").mean()),
            "markdown_bullet_rate": float(group.markdown_bullet.mean()),
            "unordered_list_marker_rate": float(group.unordered_list_marker.mean()),
            "numbered_list_marker_rate": float(group.numbered_list_marker.mean()),
            "provenance_field": provenance, "provenance_unique": len(counts),
            "largest_provenance": str(counts.index[0]),
            "largest_provenance_share": float(shares.iloc[0]),
            "effective_provenance_count": float(1 / np.square(shares.to_numpy()).sum()),
        })
    return output


def _provenance_rows(frame: pd.DataFrame, full: pd.DataFrame,
                     view: str, scenario: str) -> list[dict]:
    output = []
    for label, column in (("human", "source_name"), ("synthetic", "generator")):
        base = full.loc[full.label_name.eq(label)].copy()
        selected = frame.loc[frame.label_name.eq(label)].copy()
        base["provenance"] = base[column].fillna("<missing>")
        selected["provenance"] = selected[column].fillna("<missing>")
        base_counts = base.groupby(["broad_area", "provenance"]).size()
        selected_counts = selected.groupby(["broad_area", "provenance"]).size()
        for (area, provenance), n in base_counts.items():
            retained = int(selected_counts.get((area, provenance), 0))
            output.append({
                "view": view, "scenario": scenario, "broad_area": area,
                "label_name": label, "provenance_field": column,
                "provenance": provenance, "full_rows": int(n),
                "selected_rows": retained, "retention": retained / int(n),
            })
    return output


def _provenance_by_bin(frame: pd.DataFrame, view: str) -> list[dict]:
    output = []
    for label, column in (("human", "source_name"), ("synthetic", "generator")):
        subset = frame.loc[frame.label_name.eq(label)].copy()
        subset["provenance"] = subset[column].fillna("<missing>")
        counts = subset.groupby(["broad_area", "coarse_bin", "provenance"], observed=True).size()
        denominators = subset.groupby(["broad_area", "coarse_bin"], observed=True).size()
        for (area, bin_name, provenance), n in counts.items():
            output.append({"view": view, "broad_area": area, "length_bin": bin_name,
                           "label_name": label, "provenance": provenance,
                           "rows": int(n), "within_bin_share": int(n) / int(denominators[(area, bin_name)])})
    return output


def _probe_split(frame: pd.DataFrame, config: dict) -> np.ndarray:
    seed = int(config["seed"])
    group = frame.split_group_id.fillna("").astype(str)
    group = group.where(group.ne(""), frame.record_id)
    hashes = group.map(lambda value: int.from_bytes(
        hashlib.sha256(f"{seed}|probe|{value}".encode("utf-8")).digest()[:8], "big"
    ) / 2**64)
    train = float(config["probe_train_fraction"])
    validation = train + float(config["probe_validation_fraction"])
    return np.where(hashes < train, "train", np.where(hashes < validation, "validation", "test"))


def _probe_rows(frame: pd.DataFrame, view: str, scenario: str, config: dict) -> list[dict]:
    splits = _probe_split(frame, config)
    y = frame.label_name.eq("synthetic").astype(int).to_numpy()
    output = []
    for probe, features in PROBES.items():
        matrix = frame.loc[:, list(features)].to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        indices = {part: np.flatnonzero(splits == part) for part in ("train", "validation", "test")}
        if any(len(np.unique(y[idx])) != 2 for idx in indices.values()):
            output.append({"view": view, "scenario": scenario, "probe": probe,
                           "status": "insufficient_labels_after_group_split"})
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=int(config["seed"])
        ))
        model.fit(matrix[indices["train"]], y[indices["train"]])
        valid_scores = model.predict_proba(matrix[indices["validation"]])[:, 1]
        threshold = validation_threshold(y[indices["validation"]], valid_scores)
        test_scores = model.predict_proba(matrix[indices["test"]])[:, 1]
        output.append({
            "view": view, "scenario": scenario, "probe": probe, "status": "exploratory",
            "train_rows": len(indices["train"]), "validation_rows": len(indices["validation"]),
            "test_rows": len(indices["test"]), "validation_threshold": threshold,
            "roc_auc": float(roc_auc_score(y[indices["test"]], test_scores)),
            "balanced_accuracy": float(balanced_accuracy_score(
                y[indices["test"]], test_scores >= threshold
            )),
        })
    return output


def _plot_lengths(frame: pd.DataFrame, view: str, output: Path) -> None:
    areas = sorted(frame.broad_area.unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    right_edge = max(15, math.ceil(float(frame.log_words.max())))
    for axis, area in zip(axes.flat, areas):
        for label, color in (("human", "#1463a0"), ("synthetic", "#df6c28")):
            values = frame.loc[(frame.broad_area == area) & (frame.label_name == label), "log_words"]
            axis.hist(values, bins=np.linspace(0, right_edge, 61), density=True, alpha=.45,
                      label=label, color=color)
        axis.set_title(area)
        axis.set_xlabel("log2(1 + palavras)")
        axis.set_ylabel("Densidade")
    axes.flat[0].legend()
    fig.suptitle(f"Comprimento por área e classe · {view} · corpus completo")
    fig.tight_layout()
    fig.savefig(output / f"length_by_area_{view}.png", dpi=170)
    plt.close(fig)


def _plot_scenarios(balance: pd.DataFrame, composition: pd.DataFrame, output: Path) -> None:
    length = balance.loc[(balance.feature == "log_words") & (balance.broad_area != "ALL")]
    areas = sorted(length.broad_area.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = []
    for index, (view, scenario) in enumerate([(v, s) for v in VIEWS for s in SCENARIOS]):
        subset = length.loc[(length.view == view) & (length.scenario == scenario)]
        values = [abs(float(subset.loc[subset.broad_area == area, "smd_synthetic_minus_human"].iloc[0]))
                  if not subset.loc[subset.broad_area == area].empty else np.nan for area in areas]
        axes[0].scatter([index] * len(values), values, alpha=.6, s=22)
        labels.append(f"{view}\n{scenario}")
    axes[0].set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("|SMD| de log2 palavras, por área")
    axes[0].set_title("Equilíbrio de comprimento")
    totals = composition.groupby(["view", "scenario"]).rows.sum()
    for index, pair in enumerate([(v, s) for v in VIEWS for s in SCENARIOS]):
        axes[1].bar(index, int(totals.get(pair, 0)))
    axes[1].set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axes[1].set_ylabel("Textos retidos")
    axes[1].set_title("Custo amostral")
    fig.tight_layout()
    fig.savefig(output / "scenario_tradeoff.png", dpi=170)
    plt.close(fig)


def run(config_path: Path, n2_run: Path, n3_run: Path, output: Path) -> Path:
    config_path, n2_run, n3_run, output = [p.expanduser().resolve() for p in
                                            (config_path, n2_run, n3_run, output)]
    config = _validated_config(config_path)
    n2_corpus, n2_verification = _check_view(n2_run, "n2")
    n3_corpus, n3_verification = _check_view(n3_run, "n3")
    if n2_verification["source_sha256"] != n3_verification["source_sha256"]:
        raise ValueError("N2 and N3 must derive from the same immutable source")
    if output.exists() or output.is_relative_to(n2_run) or output.is_relative_to(n3_run):
        raise FileExistsError(f"Output must be a new directory: {output}")
    output.mkdir(parents=True)
    n2 = _load_view(n2_corpus, n2_run / "normalization_trace.parquet")
    n3 = _load_view(n3_corpus, n3_run / "normalization_trace.parquet")
    _check_paired_views(n2, n3)
    views = {"n2": n2, "n3": n3}
    balance_rows, composition_rows, probe_rows, support_rows, membership_rows = [], [], [], [], []
    provenance_rows, provenance_bin_rows = [], []
    ids_by_case: dict[tuple[str, str], set[str]] = {}
    for view, original in views.items():
        frame = _add_bins(original, config, int(config["seed"]))
        _plot_lengths(frame, view, output)
        for bin_name in ("coarse_bin", "fine_bin"):
            support = _support_table(frame, bin_name, view)
            support["bin_scheme"] = bin_name
            support_rows.append(support)
        provenance_bin_rows.extend(_provenance_by_bin(frame, view))
        for scenario in SCENARIOS:
            ids = _candidate_ids(frame, scenario)
            ids_by_case[(view, scenario)] = ids
            candidate = frame.loc[frame.record_id.isin(ids)]
            balance_rows.extend(_balance_rows(candidate, view, scenario))
            composition_rows.extend(_composition_rows(candidate, frame, view, scenario))
            provenance_rows.extend(_provenance_rows(candidate, frame, view, scenario))
            probe_rows.extend(_probe_rows(candidate, view, scenario, config))
            membership_rows.extend({"view": view, "scenario": scenario, "record_id": rid}
                                   for rid in sorted(ids))
            print(f"{view} {scenario}: {len(ids)} rows", flush=True)
    # Hold N2 candidate IDs fixed to isolate the effect of presentation rules.
    # Independent selections above answer the separate question of whether N3
    # changes *which* documents a length-based selection would retain.
    for view in ("n3",):
        frame = _add_bins(views[view], config, int(config["seed"]))
        for scenario in SCENARIOS:
            fixed_name = f"n2_fixed_{scenario}"
            ids = ids_by_case[("n2", scenario)]
            candidate = frame.loc[frame.record_id.isin(ids)]
            balance_rows.extend(_balance_rows(candidate, view, fixed_name))
            composition_rows.extend(_composition_rows(candidate, frame, view, fixed_name))
            provenance_rows.extend(_provenance_rows(candidate, frame, view, fixed_name))
            probe_rows.extend(_probe_rows(candidate, view, fixed_name, config))
    balance = pd.DataFrame(balance_rows)
    composition = pd.DataFrame(composition_rows)
    pd.concat(support_rows, ignore_index=True).to_csv(output / "common_support.csv", index=False)
    balance.to_csv(output / "balance_smd_ks.csv", index=False)
    composition.to_csv(output / "composition_retention.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(output / "provenance_retention.csv", index=False)
    pd.DataFrame(provenance_bin_rows).to_csv(output / "provenance_by_length_bin.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(output / "surface_probes.csv", index=False)
    membership = pa.Table.from_pylist(membership_rows)
    pq.write_table(membership, output / "candidate_membership.parquet", compression="zstd")
    pairwise = []
    for scenario in SCENARIOS:
        left = ids_by_case[("n2", scenario)]
        for right_view in ("n3",):
            right = ids_by_case[(right_view, scenario)]
            pairwise.append({"scenario": scenario, "left_view": "n2", "right_view": right_view,
                             "shared_ids": len(left & right), "n2_only": len(left - right),
                             "other_only": len(right - left),
                             "jaccard": len(left & right) / len(left | right)})
    pd.DataFrame(pairwise).to_csv(output / "candidate_overlap.csv", index=False)
    changed = n2.loc[:, ["label_name", "broad_area", "content_sha256_audit", "word_count_audit"]].copy()
    changed["changed_text"] = n2.content_sha256_audit.ne(n3.content_sha256_audit)
    changed["n3_minus_n2_words"] = n3.word_count_audit - n2.word_count_audit
    changed.groupby(["label_name", "broad_area"], observed=True).agg(
        rows=("changed_text", "size"), changed_rows=("changed_text", "sum"),
        mean_word_delta=("n3_minus_n2_words", "mean"),
    ).reset_index().to_csv(output / "n3_changes_by_class_area.csv", index=False)
    _plot_scenarios(balance, composition, output)
    summary = {
        "experiment_id": config["experiment_id"], "seed": config["seed"],
        "n2_run": str(n2_run), "n3_run": str(n3_run),
        "n2_rows": len(n2), "n3_rows": len(n3),
        "source_sha256": n2_verification["source_sha256"],
        "n2_corpus_sha256": sha256_file(n2_corpus),
        "n3_corpus_sha256": sha256_file(n3_corpus),
        "config_sha256": sha256_file(config_path), "code_sha256": sha256_file(Path(__file__)),
        "python": sys.version, "pandas": pd.__version__, "pyarrow": pa.__version__,
        "sklearn": sklearn.__version__,
        "selection_status": "candidate_simulations_only",
        "n3_presentation_target_labels": ["synthetic"],
        "limitations": [
            "Existing near-duplicate and parent-document leakage is not resolved.",
            "Probe group splits use source split_group_id when available and remain exploratory.",
            "No quality, theme, genre, source or generator eligibility rule is imposed.",
            "Fine and coarse bins are sensitivity scenarios, not approved criteria.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    (output / "README.md").write_text(
        "# Diagnóstico de seleção N2/N3\n\n"
        "As duas views usam os mesmos IDs e o corpus completo. "
        "N3s preserva exatamente o conteúdo humano da N2 e aplica Markdown/placeholder só aos sintéticos. "
        "Nenhum critério de seleção está aprovado e nenhum corpus final foi escrito.\n\n"
        "- `common_support.csv`: contagem e potencial de pares por área/faixa.\n"
        "- `provenance_by_length_bin.csv`: fontes e geradores em cada faixa.\n"
        "- `balance_smd_ks.csv`: diferenças padronizadas e KS por atributo/área.\n"
        "- `composition_retention.csv` e `provenance_retention.csv`: perda de cobertura.\n"
        "- `surface_probes.csv`: sondas exploratórias com ajuste no treino e limiar na validação.\n"
        "- `candidate_overlap.csv`: alteração dos IDs selecionados ao mudar a view.\n"
        "- `n3_changes_by_class_area.csv`: efeito da N3 real em cada classe.\n"
        "- `candidate_membership.parquet`: IDs das simulações; arquivo local ignorado pelo Git.\n"
        "- `length_by_area_*.png` e `scenario_tradeoff.png`: figuras.\n\n"
        "`n2_fixed_*` em tabelas de métricas aplica os IDs escolhidos na N2 "
        "às outras views, isolando o efeito da apresentação. Os demais cenários "
        "são selecionados independentemente em cada view. Os splits dos probes "
        "agrupam `split_group_id` quando presente, mas near-duplicates e documentos-pai "
        "ainda não estão inteiramente resolvidos; resultados não são teste final nem OOD.\n",
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n2-run", type=Path, required=True)
    parser.add_argument("--n3-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.n2_run, args.n3_run, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
