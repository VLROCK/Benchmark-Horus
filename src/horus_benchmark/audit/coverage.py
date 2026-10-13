"""Coverage and observable shortcut summaries for a full corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from horus_benchmark.audit.text_features import BOOLEAN_FEATURES, NUMERIC_FEATURES


LABEL_ORDER = ("human", "synthetic")
COLORS = {"human": "#286090", "synthetic": "#c65943"}


def assign_length_bin(word_count: int, boundaries: list[int]) -> str:
    """Create a stable length bin from measured whitespace word count."""

    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries != sorted(boundaries):
        raise ValueError("length_bins_words must be sorted and start at zero")
    for low, high in zip(boundaries, boundaries[1:]):
        if low <= word_count < high:
            return f"{low}-{high - 1}"
    return f"{boundaries[-1]}+"


def coverage_tables(
    features: pd.DataFrame, length_boundaries: list[int]
) -> dict[str, pd.DataFrame]:
    """Tabulate class/domain/source/generator/length coverage."""

    frame = features.copy()
    frame["length_bin"] = frame["word_count_audit"].map(
        lambda value: assign_length_bin(int(value), length_boundaries)
    )
    tables: dict[str, pd.DataFrame] = {}
    for name, columns in {
        "label_domain": ["label_name", "broad_area"],
        "label_domain_length": ["label_name", "broad_area", "length_bin"],
        "human_source_domain": ["source_name", "broad_area"],
        "synthetic_generator_domain": ["generator", "broad_area"],
        "genre_label_domain": ["label_name", "broad_area", "genre"],
    }.items():
        view = frame
        if name == "human_source_domain":
            view = frame.loc[frame["label_name"] == "human"]
        elif name == "synthetic_generator_domain":
            view = frame.loc[frame["label_name"] == "synthetic"]
        tables[name] = (
            view.groupby(columns, dropna=False, observed=True)
            .size()
            .rename("rows")
            .reset_index()
            .sort_values(["rows", *columns], ascending=[False] + [True] * len(columns))
        )

    cell_counts = tables["label_domain_length"].pivot_table(
        index=["broad_area", "length_bin"], columns="label_name", values="rows",
        fill_value=0, observed=True,
    ).reset_index()
    for label in LABEL_ORDER:
        if label not in cell_counts:
            cell_counts[label] = 0
    cell_counts["potential_pair_count"] = cell_counts[list(LABEL_ORDER)].min(axis=1)
    cell_counts["potential_balanced_rows"] = 2 * cell_counts["potential_pair_count"]
    tables["domain_length_common_support"] = cell_counts

    theme_cardinality = (
        frame.groupby(["label_name", "broad_area"], dropna=False)
        .agg(rows=("record_id", "size"), raw_theme_values=("theme", "nunique"),
             raw_subtheme_values=("subtheme", "nunique"))
        .reset_index()
    )
    tables["raw_theme_cardinality"] = theme_cardinality
    theme_overlap = []
    for domain in ("ALL", *sorted(frame["broad_area"].dropna().unique())):
        view = frame if domain == "ALL" else frame.loc[frame["broad_area"] == domain]
        human_themes = set(view.loc[view["label_name"] == "human", "theme"].dropna()) - {""}
        synthetic_themes = set(view.loc[view["label_name"] == "synthetic", "theme"].dropna()) - {""}
        shared = human_themes & synthetic_themes
        union = human_themes | synthetic_themes
        theme_overlap.append(
            {"broad_area": domain, "human_raw_themes": len(human_themes),
             "synthetic_raw_themes": len(synthetic_themes), "shared_raw_themes": len(shared),
             "raw_theme_jaccard": len(shared) / len(union) if union else 0.0}
        )
    tables["raw_theme_overlap"] = pd.DataFrame(theme_overlap)

    human_urls = frame.loc[
        (frame["label_name"] == "human")
        & frame["source_url"].notna()
        & (frame["source_url"] != "")
    ]
    url_groups = (
        human_urls.groupby(["source_name", "source_url"], dropna=False)
        .agg(rows=("record_id", "size"), splits=("split", "nunique"))
        .reset_index()
    )
    cross_url = url_groups.loc[url_groups["splits"] > 1]
    tables["cross_split_url_by_source"] = (
        cross_url.groupby("source_name", dropna=False)
        .agg(cross_split_urls=("source_url", "size"),
             rows_in_cross_split_urls=("rows", "sum"),
             largest_url_group=("rows", "max"))
        .reset_index()
        .sort_values("cross_split_urls", ascending=False)
    )

    completeness_fields = (
        "record_id", "source_record_id", "content_hash", "split_group_id",
        "theme", "subtheme", "genre", "source_name", "source_url",
        "generator", "prompt_version", "creation_date", "license",
    )
    completeness: list[dict[str, Any]] = []
    for label, group in frame.groupby("label_name"):
        for field in completeness_fields:
            values = group[field]
            populated = values.notna() & (values.astype(str).str.strip() != "")
            completeness.append(
                {"label_name": label, "field": field, "rows": len(group),
                 "populated": int(populated.sum()), "populated_rate": float(populated.mean())}
            )
    tables["metadata_completeness"] = pd.DataFrame(completeness)
    discrepancy = frame.assign(
        content_hash_mismatch=frame["content_hash"] != frame["content_sha256_audit"],
        word_count_mismatch=frame["word_count"] != frame["word_count_audit"],
    )
    tables["metadata_discrepancies_by_source"] = (
        discrepancy.groupby(["label_name", "broad_area", "source_name"], dropna=False)
        .agg(rows=("record_id", "size"),
             content_hash_mismatches=("content_hash_mismatch", "sum"),
             word_count_mismatches=("word_count_mismatch", "sum"))
        .reset_index()
        .sort_values(["content_hash_mismatches", "word_count_mismatches"], ascending=False)
    )
    return tables


def feature_summaries(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize measured features by label and domain, plus class gaps."""

    numeric_rows: list[dict[str, Any]] = []
    boolean_rows: list[dict[str, Any]] = []
    for (label, domain), group in features.groupby(["label_name", "broad_area"], dropna=False):
        for feature in NUMERIC_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            numeric_rows.append(
                {"label_name": label, "broad_area": domain, "feature": feature,
                 "rows": len(values), "mean": float(values.mean()),
                 "median": float(values.median()), "p10": float(values.quantile(0.10)),
                 "p90": float(values.quantile(0.90))}
            )
        for feature in BOOLEAN_FEATURES:
            values = group[feature].fillna(False).astype(bool)
            boolean_rows.append(
                {"label_name": label, "broad_area": domain, "feature": feature,
                 "rows": len(values), "positive": int(values.sum()),
                 "positive_rate": float(values.mean())}
            )

    gaps: list[dict[str, Any]] = []
    for domain in ("ALL", *sorted(features["broad_area"].dropna().unique())):
        view = features if domain == "ALL" else features.loc[features["broad_area"] == domain]
        human = view.loc[view["label_name"] == "human"]
        synthetic = view.loc[view["label_name"] == "synthetic"]
        if human.empty or synthetic.empty:
            continue
        for feature in NUMERIC_FEATURES:
            a = human[feature].astype(float).dropna().to_numpy()
            b = synthetic[feature].astype(float).dropna().to_numpy()
            pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            standardized_gap = (b.mean() - a.mean()) / pooled_sd if pooled_sd > 0 else 0.0
            gaps.append(
                {"broad_area": domain, "feature": feature, "feature_type": "numeric",
                 "human_n": len(a), "synthetic_n": len(b),
                 "human_value": float(np.median(a)), "synthetic_value": float(np.median(b)),
                 "signed_gap": float(b.mean() - a.mean()),
                 "standardized_mean_difference": float(standardized_gap),
                 "ks_statistic": float(ks_2samp(a, b, method="asymp").statistic)}
            )
        for feature in BOOLEAN_FEATURES:
            a = human[feature].fillna(False).astype(bool).to_numpy()
            b = synthetic[feature].fillna(False).astype(bool).to_numpy()
            gaps.append(
                {"broad_area": domain, "feature": feature, "feature_type": "boolean",
                 "human_n": len(a), "synthetic_n": len(b),
                 "human_value": float(a.mean()), "synthetic_value": float(b.mean()),
                 "signed_gap": float(b.mean() - a.mean()),
                 "standardized_mean_difference": np.nan, "ks_statistic": np.nan}
            )
    return {
        "numeric_by_label_domain": pd.DataFrame(numeric_rows),
        "boolean_by_label_domain": pd.DataFrame(boolean_rows),
        "shortcut_gaps": pd.DataFrame(gaps),
    }


def near_pair_review_tables(
    features: pd.DataFrame, pairs: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Attach provenance to verified pairs and prioritize cross-split review."""

    columns = [
        "record_id", "source_name", "source_url", "source_record_id",
        "split_group_id", "word_count_audit",
    ]
    if pairs.empty:
        return {
            "near_pair_review": pairs.copy(),
            "cross_split_near_by_source": pd.DataFrame(
                columns=["left_source_name", "right_source_name", "pairs", "same_url_pairs"]
            ),
        }
    context = features[columns]
    if context["record_id"].duplicated().any():
        raise ValueError("Record IDs must be unique for near-pair context")
    review = pairs.merge(
        context.add_prefix("left_"), on="left_record_id", validate="many_to_one"
    ).merge(
        context.add_prefix("right_"), on="right_record_id", validate="many_to_one"
    )
    review["same_source_url"] = (
        review["left_source_url"].notna()
        & (review["left_source_url"] != "")
        & (review["left_source_url"] == review["right_source_url"])
    )
    review["same_split_group"] = (
        review["left_split_group_id"].notna()
        & (review["left_split_group_id"] == review["right_split_group_id"])
    )
    review = review.sort_values(
        ["cross_split", "containment_5gram", "jaccard_5gram"],
        ascending=[False, False, False],
    )
    cross = review.loc[review["cross_split"]]
    source_summary = (
        cross.groupby(["left_source_name", "right_source_name"], dropna=False)
        .agg(pairs=("left_record_id", "size"), same_url_pairs=("same_source_url", "sum"))
        .reset_index()
        .sort_values("pairs", ascending=False)
    )
    return {"near_pair_review": review, "cross_split_near_by_source": source_summary}


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def create_figures(
    features: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    """Create compact publication-style exploratory figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    domain = tables["label_domain"].pivot(index="broad_area", columns="label_name", values="rows").fillna(0)
    domain = domain.reindex(columns=LABEL_ORDER, fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(domain))
    width = 0.38
    for offset, label in ((-width / 2, "human"), (width / 2, "synthetic")):
        ax.bar(positions + offset, domain[label], width, label=label, color=COLORS[label])
    ax.set_xticks(positions, domain.index, rotation=20, ha="right")
    ax.set_ylabel("Textos")
    ax.set_title("Cobertura por domínio e classe — corpus completo")
    ax.legend(frameon=False)
    _save_figure(fig, output_dir / "01_domain_coverage")
    outputs.append(output_dir / "01_domain_coverage.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bins = np.geomspace(1, max(2, int(features["word_count_audit"].max()) + 1), 38)
    for label in LABEL_ORDER:
        counts = features.loc[features["label_name"] == label, "word_count_audit"]
        ax.hist(np.maximum(counts, 1), bins=bins, density=True, alpha=0.55,
                color=COLORS[label], label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Palavras medidas (escala log)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição de comprimento — corpus completo")
    ax.legend(frameon=False)
    _save_figure(fig, output_dir / "02_length_distribution")
    outputs.append(output_dir / "02_length_distribution.png")

    gap = tables["shortcut_gaps"]
    numeric = gap.loc[(gap["broad_area"] == "ALL") & (gap["feature_type"] == "numeric")].copy()
    numeric = numeric.reindex(numeric["standardized_mean_difference"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(numeric["feature"], numeric["standardized_mean_difference"], color="#577b8e")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Diferença padronizada das médias (sintético − humano)")
    ax.set_title("Sinais superficiais: tamanho e forma da diferença")
    _save_figure(fig, output_dir / "03_numeric_feature_gaps")
    outputs.append(output_dir / "03_numeric_feature_gaps.png")

    binary = gap.loc[(gap["broad_area"] == "ALL") & (gap["feature_type"] == "boolean")].copy()
    binary = binary.sort_values("signed_gap")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(binary["feature"], 100 * binary["signed_gap"], color="#b56e50")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Diferença na prevalência, pontos percentuais (sintético − humano)")
    ax.set_title("Marcas de apresentação observadas no texto")
    _save_figure(fig, output_dir / "04_presentation_gaps")
    outputs.append(output_dir / "04_presentation_gaps.png")

    matrix = tables["synthetic_generator_domain"].pivot(
        index="generator", columns="broad_area", values="rows"
    ).fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Cobertura dos geradores por domínio")
    fig.colorbar(image, ax=ax, label="Textos")
    _save_figure(fig, output_dir / "05_generator_domain_coverage")
    outputs.append(output_dir / "05_generator_domain_coverage.png")

    support = tables["domain_length_common_support"].pivot(
        index="broad_area", columns="length_bin", values="potential_pair_count"
    ).fillna(0)
    length_order = sorted(
        tables["label_domain_length"]["length_bin"].astype(str).unique(),
        key=lambda value: int(value.split("-")[0].rstrip("+")),
    )
    support = support.reindex(columns=length_order, fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(support.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(support.columns)), support.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(support.index)), support.index)
    for row_index in range(len(support.index)):
        for column_index in range(len(support.columns)):
            ax.text(column_index, row_index, f"{int(support.iat[row_index, column_index]):,}",
                    ha="center", va="center", fontsize=7)
    ax.set_title("Suporte comum por domínio e comprimento")
    ax.set_xlabel("Faixa de palavras medidas")
    fig.colorbar(image, ax=ax, label="Pares humano–sintético possíveis")
    _save_figure(fig, output_dir / "06_domain_length_common_support")
    outputs.append(output_dir / "06_domain_length_common_support.png")

    if "shortcut_probes" in tables and not tables["shortcut_probes"].empty:
        probes = tables["shortcut_probes"].sort_values("roc_auc")
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.barh(probes["probe"], probes["roc_auc"], color="#577b8e")
        ax.errorbar(
            probes["roc_auc"], probes["probe"],
            xerr=[probes["roc_auc"] - probes["roc_auc_ci_low"],
                  probes["roc_auc_ci_high"] - probes["roc_auc"]],
            fmt="none", color="black", capsize=3,
        )
        ax.set_xlim(0.5, 1.01)
        ax.set_xlabel("ROC-AUC no teste (IC bootstrap de 95%)")
        ax.set_title("Sondas diagnósticas de separabilidade")
        _save_figure(fig, output_dir / "07_shortcut_probe_auc")
        outputs.append(output_dir / "07_shortcut_probe_auc.png")
    return outputs
