"""Paired comparison of two audits over the same record IDs and splits."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from horus_benchmark.audit.text_features import BOOLEAN_FEATURES, NUMERIC_FEATURES


def compare_runs(before: str | Path, after: str | Path, output: str | Path) -> Path:
    """Compare original and normalized views only when identities align exactly."""

    before_dir = Path(before).expanduser().resolve()
    after_dir = Path(after).expanduser().resolve()
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Comparison output directory is not empty: {output_dir}")

    before_features = pd.read_parquet(before_dir / "features.parquet")
    after_features = pd.read_parquet(after_dir / "features.parquet")
    if before_features["record_id"].duplicated().any() or after_features["record_id"].duplicated().any():
        raise ValueError("Paired comparison requires unique record IDs in both audits")
    before_indexed = before_features.set_index("record_id").sort_index()
    after_indexed = after_features.set_index("record_id").sort_index()
    if not before_indexed.index.equals(after_indexed.index):
        raise ValueError("The two audits must contain exactly the same record IDs")
    for field in ("label", "label_name", "split", "broad_area"):
        if not before_indexed[field].equals(after_indexed[field]):
            raise ValueError(f"The two audits differ in protected metadata: {field}")

    before_gaps = pd.read_csv(before_dir / "tables" / "shortcut_gaps.csv")
    after_gaps = pd.read_csv(after_dir / "tables" / "shortcut_gaps.csv")
    merged = before_gaps.merge(
        after_gaps,
        on=["broad_area", "feature", "feature_type"],
        suffixes=("_before", "_after"), validate="one_to_one",
    )
    if len(merged) != len(before_gaps) or len(merged) != len(after_gaps):
        raise ValueError("The two audits have incompatible feature gap tables")
    numeric = merged["feature_type"] == "numeric"
    merged["absolute_gap_before"] = np.where(
        numeric, merged["standardized_mean_difference_before"].abs(),
        merged["signed_gap_before"].abs(),
    )
    merged["absolute_gap_after"] = np.where(
        numeric, merged["standardized_mean_difference_after"].abs(),
        merged["signed_gap_after"].abs(),
    )
    merged["absolute_gap_reduction"] = (
        merged["absolute_gap_before"] - merged["absolute_gap_after"]
    )

    paired_changes = []
    for feature in (*NUMERIC_FEATURES, *BOOLEAN_FEATURES):
        first = before_indexed[feature].astype(float)
        second = after_indexed[feature].astype(float)
        paired_changes.append(
            {"feature": feature, "rows": len(first),
             "changed_rows": int((first != second).sum()),
             "changed_rate": float((first != second).mean()),
             "median_absolute_change": float((first - second).abs().median())}
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "shortcut_gap_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_changes).to_csv(
        output_dir / "paired_feature_changes.csv", index=False, encoding="utf-8-sig"
    )

    before_probe = before_dir / "tables" / "shortcut_probes.csv"
    after_probe = after_dir / "tables" / "shortcut_probes.csv"
    if before_probe.is_file() and after_probe.is_file():
        first = pd.read_csv(before_probe)
        second = pd.read_csv(after_probe)
        compared = first.merge(second, on="probe", suffixes=("_before", "_after"), validate="one_to_one")
        for field in ("train_rows", "validation_rows", "test_rows"):
            if not (compared[f"{field}_before"] == compared[f"{field}_after"]).all():
                raise ValueError(f"Probe samples differ in {field}; scores are not paired")
        compared["roc_auc_delta"] = compared["roc_auc_after"] - compared["roc_auc_before"]
        compared["balanced_accuracy_delta"] = (
            compared["balanced_accuracy_after"] - compared["balanced_accuracy_before"]
        )
        compared.to_csv(output_dir / "probe_comparison.csv", index=False, encoding="utf-8-sig")

    overall = merged.loc[merged["broad_area"] == "ALL"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, feature_type, scale, xlabel in (
        (axes[0], "numeric", 1.0, "|SMD| entre classes"),
        (axes[1], "boolean", 100.0, "Diferença absoluta, pontos percentuais"),
    ):
        view = overall.loc[overall["feature_type"] == feature_type].copy()
        view = view.nlargest(12, "absolute_gap_before").sort_values("absolute_gap_before")
        positions = np.arange(len(view))
        ax.barh(positions - 0.19, scale * view["absolute_gap_before"], height=0.38,
                label="original", color="#286090")
        ax.barh(positions + 0.19, scale * view["absolute_gap_after"], height=0.38,
                label="normalizado", color="#c65943")
        ax.set_yticks(positions, view["feature"])
        ax.set_xlabel(xlabel)
        ax.legend(frameon=False)
    fig.suptitle("Mudança de sinais superficiais nos mesmos textos")
    fig.savefig(output_dir / "shortcut_gap_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "shortcut_gap_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    (output_dir / "README.md").write_text(
        "# Comparação pareada\n\n"
        f"Antes: `{before_dir}`\n\nDepois: `{after_dir}`\n\n"
        f"IDs, rótulos, splits e domínios iguais em {len(before_indexed):,} textos.\n\n"
        "SMD é usado para medidas numéricas; diferença de prevalência para marcas binárias. "
        "As escalas aparecem em painéis separados. Redução de separabilidade "
        "não demonstra por si só preservação semântica nem qualidade do benchmark. "
        "Inspecione também `paired_feature_changes.csv` e revise amostras de texto.\n",
        encoding="utf-8",
    )
    return output_dir
