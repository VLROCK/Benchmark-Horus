"""Small descriptive figures for an already completed normalization run.

These are paired corpus summaries, not selection rules or detector scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MEASURES = (
    ("char_count_audit", "Caracteres médios por texto"),
    ("punctuation_per_100_words", "Pontuação por 100 palavras"),
    ("markdown_bullet", "Parcela com marcador de lista"),
    ("html_tag", "Parcela com tag HTML"),
)
AREA_NAMES = {
    "academic": "Acadêmica", "informational": "Informacional",
    "journalistic": "Jornalística", "juridical": "Jurídica",
    "literary": "Literária", "social": "Social",
}


def build_report(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["output_rows"] != summary["source_rows"]:
        raise ValueError("Refusing figures for a partial run")
    with (run_dir / "paired_surface_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if sum(int(row["rows"]) for row in rows) != summary["source_rows"]:
        raise ValueError("Group totals do not reconcile with corpus rows")
    figures = run_dir / "figures"
    tables = run_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    delta_columns = ["label_name", "broad_area", "rows", "measure", "before", "after", "delta"]
    with (tables / "paired_surface_delta.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=delta_columns)
        writer.writeheader()
        for row in rows:
            for measure, _ in MEASURES:
                before = float(row[f"before_{measure}"])
                after = float(row[f"after_{measure}"])
                writer.writerow({
                    "label_name": row["label_name"], "broad_area": row["broad_area"],
                    "rows": row["rows"], "measure": measure,
                    "before": before, "after": after, "delta": after - before,
                })
    by_cell = {(row["label_name"], row["broad_area"]): row for row in rows}
    gap_columns = ["broad_area", "measure", "human_before", "synthetic_before",
                   "human_after", "synthetic_after", "gap_before", "gap_after",
                   "absolute_gap_reduction"]
    with (tables / "class_gap_delta.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=gap_columns)
        writer.writeheader()
        for area in AREA_NAMES:
            if ("human", area) not in by_cell or ("synthetic", area) not in by_cell:
                continue
            human, synthetic = by_cell[("human", area)], by_cell[("synthetic", area)]
            for measure, _ in MEASURES:
                hb = float(human[f"before_{measure}"])
                sb = float(synthetic[f"before_{measure}"])
                ha = float(human[f"after_{measure}"])
                sa = float(synthetic[f"after_{measure}"])
                gap_before, gap_after = sb - hb, sa - ha
                writer.writerow({
                    "broad_area": area, "measure": measure,
                    "human_before": hb, "synthetic_before": sb,
                    "human_after": ha, "synthetic_after": sa,
                    "gap_before": gap_before, "gap_after": gap_after,
                    "absolute_gap_reduction": abs(gap_before) - abs(gap_after),
                })
    ordered_areas = [area for area in AREA_NAMES if any(row["broad_area"] == area for row in rows)]
    for measure, title in MEASURES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
        for ax, label in zip(axes, ("human", "synthetic")):
            selected = {row["broad_area"]: row for row in rows if row["label_name"] == label}
            positions = range(len(ordered_areas))
            before = [float(selected[a][f"before_{measure}"]) for a in ordered_areas]
            after = [float(selected[a][f"after_{measure}"]) for a in ordered_areas]
            ax.barh([n - 0.18 for n in positions], before, height=0.34, label="original", color="#286090")
            ax.barh([n + 0.18 for n in positions], after, height=0.34, label="normalizado", color="#c65943")
            ax.set_yticks(list(positions), [AREA_NAMES[a] for a in ordered_areas])
            ax.invert_yaxis()
            ax.set_title("Humano" if label == "human" else "Sintético")
            ax.grid(axis="x", alpha=0.2)
            if measure in {"markdown_bullet", "html_tag"}:
                ax.set_xlim(left=0, right=1)
        axes[1].legend(frameon=False)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(figures / f"{measure}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{measure}.svg", bbox_inches="tight")
        plt.close(fig)
    (run_dir / "REPORT_README.md").write_text(
        "# Relatório pareado da normalização\n\n"
        f"Fonte: `{summary['source']}`; SHA-256 `{summary['source_sha256']}`.\n\n"
        f"Etapa: `{summary['stage']}`; mesmos {summary['source_rows']} registros, sem seleção.\n\n"
        "Os gráficos mostram médias/prevalências agregadas por rótulo e área. "
        "A queda de uma pista superficial não prova melhor qualidade; verificar "
        "conteúdo, procedência, alertas de revisão e controles de gênero. "
        "Marcadores de lista e HTML podem ser legítimos em alguns gêneros.\n",
        encoding="utf-8",
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build_report(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
