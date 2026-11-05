"""Compare provenance-aware IID candidates with the unchanged v2 baseline.

This is a derived, read-only audit. It neither trains detectors nor freezes a
test split. Run on the researcher's cluster per AGENTS.md section 7.19.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from horus_benchmark.audit.prepare_selection_protocol import (
    _load_inputs, _read_n2, _read_sample_contents, _read_source_urls,
)
from horus_benchmark.audit.selection_diagnostics import _check_view
from horus_benchmark.data.normalize_corpus import sha256_file
from horus_benchmark.protocols import grouped_iid
from horus_benchmark.protocols.grouped_iid import (
    add_strata, assign_heldout_cv, check_observed_pairs, split_diagnostics,
    stable_key,
)


def _config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "horus-source-balanced-iid-v3":
        raise ValueError("Wrong v3 experiment_id")
    if not 0 < float(config["test_fraction"]) < 1 or int(config["cv_folds"]) != 5:
        raise ValueError("Expected a held-out test and five development folds")
    if float(config["provenance_weight"]) <= 0 or int(config["material_min_rows"]) < 1:
        raise ValueError("Positive soft provenance weight and material threshold required")
    if not config["split_seeds"] or len(set(config["split_seeds"])) != len(config["split_seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    return config


def provenance_shift(report: pd.DataFrame, minimum: int) -> pd.DataFrame:
    """Compare each partition's provenance share to its own full population."""
    keys = ["broad_area", "label_name", "provenance"]
    base = report.groupby(keys, dropna=False).rows.sum().rename("total_rows").reset_index()
    base["provenance_count"] = base.groupby(["broad_area", "label_name"]).provenance.transform("size")
    base["area_label_total"] = base.groupby(["broad_area", "label_name"]).total_rows.transform("sum")
    base["overall_share"] = base.total_rows / base.area_label_total
    partitions = sorted(report.split.unique())
    grid = pd.concat([base.assign(split=part) for part in partitions], ignore_index=True)
    counts = report.groupby(["split", *keys], dropna=False).rows.sum().rename("partition_rows").reset_index()
    grid = grid.merge(counts, on=["split", *keys], how="left", validate="one_to_one")
    grid["partition_rows"] = grid.partition_rows.fillna(0).astype(int)
    grid["partition_total"] = grid.groupby(["split", "broad_area", "label_name"]
                                            ).partition_rows.transform("sum")
    if grid.partition_total.eq(0).any():
        raise ValueError("An area/class cell is absent from a partition")
    grid["partition_share"] = grid.partition_rows / grid.partition_total
    grid["shift_pp"] = 100 * (grid.partition_share - grid.overall_share)
    grid["material_diverse"] = grid.total_rows.ge(minimum) & grid.provenance_count.gt(1)
    return grid


def _metrics(run_dir: Path, seed: int, minimum: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    reports = {}
    shifts = {}
    for scope, prefix in (("holdout", "holdout"), ("folds", "dev_fold")):
        strata = pd.read_csv(run_dir / f"{prefix}_strata_balance_seed{seed}.csv")
        provenance = pd.read_csv(run_dir / f"{prefix}_provenance_seed{seed}.csv")
        report = provenance_shift(provenance, minimum)
        material = report.loc[report.material_diverse]
        reports[f"{scope}_max_stratum_n100_pp"] = float(
            100 * strata.loc[strata.total_rows.ge(100), "share_minus_target"].abs().max())
        reports[f"{scope}_max_material_provenance_shift_pp"] = float(material.shift_pp.abs().max())
        reports[f"{scope}_material_provenances_missing"] = int(material.partition_rows.eq(0).sum())
        shifts[scope] = report.assign(scope=scope)
    return reports, shifts["holdout"], shifts["folds"]


def _provenance_table(split: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    joined = split[["record_id", "split"]].merge(
        selected[["record_id", "label_name", "broad_area", "source_name", "generator"]],
        on="record_id", validate="one_to_one")
    joined["provenance"] = joined.source_name.where(
        joined.label_name.eq("human"), joined.generator).fillna("<missing>")
    counts = joined.groupby(["split", "broad_area", "label_name", "provenance"],
                            dropna=False).size().rename("rows").reset_index()
    counts["within_split_area_label_share"] = counts.rows / counts.groupby(
        ["split", "broad_area", "label_name"]).rows.transform("sum")
    return counts


def _write_large_group_audit(selected: pd.DataFrame, raw: Path, output: Path,
                             seed: int, examples: int) -> dict:
    counts = selected.groupby("probe_group_id").size().sort_values(ascending=False)
    top = counts.head(5)
    rows = []
    for group_id, n in top.items():
        group = selected.loc[selected.probe_group_id.eq(group_id)]
        rows.append({"probe_group_id": group_id, "selected_rows": int(n),
                     "labels": ";".join(sorted(group.label_name.astype(str).unique())),
                     "areas": ";".join(sorted(group.broad_area.astype(str).unique())),
                     "sources": ";".join(sorted(group.source_name.fillna("<missing>").astype(str).unique())),
                     "generators": ";".join(sorted(group.generator.fillna("<missing>").astype(str).unique())),
                     "genres": ";".join(sorted(group.genre.fillna("<missing>").astype(str).unique())),
                     "selection_groups": int(group.selection_group_id.nunique()),
                     "distinct_n2_texts": int(group.content.nunique()),
                     "median_words": float(group.word_count_audit.median()),
                     "min_words": int(group.word_count_audit.min()),
                     "max_words": int(group.word_count_audit.max())})
    pd.DataFrame(rows).to_csv(output / "largest_group_summary.csv", index=False)
    largest = selected.loc[selected.probe_group_id.eq(top.index[0])].copy()
    largest["sample_key"] = largest.record_id.map(lambda rid: stable_key(seed, rid))
    sample = largest.sort_values("sample_key").head(examples)
    ids = set(sample.record_id)
    urls = _read_source_urls(raw, ids)
    manifest, blind = [], []
    for index, row in enumerate(sample.itertuples(index=False)):
        sample_id = f"G{index:03d}"
        manifest.append({"sample_id": sample_id, "record_id": row.record_id,
                         "probe_group_id": row.probe_group_id,
                         "selection_group_id": row.selection_group_id,
                         "label_name": row.label_name, "broad_area": row.broad_area,
                         "source_name": row.source_name, "generator": row.generator,
                         "genre": row.genre, "source_url": urls[row.record_id],
                         "word_count_audit": row.word_count_audit})
        blind.append({"sample_id": sample_id, "n2_content": row.content})
    pd.DataFrame(manifest).to_csv(output / "largest_group_examples_manifest.csv", index=False)
    (output / "largest_group_examples_blind.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind), encoding="utf-8")
    return rows[0]


def _write_literary_audit(selected: pd.DataFrame, selection: Path, n3_corpus: Path,
                          output: Path, seed: int, maximum: int) -> int:
    loss = pd.read_csv(selection / "dialogue_marker_loss.csv",
                       usecols=["record_id", "lost_dash_lines"])
    subset = selected.loc[selected.label_name.eq("synthetic") &
                          selected.broad_area.eq("literary")].merge(
                              loss, on="record_id", how="inner", validate="one_to_one")
    summary = subset.groupby(["generator", "genre"], dropna=False).agg(
        affected_rows=("record_id", "size"), lost_dash_lines=("lost_dash_lines", "sum")
    ).reset_index()
    summary.to_csv(output / "literary_dash_loss_by_genre_generator.csv", index=False)
    subset["sample_key"] = subset.record_id.map(lambda rid: stable_key(seed, rid))
    sample = subset.sort_values("sample_key").groupby(
        ["generator", "genre"], dropna=False).head(2).sort_values("sample_key").head(maximum)
    ids = set(sample.record_id)
    n3 = _read_sample_contents(n3_corpus, ids)
    manifest, blind = [], []
    for index, row in enumerate(sample.itertuples(index=False)):
        sample_id = f"L{index:03d}"
        manifest.append({"sample_id": sample_id, "record_id": row.record_id,
                         "generator": row.generator, "genre": row.genre,
                         "lost_dash_lines": row.lost_dash_lines})
        blind.append({"sample_id": sample_id, "n2_content": row.content,
                      "n3s_content": n3[row.record_id]})
    pd.DataFrame(manifest).to_csv(output / "literary_dash_examples_manifest.csv", index=False)
    (output / "literary_dash_examples_blind.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind), encoding="utf-8")
    return len(sample)


def run(config_path: Path, baseline_config_path: Path, selection: Path,
        baseline: Path, raw: Path,
        n2_run: Path, n3_run: Path, output: Path) -> Path:
    config = _config(config_path)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    n2_corpus, n2_ver = _check_view(n2_run, "n2")
    n3_corpus, n3_ver = _check_view(n3_run, "n3")
    candidates, pairs, selection_summary = _load_inputs(
        selection, config["selection_scenario"], n2_corpus)
    prior = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    baseline_config = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    if (prior.get("status") != "provisional_iid_holdout_cv" or
            prior["selected_rows"] != len(candidates) or
            prior["selection_scenario"] != config["selection_scenario"] or
            prior["selection_summary_sha256"] != sha256_file(selection / "summary.json") or
            prior["source_sha256"] != n2_ver["source_sha256"] or
            n3_ver["source_sha256"] != n2_ver["source_sha256"] or
            sha256_file(raw) != n2_ver["source_sha256"]):
        raise ValueError("Baseline, selection, N2, N3s and raw inputs are not paired")
    if (float(prior["test_fraction"]) != float(config["test_fraction"]) or
            int(prior["development_folds"]) != int(config["cv_folds"]) or
            baseline_config["coarse_word_edges"] != config["coarse_word_edges"] or
            sha256_file(baseline_config_path) != prior["config_sha256"]):
        raise ValueError("Baseline uses different holdout/fold proportions")
    if set(config["split_seeds"]) != {item["seed"] for item in prior["split_candidates"]}:
        raise ValueError("V2 and V3 seeds differ")
    selected = _read_n2(n2_corpus, set(candidates.record_id)).merge(
        candidates, on="record_id", validate="one_to_one")
    selected = add_strata(selected, list(config["coarse_word_edges"]))
    output.mkdir(parents=True)
    comparison = []
    summaries = []
    test_fraction = float(config["test_fraction"])
    folds = int(config["cv_folds"])
    minimum = int(config["material_min_rows"])
    for seed in config["split_seeds"]:
        split = assign_heldout_cv(selected, test_fraction, folds, int(seed),
                                   float(config["provenance_weight"]), minimum)
        pair_check = check_observed_pairs(split, pairs)
        outer_strata, outer_area = split_diagnostics(
            split, {"development": 1 - test_fraction, "test": test_fraction})
        outer_provenance = _provenance_table(split, selected)
        development = split.loc[split.split.eq("development")].copy()
        development["split"] = development.fold.map(lambda value: f"fold_{value}")
        fold_strata, fold_area = split_diagnostics(
            development, {f"fold_{value}": 1 / folds for value in range(folds)})
        fold_provenance = _provenance_table(development, selected)
        suffix = f"seed{seed}"
        split_path = output / f"iid_holdout_{folds}fold_{suffix}.parquet"
        pq.write_table(pa.Table.from_pandas(split, preserve_index=False), split_path,
                       compression="zstd")
        for name, table in (("holdout_strata_balance", outer_strata),
                            ("holdout_area_label_balance", outer_area),
                            ("holdout_provenance", outer_provenance),
                            ("dev_fold_strata_balance", fold_strata),
                            ("dev_fold_area_label_balance", fold_area),
                            ("dev_fold_provenance", fold_provenance)):
            table.to_csv(output / f"{name}_{suffix}.csv", index=False)
        split.groupby(["split", "fold", "probe_group_id"]).agg(
            rows=("record_id", "size"), areas=("broad_area", "nunique"),
            labels=("label_name", "nunique")
        ).reset_index().sort_values("rows", ascending=False).head(30).to_csv(
            output / f"largest_groups_{suffix}.csv", index=False)
        role_rows = []
        for fold in range(folds):
            roles = split[["broad_area", "label_name", "split", "fold"]].copy()
            roles["role"] = roles.split.where(roles.split.eq("test"), "train")
            roles.loc[roles.split.eq("development") & roles.fold.eq(fold), "role"] = "validation"
            role_rows.extend({"fold": fold, "broad_area": area, "label_name": label,
                              "role": role, "rows": int(count)}
                             for (area, label, role), count in roles.groupby(
                                 ["broad_area", "label_name", "role"]).size().items())
        pd.DataFrame(role_rows).to_csv(output / f"cv_round_roles_{suffix}.csv", index=False)
        source_shifts = provenance_shift(outer_provenance, minimum).assign(scope="holdout")
        fold_shifts = provenance_shift(fold_provenance, minimum).assign(scope="folds")
        pd.concat([source_shifts, fold_shifts], ignore_index=True).to_csv(
            output / f"provenance_shift_{suffix}.csv", index=False)
        for version, run_dir in (("v2", baseline), ("v3", output)):
            metrics, _, _ = _metrics(run_dir, int(seed), minimum)
            comparison.append({"version": version, "seed": seed, **metrics,
                               "test_rows": int(split.split.eq("test").sum()) if version == "v3" else
                               next(item["split_sizes"]["test"] for item in prior["split_candidates"]
                                    if item["seed"] == seed)})
        material_test = source_shifts.loc[source_shifts.material_diverse]
        material_folds = fold_shifts.loc[fold_shifts.material_diverse]
        summary = {"seed": int(seed), "rows": len(split),
                   "test_rows": int(split.split.eq("test").sum()),
                   "development_fold_sizes": {str(k): int(v) for k, v in split.loc[
                       split.split.eq("development")].fold.value_counts().sort_index().items()},
                   "split_manifest_sha256": sha256_file(split_path),
                   "max_test_material_shift_pp": float(material_test.shift_pp.abs().max()),
                   "max_fold_material_shift_pp": float(material_folds.shift_pp.abs().max()),
                   "material_provenances_missing_test": int(material_test.partition_rows.eq(0).sum()),
                   "material_provenances_missing_folds": int(material_folds.partition_rows.eq(0).sum()),
                   "max_test_stratum_n100_pp": float(100 * outer_strata.loc[
                       outer_strata.total_rows.ge(100), "share_minus_target"].abs().max()),
                   "max_fold_stratum_n100_pp": float(100 * fold_strata.loc[
                       fold_strata.total_rows.ge(100), "share_minus_target"].abs().max()),
                   **pair_check}
        summary["exploratory_gates_met"] = bool(
            summary["max_test_material_shift_pp"] <= float(config["max_test_source_shift_pp"]) and
            summary["max_fold_material_shift_pp"] <= float(config["max_fold_source_shift_pp"]) and
            summary["max_test_stratum_n100_pp"] <= float(config["max_stratum_n100_shift_pp"]) and
            summary["max_fold_stratum_n100_pp"] <= float(config["max_stratum_n100_shift_pp"]) and
            summary["material_provenances_missing_test"] == 0 and
            summary["material_provenances_missing_folds"] == 0 and
            pair_check["observed_pairs_cross_split"] == 0)
        summaries.append(summary)
        print(f"Seed {seed}: test={summary['test_rows']}, "
              f"source shift={summary['max_test_material_shift_pp']:.2f}pp, "
              f"gates={summary['exploratory_gates_met']}", flush=True)
    pd.DataFrame(comparison).to_csv(output / "v2_v3_split_comparison.csv", index=False)
    largest = _write_large_group_audit(selected, raw, output,
                                        int(config["inspection_seed"]),
                                        int(config["largest_group_examples"]))
    literary_count = _write_literary_audit(selected, selection, n3_corpus, output,
                                           int(config["inspection_seed"]),
                                           int(config["literary_examples_max"]))
    summary = {"experiment_id": config["experiment_id"],
               "status": "provisional_source_balanced_iid_candidates",
               "selection_scenario": config["selection_scenario"], "rows": len(selected),
               "source_sha256": n2_ver["source_sha256"],
               "selection_summary_sha256": sha256_file(selection / "summary.json"),
               "baseline_summary_sha256": sha256_file(baseline / "summary.json"),
               "baseline_config_sha256": sha256_file(baseline_config_path),
               "config_sha256": sha256_file(config_path),
               "code_sha256": sha256_file(Path(__file__)),
               "grouping_code_sha256": sha256_file(Path(grouped_iid.__file__)),
               "python": sys.version, "provenance_weight": config["provenance_weight"],
               "material_min_rows": minimum, "test_fraction": test_fraction,
               "development_folds": folds, "split_candidates": summaries,
               "largest_group": largest, "literary_examples": literary_count,
               "layout_control_reused_from_v2": str(baseline / "layout_control_corpus.parquet"),
               "limitations": ["Source names are preserved as supplied; aliases are not merged.",
                               "Single-source areas provide no source balancing signal.",
                               "All observed near pairs remain grouped; this is not exhaustive linkage.",
                               "No texts are excluded or normalized by this refinement.",
                               "No detector training or OOD evaluation has been run."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    (output / "README.md").write_text(
        "# Fonte/gerador no IID — comparação exploratória v2 × v3\n\n"
        "Os mesmos IDs, grupos e sementes são usados na v2 e v3. A v3 acrescenta "
        "um termo suave no objetivo de alocação, dentro de área e classe, para "
        "fontes humanas e geradores sintéticos com ao menos 100 textos. Fontes "
        "únicas não fornecem contraste. Nenhum texto é removido.\n\n"
        "`v2_v3_split_comparison.csv` compara desvios de proveniência e comprimento. "
        "`provenance_shift_*.csv` contém cada fonte/gerador e o desvio de participação "
        "em relação ao conjunto correspondente. `iid_holdout_5fold_*.parquet` contém "
        "os novos splits candidatos. `largest_group_summary.csv` e "
        "`literary_dash_loss_by_genre_generator.csv` resumem as inspeções. "
        "Os arquivos `*_blind.jsonl` contêm textos e não entram no Git. "
        "A visão de controle de layout da v2 é reutilizada, sem recomputação.\n\n"
        "Os gates do resumo são exploratórios: não há aprovação automática, "
        "freeze, treinamento ou OOD.\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "baseline-config", "selection", "baseline", "raw",
                 "n2-run", "n3-run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.baseline_config, args.selection, args.baseline, args.raw,
              args.n2_run, args.n3_run, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
