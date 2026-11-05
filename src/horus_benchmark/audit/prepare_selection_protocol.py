"""Prepare provisional grouped IID splits and a paired layout-control view.

Read-only input; all outputs are derived. No model training, OOD fold or final
benchmark release is produced. AGENTS.md requires the researcher to run this
module on the cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from horus_benchmark.protocols import grouped_iid
from horus_benchmark.audit.selection_diagnostics import _check_view
from horus_benchmark.data.normalize_corpus import sha256_file
from horus_benchmark.protocols.grouped_iid import (
    SPLITS, add_strata, assign_groups, check_observed_pairs, split_diagnostics,
    stable_key,
)


LINEBREAKS = re.compile(r"\r\n?|\n")
SPACE_RUN = re.compile(r" {2,}")
DASH_LINE = re.compile(r"(?m)^[ \t]*[-–—][ \t]+\S")


def layout_control_text(content: str) -> str:
    """Replace line breaks with spaces; no words or punctuation are changed."""
    transformed = SPACE_RUN.sub(" ", LINEBREAKS.sub(" ", content)).strip()
    if transformed.split() != content.split():
        raise ValueError("Layout control altered the whitespace token sequence")
    return transformed


def _config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "horus-grouped-iid-layout-control-v1":
        raise ValueError("Wrong protocol experiment_id")
    if config.get("selection_scenario") != "w0.25_s20261103":
        raise ValueError("This protocol requires the reviewed N2 width-0.25 candidate")
    if not config.get("split_seeds") or len(set(config["split_seeds"])) != len(config["split_seeds"]):
        raise ValueError("Nonempty unique split_seeds required")
    return config


def _load_inputs(selection: Path, scenario: str, n2_corpus: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((selection / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "exploratory_candidates_no_final_split":
        raise ValueError("Selection run is not the expected candidate audit")
    if summary["n2_corpus_sha256"] != sha256_file(n2_corpus):
        raise ValueError("N2 corpus differs from the candidate audit input")
    membership = pd.read_parquet(selection / "candidate_membership.parquet")
    chosen = membership.loc[membership.scenario.eq(scenario), "record_id"]
    if chosen.empty or chosen.duplicated().any():
        raise ValueError("Candidate scenario missing or has duplicate IDs")
    expected = next((row["rows"] for row in summary["scenarios"]
                     if row["scenario"] == scenario), None)
    if expected != len(chosen):
        raise ValueError("Candidate membership count disagrees with summary")
    lineage = pd.read_parquet(selection / "lineage_manifest.parquet",
                              columns=["record_id", "selection_group_id", "probe_group_id"])
    if lineage.record_id.duplicated().any() or lineage.probe_group_id.isna().any():
        raise ValueError("Invalid lineage manifest")
    candidates = lineage.loc[lineage.record_id.isin(set(chosen))].copy()
    if len(candidates) != len(chosen):
        raise ValueError("Candidate IDs missing from lineage")
    pairs = pd.read_csv(selection / "near_pair_decisions.csv",
                        usecols=["left_record_id", "right_record_id", "decision"])
    return candidates, pairs, summary


def _read_n2(n2_corpus: Path, ids: set[str]) -> pd.DataFrame:
    columns = ["record_id", "label_name", "broad_area", "source_name", "generator",
               "genre", "content"]
    rows = []
    for batch in ds.dataset(str(n2_corpus), format="parquet").to_batches(
            columns=columns, batch_size=2048):
        part = batch.to_pandas()
        rows.append(part.loc[part.record_id.isin(ids)])
    frame = pd.concat(rows, ignore_index=True)
    if len(frame) != len(ids) or frame.record_id.duplicated().any() or set(frame.record_id) != ids:
        raise ValueError("Selected IDs do not exactly match N2 corpus")
    if frame.content.isna().any() or not frame.label_name.isin(["human", "synthetic"]).all():
        raise ValueError("Missing text or invalid labels")
    frame["word_count_audit"] = frame.content.map(lambda value: len(value.split()))
    return frame


def _read_sample_contents(path: Path, ids: set[str]) -> dict[str, str]:
    result = {}
    for batch in ds.dataset(str(path), format="parquet").to_batches(
            columns=["record_id", "content"], batch_size=2048):
        for row in batch.to_pylist():
            if row["record_id"] in ids:
                result[row["record_id"]] = row["content"]
    if set(result) != ids:
        raise ValueError("Format sample IDs missing from source")
    return result


def _read_source_urls(raw: Path, ids: set[str]) -> dict[str, str]:
    result = {}
    for batch in ds.dataset(str(raw), format="parquet").to_batches(
            columns=["record_id", "source_url"], batch_size=8192):
        for row in batch.to_pylist():
            if row["record_id"] in ids:
                result[row["record_id"]] = row["source_url"] or ""
    if set(result) != ids:
        raise ValueError("Format sample source URLs missing")
    return result


def _format_sample(frame: pd.DataFrame, loss: pd.DataFrame,
                   config: dict) -> pd.DataFrame:
    """Directed, deterministic sample. Reasons may overlap; preserve each."""
    frame = frame.copy()
    frame["paragraphs"] = frame.content.map(
        lambda value: len(re.split(r"\n\s*\n+", value.strip())) if value.strip() else 0)
    frame["dash_lines"] = frame.content.map(lambda value: len(DASH_LINE.findall(value)))
    selected = []
    seed = int(config["sample_seed"])
    per_provenance = int(config["format_examples_per_provenance"])
    max_provenances = int(config["max_provenances_per_cell"])
    categories = (
        ("human_flat", "human", {"journalistic", "juridical", "literary"},
         lambda data: data.paragraphs.eq(1), "source_name"),
        ("synthetic_structured", "synthetic", {"journalistic", "juridical", "literary"},
         lambda data: data.paragraphs.ge(5), "generator"),
    )
    for reason, label, areas, predicate, provenance in categories:
        subset = frame.loc[frame.label_name.eq(label) & frame.broad_area.isin(areas)].copy()
        subset = subset.loc[predicate(subset)]
        for area, area_rows in subset.groupby("broad_area", sort=True):
            top = area_rows[provenance].fillna("<missing>").value_counts().head(max_provenances).index
            for value in top:
                pool = area_rows.loc[area_rows[provenance].fillna("<missing>").eq(value)].copy()
                pool["sample_key"] = pool.record_id.map(lambda rid: stable_key(seed, rid))
                for rid in pool.sort_values("sample_key").head(per_provenance).record_id:
                    selected.append((rid, reason))
    if not loss.empty:
        literary = frame.loc[frame.label_name.eq("synthetic") & frame.broad_area.eq("literary")]
        affected = literary.merge(loss[["record_id", "lost_dash_lines"]], on="record_id")
        for generator, group in affected.groupby("generator", sort=True):
            group = group.copy()
            group["sample_key"] = group.record_id.map(lambda rid: stable_key(seed, rid))
            for rid in group.sort_values("sample_key").head(per_provenance).record_id:
                selected.append((rid, "literary_dash_loss"))
    reasons: dict[str, set[str]] = {}
    for rid, reason in selected:
        reasons.setdefault(rid, set()).add(reason)
    result = frame.loc[frame.record_id.isin(reasons),
                       ["record_id", "label_name", "broad_area", "source_name", "generator",
                        "genre", "paragraphs", "dash_lines"]].copy()
    result["sample_reason"] = result.record_id.map(lambda rid: ";".join(sorted(reasons[rid])))
    result["sample_key"] = result.record_id.map(lambda rid: stable_key(seed, rid))
    return result.sort_values("sample_key").drop(columns="sample_key").reset_index(drop=True)


def _write_format_sample(sample: pd.DataFrame, raw: Path, n2: pd.DataFrame,
                         n3: Path, output: Path) -> None:
    ids = set(sample.record_id)
    raw_text = _read_sample_contents(raw, ids)
    n3_text = _read_sample_contents(n3, ids)
    n2_text = n2.set_index("record_id").content.to_dict()
    urls = _read_source_urls(raw, ids)
    manifest, blind = [], []
    for index, row in enumerate(sample.itertuples(index=False)):
        rid = row.record_id
        sample_id = f"S{index:04d}"
        raw_value = raw_text[rid]
        n3_value = n3_text[rid]
        manifest.append({"sample_id": sample_id, "record_id": rid,
                         "label_name": row.label_name, "broad_area": row.broad_area,
                         "source_name": row.source_name, "generator": row.generator,
                         "genre": row.genre, "source_url": urls[rid],
                         "n2_paragraphs": row.paragraphs, "n2_dash_lines": row.dash_lines,
                         "raw_paragraphs": len(re.split(r"\n\s*\n+", raw_value.strip())) if raw_value.strip() else 0,
                         "n3s_paragraphs": len(re.split(r"\n\s*\n+", n3_value.strip())) if n3_value.strip() else 0,
                         "raw_dash_lines": len(DASH_LINE.findall(raw_value)),
                         "n3s_dash_lines": len(DASH_LINE.findall(n3_value)),
                         "sample_reason": row.sample_reason})
        blind.append({"sample_id": sample_id, "raw_content": raw_value,
                      "n2_content": n2_text[rid], "n3s_content": n3_value})
    pd.DataFrame(manifest).to_csv(output / "format_samples_manifest.csv", index=False)
    (output / "format_samples_blind.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind), encoding="utf-8")


def _provenance_table(split: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    frame = split[["record_id", "split"]].merge(
        selected[["record_id", "label_name", "broad_area", "source_name", "generator"]],
        on="record_id", validate="one_to_one")
    frame["provenance"] = frame.source_name.where(frame.label_name.eq("human"), frame.generator
                                                 ).fillna("<missing>")
    counts = frame.groupby(["split", "broad_area", "label_name", "provenance"],
                           dropna=False).size().rename("rows").reset_index()
    counts["within_split_area_label_share"] = counts.rows / counts.groupby(
        ["split", "broad_area", "label_name"]).rows.transform("sum")
    return counts


def run(config_path: Path, selection: Path, raw: Path, n2_run: Path,
        n3_run: Path, output: Path) -> Path:
    config = _config(config_path)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    n2_corpus, n2_ver = _check_view(n2_run, "n2")
    n3_corpus, n3_ver = _check_view(n3_run, "n3")
    candidates, pairs, selection_summary = _load_inputs(
        selection, config["selection_scenario"], n2_corpus)
    if n2_ver["source_sha256"] != n3_ver["source_sha256"] or (
            selection_summary["source_sha256"] != n2_ver["source_sha256"]):
        raise ValueError("N2, N3s and selection source hashes disagree")
    if sha256_file(raw) != n2_ver["source_sha256"]:
        raise ValueError("Raw source hash disagrees with selection")
    selected = _read_n2(n2_corpus, set(candidates.record_id)).merge(
        candidates, on="record_id", validate="one_to_one")
    selected = add_strata(selected, list(config["coarse_word_edges"]))
    output.mkdir(parents=True)
    print(f"Selected N2 documents: {len(selected)}", flush=True)
    fractions = {part: float(config[f"{part}_fraction"]) for part in SPLITS}
    split_summaries = []
    for seed in config["split_seeds"]:
        split = assign_groups(selected, fractions, int(seed))
        pair_check = check_observed_pairs(split, pairs)
        strata, area = split_diagnostics(split, fractions)
        provenance = _provenance_table(split, selected)
        suffix = f"seed{seed}"
        split_path = output / f"iid_split_{suffix}.parquet"
        pq.write_table(pa.Table.from_pandas(split, preserve_index=False),
                       split_path, compression="zstd")
        strata.to_csv(output / f"strata_balance_{suffix}.csv", index=False)
        area.to_csv(output / f"area_label_balance_{suffix}.csv", index=False)
        provenance.to_csv(output / f"provenance_by_split_{suffix}.csv", index=False)
        split.groupby(["split", "probe_group_id"]).agg(
            rows=("record_id", "size"), areas=("broad_area", "nunique"),
            labels=("label_name", "nunique")
        ).reset_index().sort_values("rows", ascending=False).head(30).to_csv(
            output / f"largest_groups_{suffix}.csv", index=False)
        split_sizes = split.split.value_counts().to_dict()
        missing_area_label = (area.groupby(["broad_area", "label_name"]).split.nunique() < 3).sum()
        maximum_group = int(split.groupby("probe_group_id").size().max())
        summary = {"seed": int(seed), "rows": len(split), "split_sizes": split_sizes,
                   "split_manifest_sha256": sha256_file(split_path),
                   "missing_area_label_cells": int(missing_area_label),
                   "max_group_size": maximum_group,
                   "max_stratum_share_deviation": float(strata.share_minus_target.abs().max()),
                   **pair_check}
        split_summaries.append(summary)
        print(f"IID candidate seed {seed}: {split_sizes}; near-pair crossings=0", flush=True)
    control_rows, diagnostic_rows = [], []
    for row in selected.itertuples(index=False):
        control = layout_control_text(row.content)
        control_rows.append({"record_id": row.record_id, "content": control})
        diagnostic_rows.append({"record_id": row.record_id, "label_name": row.label_name,
                                "broad_area": row.broad_area, "n2_words": row.word_count_audit,
                                "control_words": len(control.split()),
                                "n2_linebreaks": row.content.count("\n"),
                                "control_linebreaks": control.count("\n"),
                                "n2_chars": len(row.content), "control_chars": len(control),
                                "text_changed": control != row.content})
    pq.write_table(pa.Table.from_pylist(control_rows), output / "layout_control_corpus.parquet",
                   compression="zstd")
    diagnostics = pd.DataFrame(diagnostic_rows)
    if diagnostics.control_words.ne(diagnostics.n2_words).any() or diagnostics.control_linebreaks.any():
        raise AssertionError("Layout view failed invariants")
    diagnostics.groupby(["label_name", "broad_area"]).agg(
        rows=("record_id", "size"), changed_rows=("text_changed", "sum"),
        n2_words=("n2_words", "sum"), control_words=("control_words", "sum"),
        removed_linebreaks=("n2_linebreaks", "sum"),
        n2_mean_chars=("n2_chars", "mean"), control_mean_chars=("control_chars", "mean"),
    ).reset_index().to_csv(output / "layout_control_summary.csv", index=False)
    trace = pd.read_parquet(n2_run / "normalization_trace.parquet",
                            columns=["record_id", "normalization_flags"])
    trace = trace.loc[trace.record_id.isin(set(selected.record_id))]
    code_examples = int(trace.normalization_flags.fillna("").str.contains(
        r"(?:^|;)CODE_EXAMPLE(?:;|$)", regex=True).sum())
    loss = pd.read_csv(selection / "dialogue_marker_loss.csv")
    sample = _format_sample(selected, loss, config)
    _write_format_sample(sample, raw, selected, n3_corpus, output)
    summary = {"experiment_id": config["experiment_id"], "status": "provisional_iid_candidates",
               "selection_scenario": config["selection_scenario"],
               "selected_rows": len(selected), "source_sha256": n2_ver["source_sha256"],
               "selection_summary_sha256": sha256_file(selection / "summary.json"),
               "candidate_membership_sha256": sha256_file(selection / "candidate_membership.parquet"),
               "lineage_sha256": sha256_file(selection / "lineage_manifest.parquet"),
               "n2_corpus_sha256": sha256_file(n2_corpus),
               "layout_control_sha256": sha256_file(output / "layout_control_corpus.parquet"),
               "config_sha256": sha256_file(config_path), "code_sha256": sha256_file(Path(__file__)),
               "grouping_code_sha256": sha256_file(Path(grouped_iid.__file__)),
               "python": sys.version, "format_samples": len(sample),
               "n2_code_example_flagged_rows_in_selection": code_examples,
               "split_candidates": split_summaries,
               "limitations": ["No parent-document IDs; observed near-pair check is not exhaustive.",
                               "IID groups are derived from existing split metadata and observed near pairs.",
                               "Layout control can change the readability of code, lists and poetry; diagnostic only.",
                               "Raw corpus preserves the collected text, not the source site's original HTML layout.",
                               "No detector training or OOD evaluation has been run."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    (output / "README.md").write_text(
        "# Protocolo IID agrupado e controle de layout — candidatos\n\n"
        "A seleção N2 `w0.25_s20261103` fornece os mesmos IDs para as duas condições. "
        "`iid_split_seed*.parquet` mantém todos os pares aproximados observados na mesma partição. "
        "`layout_control_corpus.parquet` troca quebras de linha por espaços em ambas as classes; "
        "a sequência de tokens separados por whitespace é idêntica à N2. "
        "A N2 original continua sendo a condição principal.\n\n"
        "`strata_balance_*.csv`, `area_label_balance_*.csv`, `provenance_by_split_*.csv` "
        "e `largest_groups_*.csv` "
        "permitem avaliar a composição antes de congelar os splits. "
        "`format_samples_manifest.csv` identifica amostras por fonte, área e motivo; "
        "`format_samples_blind.jsonl` contém os textos bruto/N2/N3s e fica fora do Git. "
        "O layout original dos sites não está nos Parquets: a causa do parágrafo único "
        "não pode ser concluída apenas com este material.\n\n"
        "Estes splits são candidatos IID. Não há folds OOD, treinamento ou release final.\n",
        encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "selection", "raw", "n2-run", "n3-run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.selection, args.raw, args.n2_run, args.n3_run, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
