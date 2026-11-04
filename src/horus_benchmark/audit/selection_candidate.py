"""Quality-gated, lineage-aware *candidate* selection on N2, compared on N3s.

This command writes derived manifests and diagnostics, never a final benchmark or
new source corpus. Run on the cluster, as required by AGENTS.md section 7.19.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import sklearn
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from horus_benchmark.audit.selection_diagnostics import (
    NUMBERED_LIST, UNORDERED_LIST, _balance_rows, _check_paired_views, _check_view, _load_view,
    _probe_rows, _provenance_rows,
)
from horus_benchmark.audit.text_features import extract_text_features
from horus_benchmark.data.normalize_corpus import sha256_file


LABELS = ("human", "synthetic")
LINEBREAKS = re.compile(r"\n[ \t]*\n+")
PUNCTUATION_LINE = re.compile(r"(?m)^[ \t]*[-–—][ \t]+\S")


class UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {value: value for value in ids}
        self.size = {value: 1 for value in ids}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        if self.size[a] < self.size[b] or (self.size[a] == self.size[b] and a > b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


def _config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "n2-lineage-selection-candidates-v1":
        raise ValueError("Wrong candidate-selection experiment_id")
    widths = value.get("log2_bin_widths")
    seeds = value.get("seeds")
    if not isinstance(widths, list) or not widths or any(float(x) <= 0 for x in widths):
        raise ValueError("log2_bin_widths must contain positive widths")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique list")
    if not 0 <= float(value["min_global_jaccard"]) <= 1:
        raise ValueError("min_global_jaccard must be in [0, 1]")
    fractions = [float(value[f"probe_{part}_fraction"]) for part in ("train", "validation", "test")]
    if any(x <= 0 for x in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Probe fractions must be positive and sum to one")
    return value


def _flags(value: object) -> set[str]:
    return set(str(value).split(";")) - {"", "nan", "None"}


def _read_raw_metadata(raw_path: Path, ids: set[str]) -> pd.DataFrame:
    columns = ["record_id", "label_name", "source_record_id", "source_url",
               "duplicate_group_id", "split_group_id", "source_name", "broad_area"]
    schema = ds.dataset(str(raw_path), format="parquet").schema
    missing = set(columns) - set(schema.names)
    if missing:
        raise ValueError(f"Raw metadata fields missing: {sorted(missing)}")
    parts = []
    for batch in ds.dataset(str(raw_path), format="parquet").to_batches(columns=columns, batch_size=16384):
        parts.append(batch.to_pandas())
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != len(ids) or frame.record_id.duplicated().any() or set(frame.record_id) != ids:
        raise ValueError("Raw IDs do not exactly match normalized corpus")
    return frame


def _trace(path: Path, ids: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["record_id", "normalization_flags",
                                           "normalization_review_reasons"])
    if frame.record_id.duplicated().any() or set(frame.record_id) != ids:
        raise ValueError("Trace IDs do not exactly match corpus")
    return frame


def _group_values(frame: pd.DataFrame, uf: UnionFind, field: str, limit: int) -> tuple[int, int]:
    edges = skipped = 0
    usable = frame.loc[frame[field].notna() & frame[field].astype(str).str.strip().ne("")]
    for _, group in usable.groupby(["label_name", "source_name", field], dropna=False, sort=False):
        if len(group) > limit:
            skipped += len(group)
            continue
        ids = sorted(group.record_id.astype(str))
        for other in ids[1:]:
            edges += uf.union(ids[0], other)
    return edges, skipped


def _article_url(value: object) -> str:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return ""
    parsed = urlparse(value)
    segments = [part for part in parsed.path.split("/") if part]
    # Homepage, section and search URLs are not document identifiers.
    if parsed.query or len(segments) < 2 or any(part.lower() in {"search", "busca"} for part in segments):
        return ""
    return f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


def build_lineage(frame: pd.DataFrame, pairs: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Conservative known-link graph; unresolved pairs remain visible."""
    ids = set(frame.record_id)
    uf = UnionFind(list(ids))
    counts: Counter[str] = Counter()
    limit = int(config["max_metadata_group_size"])
    for field in ("split_group_id", "duplicate_group_id", "source_record_id"):
        field_limit = len(frame) if field == "split_group_id" else limit
        counts[f"{field}_unions"], counts[f"{field}_skipped_rows"] = _group_values(
            frame, uf, field, field_limit)
    with_urls = frame.assign(article_url=frame.source_url.map(_article_url))
    counts["article_url_unions"], counts["article_url_skipped_rows"] = _group_values(
        with_urls, uf, "article_url", limit)
    # Exact text equality can be grouped across source names. Cross-label
    # equality is a conflict, not evidence that one class is correct.
    conflicts: set[str] = set()
    for _, group in frame.groupby("content_sha256_audit", sort=False):
        ids_here = sorted(group.record_id.astype(str))
        if len(group.label_name.unique()) > 1:
            conflicts.update(ids_here)
            counts["cross_label_exact_groups"] += 1
        else:
            for other in ids_here[1:]:
                counts["exact_text_unions"] += uf.union(ids_here[0], other)
    pair_rows = []
    required = {"left_record_id", "right_record_id", "jaccard_5gram", "containment_5gram"}
    if required - set(pairs.columns):
        raise ValueError(f"Near-pair table missing {sorted(required - set(pairs.columns))}")
    metadata = frame.set_index("record_id")
    for pair in pairs.itertuples(index=False):
        left, right = str(pair.left_record_id), str(pair.right_record_id)
        if left not in ids or right not in ids:
            raise ValueError("Near-pair table has IDs outside this corpus")
        same_label = metadata.at[left, "label_name"] == metadata.at[right, "label_name"]
        jaccard = float(pair.jaccard_5gram)
        if not same_label:
            conflicts.update((left, right))
            decision = "cross_label_review"
        elif jaccard >= float(config["min_global_jaccard"]):
            counts["high_jaccard_unions"] += uf.union(left, right)
            decision = "grouped_high_jaccard"
        else:
            decision = "unresolved_containment_only"
        counts[decision] += 1
        pair_rows.append({"left_record_id": left, "right_record_id": right,
                          "jaccard_5gram": jaccard,
                          "containment_5gram": float(pair.containment_5gram),
                          "decision": decision})
    roots = {rid: uf.find(rid) for rid in ids}
    members: dict[str, list[str]] = defaultdict(list)
    for rid, root in roots.items():
        members[root].append(rid)
    root_ids = {root: hashlib.sha256("|".join(sorted(group)).encode()).hexdigest()[:24]
                for root, group in members.items()}
    group_ids = {rid: root_ids[root] for rid, root in roots.items()}
    # For exploratory probe partitions, all *observed* near pairs share a
    # partition, including containment-only pairs that are too weak to assert
    # document identity. This prevents those observed pairs crossing the probe
    # train/validation/test boundary without treating them as true duplicates.
    probe_uf = UnionFind(list(ids))
    first_by_group: dict[str, str] = {}
    for rid in sorted(ids):
        group_id = group_ids[rid]
        if group_id in first_by_group:
            probe_uf.union(first_by_group[group_id], rid)
        else:
            first_by_group[group_id] = rid
    for pair in pair_rows:
        probe_uf.union(pair["left_record_id"], pair["right_record_id"])
    probe_members: dict[str, list[str]] = defaultdict(list)
    for rid in ids:
        probe_members[probe_uf.find(rid)].append(rid)
    probe_root_ids = {root: hashlib.sha256("|".join(sorted(group)).encode()).hexdigest()[:24]
                      for root, group in probe_members.items()}
    probe_ids = {rid: probe_root_ids[probe_uf.find(rid)] for rid in ids}
    lineage = frame[["record_id", "label_name", "broad_area", "source_name", "generator",
                     "content_sha256_audit"]].copy()
    lineage["selection_group_id"] = lineage.record_id.map(group_ids)
    lineage["probe_group_id"] = lineage.record_id.map(probe_ids)
    lineage["group_size"] = lineage.record_id.map(lambda rid: len(members[roots[rid]]))
    lineage["cross_label_conflict"] = lineage.record_id.isin(conflicts)
    counts["groups"] = len(members)
    counts["multirow_groups"] = sum(len(v) > 1 for v in members.values())
    counts["probe_partition_groups"] = len(probe_members)
    counts["cross_label_conflict_rows"] = len(conflicts)
    return lineage, pd.DataFrame(pair_rows), dict(counts)


def quality_table(n2: pd.DataFrame, trace: pd.DataFrame, lineage: pd.DataFrame,
                  config: dict) -> pd.DataFrame:
    frame = n2[["record_id", "label_name", "broad_area", "word_count_audit"]].merge(
        trace, on="record_id", validate="one_to_one").merge(
        lineage[["record_id", "selection_group_id", "cross_label_conflict"]],
        on="record_id", validate="one_to_one")
    severe = set(config["hold_flags"])
    reasons = []
    for row in frame.itertuples(index=False):
        active = sorted(_flags(row.normalization_flags) & severe)
        if row.word_count_audit == 0:
            active.append("EMPTY_TEXT")
        if row.cross_label_conflict:
            active.append("CROSS_LABEL_CONFLICT")
        reasons.append(";".join(sorted(set(active))))
    frame["hold_reasons"] = reasons
    frame["quality_state"] = np.where(frame.hold_reasons.eq(""), "eligible", "hold_for_review")
    # Exact copies add no independent evidence. Keep one deterministic
    # representative per same-label hash; all rows remain in the manifest.
    hashes = lineage[["record_id", "content_sha256_audit"]]
    frame = frame.merge(hashes, on="record_id", validate="one_to_one")
    eligible_only = frame.loc[frame.quality_state.eq("eligible")]
    canonical = eligible_only.groupby(["label_name", "content_sha256_audit"]).record_id.transform("min")
    secondary_idx = eligible_only.index[eligible_only.record_id.ne(canonical)]
    frame.loc[secondary_idx, "quality_state"] = "redundant_exact"
    frame.loc[secondary_idx, "hold_reasons"] = "EXACT_DUPLICATE_SECONDARY"
    # A nonempty review flag not in hold_flags is intentionally diagnostic.
    return frame


def _sample_key(seed: int, rid: str) -> str:
    return hashlib.sha256(f"{seed}|{rid}".encode()).hexdigest()


def select_cell_balanced(frame: pd.DataFrame, width: float, seed: int) -> pd.DataFrame:
    eligible = frame.loc[frame.quality_state.eq("eligible")].copy()
    eligible["length_bin"] = np.floor(np.log2(1 + eligible.word_count_audit) / width).astype(int)
    eligible["sample_key"] = eligible.record_id.map(lambda rid: _sample_key(seed, rid))
    selected = []
    for _, cell in eligible.groupby(["broad_area", "length_bin"], sort=True):
        counts = cell.label_name.value_counts()
        n = min(int(counts.get(label, 0)) for label in LABELS)
        if n == 0:
            continue
        for label in LABELS:
            picked = cell.loc[cell.label_name.eq(label)].sort_values(
                ["sample_key", "record_id"], kind="stable").head(n)
            selected.append(picked)
    if not selected:
        raise ValueError("No area-length cells have both classes")
    return pd.concat(selected, ignore_index=True).drop(columns="sample_key")


def _coverage(selected: pd.DataFrame, quality: pd.DataFrame, n2: pd.DataFrame,
              config: dict, scenario: str) -> pd.DataFrame:
    base = n2[["record_id", "label_name", "broad_area", "source_name", "generator"]].merge(
        quality[["record_id", "quality_state"]], on="record_id", validate="one_to_one")
    base = base.loc[base.quality_state.eq("eligible")].copy()
    chosen = base.loc[base.record_id.isin(set(selected.record_id))]
    rows = []
    for label, column in (("human", "source_name"), ("synthetic", "generator")):
        subset = base.loc[base.label_name.eq(label)].copy()
        subset["provenance"] = subset[column].fillna("<missing>")
        chosen_subset = chosen.loc[chosen.label_name.eq(label)].copy()
        chosen_subset["provenance"] = chosen_subset[column].fillna("<missing>")
        base_counts = subset.groupby(["broad_area", "provenance"], dropna=False).size()
        chosen_counts = chosen_subset.groupby(["broad_area", "provenance"], dropna=False).size()
        for (area, provenance), count in base_counts.items():
            retained = int(chosen_counts.get((area, provenance), 0))
            material = int(count) >= int(config["material_provenance_min_rows"])
            target = int(config["min_retained_per_material_provenance"])
            gate = (not material or retained >= min(target, int(count)))
            rows.append({"scenario": scenario, "broad_area": area, "label_name": label,
                         "provenance_field": column, "provenance": provenance,
                         "eligible_rows": int(count), "selected_rows": retained,
                         "retention": retained / int(count), "material": material,
                         "coverage_gate_met": gate})
    return pd.DataFrame(rows)


def _raw_format(raw_path: Path, n2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ["record_id", "content"]
    for batch in ds.dataset(str(raw_path), format="parquet").to_batches(columns=columns, batch_size=512):
        for row in batch.to_pylist():
            content = row["content"]
            features = extract_text_features(content if isinstance(content, str) else "")
            rows.append({"record_id": row["record_id"],
                         "paragraph_count": features["paragraph_count"],
                         "nonempty_line_count": features["nonempty_line_count"],
                         "newline_per_100_words": features["newline_per_100_words"],
                         "markdown_bullet": features["markdown_bullet"]})
    frame = pd.DataFrame(rows)
    if frame.record_id.duplicated().any() or set(frame.record_id) != set(n2.record_id):
        raise ValueError("Raw format features do not match N2 IDs")
    meta = n2[["record_id", "label_name", "broad_area", "source_name", "generator"]]
    return meta.merge(frame, on="record_id", validate="one_to_one")


def _dialogue_lines(corpus: Path) -> pd.DataFrame:
    rows = []
    for batch in ds.dataset(str(corpus), format="parquet").to_batches(
            columns=["record_id", "content"], batch_size=512):
        for row in batch.to_pylist():
            rows.append((row["record_id"], len(PUNCTUATION_LINE.findall(row["content"]))))
    return pd.DataFrame(rows, columns=["record_id", "dialogue_dash_lines"])


def _format_rows(raw: pd.DataFrame, n2: pd.DataFrame, n3: pd.DataFrame,
                 selected_ids: set[str]) -> pd.DataFrame:
    frames = []
    for view, frame in (("raw", raw), ("n2", n2), ("n3s", n3)):
        for scope, subset in (("full_corpus", frame),
                              ("selected", frame.loc[frame.record_id.isin(selected_ids)])):
            for label, field in (("human", "source_name"), ("synthetic", "generator")):
                local = subset.loc[subset.label_name.eq(label)].copy()
                local["provenance"] = local[field].fillna("<missing>")
                grouped = local.groupby(["broad_area", "provenance"], sort=True)
                rows = grouped.agg(rows=("record_id", "size"),
                                   mean_paragraphs=("paragraph_count", "mean"),
                                   median_paragraphs=("paragraph_count", "median"),
                                   mean_nonempty_lines=("nonempty_line_count", "mean"),
                                   mean_newlines_per_100_words=("newline_per_100_words", "mean"),
                                   markdown_bullet_rate=("markdown_bullet", "mean")).reset_index()
                rows.insert(0, "scope", scope)
                rows.insert(0, "view", view)
                rows.insert(3, "label_name", label)
                rows.insert(4, "provenance_field", field)
                frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def _plot_tradeoff(balance: pd.DataFrame, coverage: pd.DataFrame, output: Path) -> None:
    length = balance.loc[(balance.view == "n2") &
                         (balance.feature == "log_words") &
                         (balance.broad_area != "ALL")].copy()
    worst = length.groupby("scenario").smd_synthetic_minus_human.apply(
        lambda values: float(np.nanmax(np.abs(values)))).sort_index()
    retention = coverage.groupby("scenario").apply(
        lambda group: group.selected_rows.sum() / group.eligible_rows.sum(),
        include_groups=False).reindex(worst.index)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(worst.index, worst.values, color="#245f98")
    axes[0].set_ylabel("Maior |SMD| de log palavras por área")
    axes[1].bar(retention.index, retention.values, color="#cb7831")
    axes[1].set_ylabel("Retenção dos elegíveis")
    for axis in axes:
        axis.tick_params(axis="x", rotation=60)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Compromisso exploratório: comprimento e cobertura")
    fig.tight_layout()
    fig.savefig(output / "selection_tradeoff.png", dpi=170)
    plt.close(fig)


def _read_texts(corpus: Path, wanted: set[str]) -> dict[str, str]:
    result = {}
    for batch in ds.dataset(str(corpus), format="parquet").to_batches(
            columns=["record_id", "content"], batch_size=1024):
        for row in batch.to_pylist():
            if row["record_id"] in wanted:
                result[row["record_id"]] = row["content"]
    if set(result) != wanted:
        raise ValueError("Requested sample IDs are absent from corpus")
    return result


def _linebreak_features(n2_corpus: Path, selected: pd.DataFrame) -> pd.DataFrame:
    wanted = set(selected.record_id)
    text = _read_texts(n2_corpus, wanted)
    rows = []
    for row in selected.itertuples(index=False):
        content = text[row.record_id]
        collapsed = LINEBREAKS.sub("\n", content)
        features = extract_text_features(collapsed)
        rows.append({"record_id": row.record_id, **features,
                     "unordered_list_marker": bool(UNORDERED_LIST.search(collapsed)),
                     "numbered_list_marker": bool(NUMBERED_LIST.search(collapsed))})
    frame = pd.DataFrame(rows)
    frame["log_words"] = np.log2(frame.word_count_audit + 1)
    frame["log_chars"] = np.log2(frame.char_count_audit + 1)
    return frame


def _sample_format_examples(n2_corpus: Path, n3_corpus: Path, n2: pd.DataFrame,
                            n3: pd.DataFrame, config: dict, output: Path) -> None:
    """Local-only blind excerpts; manifest with labels is separate."""
    joined = n2[["record_id", "label_name", "broad_area", "source_name", "generator",
                 "paragraph_count", "nonempty_line_count", "markdown_bullet"]].copy()
    joined["n3_paragraph_count"] = n3.paragraph_count
    joined["n3_bullet"] = n3.markdown_bullet
    joined["changed_bullet"] = joined.markdown_bullet.ne(joined.n3_bullet)
    candidates = []
    for (label, area), group in joined.groupby(["label_name", "broad_area"], sort=True):
        for kind, subset in (("one_paragraph", group.loc[group.paragraph_count.eq(1)]),
                             ("many_paragraphs", group.loc[group.paragraph_count.ge(5)]),
                             ("n3_bullet_changed", group.loc[group.changed_bullet])):
            subset = subset.copy()
            subset["key"] = subset.record_id.map(lambda rid: _sample_key(int(config["sample_seed"]), rid))
            candidates.extend((kind, record) for record in subset.sort_values("key").head(
                int(config["format_examples_per_cell"])).itertuples(index=False))
    wanted = {record.record_id for _, record in candidates}
    n2_text = _read_texts(n2_corpus, wanted)
    n3_text = _read_texts(n3_corpus, wanted)
    blind, manifest = [], []
    for index, (kind, row) in enumerate(candidates):
        rid = row.record_id
        sample_id = f"F{index:04d}"
        blind.append({"sample_id": sample_id, "n2_content": n2_text[rid],
                      "n3s_content": n3_text[rid]})
        manifest.append({"sample_id": sample_id, "record_id": rid, "kind": kind,
                         "label_name": row.label_name, "broad_area": row.broad_area,
                         "source_name": row.source_name, "generator": row.generator,
                         "n2_dialogue_line_count": len(PUNCTUATION_LINE.findall(n2_text[rid])),
                         "n3_dialogue_line_count": len(PUNCTUATION_LINE.findall(n3_text[rid]))})
    (output / "format_examples_blind.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind), encoding="utf-8")
    pd.DataFrame(manifest).to_csv(output / "format_examples_manifest.csv", index=False)


def run(config_path: Path, raw_path: Path, n2_run: Path, n3_run: Path,
        near_pairs_path: Path, output: Path) -> Path:
    config = _config(config_path)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    n2_corpus, n2_verified = _check_view(n2_run, "n2")
    n3_corpus, n3_verified = _check_view(n3_run, "n3")
    raw_sha = sha256_file(raw_path)
    if raw_sha != n2_verified["source_sha256"] or raw_sha != n3_verified["source_sha256"]:
        raise ValueError("Raw, N2 and N3s source hashes disagree")
    audit_dir = near_pairs_path.parent.parent
    audited_inputs = json.loads((audit_dir / "input_files.json").read_text(encoding="utf-8"))
    literal = json.loads((audit_dir / "near_duplicate_literal_verification.json").read_text(
        encoding="utf-8"))
    if audited_inputs["corpus"]["sha256"] != raw_sha or literal.get("passed") is not True:
        raise ValueError("Near-pair audit was not literally verified on this raw source")
    n2 = _load_view(n2_corpus, n2_run / "normalization_trace.parquet")
    n3 = _load_view(n3_corpus, n3_run / "normalization_trace.parquet")
    print(f"Loaded paired views: {len(n2)} rows", flush=True)
    _check_paired_views(n2, n3)
    ids = set(n2.record_id)
    raw = _read_raw_metadata(raw_path, ids)
    protected = ["label_name", "broad_area", "source_name", "split_group_id"]
    check = n2[["record_id", *protected]].merge(
        raw[["record_id", *protected]], on="record_id", suffixes=("_n2", "_raw"),
        validate="one_to_one")
    for field in protected:
        if not check[f"{field}_n2"].fillna("<NULL>").equals(check[f"{field}_raw"].fillna("<NULL>")):
            raise ValueError(f"Raw/N2 protected metadata changed: {field}")
    combined = n2.merge(raw[["record_id", "source_record_id", "source_url", "duplicate_group_id"]],
                        on="record_id", validate="one_to_one")
    pairs = pd.read_csv(near_pairs_path)
    lineage, near_decisions, lineage_counts = build_lineage(combined, pairs, config)
    trace = _trace(n2_run / "normalization_trace.parquet", ids)
    quality = quality_table(n2, trace, lineage, config)
    print(f"Lineage and quality: {lineage_counts['groups']} groups, "
          f"{int(quality.quality_state.eq('eligible').sum())} eligible", flush=True)
    eligible = n2.merge(quality[["record_id", "quality_state", "selection_group_id"]],
                        on="record_id", validate="one_to_one")
    eligible = eligible.merge(lineage[["record_id", "probe_group_id"]],
                              on="record_id", validate="one_to_one")
    output.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(quality, preserve_index=False), output / "quality_manifest.parquet",
                   compression="zstd")
    pq.write_table(pa.Table.from_pandas(lineage, preserve_index=False), output / "lineage_manifest.parquet",
                   compression="zstd")
    near_decisions.to_csv(output / "near_pair_decisions.csv", index=False)
    lineage.groupby(["label_name", "broad_area", "group_size"], dropna=False).size(
    ).rename("rows").reset_index().to_csv(output / "lineage_group_sizes.csv", index=False)
    quality.groupby(["label_name", "broad_area", "quality_state", "hold_reasons"],
                    dropna=False).size().rename("rows").reset_index().to_csv(
        output / "quality_counts.csv", index=False)
    flag_rows = []
    for row in quality.itertuples(index=False):
        for flag in _flags(row.normalization_flags):
            flag_rows.append((row.label_name, row.broad_area, flag))
    pd.DataFrame(flag_rows, columns=["label_name", "broad_area", "flag"]).groupby(
        ["label_name", "broad_area", "flag"]).size().rename("rows").reset_index().to_csv(
        output / "flag_prevalence.csv", index=False)
    n3_trace = _trace(n3_run / "normalization_trace.parquet", ids)
    n3_flags = dict(zip(n3_trace.record_id, n3_trace.normalization_flags))
    membership, coverage, balance, probes, provenance, format_tables = [], [], [], [], [], []
    composition = []
    near_exposure, n3_quality_exposure = [], []
    summary_scenarios = []
    reference_ids: set[str] | None = None
    for width in config["log2_bin_widths"]:
        for seed in config["seeds"]:
            scenario = f"w{float(width):g}_s{int(seed)}"
            selected = select_cell_balanced(eligible, float(width), int(seed))
            chosen = set(selected.record_id)
            membership.extend({"scenario": scenario, "record_id": rid} for rid in sorted(chosen))
            composition.extend({"scenario": scenario, "broad_area": area, "label_name": label,
                                "selected_rows": int(count)}
                               for (area, label), count in selected.groupby(
                                   ["broad_area", "label_name"]).size().items())
            near_status = near_decisions.assign(
                both_selected=near_decisions.left_record_id.isin(chosen) &
                              near_decisions.right_record_id.isin(chosen))
            for decision, group in near_status.groupby("decision"):
                near_exposure.append({"scenario": scenario, "decision": decision,
                                      "pairs_both_selected": int(group.both_selected.sum()),
                                      "pairs_total": len(group)})
            for flag in sorted(set(config["hold_flags"]) | set(config["n3_diagnostic_flags"])):
                n3_quality_exposure.append({"scenario": scenario, "flag": flag,
                                            "selected_rows": sum(
                                                flag in _flags(n3_flags[rid]) for rid in chosen)})
            cov = _coverage(selected, quality, n2, config, scenario)
            coverage.append(cov)
            for view, frame in (("n2", n2), ("n3s", n3)):
                subset = frame.loc[frame.record_id.isin(chosen)].copy()
                subset["split_group_id"] = subset.record_id.map(
                    dict(zip(lineage.record_id, lineage.probe_group_id)))
                balance.extend(_balance_rows(subset, view, scenario))
                provenance.extend(_provenance_rows(subset, frame, view, scenario))
                probe_config = {**config, "seed": int(seed)}
                probes.extend(_probe_rows(subset, view, scenario, probe_config))
            summary_scenarios.append({"scenario": scenario, "rows": len(chosen),
                                      "per_class": len(chosen) // 2,
                                      "coverage_gate_met": bool(cov.coverage_gate_met.all())})
            print(f"Candidate {scenario}: {len(chosen)} rows; "
                  f"coverage gate={bool(cov.coverage_gate_met.all())}", flush=True)
            if float(width) == float(config["reference_width"]) and int(seed) == int(config["reference_seed"]):
                reference_ids = chosen
    if reference_ids is None:
        raise ValueError("Reference width/seed must be among configured scenarios")
    selected_ref = eligible.loc[eligible.record_id.isin(reference_ids)]
    collapsed = _linebreak_features(n2_corpus, selected_ref)
    collapsed = selected_ref.drop(columns=[column for column in collapsed.columns if column != "record_id"
                                          and column in selected_ref.columns]).merge(
        collapsed, on="record_id", validate="one_to_one")
    collapsed["split_group_id"] = collapsed.probe_group_id
    raw_format = _raw_format(raw_path, n2)
    format_tables.append(_format_rows(raw_format, n2, n3, reference_ids))
    n2_dialogue = _dialogue_lines(n2_corpus).rename(columns={"dialogue_dash_lines": "n2_dash_lines"})
    n3_dialogue = _dialogue_lines(n3_corpus).rename(columns={"dialogue_dash_lines": "n3s_dash_lines"})
    dialogue = n2[["record_id", "label_name", "broad_area", "source_name", "generator"]].merge(
        n2_dialogue, on="record_id", validate="one_to_one").merge(
        n3_dialogue, on="record_id", validate="one_to_one")
    dialogue["lost_dash_lines"] = (dialogue.n2_dash_lines - dialogue.n3s_dash_lines).clip(lower=0)
    dialogue.loc[dialogue.lost_dash_lines.gt(0)].to_csv(output / "dialogue_marker_loss.csv", index=False)
    balance.extend(_balance_rows(collapsed, "n2_blank_lines_collapsed", "reference"))
    probes.extend(_probe_rows(collapsed, "n2_blank_lines_collapsed", "reference",
                              {**config, "seed": int(config["reference_seed"])}))
    _sample_format_examples(n2_corpus, n3_corpus, n2, n3, config, output)
    pq.write_table(pa.Table.from_pylist(membership), output / "candidate_membership.parquet",
                   compression="zstd")
    coverage_table = pd.concat(coverage, ignore_index=True)
    balance_table = pd.DataFrame(balance)
    coverage_table.to_csv(output / "provenance_coverage.csv", index=False)
    balance_table.to_csv(output / "paired_balance_smd_ks.csv", index=False)
    pd.DataFrame(probes).to_csv(output / "paired_surface_probes.csv", index=False)
    pd.DataFrame(provenance).to_csv(output / "paired_provenance_retention.csv", index=False)
    pd.DataFrame(composition).to_csv(output / "candidate_composition.csv", index=False)
    pd.DataFrame(near_exposure).to_csv(output / "near_pair_selection_exposure.csv", index=False)
    pd.DataFrame(n3_quality_exposure).to_csv(output / "n3s_quality_exposure.csv", index=False)
    pd.concat(format_tables, ignore_index=True).to_csv(output / "format_by_provenance.csv", index=False)
    # N3s may merge different N2 texts. Report these collisions before anyone
    # decides whether N3s is suitable as a paired benchmark view.
    n3_dup = n3.loc[n3.content_sha256_audit.duplicated(keep=False),
                    ["record_id", "label_name", "content_sha256_audit"]].copy()
    n3_dup = n3_dup.merge(n2[["record_id", "content_sha256_audit"]].rename(
        columns={"content_sha256_audit": "n2_content_sha256_audit"}),
        on="record_id", validate="one_to_one")
    n3_dup["new_n3s_collision"] = n3_dup.groupby(
        "content_sha256_audit").n2_content_sha256_audit.transform("nunique").gt(1)
    n3_dup["selected_reference"] = n3_dup.record_id.isin(reference_ids)
    n3_dup.to_csv(output / "n3s_exact_collisions.csv", index=False)
    _plot_tradeoff(balance_table, coverage_table, output)
    summary = {"experiment_id": config["experiment_id"],
               "status": "exploratory_candidates_no_final_split",
               "raw_path": str(raw_path), "n2_run": str(n2_run), "n3s_run": str(n3_run),
               "near_pairs_path": str(near_pairs_path),
               "source_sha256": raw_sha, "n2_corpus_sha256": sha256_file(n2_corpus),
               "n3s_corpus_sha256": sha256_file(n3_corpus),
               "near_pairs_sha256": sha256_file(near_pairs_path),
               "config_sha256": sha256_file(config_path), "code_sha256": sha256_file(Path(__file__)),
               "python": sys.version, "pandas": pd.__version__, "pyarrow": pa.__version__,
               "sklearn": sklearn.__version__, "raw_rows": len(n2),
               "quality_eligible": int(quality.quality_state.eq("eligible").sum()),
               "quality_hold": int(quality.quality_state.eq("hold_for_review").sum()),
               "redundant_exact": int(quality.quality_state.eq("redundant_exact").sum()),
               "n3s_exact_collision_rows": len(n3_dup),
               "new_n3s_collision_rows": int(n3_dup.new_n3s_collision.sum()),
               "dialogue_dash_loss_rows": int(dialogue.lost_dash_lines.gt(0).sum()),
               "lineage": lineage_counts, "scenarios": summary_scenarios,
               "limitations": ["Near-duplicate search was approximate and missed short documents.",
                               "Containment-only pairs remain unresolved; parent-document IDs are unavailable.",
                               "No final split, detector benchmark or OOD estimate is produced.",
                               "Selection conditions on both labels and probes are exploratory."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    (output / "README.md").write_text(
        "# Candidatos de seleção N2\n\n"
        "A seleção usa elegibilidade conservadora e faixas de área × log2(1+palavras) da N2. "
        "Aplica exatamente os mesmos IDs à N3s. Nenhum split final foi criado.\n\n"
        "`quality_manifest.parquet` registra elegibilidade, cópias exatas e motivos de revisão. "
        "`flag_prevalence.csv` mostra também as flags que não excluem. "
        "`lineage_manifest.parquet` contém grupos de seleção conservadores e grupos "
        "de partição dos probes. Estes últimos unem todos os pares observados para evitar "
        "sua passagem entre partições, sem considerá-los duplicatas confirmadas. "
        "`near_pair_decisions.csv` mantém pares de contenção não resolvidos. "
        "`candidate_membership.parquet` contém todos os cenários por semente/faixa. "
        "`candidate_composition.csv` e `lineage_group_sizes.csv` resumem cobertura e agrupamento. "
        "`provenance_coverage.csv`, `paired_balance_smd_ks.csv`, `paired_surface_probes.csv`, "
        "`paired_provenance_retention.csv` e `format_by_provenance.csv` são diagnósticos. "
        "`dialogue_marker_loss.csv` e `n3s_exact_collisions.csv` verificam efeitos da N3s. "
        "`near_pair_selection_exposure.csv` e `n3s_quality_exposure.csv` mostram "
        "pendências que permanecem entre os IDs selecionados. "
        "`format_examples_blind.jsonl` contém textos e deve ficar local; "
        "`format_examples_manifest.csv` dá os metadados separadamente.\n\n"
        "A ablação `n2_blank_lines_collapsed` altera apenas quebras vazias, em memória. "
        "Sondas usam grupos conhecidos e limiar de validação, mas a seleção foi feita no corpus "
        "completo usando rótulos; seus resultados não são métricas finais. "
        "Restrições de cobertura são relatadas, nunca reparadas automaticamente. "
        "Sem ID de documento-pai e busca exaustiva de near-duplicates, não se pode declarar "
        "ausência de leakage documental.\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "raw", "n2-run", "n3-run", "near-pairs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.raw, args.n2_run, args.n3_run, args.near_pairs, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
