"""Paired exploratory shortcut probes for raw, N1 and N2 corpus views.

The inherited splits are not certified leakage-free. This pipeline does not
select benchmark rows, train final detectors, or evaluate OOD generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as ds
import sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from horus_benchmark.audit.middle_segments import align_segment, select_middle
from horus_benchmark.audit.probes import SURFACE_GROUPS, validation_threshold
from horus_benchmark.audit.text_features import extract_text_features
from horus_benchmark.data.freeze_normalization import check as check_freeze
from horus_benchmark.data.normalize_corpus import sha256_file


VIEWS = ("raw", "n1", "n2")
SPLITS = ("train", "validation", "test")
KINDS = ("document", "paragraph", "sentence")


def _sample_ids(raw: Path, limits: dict[str, int], seed: int) -> pd.DataFrame:
    columns = ["record_id", "label", "split", "broad_area"]
    frame = ds.dataset(str(raw), format="parquet").to_table(columns=columns).to_pandas()
    if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise ValueError("Raw corpus has missing or repeated record IDs")
    if set(frame["label"].dropna().unique()) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1")
    if not set(frame["split"]).issubset(SPLITS):
        raise ValueError("Unexpected split in raw corpus")
    selected = []
    for split in SPLITS:
        for label in (0, 1):
            group = frame.loc[(frame["split"] == split) & (frame["label"] == label)].copy()
            if len(group) < limits[split]:
                raise ValueError(f"Too few {split} rows of label {label}")
            group["sample_key"] = group["record_id"].map(
                lambda rid: hashlib.sha256(f"{seed}|{rid}".encode()).hexdigest()
            )
            # SHA-256 digests are strings: sort lexicographically, with record_id
            # as a deterministic tie-breaker independent of Parquet row order.
            selected.append(
                group.sort_values(["sample_key", "record_id"], kind="stable")
                .head(limits[split])
            )
    return pd.concat(selected, ignore_index=True).drop(columns="sample_key")


def _load_selected(path: Path, expected: pd.DataFrame, batch_size: int) -> dict[str, str]:
    expected_by_id = expected.set_index("record_id")
    contents: dict[str, str] = {}
    for batch in ds.dataset(str(path), format="parquet").to_batches(
        columns=["record_id", "label", "split", "broad_area", "content"],
        batch_size=batch_size,
    ):
        for row in batch.to_pylist():
            rid = row["record_id"]
            if rid not in expected_by_id.index:
                continue
            if rid in contents:
                raise ValueError(f"Repeated sampled record_id in {path}: {rid}")
            meta = expected_by_id.loc[rid]
            if any(row[key] != meta[key] for key in ("label", "split", "broad_area")):
                raise ValueError(f"Protected metadata differ for {rid} in {path}")
            if not isinstance(row["content"], str):
                raise ValueError(f"Invalid content for {rid} in {path}")
            contents[rid] = row["content"]
    if len(contents) != len(expected):
        raise ValueError(f"Missing {len(expected) - len(contents)} selected IDs in {path}")
    return contents


def _segment_triplets(
    sample: pd.DataFrame, texts: dict[str, dict[str, str]], kind: str,
    segment_config: dict, seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], pd.DataFrame]:
    if kind == "document":
        raise ValueError("Documents are compared directly, without segment alignment")
    bounds = segment_config[kind]
    rows = []
    selected = {view: {} for view in VIEWS}
    for meta in sample.itertuples(index=False):
        rid = meta.record_id
        raw = select_middle(
            texts["raw"][rid], kind, rid,
            min_words=int(bounds["min_words"]), max_words=int(bounds["max_words"]),
            middle_start=float(segment_config["middle_start"]),
            middle_end=float(segment_config["middle_end"]),
            max_fraction_of_document=float(bounds["max_fraction_of_document"]),
            seed=seed,
        )
        row = {"record_id": rid, "label": meta.label, "split": meta.split,
               "broad_area": meta.broad_area,
               "raw_status": "selected" if raw else "no_middle_unit",
               "raw_midpoint": raw.midpoint if raw else np.nan}
        aligned = {}
        for view in ("n1", "n2"):
            if raw is None:
                row[f"{view}_status"] = "not_attempted"
                row[f"{view}_similarity"] = np.nan
                continue
            unit, status, similarity = align_segment(
                raw, texts[view][rid], kind,
                minimum_similarity=float(segment_config["minimum_similarity"]),
                minimum_margin=float(segment_config["minimum_margin"]),
            )
            row[f"{view}_status"] = status
            row[f"{view}_similarity"] = similarity
            if unit is not None:
                aligned[view] = unit.text
        row["paired_eligible"] = raw is not None and len(aligned) == 2
        if row["paired_eligible"]:
            selected["raw"][rid] = raw.text
            selected["n1"][rid] = aligned["n1"]
            selected["n2"][rid] = aligned["n2"]
        rows.append(row)
    coverage = pd.DataFrame(rows)
    return sample.loc[sample["record_id"].isin(selected["raw"])].copy(), selected, coverage


def _fpr_threshold(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    negatives = np.sort(scores[y == 0])
    if len(negatives) == 0:
        raise ValueError("Validation requires negatives")
    # Strict greater-than makes empirical validation FPR at most target.
    index = max(0, int(np.ceil((1 - target) * len(negatives))) - 1)
    return float(negatives[index])


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float,
             low_fpr_threshold: float) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, scores >= threshold)),
        "tpr_at_fixed_fpr": float(np.mean(scores[y == 1] > low_fpr_threshold)),
        "observed_fpr": float(np.mean(scores[y == 0] > low_fpr_threshold)),
    }


def paired_intervals(
    y: np.ndarray, predictions: dict[str, np.ndarray], thresholds: dict[str, tuple[float, float]],
    *, replicates: int, confidence: float, seed: int,
) -> pd.DataFrame:
    """Stratified paired bootstrap: resample the same test IDs in every view."""

    if replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    indices = {label: np.flatnonzero(y == label) for label in (0, 1)}
    if any(len(value) == 0 for value in indices.values()):
        raise ValueError("Both labels required for paired bootstrap")
    rng = np.random.default_rng(seed)
    draws: dict[tuple[str, str], list[float]] = {}
    for _ in range(replicates):
        take = np.concatenate([rng.choice(indices[label], len(indices[label]), replace=True)
                               for label in (0, 1)])
        for view in VIEWS:
            values = _metrics(y[take], predictions[view][take], *thresholds[view])
            for metric, value in values.items():
                draws.setdefault((view, metric), []).append(value)
    alpha = (1 - confidence) / 2
    rows = []
    for metric in ("roc_auc", "balanced_accuracy", "tpr_at_fixed_fpr", "observed_fpr"):
        for before, after in (("raw", "n1"), ("n1", "n2"), ("raw", "n2")):
            first = np.asarray(draws[before, metric])
            second = np.asarray(draws[after, metric])
            delta = second - first
            rows.append({"metric": metric, "before": before, "after": after,
                         "delta_ci_low": float(np.quantile(delta, alpha)),
                         "delta_ci_high": float(np.quantile(delta, 1 - alpha))})
    return pd.DataFrame(rows)


def _fit_probe(
    frame: pd.DataFrame, content: dict[str, str], view: str, kind: str,
    probe: str, config: dict, seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    splits = {split: frame.loc[frame["split"] == split] for split in SPLITS}
    y = {split: splits[split]["label"].to_numpy(dtype=int) for split in SPLITS}
    if probe in SURFACE_GROUPS:
        columns = SURFACE_GROUPS[probe]
        matrix = {}
        for split in SPLITS:
            matrix[split] = np.asarray([
                [float(extract_text_features(content[rid])[name]) for name in columns]
                for rid in splits[split]["record_id"]
            ])
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=1000, random_state=seed))
    else:
        if probe == "char_4gram":
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(4, 4),
                                         max_features=int(config["max_features"]),
                                         min_df=int(config["min_df"]), sublinear_tf=True,
                                         dtype=np.float32)
        elif probe == "word_1_2gram":
            vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                         max_features=int(config["max_features"]),
                                         min_df=int(config["min_df"]), sublinear_tf=True,
                                         dtype=np.float32)
        else:
            raise ValueError(f"Unknown probe: {probe}")
        matrix = {split: [content[rid] for rid in splits[split]["record_id"]] for split in SPLITS}
        matrix["train"] = vectorizer.fit_transform(matrix["train"])
        matrix["validation"] = vectorizer.transform(matrix["validation"])
        matrix["test"] = vectorizer.transform(matrix["test"])
        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
    model.fit(matrix["train"], y["train"])
    val_scores = model.predict_proba(matrix["validation"])[:, 1]
    test_scores = model.predict_proba(matrix["test"])[:, 1]
    if probe in SURFACE_GROUPS:
        feature_names = np.asarray(SURFACE_GROUPS[probe])
        coefficients = model.named_steps["logisticregression"].coef_[0]
    else:
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
    count = min(20, len(coefficients))
    feature_indices = np.unique(np.concatenate((
        np.argsort(coefficients)[:count], np.argsort(coefficients)[-count:]
    )))
    top_features = pd.DataFrame({
        "kind": kind, "probe": probe, "view": view,
        "feature": feature_names[feature_indices],
        "coefficient": coefficients[feature_indices],
    }).sort_values("coefficient")
    return val_scores, test_scores, y, top_features


def run(config_path: Path, raw: Path, n1: Path, n2: Path, output: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    raw, n1, n2, output = (path.expanduser().resolve() for path in (raw, n1, n2, output))
    if output.exists():
        raise FileExistsError(f"Output path must be new: {output}")
    check_freeze(n2)
    manifests = {}
    for stage, path in (("n1", n1), ("n2", n2)):
        verification = json.loads((path / "verification.json").read_text(encoding="utf-8"))
        if not verification.get("verified") or verification.get("stage") != stage:
            raise ValueError(f"{stage} is not verified")
        manifests[stage] = verification
    raw_sha = sha256_file(raw)
    if any(item["source_sha256"] != raw_sha for item in manifests.values()):
        raise ValueError("Raw source hash differs from N1/N2 verification")
    raw_rows = ds.dataset(str(raw), format="parquet").count_rows()
    if any(item["rows"] != raw_rows for item in manifests.values()):
        raise ValueError("N1/N2 verified row counts differ from the raw corpus")
    sample = _sample_ids(raw, config["sample_per_label"], int(config["seed"]))
    paths = {"raw": raw, "n1": n1 / "normalized_corpus.parquet",
             "n2": n2 / "normalized_corpus.parquet"}
    texts = {view: _load_selected(path, sample, int(config["batch_size"]))
             for view, path in paths.items()}
    output.mkdir(parents=True)
    sample.to_csv(output / "sampled_documents.csv", index=False)
    results = []
    intervals = []
    predictions = []
    subgroup = []
    top_terms = []
    coverage_summaries = []
    paired_counts = {"document": len(sample)}
    skipped = []
    for kind in KINDS:
        if kind == "document":
            eligible, selected = sample.copy(), texts
        else:
            eligible, selected, coverage = _segment_triplets(
                sample, texts, kind, config["segments"], int(config["seed"]))
            coverage.to_csv(output / f"{kind}_alignment.csv", index=False)
            paired_counts[kind] = len(eligible)
            summary = coverage.groupby(["split", "label", "raw_status", "n1_status", "n2_status"],
                                       dropna=False).size().reset_index(name="rows")
            summary.insert(0, "kind", kind)
            coverage_summaries.append(summary)
        counts = eligible.groupby(["split", "label"]).size()
        low_cells = [f"{split}:{label}={counts.get((split, label), 0)}"
                     for split in SPLITS for label in (0, 1)
                     if counts.get((split, label), 0) < int(config["minimum_rows_per_label_split"])]
        if low_cells:
            skipped.append({"kind": kind, "reason": "insufficient_paired_triplets",
                            "cells": ", ".join(low_cells)})
            continue
        probes = ["word_1_2gram", "char_4gram"]
        if kind == "document":
            probes += list(SURFACE_GROUPS)
        for probe in probes:
            scores = {}
            thresholds = {}
            label_arrays = None
            for view in VIEWS:
                val, test, labels, top = _fit_probe(
                    eligible, selected[view], view, kind, probe,
                    config["tfidf"], int(config["seed"])
                )
                top_terms.append(top)
                if label_arrays is None:
                    label_arrays = labels
                elif any(not np.array_equal(labels[split], label_arrays[split]) for split in SPLITS):
                    raise ValueError("Labels differ across views")
                threshold = validation_threshold(labels["validation"], val)
                low_fpr = _fpr_threshold(labels["validation"], val, float(config["target_fpr"]))
                scores[view] = test
                thresholds[view] = (threshold, low_fpr)
                values = _metrics(labels["test"], test, threshold, low_fpr)
                results.append({"kind": kind, "probe": probe, "view": view,
                                "train_rows": len(labels["train"]),
                                "validation_rows": len(labels["validation"]),
                                "test_rows": len(labels["test"]),
                                "validation_threshold": threshold,
                                "validation_fixed_fpr_threshold": low_fpr,
                                **values})
            test_meta = eligible.loc[eligible["split"] == "test"].reset_index(drop=True)
            for view in VIEWS:
                predictions.extend({"kind": kind, "probe": probe, "view": view,
                                    "record_id": rid, "label": label, "score": float(score)}
                                   for rid, label, score in zip(test_meta["record_id"],
                                                                 test_meta["label"], scores[view]))
            ci = paired_intervals(label_arrays["test"], scores, thresholds,
                                  replicates=int(config["bootstrap_replicates"]),
                                  confidence=float(config["bootstrap_confidence"]),
                                  seed=int(config["seed"]))
            ci.insert(0, "probe", probe)
            ci.insert(0, "kind", kind)
            intervals.append(ci)
            # Nuisance controls: fixed raw-document length strata and domain;
            # never feed source/generator/area metadata into the classifier.
            raw_words = test_meta["record_id"].map(lambda rid: len(texts["raw"][rid].split()))
            bins = pd.cut(raw_words, bins=config["raw_word_bins"], include_lowest=True)
            for grouping, groups in (("area", test_meta["broad_area"].fillna("MISSING").astype(str)),
                                     ("raw_length_bin", bins.astype(str))):
                for group in sorted(groups.unique()):
                    mask = (groups == group).to_numpy()
                    y_group = label_arrays["test"][mask]
                    if min((y_group == 0).sum(), (y_group == 1).sum()) < int(config["minimum_subgroup_per_label"]):
                        continue
                    for view in VIEWS:
                        values = _metrics(y_group, scores[view][mask], *thresholds[view])
                        subgroup.append({"kind": kind, "probe": probe, "view": view,
                                         "grouping": grouping, "group": str(group),
                                         "rows": len(y_group), **values})
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / "probe_metrics.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(output / "paired_delta_intervals.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(subgroup).to_csv(output / "nuisance_strata.csv", index=False)
    pd.concat(top_terms, ignore_index=True).to_csv(output / "top_probe_features.csv", index=False)
    pd.DataFrame(skipped, columns=["kind", "reason", "cells"]).to_csv(
        output / "skipped_granularities.csv", index=False)
    pd.concat(coverage_summaries, ignore_index=True).to_csv(output / "segment_coverage.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("roc_auc", "balanced_accuracy")):
        for (kind, probe), group in metrics.groupby(["kind", "probe"]):
            if probe not in {"word_1_2gram", "char_4gram", "all_surface"}:
                continue
            ordered = group.set_index("view").loc[list(VIEWS)]
            ax.plot(list(VIEWS), ordered[metric], marker="o", label=f"{kind}: {probe}")
        ax.set_title(metric)
        ax.set_ylim(0, 1)
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.savefig(output / "probe_trajectory.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (output / "provenance.json").write_text(json.dumps({
        "raw_path": str(raw), "raw_sha256": raw_sha,
        "n1_run": str(n1), "n2_run": str(n2),
        "n1_corpus_sha256": sha256_file(paths["n1"]),
        "n2_corpus_sha256": sha256_file(paths["n2"]),
        "n2_freeze_sha256": sha256_file(n2 / "freeze_manifest.json"),
        "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "segmenter_sha256": sha256_file(Path(select_middle.__code__.co_filename)),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__, "sklearn_version": sklearn.__version__,
        "sampled_documents": len(sample),
        "paired_segment_counts": paired_counts,
        "skipped_granularities": skipped,
    }, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Comparação exploratória raw → N1 → N2\n\n"
        "Modelos separados por view, mesmos IDs e rótulos em cada granularidade. "
        "Vocabulário e scaler ajustados somente no treino; thresholds somente na validação. "
        "Frase significa sentença completa; parágrafo significa bloco separado por linha em branco. "
        "Frases/parágrafos vêm do miolo do original; só trechos alinhados com alta confiança "
        "entram na comparação pareada. Ausências e ambiguidades estão nas tabelas de alinhamento. "
        "Apenas um segmento por documento evita sobrepeso dos documentos longos. "
        "Um rótulo de documento herdado não garante autoria uniforme em cada frase. "
        "Splits herdados ainda têm pares near-duplicate e falta de lineage documental: "
        "resultados são exploratórios, não uma avaliação final leakage-free ou OOD. "
        "Nuisance é sondado por características superficiais e estratos de domínio/comprimento; "
        "fonte e gerador não são usados como features.\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--n1-run", type=Path, required=True)
    parser.add_argument("--n2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.raw, args.n1_run, args.n2_run, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
