"""Blind human review, threshold validation and derived theme tables."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.metrics import cohen_kappa_score

from horus_benchmark.audit.theme_common import (
    ThemeContext, read_run_metadata, sha256_file, validate_artifact_context,
)
from horus_benchmark.data import DatasetLoader


REVIEW_COLUMNS = ["record_id", "broad_area", "content", "theme_human", "annotation_note"]
SPECIAL_GOLD = {"incerto", "fora_taxonomia"}


def proportional_allocation(source_counts: pd.Series, total: int) -> pd.Series:
    """Largest-remainder allocation with positive support for every source."""
    if (total < 1 or total > int(source_counts.sum())
            or len(source_counts) > total or (source_counts <= 0).any()):
        raise ValueError("Invalid proportional sample allocation")
    quotas = source_counts * (total / int(source_counts.sum()))
    allocations = np.maximum(np.floor(quotas).astype(int), 1)
    while int(allocations.sum()) > total:
        reducible = allocations.loc[allocations > 1]
        if reducible.empty:
            raise ValueError("Could not preserve source coverage")
        source = max(reducible.index, key=lambda name: (allocations[name] - quotas[name], str(name)))
        allocations[source] -= 1
    slots = total - int(allocations.sum())
    for source in sorted(
        source_counts.index,
        key=lambda name: (-(quotas[name] - allocations[name]), str(name)),
    ):
        if slots == 0:
            break
        if allocations[source] < source_counts[source]:
            allocations[source] += 1
            slots -= 1
    if slots:
        raise ValueError("Could not fill proportional sample allocation")
    return allocations


def export_theme_inventory(context: ThemeContext) -> Path:
    loader = DatasetLoader(context.catalog.resource(context.config["corpus_resource"]))
    table = loader.read_table(
        columns=["broad_area", "source_name", "theme"],
        filter_expression=ds.field("label_name") == "human",
    )
    frame = table.to_pandas()
    if not set(frame["broad_area"]).issubset(context.taxonomy):
        raise ValueError("Unknown human broad area")
    counts = frame.groupby(["broad_area", "source_name", "theme"], dropna=False).size()
    output_frame = counts.reset_index(name="rows")
    resolved = [
        context.canonical_existing(area, raw if isinstance(raw, str) else None)
        for area, raw in zip(output_frame["broad_area"], output_frame["theme"])
    ]
    output_frame["theme_canonical"] = [value for value, _ in resolved]
    output_frame["mapping_method"] = [method for _, method in resolved]
    run = context.new_run("inventory", extra={
        "corpus_sha256": sha256_file(context.catalog.resource(context.config["corpus_resource"]).path),
    })
    output = run.directory / "human_theme_inventory.csv"
    output_frame.to_csv(output, index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"broad_area": area, "theme": theme}
        for area, themes in context.taxonomy.items() for theme in themes
    ]).to_csv(run.directory / "taxonomy_themes.csv", index=False, encoding="utf-8-sig")
    return output


def export_synthetic_review(context: ThemeContext) -> Path:
    """Small balanced sample to check whether texts follow their prompt themes."""
    loader = DatasetLoader(context.catalog.resource(context.config["corpus_resource"]))
    rng = np.random.default_rng(int(context.config["seed"]) + 2)
    per_cell = int(context.config["sampling"]["synthetic_review_per_area_generator"])
    if per_cell < 1:
        raise ValueError("synthetic_review_per_area_generator must be positive")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    counts: dict[tuple[str, str], int] = {}
    for batch in loader.iter_batches(
        columns=["record_id", "broad_area", "generator", "theme", "content"],
        filter_expression=ds.field("label_name") == "synthetic",
        batch_size=int(context.config["batch_size"]),
    ):
        for row in batch.to_pylist():
            area, generator = row["broad_area"], row["generator"] or "<missing>"
            if area not in context.taxonomy or row["theme"] not in context.taxonomy[area]:
                raise ValueError(f"Invalid synthetic theme in {area} / {generator}")
            key = (area, generator)
            counts[key] = counts.get(key, 0) + 1
            bucket = buckets.setdefault(key, [])
            if len(bucket) < per_cell:
                bucket.append(row)
            else:
                slot = int(rng.integers(counts[key]))
                if slot < per_cell:
                    bucket[slot] = row
    selected = pd.DataFrame.from_records(
        row for key in sorted(buckets) for row in buckets[key]
    )
    selected["theme_supported"] = ""
    selected["annotation_note"] = ""
    run = context.new_run("synthetic-review", extra={
        "corpus_sha256": sha256_file(context.catalog.resource(context.config["corpus_resource"]).path),
    })
    output = run.directory / "synthetic_theme_review.csv"
    selected.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _human_metadata(context: ThemeContext) -> pd.DataFrame:
    loader = DatasetLoader(context.catalog.resource(context.config["corpus_resource"]))
    condition = ds.field("label_name") == "human"
    table = loader.read_table(
        columns=["record_id", "broad_area", "source_name", "split"],
        filter_expression=condition,
    )
    frame = table.to_pandas()
    if frame["record_id"].duplicated().any():
        raise ValueError("Duplicate human record ID")
    if set(frame["broad_area"]) != set(context.taxonomy):
        raise ValueError("Human broad areas differ from taxonomy")
    return frame


def _blind_sheet(context: ThemeContext, selected: pd.DataFrame) -> pd.DataFrame:
    ids = selected["record_id"].astype(str).tolist()
    loader = DatasetLoader(context.catalog.resource(context.config["corpus_resource"]))
    table = loader.read_table(
        columns=["record_id", "broad_area", "content"],
        filter_expression=ds.field("record_id").isin(ids),
    )
    text = table.to_pandas()
    if len(text) != len(ids) or text["record_id"].duplicated().any():
        raise ValueError("Could not retrieve every selected text exactly once")
    sheet = selected[["record_id"]].merge(text, on="record_id", validate="one_to_one")
    if sheet["content"].isna().any() or (sheet["content"].str.strip() == "").any():
        raise ValueError("A selected text is empty")
    sheet["theme_human"] = ""
    sheet["annotation_note"] = ""
    return sheet[REVIEW_COLUMNS]


def export_pilot(context: ThemeContext) -> Path:
    frame = _human_metadata(context)
    rng = np.random.default_rng(int(context.config["seed"]))
    per_area = int(context.config["sampling"]["pilot_per_area"])
    parts = []
    for area in context.taxonomy:
        group = frame.loc[frame["broad_area"] == area].sort_values("record_id")
        if len(group) < per_area:
            raise ValueError(f"Too few human texts for pilot in {area}")
        parts.append(group.iloc[rng.choice(len(group), per_area, replace=False)])
    selected = pd.concat(parts, ignore_index=True)
    run = context.new_run("pilot", extra={
        "corpus_sha256": sha256_file(context.catalog.resource(context.config["corpus_resource"]).path),
    })
    output = run.directory / "pilot_annotation.csv"
    _blind_sheet(context, selected).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _predictions(context: ThemeContext, path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    validate_artifact_context(context, path)
    meta = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    corpus_path = context.catalog.resource(context.config["corpus_resource"]).path
    if meta["extra"].get("corpus_sha256") != sha256_file(corpus_path):
        raise ValueError("Corpus content differs from prediction source")
    frame = pd.read_parquet(path)
    if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise ValueError("Predictions contain missing or duplicate record IDs")
    expected_humans = DatasetLoader(
        context.catalog.resource(context.config["corpus_resource"])
    ).count_rows(filter_expression=ds.field("label_name") == "human")
    if len(frame) != expected_humans:
        raise ValueError(f"Predictions cover {len(frame)} humans, expected {expected_humans}")
    if set(frame["broad_area"]) != set(context.taxonomy):
        raise ValueError("Prediction broad areas differ from taxonomy")
    for area, group in frame.groupby("broad_area"):
        if not set(group["theme_predicted"]).issubset(context.taxonomy[area]):
            raise ValueError(f"Prediction outside taxonomy: {area}")
        mapped = set(group["mapped_theme"].dropna())
        if not mapped.issubset(context.taxonomy[area]):
            raise ValueError(f"Mapped theme outside taxonomy: {area}")
    return frame


def export_review_sample(context: ThemeContext, predictions_path: Path, pilot: Path) -> Path:
    predictions = _predictions(context, predictions_path)
    pilot_meta = read_run_metadata(pilot)
    if pilot_meta["dataset_version"] != context.catalog.dataset_version:
        raise ValueError("Pilot came from another dataset version")
    current_corpus_hash = sha256_file(
        context.catalog.resource(context.config["corpus_resource"]).path
    )
    if pilot_meta["extra"].get("corpus_sha256") != current_corpus_hash:
        raise ValueError("Pilot came from another corpus file")
    pilot_sheet = pd.read_csv(pilot, dtype=str)
    pilot_ids = set(pilot_sheet["record_id"])
    expected_pilot = int(context.config["sampling"]["pilot_per_area"]) * len(context.taxonomy)
    if (len(pilot_ids) != len(pilot_sheet) or len(pilot_ids) != expected_pilot
            or not pilot_ids.issubset(set(predictions["record_id"]))):
        raise ValueError("Pilot IDs are missing, duplicated or outside human predictions")
    rng = np.random.default_rng(int(context.config["seed"]) + 1)
    sample_cfg = context.config["sampling"]
    random_n = int(sample_cfg["random_per_area"])
    second_n = int(sample_cfg["second_annotator_per_area"])
    if second_n > random_n or random_n % 2:
        raise ValueError("Second annotator count must not exceed random count; random count must be even")
    eligible = predictions.loc[~predictions["record_id"].isin(pilot_ids)].copy()
    selected: list[pd.DataFrame] = []
    second_ids: set[str] = set()
    for area in context.taxonomy:
        group = eligible.loc[eligible["broad_area"] == area].sort_values("record_id")
        if len(group) < random_n:
            raise ValueError(f"Too few human texts for random sample in {area}")
        group = group.copy()
        group["source_stratum"] = group["source_name"].fillna("<missing>")
        source_counts = group["source_stratum"].value_counts().sort_index()
        allocations = proportional_allocation(source_counts, random_n)
        source_draws = []
        for source, count in allocations.items():
            if count == 0:
                continue
            source_group = group.loc[group["source_stratum"] == source]
            drawn = source_group.iloc[
                rng.choice(len(source_group), int(count), replace=False)
            ].copy()
            drawn["inclusion_probability"] = count / len(source_group)
            source_draws.append(drawn)
        draw = pd.concat(source_draws, ignore_index=True)
        draw = draw.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)
        half = random_n // 2
        draw["sample_role"] = ["calibration"] * half + ["evaluation"] * half
        selected.append(draw)
        for role in ("calibration", "evaluation"):
            role_ids = draw.loc[draw["sample_role"] == role, "record_id"].tolist()
            take = second_n // 2 + (1 if role == "calibration" and second_n % 2 else 0)
            second_ids.update(rng.choice(role_ids, take, replace=False).tolist())
    random_sample = pd.concat(selected, ignore_index=True)
    remaining = eligible.loc[~eligible["record_id"].isin(random_sample["record_id"])].copy()
    diagnostic_n = int(sample_cfg["diagnostic_total"])
    allocations = {
        area: diagnostic_n // len(context.taxonomy) + (i < diagnostic_n % len(context.taxonomy))
        for i, area in enumerate(context.taxonomy)
    }
    diagnostic_parts = []
    for area, count in allocations.items():
        group = remaining.loc[remaining["broad_area"] == area].copy()
        if len(group) < count:
            raise ValueError(f"Too few diagnostic candidates in {area}")
        # Half the cases are low confidence; the rest favor rare predicted themes.
        low = group.sort_values(["top_probability", "record_id"]).head(count // 2)
        rest = group.loc[~group["record_id"].isin(low["record_id"])].copy()
        frequency = group["theme_predicted"].value_counts()
        rest["predicted_frequency"] = rest["theme_predicted"].map(frequency)
        rare = rest.sort_values(
            ["predicted_frequency", "top_probability", "record_id"]
        ).head(count - len(low))
        drawn = pd.concat([low, rare], ignore_index=True)
        drawn["sample_role"] = "diagnostic"
        drawn["inclusion_probability"] = np.nan
        diagnostic_parts.append(drawn)
    manifest = pd.concat([random_sample, *diagnostic_parts], ignore_index=True)
    if manifest["record_id"].duplicated().any():
        raise ValueError("Review sample contains duplicate records")
    manifest["second_annotator"] = manifest["record_id"].isin(second_ids)
    run = context.new_run("sample", extra={
        "predictions_sha256": sha256_file(predictions_path),
        "pilot_sha256": sha256_file(pilot),
    })
    manifest_path = run.directory / "sample_manifest.csv"
    manifest.drop(columns=["predicted_frequency"], errors="ignore").to_csv(
        manifest_path, index=False, encoding="utf-8-sig",
    )
    _blind_sheet(context, manifest).to_csv(
        run.directory / "primary_annotation.csv", index=False, encoding="utf-8-sig",
    )
    _blind_sheet(context, manifest.loc[manifest["second_annotator"]]).to_csv(
        run.directory / "second_annotation.csv", index=False, encoding="utf-8-sig",
    )
    return manifest_path


def _annotation(path: Path, context: ThemeContext, expected_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not {"record_id", "broad_area", "theme_human"}.issubset(frame.columns):
        raise ValueError(f"Annotation sheet lacks required columns: {path}")
    if frame["record_id"].duplicated().any() or set(frame["record_id"]) != expected_ids:
        raise ValueError(f"Annotation IDs differ from sample: {path}")
    frame["theme_human"] = frame["theme_human"].str.strip()
    for row in frame.itertuples(index=False):
        if row.broad_area not in context.taxonomy:
            raise ValueError(f"Unknown broad area in annotation: {row.broad_area}")
        if row.theme_human not in context.taxonomy[row.broad_area] and row.theme_human not in SPECIAL_GOLD:
            raise ValueError(f"Invalid or blank human theme for {row.record_id}: {row.theme_human!r}")
    return frame[["record_id", "broad_area", "theme_human"]]


def _wilson(successes: int, n: int, confidence: float) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return float(center - half), float(center + half)


def choose_threshold(
    scores: np.ndarray, correct: np.ndarray, *, target_precision: float,
    min_accepted: int, confidence: float,
) -> dict[str, Any]:
    if len(scores) != len(correct) or not 0 < target_precision < 1:
        raise ValueError("Invalid threshold calibration inputs")
    if min_accepted < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid threshold policy")
    best = None
    for threshold in sorted(set(float(value) for value in scores)):
        accepted = scores >= threshold
        n = int(accepted.sum())
        if n < min_accepted:
            continue
        right = int(correct[accepted].sum())
        lower, upper = _wilson(right, n, confidence)
        if lower is not None and lower >= target_precision:
            best = {
                "threshold": threshold, "accepted": n, "correct": right,
                "precision": right / n, "precision_ci_low": lower,
                "precision_ci_high": upper,
            }
            break
    return best or {
        "threshold": 1.01, "accepted": 0, "correct": 0,
        "precision": None, "precision_ci_low": None,
        "precision_ci_high": None,
        "note": "No score met the prespecified precision lower-bound target",
    }


def _assigned(frame: pd.DataFrame, threshold: float) -> pd.Series:
    mapped = frame["mapped_theme"].notna() & (frame["mapped_theme"] != "")
    confident = frame["top_probability"] >= threshold
    return pd.Series(
        np.where(mapped, frame["mapped_theme"],
                 np.where(confident, frame["theme_predicted"], "incerto")),
        index=frame.index,
    )


def _metrics(frame: pd.DataFrame, threshold: float, confidence: float) -> dict[str, Any]:
    assigned = _assigned(frame, threshold)
    accepted = assigned != "incerto"
    correct = assigned == frame["theme_human"]
    n = int(accepted.sum())
    right = int((accepted & correct).sum())
    low, high = _wilson(right, n, confidence)
    return {
        "rows": len(frame), "accepted": n,
        "coverage": n / len(frame) if len(frame) else None,
        "accepted_precision": right / n if n else None,
        "accepted_precision_ci_low": low,
        "accepted_precision_ci_high": high,
        "gold_incerto_or_outside": int(frame["theme_human"].isin(SPECIAL_GOLD).sum()),
        "mapped_rows": int(frame["mapped_theme"].notna().sum()),
    }


def evaluate_annotations(
    context: ThemeContext, predictions_path: Path, sample_manifest: Path,
    primary_path: Path, second_path: Path | None = None,
    adjudicated_path: Path | None = None,
    baseline_predictions_path: Path | None = None,
) -> Path:
    predictions = _predictions(context, predictions_path)
    validate_artifact_context(context, sample_manifest)
    sample_meta = json.loads((sample_manifest.parent / "run.json").read_text(encoding="utf-8"))
    if sample_meta["extra"].get("predictions_sha256") != sha256_file(predictions_path):
        raise ValueError("Sample manifest was created from other predictions")
    manifest = pd.read_csv(sample_manifest, dtype={"record_id": str})
    manifest["second_annotator"] = (
        manifest["second_annotator"].astype(str).str.lower() == "true"
    )
    if manifest["record_id"].duplicated().any():
        raise ValueError("Sample manifest has duplicate record IDs")
    joined = manifest[["record_id", "broad_area", "sample_role", "second_annotator",
                       "inclusion_probability"]].merge(
        predictions, on=["record_id", "broad_area"], validate="one_to_one",
    )
    if len(joined) != len(manifest):
        raise ValueError("Sample records missing predictions")
    primary = _annotation(primary_path, context, set(manifest["record_id"]))
    joined = joined.merge(
        primary[["record_id", "broad_area", "theme_human"]],
        on=["record_id", "broad_area"], validate="one_to_one",
    )
    if len(joined) != len(manifest):
        raise ValueError("Primary annotation areas differ from sample")
    if baseline_predictions_path is not None:
        validate_artifact_context(context, baseline_predictions_path)
        baseline_meta = read_run_metadata(baseline_predictions_path)
        current_hash = sha256_file(
            context.catalog.resource(context.config["corpus_resource"]).path
        )
        if baseline_meta["extra"].get("corpus_sha256") != current_hash:
            raise ValueError("Baseline predictions came from another corpus")
        baseline = pd.read_parquet(baseline_predictions_path)
        if baseline["record_id"].duplicated().any():
            raise ValueError("Baseline predictions contain duplicate IDs")
        joined = joined.merge(
            baseline[["record_id", "broad_area", "theme_baseline"]],
            on=["record_id", "broad_area"], validate="one_to_one",
        )
        if len(joined) != len(manifest):
            raise ValueError("Some sampled texts lack baseline predictions")
    agreement = None
    if second_path is not None:
        second_ids = set(manifest.loc[manifest["second_annotator"], "record_id"])
        second = _annotation(second_path, context, second_ids)
        paired = primary.merge(second, on=["record_id", "broad_area"],
                               suffixes=("_primary", "_second"), validate="one_to_one")
        paired = paired.loc[paired["record_id"].isin(second_ids)]
        if len(paired) != len(second_ids):
            raise ValueError("Second annotator overlap is incomplete")
        primary_labels = paired["broad_area"] + "::" + paired["theme_human_primary"]
        second_labels = paired["broad_area"] + "::" + paired["theme_human_second"]
        kappa = float(cohen_kappa_score(primary_labels, second_labels))
        rng = np.random.default_rng(int(context.config["seed"]) + 3)
        strata = [group.index.to_numpy() for _, group in paired.groupby("broad_area")]
        bootstrapped = []
        for _ in range(500):
            drawn = np.concatenate([
                rng.choice(index, len(index), replace=True) for index in strata
            ])
            value = float(cohen_kappa_score(
                primary_labels.loc[drawn], second_labels.loc[drawn],
            ))
            if np.isfinite(value):
                bootstrapped.append(value)
        agreement = {
            "rows": len(paired),
            "raw_agreement": float((paired["theme_human_primary"] == paired["theme_human_second"]).mean()),
            "cohen_kappa": kappa if np.isfinite(kappa) else None,
            "cohen_kappa_bootstrap_95": (
                [float(value) for value in np.quantile(bootstrapped, [0.025, 0.975])]
                if bootstrapped else None
            ),
            "note": "Area-stratified bootstrap; sparse themes limit per-theme conclusions",
        }
        disagreements = paired.loc[
            paired["theme_human_primary"] != paired["theme_human_second"]
        ]
    else:
        disagreements = pd.DataFrame()
    if adjudicated_path is not None:
        if second_path is None:
            raise ValueError("Adjudication requires a second annotator sheet")
        override = _annotation(adjudicated_path, context, set(disagreements["record_id"]))
        checked = override.merge(
            disagreements[["record_id", "broad_area"]],
            on=["record_id", "broad_area"], validate="one_to_one",
        )
        if len(checked) != len(override):
            raise ValueError("Adjudication areas differ from disagreement records")
        overrides = override.set_index("record_id")["theme_human"].to_dict()
        joined["theme_human"] = joined.apply(
            lambda row: overrides.get(row["record_id"], row["theme_human"]), axis=1,
        )
    calibration = joined.loc[joined["sample_role"] == "calibration"]
    evaluation = joined.loc[joined["sample_role"] == "evaluation"]
    if calibration.empty or evaluation.empty:
        raise ValueError("Both calibration and independent evaluation are required")
    model_only = calibration.loc[
        calibration["mapped_theme"].isna()
    ]
    correct = (model_only["theme_predicted"] == model_only["theme_human"]).to_numpy()
    policy = context.config["threshold"]
    chosen = choose_threshold(
        model_only["top_probability"].to_numpy(dtype=float), correct,
        target_precision=float(policy["target_precision"]),
        min_accepted=int(policy["min_accepted_calibration"]),
        confidence=float(policy["wilson_confidence"]),
    )
    threshold = chosen["threshold"]
    report = {
        "threshold_calibration": chosen,
        "evaluation_area_balanced": _metrics(evaluation, threshold, float(policy["wilson_confidence"])),
        "evaluation_by_area": {
            area: _metrics(group, threshold, float(policy["wilson_confidence"]))
            for area, group in evaluation.groupby("broad_area")
        },
        "evaluation_by_source": {
            str(source): _metrics(group, threshold, float(policy["wilson_confidence"]))
            for source, group in evaluation.groupby("source_name", dropna=False)
        },
        "evaluation_mapped_only": _metrics(
            evaluation.loc[evaluation["mapped_theme"].notna()],
            threshold, float(policy["wilson_confidence"]),
        ),
        "evaluation_model_only": _metrics(
            evaluation.loc[evaluation["mapped_theme"].isna()],
            threshold, float(policy["wilson_confidence"]),
        ),
        "agreement": agreement,
        "gold_reference": "adjudicated disagreements" if adjudicated_path else "primary annotator",
    }
    report["unthresholded_human_accuracy"] = {
        "bert_overall_area_balanced": float(
            (evaluation["theme_predicted"] == evaluation["theme_human"]).mean()
        ),
        "bert_by_area": {
            area: float((group["theme_predicted"] == group["theme_human"]).mean())
            for area, group in evaluation.groupby("broad_area")
        },
        "baseline_overall_area_balanced": (
            float((evaluation["theme_baseline"] == evaluation["theme_human"]).mean())
            if baseline_predictions_path is not None else None
        ),
        "baseline_by_area": (
            {
                area: float((group["theme_baseline"] == group["theme_human"]).mean())
                for area, group in evaluation.groupby("broad_area")
            } if baseline_predictions_path is not None else None
        ),
        "note": "Gold incerto/fora_taxonomia count as errors for both forced classifiers",
    }
    length_bins = [0, 150, 400, 800, 1800, 5000, float("inf")]
    evaluation = evaluation.copy()
    evaluation["length_bin"] = pd.cut(
        pd.to_numeric(evaluation["word_count"], errors="coerce"),
        bins=length_bins, right=False,
    ).astype(str)
    report["evaluation_by_length"] = {
        length: _metrics(group, threshold, float(policy["wilson_confidence"]))
        for length, group in evaluation.groupby("length_bin")
    }
    weights = 1 / evaluation["inclusion_probability"].to_numpy(dtype=float)
    assigned_evaluation = _assigned(evaluation, threshold)
    accepted_evaluation = (assigned_evaluation != "incerto").to_numpy()
    correct_evaluation = (assigned_evaluation == evaluation["theme_human"]).to_numpy()
    weighted_accepted = float(weights[accepted_evaluation].sum())
    report["evaluation_corpus_weighted"] = {
        "coverage": weighted_accepted / float(weights.sum()),
        "accepted_precision": (
            float(weights[accepted_evaluation & correct_evaluation].sum()) / weighted_accepted
            if weighted_accepted else None
        ),
        "note": "Point estimates use inverse area/source-stratum sampling weights; no CI is asserted",
    }
    run = context.new_run("evaluate", extra={
        "predictions_sha256": sha256_file(predictions_path),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "primary_annotation_sha256": sha256_file(primary_path),
        "second_annotation_sha256": sha256_file(second_path) if second_path else None,
        "adjudicated_sha256": sha256_file(adjudicated_path) if adjudicated_path else None,
        "baseline_predictions_sha256": (
            sha256_file(baseline_predictions_path) if baseline_predictions_path else None
        ),
    })
    joined["theme_assigned"] = _assigned(joined, threshold)
    joined.loc[joined["sample_role"] == "evaluation"].groupby(
        ["broad_area", "theme_human", "theme_assigned"], dropna=False,
    ).size().reset_index(name="rows").to_csv(
        run.directory / "confusion_evaluation.csv", index=False, encoding="utf-8-sig",
    )
    detail_columns = [
        "record_id", "broad_area", "sample_role", "theme_human", "theme_assigned",
        "mapped_theme", "theme_predicted", "top_probability", "source_name",
    ]
    if baseline_predictions_path is not None:
        detail_columns.append("theme_baseline")
    joined[detail_columns].to_csv(
        run.directory / "human_evaluation_rows.csv", index=False, encoding="utf-8-sig",
    )
    if second_path is not None:
        disagreements.to_csv(run.directory / "annotator_disagreements.csv", index=False,
                             encoding="utf-8-sig")
    run.write_result("evaluation", report)
    threshold_path = run.write_result("threshold", {
        **chosen,
        "predictions_sha256": sha256_file(predictions_path),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "evaluation_run_id": run.run_id,
    })
    return threshold_path


def materialize_themes(
    context: ThemeContext, predictions_path: Path, threshold_path: Path,
) -> Path:
    predictions = _predictions(context, predictions_path)
    validate_artifact_context(context, threshold_path)
    threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
    if threshold_data["predictions_sha256"] != sha256_file(predictions_path):
        raise ValueError("Threshold was calibrated for another prediction artifact")
    threshold = float(threshold_data["threshold"])
    human = predictions.set_index("record_id").to_dict(orient="index")
    corpus_path = context.catalog.resource(context.config["corpus_resource"]).path
    run = context.new_run("materialize", extra={
        "corpus_sha256": sha256_file(corpus_path),
        "predictions_sha256": sha256_file(predictions_path),
        "threshold_sha256": sha256_file(threshold_path),
    })
    output = run.directory / "corpus_with_canonical_theme.parquet"
    partial = run.directory / "corpus_with_canonical_theme.partial.parquet"
    assignments: list[dict[str, Any]] = []
    writer = None
    seen_human: set[str] = set()
    seen_all: set[str] = set()
    loader = DatasetLoader(context.catalog.resource(context.config["corpus_resource"]))
    for batch in loader.iter_batches(
        batch_size=int(context.config.get("materialize_batch_size", context.config["batch_size"]))
    ):
        table = pa.Table.from_batches([batch])
        canonicals: list[str] = []
        originals: list[str | None] = []
        methods: list[str] = []
        probabilities: list[float | None] = []
        model_predictions: list[str | None] = []
        model_runs: list[str | None] = []
        for row in table.select([
            "record_id", "label_name", "broad_area", "theme", "split",
            "benchmark_eligible", "source_name", "generator",
        ]).to_pylist():
            record_id = row["record_id"]
            if record_id in seen_all:
                raise ValueError(f"Duplicate corpus record ID: {record_id}")
            seen_all.add(record_id)
            area = row["broad_area"]
            original = row["theme"]
            if area not in context.taxonomy:
                raise ValueError(f"Unknown broad area: {area}")
            if row["label_name"] == "synthetic":
                if original not in context.taxonomy[area]:
                    raise ValueError(f"Synthetic theme outside taxonomy: {record_id}")
                canonical, method, score, model_run = original, "generation_prompt", None, None
                model_prediction = None
            elif row["label_name"] == "human":
                if record_id not in human:
                    raise ValueError(f"Missing human prediction: {record_id}")
                prediction = human[record_id]
                seen_human.add(record_id)
                predicted_original = prediction["theme_original"]
                same_original = (
                    predicted_original == original
                    if isinstance(predicted_original, str) or isinstance(original, str)
                    else pd.isna(predicted_original) and pd.isna(original)
                )
                if prediction["broad_area"] != area or not same_original:
                    raise ValueError(f"Prediction metadata differs from corpus: {record_id}")
                mapped = prediction["mapped_theme"]
                model_score = float(prediction["top_probability"])
                model_prediction = prediction["theme_predicted"]
                model_run = prediction["model_run_id"]
                if isinstance(mapped, str) and mapped:
                    canonical, method = mapped, prediction["mapping_method"]
                elif model_score >= threshold:
                    canonical, method = prediction["theme_predicted"], "model"
                else:
                    canonical, method = "incerto", "uncertain"
                score = model_score
            else:
                raise ValueError(f"Unknown authorship label: {row['label_name']}")
            originals.append(original)
            canonicals.append(canonical)
            methods.append(method)
            probabilities.append(score)
            model_predictions.append(model_prediction)
            model_runs.append(model_run)
            assignments.append({
                "record_id": record_id, "label_name": row["label_name"],
                "broad_area": area, "split": row["split"],
                "benchmark_eligible": row["benchmark_eligible"],
                "source_name": row["source_name"], "generator": row["generator"],
                "theme_original": original, "theme_canonical": canonical,
                "theme_assignment_method": method,
                "theme_model_prediction": model_prediction,
                "theme_model_top_probability": score, "theme_model_run_id": model_run,
            })
        table = table.set_column(
            table.schema.get_field_index("theme"), "theme",
            pa.array(canonicals, type=pa.large_string()),
        )
        table = table.append_column("theme_original", pa.array(originals, type=pa.large_string()))
        table = table.append_column("theme_canonical", pa.array(canonicals, type=pa.large_string()))
        table = table.append_column("theme_assignment_method", pa.array(methods, type=pa.large_string()))
        table = table.append_column("theme_model_prediction", pa.array(model_predictions, type=pa.large_string()))
        table = table.append_column("theme_model_top_probability", pa.array(probabilities, type=pa.float64()))
        table = table.append_column("theme_model_run_id", pa.array(model_runs, type=pa.large_string()))
        if writer is None:
            writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
        writer.write_table(table)
    if writer is None:
        raise ValueError("Source corpus has no rows")
    writer.close()
    if seen_human != set(human):
        raise ValueError("Predictions include records outside the source corpus")
    expected = context.catalog.resource(context.config["corpus_resource"]).expected_rows
    if expected is not None and len(assignments) != expected:
        raise ValueError(f"Output has {len(assignments)} rows, expected {expected}")
    partial.rename(output)
    frame = pd.DataFrame.from_records(assignments)
    frame.to_parquet(run.directory / "theme_assignments.parquet", index=False)
    keys = ["label_name", "broad_area", "theme_canonical"]

    def counts_with_shares(source_frame: pd.DataFrame) -> pd.DataFrame:
        counts = source_frame.groupby(keys, dropna=False).size().reset_index(name="rows")
        counts["share_within_class_area"] = counts["rows"] / counts.groupby(
            ["label_name", "broad_area"]
        )["rows"].transform("sum")
        return counts

    table = counts_with_shares(frame)
    table.to_csv(run.directory / "theme_counts_full.csv", index=False, encoding="utf-8-sig")
    frame.groupby(
        [*keys, "theme_assignment_method"], dropna=False,
    ).size().reset_index(name="rows").to_csv(
        run.directory / "theme_counts_by_method.csv", index=False, encoding="utf-8-sig",
    )
    comparison = table.pivot_table(
        index=["broad_area", "theme_canonical"], columns="label_name",
        values="rows", aggfunc="sum", fill_value=0,
    ).reset_index()
    comparison.columns.name = None
    for label in ("human", "synthetic"):
        if label not in comparison:
            comparison[label] = 0
        comparison[f"{label}_share_in_area"] = comparison[label] / comparison.groupby(
            "broad_area"
        )[label].transform("sum")
    comparison.rename(columns={"human": "human_rows", "synthetic": "synthetic_rows"}).to_csv(
        run.directory / "theme_comparison_full.csv", index=False, encoding="utf-8-sig",
    )
    eligible = frame.loc[frame["benchmark_eligible"] == True]  # noqa: E712
    counts_with_shares(eligible).to_csv(
        run.directory / "theme_counts_eligible.csv", index=False, encoding="utf-8-sig",
    )
    run.write_result("materialization_summary", {
        "rows": len(frame), "humans": len(human),
        "uncertain_humans": int((frame["theme_assignment_method"] == "uncertain").sum()),
        "threshold": threshold,
        "source_sha256": sha256_file(corpus_path),
        "derived_sha256": sha256_file(output),
        "note": "Eligible rows are not the frozen balanced benchmark",
    })
    return output
