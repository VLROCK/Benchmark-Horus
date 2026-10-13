"""Six independent synthetic-trained BERT theme models and human inference."""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
from sklearn.metrics import accuracy_score, f1_score

from horus_benchmark.audit.theme_common import (
    ThemeContext, sha256_file,
)
from horus_benchmark.data import DatasetLoader


def _generator_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["generator"] or "<missing>") for row in rows).items()))


def train_baselines(context: ThemeContext) -> Path:
    """Diagnostic TF-IDF reference; its models never label the corpus."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    cfg = context.config
    train_spec = context.catalog.resource(cfg["train_resource"])
    validation_spec = context.catalog.resource(cfg["validation_resource"])
    run = context.new_run("baseline", extra={
        "train_sha256": sha256_file(train_spec.path),
        "validation_sha256": sha256_file(validation_spec.path),
    })
    train_loader = DatasetLoader(train_spec)
    validation_loader = DatasetLoader(validation_spec)
    heldout = cfg["model"].get("heldout_generator")
    summary = {}
    for area, themes in context.taxonomy.items():
        train_rows = _synthetic_records(
            train_loader, area, themes, exclude_generator=heldout,
        )
        validation_rows = _synthetic_records(
            validation_loader, area, themes, exclude_generator=heldout,
        )
        holdout_rows = _synthetic_records(
            validation_loader, area, themes, generator=heldout,
        ) if heldout else []
        if set(themes) - {row["theme"] for row in train_rows}:
            raise ValueError(f"Baseline lacks a training theme in {area}")
        options = cfg["baseline"]
        pipeline = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=int(options["min_df"]),
                max_features=int(options["max_features"]), sublinear_tf=True,
            ),
            LogisticRegression(C=float(options["c"]), max_iter=1000),
        )
        pipeline.fit([row["content"] for row in train_rows],
                     [row["theme"] for row in train_rows])

        def scores(rows):
            if not rows:
                return {"rows": 0, "accuracy": None, "macro_f1": None}
            truth = [row["theme"] for row in rows]
            guess = pipeline.predict([row["content"] for row in rows])
            return {
                "rows": len(rows),
                "accuracy": float(accuracy_score(truth, guess)),
                "macro_f1": float(f1_score(
                    truth, guess, labels=list(themes), average="macro", zero_division=0,
                )),
            }

        summary[area] = {
            "train_rows": len(train_rows),
            "train_by_generator": _generator_counts(train_rows),
            "validation_by_generator": _generator_counts(validation_rows),
            "validation": scores(validation_rows),
            "heldout_generator": heldout,
            "heldout_validation": scores(holdout_rows),
        }
        joblib.dump(pipeline, run.directory / f"tfidf_{area}.joblib")
    run.write_result("baseline_summary", summary)
    return run.directory


def predict_baselines(context: ThemeContext, baseline_run: Path) -> Path:
    import joblib
    import pandas as pd

    baseline_run = baseline_run.expanduser().resolve()
    metadata = json.loads((baseline_run / "run.json").read_text(encoding="utf-8"))
    if metadata["dataset_version"] != context.catalog.dataset_version:
        raise ValueError("Baseline was trained for another dataset version")
    if metadata["extra"].get("taxonomy_sha256") != sha256_file(context.taxonomy_path):
        raise ValueError("Baseline taxonomy differs")
    for key, resource in (("train_sha256", "train_resource"),
                          ("validation_sha256", "validation_resource")):
        if metadata["extra"].get(key) != sha256_file(
            context.catalog.resource(context.config[resource]).path
        ):
            raise ValueError(f"Baseline source {resource} differs")
    corpus_spec = context.catalog.resource(context.config["corpus_resource"])
    run = context.new_run("baseline-predict", extra={
        "corpus_sha256": sha256_file(corpus_spec.path),
        "baseline_run_id": metadata["run_id"],
    })
    loader = DatasetLoader(corpus_spec)
    results = []
    for area in context.taxonomy:
        pipeline = joblib.load(baseline_run / f"tfidf_{area}.joblib")
        condition = (ds.field("label_name") == "human") & (ds.field("broad_area") == area)
        for batch in loader.iter_batches(
            columns=["record_id", "content"], filter_expression=condition,
            batch_size=int(context.config["batch_size"]),
        ):
            rows = batch.to_pylist()
            texts = [row["content"] or "" for row in rows]
            predicted = pipeline.predict(texts)
            probabilities = pipeline.predict_proba(texts).max(axis=1)
            results.extend({
                "record_id": row["record_id"], "broad_area": area,
                "theme_baseline": str(theme),
                "baseline_probability": float(score),
            } for row, theme, score in zip(rows, predicted, probabilities))
    frame = pd.DataFrame.from_records(results)
    if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise ValueError("Baseline predictions have missing or duplicate IDs")
    expected = loader.count_rows(filter_expression=ds.field("label_name") == "human")
    if len(frame) != expected:
        raise ValueError(f"Baseline predicted {len(frame)} humans, expected {expected}")
    output = run.directory / "human_baseline_predictions.parquet"
    frame.to_parquet(output, index=False)
    return output


def _device(config: dict[str, Any], torch):
    requested = config["device"]
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _synthetic_records(
    loader: DatasetLoader, area: str, themes: tuple[str, ...],
    *, generator: str | None = None, exclude_generator: str | None = None,
    batch_size: int = 2048,
) -> list[dict[str, str]]:
    condition = (ds.field("label_name") == "synthetic") & (ds.field("broad_area") == area)
    records: list[dict[str, str]] = []
    valid = set(themes)
    for batch in loader.iter_batches(
        columns=["record_id", "content", "theme", "generator"],
        filter_expression=condition, batch_size=batch_size,
    ):
        for row in batch.to_pylist():
            if generator is not None and row["generator"] != generator:
                continue
            if exclude_generator is not None and row["generator"] == exclude_generator:
                continue
            if row["theme"] not in valid:
                raise ValueError(f"Synthetic theme outside taxonomy: {area} / {row['theme']}")
            if not isinstance(row["content"], str) or not row["content"].strip():
                raise ValueError(f"Empty synthetic training text: {row['record_id']}")
            records.append(row)
    return records


def _loader(records, tokenizer, themes, batch_size, shuffle, max_length, torch):
    from torch.utils.data import DataLoader

    label_to_id = {theme: i for i, theme in enumerate(themes)}

    def collate(rows):
        encoded = tokenizer(
            [row["content"] for row in rows], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [label_to_id[row["theme"]] for row in rows], dtype=torch.long,
        )
        return encoded

    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def _evaluate(model, loader, device, torch, themes):
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if not targets:
        return {"rows": 0, "accuracy": None, "macro_f1": None}
    return {
        "rows": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(
            targets, predictions, labels=list(range(len(themes))),
            average="macro", zero_division=0,
        )),
    }


def train_models(context: ThemeContext) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = context.config
    model_cfg = cfg["model"]
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_spec = context.catalog.resource(cfg["train_resource"])
    validation_spec = context.catalog.resource(cfg["validation_resource"])
    run = context.new_run("train", extra={
        "train_sha256": sha256_file(train_spec.path),
        "validation_sha256": sha256_file(validation_spec.path),
        "model_name": model_cfg["name"],
        "model_revision_requested": model_cfg["revision"],
    })
    train_loader = DatasetLoader(train_spec)
    validation_loader = DatasetLoader(validation_spec)
    device = _device(model_cfg, torch)
    heldout = model_cfg.get("heldout_generator")
    summary: dict[str, Any] = {}
    for area, themes in context.taxonomy.items():
        area_started = time.perf_counter()
        train_rows = _synthetic_records(
            train_loader, area, themes, exclude_generator=heldout,
        )
        validation_rows = _synthetic_records(
            validation_loader, area, themes, exclude_generator=heldout,
        )
        heldout_rows = _synthetic_records(
            validation_loader, area, themes, generator=heldout,
        ) if heldout else []
        missing = set(themes) - {row["theme"] for row in train_rows}
        if missing or not validation_rows:
            raise ValueError(f"Insufficient synthetic support in {area}: {sorted(missing)}")
        print(
            f"[theme/train] {area}: {len(train_rows)} treino, "
            f"{len(validation_rows)} validação, {len(themes)} temas",
            file=sys.stderr,
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"], revision=model_cfg["revision"],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["name"], revision=model_cfg["revision"],
            num_labels=len(themes),
            id2label={i: theme for i, theme in enumerate(themes)},
            label2id={theme: i for i, theme in enumerate(themes)},
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(model_cfg["learning_rate"]),
            weight_decay=float(model_cfg["weight_decay"]),
        )
        max_length = int(model_cfg["max_length"])
        if max_length > tokenizer.model_max_length:
            raise ValueError(f"max_length exceeds tokenizer limit in {area}")
        train_batches = _loader(
            train_rows, tokenizer, themes, int(model_cfg["train_batch_size"]),
            True, max_length, torch,
        )
        val_batches = _loader(
            validation_rows, tokenizer, themes, int(model_cfg["inference_batch_size"]),
            False, max_length, torch,
        )
        area_dir = run.directory / "models" / area
        area_dir.mkdir(parents=True)
        best_score = -1.0
        stale = 0
        epochs: list[dict[str, Any]] = []
        for epoch in range(1, int(model_cfg["epochs"]) + 1):
            epoch_started = time.perf_counter()
            model.train()
            losses = []
            for batch in train_batches:
                optimizer.zero_grad(set_to_none=True)
                output = model(**{key: value.to(device) for key, value in batch.items()})
                output.loss.backward()
                optimizer.step()
                losses.append(float(output.loss.detach().cpu()))
            val_metrics = _evaluate(model, val_batches, device, torch, themes)
            epochs.append({
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "elapsed_seconds": round(time.perf_counter() - epoch_started, 1),
                **val_metrics,
            })
            print(
                f"[theme/train] {area}: época {epoch}, "
                f"macro-F1 validação={val_metrics['macro_f1']:.4f}, "
                f"tempo={epochs[-1]['elapsed_seconds']:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            if val_metrics["macro_f1"] > best_score:
                best_score = val_metrics["macro_f1"]
                model.save_pretrained(area_dir, safe_serialization=True)
                tokenizer.save_pretrained(area_dir)
                stale = 0
            else:
                stale += 1
                if stale >= int(model_cfg["patience"]):
                    break
        best_model = AutoModelForSequenceClassification.from_pretrained(area_dir).to(device)
        if heldout_rows:
            hold_loader = _loader(
                heldout_rows, tokenizer, themes, int(model_cfg["inference_batch_size"]),
                False, max_length, torch,
            )
            ood = _evaluate(best_model, hold_loader, device, torch, themes)
        else:
            ood = {"rows": 0, "accuracy": None, "macro_f1": None}
        summary[area] = {
            "themes": list(themes), "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_by_generator": _generator_counts(train_rows),
            "validation_by_generator": _generator_counts(validation_rows),
            "heldout_generator": heldout, "heldout_validation": ood,
            "best_validation_macro_f1": best_score, "epochs": epochs,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "max_length": max_length,
            "elapsed_seconds": round(time.perf_counter() - area_started, 1),
        }
        del model, best_model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    run.write_result("training_summary", summary)
    return run.directory


def predict_humans(context: ThemeContext, model_run: Path) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_run = model_run.expanduser().resolve()
    train_meta = json.loads((model_run / "run.json").read_text(encoding="utf-8"))
    if train_meta["dataset_version"] != context.catalog.dataset_version:
        raise ValueError("Model was trained for another dataset version")
    if train_meta["extra"]["taxonomy_sha256"] != sha256_file(context.taxonomy_path):
        raise ValueError("Model taxonomy differs from current taxonomy")
    for key, resource in (("train_sha256", "train_resource"),
                          ("validation_sha256", "validation_resource")):
        current_hash = sha256_file(context.catalog.resource(context.config[resource]).path)
        if train_meta["extra"].get(key) != current_hash:
            raise ValueError(f"Model source {resource} differs from current dataset")
    corpus_spec = context.catalog.resource(context.config["corpus_resource"])
    run = context.new_run("predict", extra={
        "corpus_sha256": sha256_file(corpus_spec.path),
        "model_run_id": train_meta["run_id"],
    })
    device = _device(context.config["model"], torch)
    result: list[dict[str, Any]] = []
    loader = DatasetLoader(corpus_spec)
    for area, themes in context.taxonomy.items():
        area_started = time.perf_counter()
        before_area = len(result)
        area_dir = model_run / "models" / area
        tokenizer = AutoTokenizer.from_pretrained(area_dir)
        model = AutoModelForSequenceClassification.from_pretrained(area_dir).to(device)
        if tuple(model.config.id2label[i] for i in range(len(themes))) != themes:
            raise ValueError(f"Model label order differs from taxonomy: {area}")
        model.eval()
        condition = (ds.field("label_name") == "human") & (ds.field("broad_area") == area)
        for batch in loader.iter_batches(
            columns=["record_id", "source_name", "split", "benchmark_eligible",
                     "word_count", "theme", "content"],
            filter_expression=condition, batch_size=int(context.config["batch_size"]),
        ):
            rows = batch.to_pylist()
            step = int(context.config["model"]["inference_batch_size"])
            for start in range(0, len(rows), step):
                subset = rows[start:start + step]
                texts = [row["content"] or "" for row in subset]
                encoded = tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=int(context.config["model"]["max_length"]),
                    return_tensors="pt",
                )
                with torch.no_grad():
                    logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
                    scores = torch.softmax(logits, dim=1).cpu().numpy()
                for row, distribution in zip(subset, scores):
                    rank = np.argsort(distribution)[::-1]
                    mapped, method = context.canonical_existing(area, row["theme"])
                    result.append({
                        "record_id": row["record_id"], "label_name": "human",
                        "broad_area": area, "source_name": row["source_name"],
                        "split": row["split"],
                        "benchmark_eligible": row["benchmark_eligible"],
                        "word_count": row["word_count"],
                        "theme_original": row["theme"],
                        "mapped_theme": mapped, "mapping_method": method,
                        "theme_predicted": themes[int(rank[0])],
                        "top_probability": float(distribution[rank[0]]),
                        "probability_margin": float(distribution[rank[0]] - distribution[rank[1]]),
                        "model_run_id": train_meta["run_id"],
                    })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[theme/predict] {area}: {len(result) - before_area} humanos, "
            f"tempo={time.perf_counter() - area_started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    import pandas as pd

    frame = pd.DataFrame.from_records(result)
    if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise ValueError("Human predictions have missing or duplicate record IDs")
    expected = loader.count_rows(filter_expression=ds.field("label_name") == "human")
    if len(frame) != expected:
        raise ValueError(f"Predicted {len(frame)} humans, expected {expected}")
    output = run.directory / "human_predictions.parquet"
    frame.to_parquet(output, index=False)
    return output
