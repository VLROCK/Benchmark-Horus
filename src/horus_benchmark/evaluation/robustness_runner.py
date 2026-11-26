"""Replay frozen IID detectors on clean-linked query-free attack conditions.

One job loads or reconstructs one clean fitted detector and scores all eleven
attack conditions. It never retunes parameters or thresholds on attacked text.
The source task, model bytes, clean predictions and attack manifests are pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from horus_benchmark.evaluation import advanced_runner as advanced
from horus_benchmark.evaluation import benchmark_runner as baseline
from horus_benchmark.evaluation.advanced_data import config, load_release, partition
from horus_benchmark.evaluation.research_io import read_json, sha256, write_json
from horus_benchmark.robustness.prepare import check as check_attacks


IDENTITY = ("family", "model", "seed", "fold", "source_task_id", "source_plan",
            "source_plan_sha256", "config_sha256", "attack_manifest_sha256")
THRESHOLDS = {"youden": "threshold_youden", "fpr_0.01": "threshold_fpr_0.01",
              "fpr_0.05": "threshold_fpr_0.05"}


def task_id(row: dict) -> str:
    value = {key: str(row[key]) for key in IDENTITY}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def create_plan(config_path: Path, advanced_plan: Path, output_dir: Path) -> Path:
    cfg = config(config_path)
    if output_dir.exists():
        raise FileExistsError(f"Robustness plans already exist: {output_dir}")
    attack_dir = Path(cfg["output_root"]) / "attacks"
    check_attacks(attack_dir)
    manifest_path = attack_dir / "attack_manifest.json"
    marker = read_json(advanced_plan.with_suffix(".json"))
    if (marker.get("sha256") != sha256(advanced_plan) or
            marker.get("config_sha256") != sha256(config_path)):
        raise ValueError("Advanced IID execution plan changed")
    adv = pd.read_csv(advanced_plan, keep_default_na=False)
    selected = adv.loc[adv.protocol.eq("iid") & adv.target.eq("all") & adv.fold.eq(0)]
    if (selected.duplicated(["model", "seed", "fold"]).any() or
            set(zip(selected.model, selected.seed)) !=
            {(model, seed) for model in cfg["models_iid"] for seed in cfg["seeds"]} or
            not selected.text_view.eq("native").all()):
        raise ValueError("Expected all advanced native IID fold-zero tasks")
    sources = [("advanced", advanced_plan, selected)]
    for name in ("iid_cpu.csv", "iid_bert.csv"):
        path = Path(cfg["baseline_plans"]) / name
        old = pd.read_csv(path, keep_default_na=False)
        subset = old.loc[old.protocol.eq("iid") & old.target.eq("all") & old.fold.eq(0)]
        sources.append(("baseline", path, subset))
    legacy = pd.concat([entry[2] for entry in sources[1:]], ignore_index=True)
    if (legacy.duplicated(["model", "seed", "fold"]).any() or
            set(zip(legacy.model, legacy.seed)) !=
            {(model, seed) for model in baseline.ALL_MODELS for seed in cfg["seeds"]}):
        raise ValueError("Expected all nine baseline IID fold-zero tasks")
    rows = []
    for family, source_plan, frame in sources:
        for source in frame.to_dict("records"):
            expected = advanced.task_id(source) if family == "advanced" else baseline._task_id(source)
            if expected != source["task_id"]:
                raise ValueError("Original source task identity changed")
            row = {"family": family, "model": source["model"], "seed": int(source["seed"]),
                   "fold": 0, "source_task_id": source["task_id"],
                   "source_plan": str(source_plan), "source_plan_sha256": sha256(source_plan),
                   "config_sha256": sha256(config_path),
                   "attack_manifest_sha256": sha256(manifest_path)}
            row["task_id"] = task_id(row)
            rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = pd.DataFrame(rows)
    outputs = {}
    for kind, is_bert in (("cpu", False), ("bert", True)):
        table = plan.loc[plan.model.eq("bertimbau").eq(is_bert)].reset_index(drop=True)
        path = output_dir / f"{kind}.csv"
        table.to_csv(path, index=False)
        outputs[kind] = {"path": str(path), "rows": len(table), "sha256": sha256(path)}
    target = output_dir / "manifest.json"
    write_json(target, {"kind": "horus_frozen_robustness_plan", "rows": len(rows),
                        "config_sha256": sha256(config_path),
                        "attack_manifest_sha256": sha256(manifest_path), "plans": outputs,
                        "note": "All advanced and original IID models; fold 0; three seeds. "
                                "Each task scores every attack condition using clean thresholds."})
    return target


def _source(cfg: dict, task: dict) -> tuple[dict, dict, pd.DataFrame, Path]:
    source_plan = Path(task["source_plan"])
    if sha256(source_plan) != task["source_plan_sha256"]:
        raise ValueError("Clean source execution plan changed")
    frame = pd.read_csv(source_plan, keep_default_na=False)
    selected = frame.loc[frame.task_id.eq(task["source_task_id"])]
    if len(selected) != 1:
        raise ValueError("Clean source plan lacks a unique task")
    source = selected.iloc[0].to_dict()
    if any(str(source[field]) != str(task[field]) for field in ("model", "seed", "fold")):
        raise ValueError("Clean source model/seed/fold differs from robustness task")
    directory = ((Path(cfg["output_root"]) / "runs") if task["family"] == "advanced"
                 else Path(cfg["baseline_runs"])) / source["task_id"]
    record = read_json(directory / "metrics.json")
    if record.get("task", {}).get("task_id") != source["task_id"]:
        raise ValueError("Clean result identity differs from source plan")
    if task["family"] == "advanced" and record.get("status") != "complete":
        raise ValueError("Advanced clean task must be complete before robustness evaluation")
    predictions_path = directory / "predictions.parquet"
    if sha256(predictions_path) != record["predictions_sha256"]:
        raise ValueError("Clean predictions changed")
    predictions = pd.read_parquet(predictions_path)
    if (predictions.record_id.duplicated().any() or
            set(predictions.role) != {"test_target"}):
        raise ValueError("Invalid clean IID prediction identity or role")
    return source, record, predictions, directory


def _reference(test: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    frame = test[["record_id", "label_name", "probe_group_id"]].merge(
        prior[["record_id", "label_name", "probe_group_id", "score_n2"]],
        on="record_id", how="left", validate="one_to_one", suffixes=("", "_prior"))
    if (set(test.record_id) != set(prior.record_id) or
            not frame.label_name.eq(frame.label_name_prior).all() or
            not frame.probe_group_id.eq(frame.probe_group_id_prior).all()):
        raise ValueError("Clean reference scores and frozen test documents differ")
    return frame[["record_id", "score_n2"]].rename(columns={"score_n2": "score_clean"})


def _joined(test: pd.DataFrame, view: pd.DataFrame, reference: pd.DataFrame,
            scores: np.ndarray, task: dict, thresholds: dict) -> pd.DataFrame:
    metadata = ["record_id", "label_name", "probe_group_id", "broad_area", "source_name", "generator"]
    keep = ["record_id", "parent_record_id", "variant_id", "content_hash", "parent_content_hash",
            "attack", "intensity", "status", "reason", "eligible_positions", "num_edits"]
    output = test[metadata].merge(view[keep], on="record_id", validate="one_to_one").merge(
        reference, on="record_id", validate="one_to_one")
    if not output.record_id.tolist() == test.record_id.tolist():
        raise ValueError("Robustness score order changed")
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(test),):
        raise ValueError("Attack scoring returned an unaligned vector")
    output["score_attacked"] = values
    output["score_status"] = np.where(np.isfinite(values) & np.isfinite(output.score_clean),
                                       "ok", "invalid")
    for column in ("model", "seed", "fold"):
        output[column] = task[column]
    output["protocol"], output["target"] = "iid", "all"
    for name, column in THRESHOLDS.items():
        output[column] = float(thresholds[name])
    return output


def _views(cfg: dict, test: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict, str]:
    directory = Path(cfg["output_root"]) / "attacks"
    manifest_path = directory / "attack_manifest.json"
    manifest = read_json(manifest_path)
    # Full prepare/check is run once before planning. Every task rechecks the
    # pinned artifact bytes and its clean parent identity without rebuilding it.
    views = {}
    inputs = {}
    for name, entry in manifest["artifacts"].items():
        if entry.get("kind") != "attack_texts":
            continue
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Attack text artifact changed: {name}")
        view = pd.read_parquet(path)
        view = view.loc[view.record_id.isin(set(test.record_id))].sort_values("record_id")
        if (view.record_id.duplicated().any() or
                view.record_id.tolist() != test.record_id.tolist() or
                not view.record_id.eq(view.parent_record_id).all() or
                not view.parent_content_hash.reset_index(drop=True).eq(
                    test.content_hash.reset_index(drop=True)).all()):
            raise ValueError(f"Attack parent identity differs from clean test: {name}")
        views[name] = view.reset_index(drop=True)
        inputs[name] = entry["sha256"]
    if len(views) != 11:
        raise ValueError("Expected all eleven approved attack/dose conditions")
    return views, inputs, sha256(manifest_path)


def _score_advanced(task: dict, cfg: dict, partitioned: pd.DataFrame,
                    test: pd.DataFrame, views: dict[str, pd.DataFrame],
                    prior: dict, previous: pd.DataFrame, directory: Path
                    ) -> tuple[dict[str, np.ndarray], dict]:
    expected_inputs = {"config": task["config_sha256"], "model_lock": sha256(cfg["lm_lock"]),
                       "classical_config": sha256(cfg["classical_config"])}
    if any(prior.get("inputs", {}).get(key) != value for key, value in expected_inputs.items()):
        raise ValueError("Clean fitted detector uses another campaign/model configuration")
    model = task["model"]
    clean_frame, clean_inputs = advanced.enrich(partitioned, model, cfg)
    if any(prior.get("inputs", {}).get(key) != value for key, value in clean_inputs.items()):
        raise ValueError("Clean fitted detector cache inputs changed")
    enriched_test = clean_frame.loc[clean_frame.role.eq("test_target")].sort_values("record_id")
    estimator = None
    if model not in advanced.ZERO_SHOT:
        path = directory / "model" / "estimator.joblib"
        if sha256(path) != prior["model_info"]["estimator_sha256"]:
            raise ValueError("Frozen fitted estimator changed")
        estimator = joblib.load(path)
        valid = enriched_test.feature_status.eq("ok").to_numpy()
        replayed = estimator.score_frame(enriched_test.loc[valid])
        expected = enriched_test.loc[valid, ["record_id"]].merge(
            previous[["record_id", "score_n2"]], on="record_id", validate="one_to_one").score_n2
        if not np.allclose(replayed, expected.to_numpy(), rtol=1e-6, atol=1e-6):
            raise ValueError("Frozen estimator does not reproduce clean scores")
    output, inputs = {}, {"clean_cache": clean_inputs}
    for name, view in views.items():
        attacked = test.drop(columns=["content", "content_hash"]).merge(
            view[["record_id", "content", "content_hash"]], on="record_id", validate="one_to_one")
        enriched, entry = advanced.enrich(
            attacked, model, cfg,
            lm_root=Path(cfg["output_root"]) / "cache" / "attacks" / name,
            pos_root=Path(cfg["output_root"]) / "linguistic" / "attacks" / name)
        if enriched.record_id.tolist() != test.record_id.tolist():
            raise ValueError("Enriched attack text order changed")
        valid = enriched.feature_status.eq("ok").to_numpy()
        scores = np.full(len(enriched), np.nan)
        if valid.any():
            if model in advanced.ZERO_SHOT:
                scores[valid] = enriched.loc[valid, advanced.ZERO_SHOT[model][1]].to_numpy(dtype=float)
            else:
                scores[valid] = estimator.score_frame(enriched.loc[valid])
        output[name], inputs[name] = scores, entry
    return output, inputs


def _score_baseline(task: dict, cfg: dict, partitioned: pd.DataFrame,
                    test: pd.DataFrame, views: dict[str, pd.DataFrame],
                    prior: dict, previous: pd.DataFrame, directory: Path
                    ) -> tuple[dict[str, np.ndarray], dict]:
    original_config = baseline._config(Path(cfg["baseline_config"]))
    seed = int(hashlib.sha256(
        f"{task['model']}|{task['seed']}|{task['fold']}".encode()).hexdigest()[:8], 16)
    if prior["model_seed"] != seed:
        raise ValueError("Original fitted model seed differs")
    model_config = Path(original_config["bertimbau_config"] if task["model"] == "bertimbau"
                        else original_config["text_model_config"])
    if (prior["input_hashes"]["evaluation_config"] != sha256(cfg["baseline_config"]) or
            prior["input_hashes"]["model_config"] != sha256(model_config)):
        raise ValueError("Original model configuration changed")
    if task["model"] == "bertimbau":
        from horus_benchmark.detectors.bertimbau_baseline import score_saved_checkpoint
        sample = pd.concat([test.loc[test.label_name.eq(label)].head(128)
                            for label in ("human", "synthetic")])
        expected = sample[["record_id"]].merge(previous[["record_id", "score_n2"]],
                                                on="record_id", validate="one_to_one")
        result = score_saved_checkpoint(
            directory / "model", {"clean_check": sample.content.tolist(),
                                  **{name: view.content.tolist() for name, view in views.items()}},
            yaml.safe_load(model_config.read_text(encoding="utf-8")),
            expected_sha256=prior["model_info"]["checkpoint_sha256"])
        check = result["scores"].pop("clean_check")
        if not np.allclose(check, expected.score_n2, rtol=1e-4, atol=1e-4):
            raise ValueError("Frozen BERT checkpoint does not reproduce clean scores")
        return result["scores"], {"checkpoint_sha256": result["checkpoint_sha256"],
                                   "token_coverage": result["token_coverage"],
                                   "clean_replay_rows": len(check)}
    from horus_benchmark.detectors.text_baselines import build_estimator, load_config, synthetic_scores
    estimator = build_estimator(task["model"], c=float(prior["model_info"].get("selected_c") or 1),
                                 seed=seed, config=load_config(model_config))
    train = partitioned.loc[partitioned.role.eq("train")].sort_values("record_id")
    estimator.fit(train.content.tolist(), baseline._labels(train))
    check = synthetic_scores(estimator, test.content.tolist())
    expected = test[["record_id"]].merge(previous[["record_id", "score_n2"]],
                                         on="record_id", validate="one_to_one")
    if not np.allclose(check, expected.score_n2, rtol=1e-6, atol=1e-6):
        raise ValueError("Reconstructed original CPU model does not reproduce clean scores")
    return ({name: synthetic_scores(estimator, view.content.tolist())
             for name, view in views.items()}, {"clean_replay_rows": len(check),
                                               "clean_max_abs_delta": float(np.max(
                                                   np.abs(check - expected.score_n2.to_numpy())))})


def run_task(config_path: Path, plan_path: Path, index: int, output: Path) -> Path:
    cfg = config(config_path)
    marker = read_json(plan_path.parent / "manifest.json")
    matching = [entry for entry in marker["plans"].values() if Path(entry["path"]) == plan_path]
    if (len(matching) != 1 or matching[0]["sha256"] != sha256(plan_path) or
            marker["config_sha256"] != sha256(config_path)):
        raise ValueError("Robustness plan/configuration changed")
    plan = pd.read_csv(plan_path, keep_default_na=False)
    if not 0 <= index < len(plan):
        raise IndexError("Robustness index outside plan")
    task = plan.iloc[index].to_dict()
    if task_id(task) != task["task_id"] or int(task["fold"]) != 0:
        raise ValueError("Invalid robustness task identity or fold")
    target = output / task["task_id"]
    if (target / "metrics.json").exists():
        saved = read_json(target / 'metrics.json')
        if (saved.get('task') != task or saved.get('status') != 'complete' or
                sha256(target / 'predictions.parquet') != saved['predictions_sha256']):
            raise ValueError('Existing robustness result differs from frozen task')
        return target / 'metrics.json'
    source, prior, previous, source_dir = _source(cfg, task)
    texts, _ = load_release(cfg)
    partitioned = partition(cfg, texts, source)
    test = partitioned.loc[partitioned.role.eq("test_target")].sort_values("record_id").reset_index(drop=True)
    reference = _reference(test, previous)
    views, artifact_inputs, manifest_sha = _views(cfg, test)
    if manifest_sha != task["attack_manifest_sha256"]:
        raise ValueError("Attack manifest changed after planning")
    if task["family"] == "advanced":
        scores, model_inputs = _score_advanced(task, cfg, partitioned, test, views,
                                               prior, previous, source_dir)
    elif task["family"] == "baseline":
        scores, model_inputs = _score_baseline(task, cfg, partitioned, test, views,
                                               prior, previous, source_dir)
    else:
        raise ValueError("Unknown robustness model family")
    thresholds = {key: float(value) for key, value in prior["evaluation"]["validation_thresholds"].items()}
    predictions = pd.concat([_joined(test, view, reference, scores[name], task, thresholds)
                             for name, view in views.items()], ignore_index=True)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "predictions.parquet"
    predictions.to_parquet(path, index=False, compression="zstd")
    result = {"kind": "horus_frozen_robustness_result", "status": "complete", "task": task,
              "clean_metrics_sha256": sha256(source_dir / "metrics.json"),
              "clean_predictions_sha256": prior["predictions_sha256"],
              "model_inputs": model_inputs, "attack_artifacts": artifact_inputs,
              "predictions_sha256": sha256(path), "rows": len(predictions),
              "conditions": len(views), "test_parents": len(test),
              "invalid_score_rows": int(predictions.score_status.ne("ok").sum()),
              "thresholds": thresholds}
    write_json(target / "metrics.json", result)
    return target / "metrics.json"


def check(plan_dir: Path, output: Path) -> dict:
    marker = read_json(plan_dir / "manifest.json")
    counts, missing = {}, []
    for name, entry in marker["plans"].items():
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Robustness execution plan changed")
        plan = pd.read_csv(path, keep_default_na=False)
        complete = 0
        for index, task in enumerate(plan.to_dict("records")):
            folder = output / task["task_id"]
            if not (folder / "metrics.json").exists():
                missing.append({"plan": str(path), "index": index})
                continue
            record = read_json(folder / "metrics.json")
            if (record.get("task", {}).get("task_id") != task["task_id"] or
                    sha256(folder / "predictions.parquet") != record["predictions_sha256"]):
                raise ValueError("Robustness result identity or prediction hash mismatch")
            complete += 1
        counts[name] = {"complete": complete, "planned": len(plan)}
    return {"complete": not missing, "counts": counts, "missing": missing}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("plan")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--advanced-plan", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = sub.add_parser("run-task")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--index", type=int, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("check")
    command.add_argument("--plan-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(create_plan(args.config, args.advanced_plan, args.output_dir))
    elif args.command == "run-task":
        print(run_task(args.config, args.plan, args.index, args.output))
    else:
        result = check(args.plan_dir, args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["complete"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
