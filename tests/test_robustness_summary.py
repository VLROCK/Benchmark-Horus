"""Paired robustness tests on scores, not detector fits or benchmark text."""

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.robustness.summarize import error_agreement, summarize_group


def _scores() -> pd.DataFrame:
    parts = []
    for fold in (0, 1):
        parts.append(pd.DataFrame({
            "parent_record_id": [f"id-{i}" for i in range(8)],
            "probe_group_id": [f"group-{i}" for i in range(8)],
            "label_name": ["human"] * 4 + ["synthetic"] * 4,
            "model": "detector", "protocol": "iid", "target": "all",
            "seed": 20261105, "fold": fold, "attack": "upper_lower", "intensity": .05,
            "score_clean": [.1, .2, .3, .4, .6, .7, .8, .9],
            "score_attacked": [.1, .2, .3, .4, .6, .7, .8, .9],
            "threshold_youden": .5, "threshold_fpr_0.01": .8,
            "threshold_fpr_0.05": .7, "status": "changed",
        }))
    return pd.concat(parts, ignore_index=True)


def test_unchanged_scores_have_zero_paired_delta_and_flip_rate() -> None:
    metrics, effects, coverage = summarize_group(_scores(), replicates=17, batch_size=4)
    assert coverage["valid_common_rows"] == 8
    assert coverage["folds"] == 2
    assert len(metrics) == 3 * 8
    assert all(row["delta_attacked_minus_clean"] == pytest.approx(0) for row in metrics)
    assert all(row["delta_ci_low"] == pytest.approx(0) and
               row["delta_ci_high"] == pytest.approx(0) for row in metrics)
    assert all(row["value"] == pytest.approx(0) for row in effects)


def test_synthetic_only_and_human_only_distinguish_evasion_from_false_positives() -> None:
    frame = _scores()
    frame["score_attacked"] = 1 - frame.score_clean
    metrics, effects, _ = summarize_group(frame, replicates=12)
    table = pd.DataFrame(metrics)
    synth = table.loc[table.condition.eq("synthetic_only") & table.metric.eq("balanced_accuracy")].iloc[0]
    human = table.loc[table.condition.eq("human_only") & table.metric.eq("balanced_accuracy")].iloc[0]
    assert synth.attacked_value == pytest.approx(.5)
    assert human.attacked_value == pytest.approx(.5)
    effects = pd.DataFrame(effects)
    evasion = effects.loc[effects.condition.eq("synthetic_only") &
                          effects.effect.eq("evasion_given_clean_detected_youden")].iloc[0]
    assert evasion.value == pytest.approx(1)
    false_positive = effects.loc[effects.condition.eq("human_only") &
                                effects.effect.eq("new_false_positive_rate_youden")].iloc[0]
    assert false_positive.value == pytest.approx(1)


def test_no_ops_stay_in_primary_population_and_changed_only_is_labeled() -> None:
    frame = _scores()
    frame.loc[frame.parent_record_id.isin({"id-0", "id-4"}), "status"] = "no_op"
    _, _, primary = summarize_group(frame, replicates=5)
    _, _, changed = summarize_group(frame, replicates=5, population="changed_only")
    assert primary["valid_common_rows"] == 8
    assert primary["no_op_rows"] == 2
    assert changed["valid_common_rows"] == 6
    frame.loc[frame.parent_record_id.eq("id-1") & frame.fold.eq(1), "score_attacked"] = np.nan
    _, _, common = summarize_group(frame, replicates=5)
    assert common["valid_common_rows"] == 7
    assert common["nonfinite_score_rows"] == 1


def test_summary_refuses_changed_parent_lineage_or_thresholds() -> None:
    frame = _scores()
    frame.loc[frame.fold.eq(1) & frame.parent_record_id.eq("id-0"), "probe_group_id"] = "wrong"
    with pytest.raises(ValueError, match="Fold source IDs"):
        summarize_group(frame, replicates=2)
    frame = _scores()
    frame.loc[0, "threshold_youden"] = .6
    with pytest.raises(ValueError, match="one frozen clean threshold"):
        summarize_group(frame, replicates=2)


def test_detector_error_agreement_uses_common_documents() -> None:
    first = _scores().query("fold == 0").copy()
    other = first.copy()
    other["model"] = "second"
    other["score_attacked"] = 1 - other.score_clean
    agreement = error_agreement(pd.concat([first, other], ignore_index=True))
    assert len(agreement) == 2
    clean = agreement.loc[agreement.view.eq("score_clean")].iloc[0]
    changed = agreement.loc[agreement.view.eq("score_attacked")].iloc[0]
    assert clean.both_correct == 8
    assert changed.only_b_wrong == 8
    assert changed.common_rows == 8


@pytest.mark.parametrize('empty_metrics', [False, True])
def test_merge_preserves_coverage_and_checks_every_shard_identity(tmp_path, monkeypatch,
                                                                 empty_metrics) -> None:
    from horus_benchmark.robustness import merge as merger
    from horus_benchmark.evaluation.research_io import read_json, sha256, write_json
    plans, runs, shards = (tmp_path / name for name in ('plans', 'runs', 'shards'))
    plans.mkdir()
    write_json(plans / 'manifest.json', {'fixture': True})
    monkeypatch.setattr(merger, 'check', lambda *_args: {'complete': True})
    for task_id, kind, model in [('cpu-task', 'cpu', 'first'), ('bert-task', 'bert', 'second')]:
        pd.DataFrame([{'task_id': task_id, 'model': model, 'seed': 20261105}]).to_csv(
            plans / f'{kind}.csv', index=False)
        scores = _scores().query('fold == 0').copy()
        scores['model'] = model
        score_path = runs / task_id / 'predictions.parquet'
        score_path.parent.mkdir(parents=True)
        scores.to_parquet(score_path, index=False)
        directory = shards / task_id
        directory.mkdir(parents=True)
        metrics = pd.DataFrame() if empty_metrics else pd.DataFrame([{
            'model': model, 'seed': 20261105, 'population': 'all_valid',
            'condition': 'both', 'metric': 'roc_auc', 'attack': 'upper_lower',
            'intensity': .05, 'delta_attacked_minus_clean': 0.0}])
        frames = {'paired_metrics.csv': metrics, 'paired_effects.csv': pd.DataFrame(),
                  'coverage.csv': pd.DataFrame([{'model': model, 'seed': 20261105,
                                                'valid_common_rows': 0 if empty_metrics else 8}])}
        artifacts = {}
        for name, frame in frames.items():
            frame.to_csv(directory / name, index=False)
            artifacts[name] = {'sha256': sha256(directory / name)}
        write_json(directory / 'summary.json', {'bootstrap_replicates': 2000,
                   'inputs': [{'sha256': sha256(score_path)}], 'artifacts': artifacts})
    target = merger.merge(plans, runs, shards, tmp_path / 'output')
    assert read_json(target)['tasks'] == 2
    assert len(pd.read_csv(target.parent / 'coverage.csv')) == 2
    assert len(pd.read_csv(target.parent / 'detector_error_agreement.csv')) == 2
    (shards / 'cpu-task' / 'coverage.csv').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='summary table changed'):
        merger.merge(plans, runs, shards, tmp_path / 'other-output')
