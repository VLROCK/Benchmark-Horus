"""Small campaign integrity checks; no corpus, model download or scheduler."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from horus_benchmark.evaluation.advanced_campaign import pending, worker
from horus_benchmark.evaluation.advanced_runner import merge_features, task_id
from horus_benchmark.evaluation.research_io import read_json, sha256, write_json


def test_array_chunks_use_local_indices_and_skip_only_pinned_successes(tmp_path):
    plan = tmp_path / 'work.csv'
    pd.DataFrame({'index': range(805), 'module': 'bad', 'argv_json': '[]'}).to_csv(plan, index=False)
    write_json(plan.with_suffix('.json'), {'kind': 'horus_abaporu_job_list', 'sha256': sha256(plan)})
    rows = pending(plan)
    assert [offset for offset, _ in rows] == [0, 400, 800]
    assert rows[-1][1] == [0, 1, 2, 3, 4]
    done = tmp_path / 'completed' / 'work' / '00800.json'
    write_json(done, {'plan_sha256': sha256(plan), 'returncode': 0})
    assert pending(plan)[-1] == (800, [1, 2, 3, 4])
    write_json(done, {'plan_sha256': sha256(plan), 'returncode': 1})
    with pytest.raises(ValueError, match='not a successful command'):
        pending(plan)
    write_json(done, {'plan_sha256': 'different'})
    with pytest.raises(ValueError, match='another job plan'):
        pending(plan)


def test_worker_never_executes_unknown_module_or_shell(tmp_path):
    plan = tmp_path / 'work.csv'
    pd.DataFrame([{'index': 0, 'module': 'os', 'argv_json': '["rm"]'}]).to_csv(plan, index=False)
    write_json(plan.with_suffix('.json'), {'kind': 'horus_abaporu_job_list', 'sha256': sha256(plan)})
    with pytest.raises(ValueError, match='Unknown campaign worker'):
        worker(plan, 0)


def test_feature_join_rejects_old_probabilities_for_changed_text(tmp_path):
    features = pd.DataFrame({'record_id': ['h', 's'], 'content_hash': ['old', 'right'],
                             'status': ['ok', 'ok'], 'value': [1., 2.]})
    path = tmp_path / 'features.parquet'
    features.to_parquet(path, index=False)
    texts = pd.DataFrame({'record_id': ['s', 'h'], 'content_hash': ['right', 'changed']})
    with pytest.raises(ValueError, match='text identity'):
        merge_features(texts, path, sha256(path), ['value'])
    texts.loc[texts.record_id.eq('h'), 'content_hash'] = 'old'
    result = merge_features(texts, path, sha256(path), ['value'])
    assert result.value.tolist() == [2., 1.]


def test_task_identity_includes_model_parameters_and_source_area():
    row = {key: 'x' for key in ('model', 'protocol', 'target', 'source_area', 'seed',
           'fold', 'manifest_path', 'manifest_sha256', 'config_sha256', 'text_view',
           'text_view_sha256', 'classical_config_sha256', 'lm_lock_sha256', 'implementation_sha256')}
    assert task_id(row) != task_id({**row, 'lm_lock_sha256': 'new'})
    assert task_id(row) != task_id({**row, 'source_area': 'other'})


def test_infinite_thresholds_are_serialized_without_changing_decisions(tmp_path):
    path = tmp_path / 'metrics.json'
    write_json(path, {'thresholds': {'youden': float('inf')}})
    assert float(read_json(path)['thresholds']['youden']) == float('inf')
    assert 'Infinity' not in path.read_text()


def test_download_never_overwrites_frozen_stanza_resources(tmp_path, monkeypatch):
    from horus_benchmark.evaluation import linguistic_cache
    resource = tmp_path / 'resources.json'
    resource.write_text('{"fixed": true}', encoding='utf-8')
    lock = tmp_path / 'horus_pt_resources.lock.json'
    write_json(lock, {'stanza': 'fixture', 'artifacts': {'resources.json': sha256(resource)}})
    monkeypatch.setattr(linguistic_cache.importlib.metadata, 'version', lambda _name: 'fixture')
    assert linguistic_cache.download(tmp_path) == lock
    resource.write_text('{"changed": true}', encoding='utf-8')
    with pytest.raises(ValueError, match='Stanza resource changed'):
        linguistic_cache.download(tmp_path)


def test_common_prefix_is_literal_reusable_and_rejects_corruption(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from horus_benchmark.evaluation.advanced_data import common_prefix
    class Tokenizer:
        def __call__(self, text, **kwargs):
            end = min(len(text), kwargs['max_length']) if kwargs.get('truncation') else len(text)
            return {'input_ids': list(range(end)),
                    'offset_mapping': [(i, i + 1) for i in range(end)]}
    fake = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Tokenizer())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=fake))
    texts = tmp_path / 'texts.parquet'
    pd.DataFrame({'record_id': ['long', 'short'], 'content': ['a' * 600, 'frase.']}).to_parquet(
        texts, index=False)
    lock = tmp_path / 'lock.json'
    write_json(lock, {'models': {'fixture': {'snapshot_path': 'fixture'}}})
    target = tmp_path / 'common.parquet'
    assert common_prefix(texts, lock, target) == target
    derived = pd.read_parquet(target)
    assert derived.content.tolist() == ['a' * 512, 'frase.']
    assert common_prefix(texts, lock, target) == target
    derived.loc[0, 'content'] = 'changed'
    derived.to_parquet(target, index=False)
    with pytest.raises(ValueError, match='artifact identity'):
        common_prefix(texts, lock, target)


def test_tiny_plan_fit_resume_and_model_config_guard(tmp_path, monkeypatch):
    from horus_benchmark.evaluation import advanced_runner as runner
    from horus_benchmark.metrics.binary_detection import evaluate_binary as real_evaluate
    rows = []
    split_rows = []
    for label, word in [('human', 'conversa'), ('synthetic', 'metodologia')]:
        for index in range(18):
            rid = f'{label}-{index:02d}'
            text = f'{word} {word} texto exemplo {index}.'
            rows.append({'record_id': rid, 'content': text, 'content_hash': runner.hashlib.sha256(text.encode()).hexdigest(),
                         'probe_group_id': rid, 'label_name': label, 'broad_area': 'social',
                         'source_name': 'source', 'generator': label, 'word_count_audit': 6})
            split_rows.append({'record_id': rid, 'probe_group_id': rid, 'label_name': label,
                               'split': 'development' if index < 15 else 'test',
                               'fold': index % 5 if index < 15 else -1})
    texts = pd.DataFrame(rows)
    directory = tmp_path / 'release'
    directory.mkdir()
    artifacts = {}
    for seed in (20261105, 20261106, 20261107):
        path = directory / f'split_seed{seed}.parquet'
        pd.DataFrame(split_rows).to_parquet(path, index=False)
        artifacts[f'split_seed{seed}'] = {'sha256': sha256(path)}
    classical = tmp_path / 'classical.yaml'
    classical.write_text(yaml.safe_dump({'tfidf': {'min_df': 1, 'max_df': 1.0,
                                                'max_features': 1000, 'sublinear_tf': True,
                                                'ngram_range': [1, 2]}}), encoding='utf-8')
    lm_config = tmp_path / 'lm.yaml'
    lm_config.write_text('model: fixture\n', encoding='utf-8')
    lock = tmp_path / 'lock.json'
    write_json(lock, {'config_sha256': sha256(lm_config)})
    cfg = {'schema_version': 1, 'experiment_id': 'horus-n2-advanced-detectors-v1',
           'release_dir': str(directory), 'seeds': [20261105, 20261106, 20261107],
           'iid_folds': list(range(5)), 'ood_folds': [0], 'fpr_targets': [.01, .05],
           'bootstrap_replicates': 2000, 'models_iid': ['alta_nb'], 'models_ood': [],
           'classical_config': str(classical), 'lm_config': str(lm_config),
           'lm_lock': str(lock)}
    path = tmp_path / 'campaign.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    monkeypatch.setattr(runner, 'load_release', lambda _cfg: (texts, {'artifacts': artifacts}))
    def small_evaluate(*args, **kwargs):
        assert kwargs['bootstrap_replicates'] == 2000
        return real_evaluate(*args, **{**kwargs, 'bootstrap_replicates': 8})
    monkeypatch.setattr(runner, 'evaluate_binary', small_evaluate)
    plan = tmp_path / 'iid.csv'
    runner.create_plan(path, plan, ['iid'])
    result = runner.run_task(path, plan, 0, tmp_path / 'runs')
    data = read_json(result)
    assert data['status'] == 'complete'
    assert data['counts']['binary_test'] == 6
    assert runner.run_task(path, plan, 0, tmp_path / 'runs') == result
    assert runner.check_plan(plan, tmp_path / 'runs')['missing_indices'] == list(range(1, 15))
    classical.write_text('nb_alpha: 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changed after planning'):
        runner.read_task(path, plan, 0)
