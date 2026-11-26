"""Fit classical/adapted detectors and calibrate frozen LM scores on N2.

Execution is controlled by immutable plans. All fitting is train-only, every
threshold uses validation, and incomplete/invalid scoring is reported explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from horus_benchmark.evaluation.advanced_data import config, load_release, partition
from horus_benchmark.evaluation.research_io import (
    checked_parquet, read_json, sha256, write_json,
)
from horus_benchmark.metrics.binary_detection import evaluate_binary


ZERO_SHOT = {
    'binoculars_falcon': ('falcon', 'binoculars_score'),
    'fastdetect_gptj_neo': ('gptj_neo', 'fastdetect_score'),
    'fastdetect_falcon': ('falcon', 'fastdetect_score'),
    'binoculars_qwen_adapted': ('qwen', 'binoculars_score'),
    'fastdetect_qwen_adapted': ('qwen', 'fastdetect_score'),
}
CLASSICAL = {'stylo_lr', 'stylo_svm', 'stylo_lr_no_structure',
             'stylo_svm_no_structure', 'alta_nb', 'alta_delta',
             'alta_readability', 'alta_pos', 'alta_entropy', 'alta_vote'}
GHOST = 'ghostbuster_ptbr_adapted'
NEEDS_POS = {'stylo_lr', 'stylo_svm', 'stylo_lr_no_structure',
             'stylo_svm_no_structure', 'alta_pos', 'alta_vote'}
META = ['record_id', 'label_name', 'probe_group_id', 'broad_area',
        'source_name', 'generator', 'word_count_audit', 'role', 'paired_iid_test']


def implementation_hash() -> str:
    package = Path(__file__).resolve().parents[1]
    paths = ['evaluation/advanced_runner.py', 'detectors/classical_adapted.py',
             'features/stylometry.py', 'detectors/ghostbuster_adapted.py',
             'metrics/binary_detection.py']
    return hashlib.sha256('|'.join(sha256(package / path) for path in paths).encode()).hexdigest()


def task_id(row: dict) -> str:
    fields = ('model', 'protocol', 'target', 'source_area', 'seed', 'fold',
              'manifest_path', 'manifest_sha256', 'config_sha256',
              'text_view', 'text_view_sha256', 'classical_config_sha256', 'lm_lock_sha256',
              'implementation_sha256')
    identity = {key: str(row[key]) for key in fields}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]


def create_plan(config_path: Path, output: Path, protocols: list[str],
                text_view: Path | None = None) -> Path:
    cfg = config(config_path)
    _, release = load_release(cfg)
    if output.exists():
        raise FileExistsError(f'Plan already exists: {output}')
    if set(protocols) - {'iid', 'logo', 'lodo', 'loso', 'logo_lodo'}:
        raise ValueError('Unknown protocol')
    view = 'native' if text_view is None else str(text_view)
    view_sha = '' if text_view is None else sha256(text_view)
    common = {'config_sha256': sha256(config_path), 'text_view': view,
              'text_view_sha256': view_sha,
              'classical_config_sha256': sha256(cfg['classical_config']),
              'lm_lock_sha256': sha256(cfg['lm_lock']),
              'implementation_sha256': implementation_hash()}
    if read_json(cfg['lm_lock'])['config_sha256'] != sha256(cfg['lm_config']):
        raise ValueError('LM configuration changed after model lock')
    rows = []
    if 'iid' in protocols:
        for seed in cfg['seeds']:
            path = Path(cfg['release_dir']) / f'split_seed{seed}.parquet'
            for model in cfg['models_iid']:
                # A training-free score needs no repeated model fitting, but
                # each validation fold has its own operating threshold.
                for fold in cfg['iid_folds']:
                    rows.append({**common, 'model': model, 'protocol': 'iid',
                                 'target': 'all', 'source_area': '', 'seed': seed,
                                 'fold': fold, 'manifest_path': str(path),
                                 'manifest_sha256': release['artifacts'][f'split_seed{seed}']['sha256']})
    indexes = []
    if set(protocols) & {'logo', 'lodo', 'loso'}:
        indexes.append(Path(cfg['ood_index']))
    if 'logo_lodo' in protocols:
        indexes.append(Path(cfg['compound_index']))
    for path in indexes:
        index = pd.read_csv(path, keep_default_na=False)
        if index.duplicated(['seed', 'protocol', 'source_area', 'target']).any():
            raise ValueError('Duplicate OOD identity (including source area)')
        for row in index.to_dict('records'):
            if row['status'] != 'eligible' or row['protocol'] not in protocols:
                continue
            if int(row['seed']) not in cfg['seeds']:
                raise ValueError('Unreviewed split seed in OOD index')
            if sha256(row['manifest_path']) != row['manifest_sha256']:
                raise ValueError('Changed OOD manifest')
            for model in cfg['models_ood']:
                for fold in cfg['ood_folds']:
                    rows.append({**common, 'model': model, 'protocol': row['protocol'],
                                 'target': row['target'], 'source_area': row['source_area'],
                                 'seed': int(row['seed']), 'fold': fold,
                                 'manifest_path': row['manifest_path'],
                                 'manifest_sha256': row['manifest_sha256']})
    if not rows:
        raise ValueError('Empty evaluation plan')
    for row in rows:
        if row['model'] not in CLASSICAL | set(ZERO_SHOT) | {GHOST}:
            raise ValueError(f"Unknown detector {row['model']}")
        row['task_id'] = task_id(row)
    plan = pd.DataFrame(rows)
    if plan.task_id.duplicated().any():
        raise ValueError('Plan task identity collision')
    output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output, index=False)
    write_json(output.with_suffix('.json'), {
        'kind': 'horus_advanced_plan', 'rows': len(plan), 'sha256': sha256(output),
        'config_sha256': sha256(config_path),
        'counts': {str(k): int(v) for k, v in plan.groupby('model').size().items()},
        'protocols': protocols, 'iid_folds': cfg['iid_folds'], 'ood_folds': cfg['ood_folds']})
    return output


def merge_features(frame: pd.DataFrame, path: Path, expected: str,
                   columns: list[str]) -> pd.DataFrame:
    features = checked_parquet(path, expected)
    keep = ['record_id', 'content_hash', 'status', *columns]
    if set(keep) - set(features):
        raise ValueError(f'Feature cache has missing fields: {path}')
    merged = frame.merge(features[keep], on='record_id', how='left',
                          validate='one_to_one', suffixes=('', '_feature'))
    if (merged.content_hash_feature.isna().any() or
            not merged.content_hash.eq(merged.content_hash_feature).all()):
        raise ValueError('Feature cache text identity does not match scored view')
    return merged.rename(columns={'status': 'feature_status'}).drop(
        columns='content_hash_feature')


def enrich(frame: pd.DataFrame, model: str, cfg: dict, *,
           lm_root: Path | None = None, pos_root: Path | None = None
           ) -> tuple[pd.DataFrame, dict]:
    inputs = {}
    if model in NEEDS_POS:
        directory = pos_root or Path(cfg['linguistic_cache'])
        manifest = read_json(directory / 'manifest.json')
        frame = merge_features(frame, directory / 'annotations.parquet',
                               manifest['annotations_sha256'], ['upos'])
        inputs['pos_manifest'] = sha256(directory / 'manifest.json')
    elif model in ZERO_SHOT or model == GHOST:
        pair = 'ghost_weak' if model == GHOST else ZERO_SHOT[model][0]
        directory = (lm_root or Path(cfg['lm_cache'])) / pair
        manifest = read_json(directory / 'manifest.json')
        if manifest['identity']['lock_sha256'] != sha256(cfg['lm_lock']):
            raise ValueError('Probability cache uses a different model lock')
        columns = (['token_ids', 'weak_logprobs_0', 'weak_logprobs_1', 'weak_token_strings'] if model == GHOST
                   else [ZERO_SHOT[model][1]])
        frame = merge_features(frame, Path(manifest['scores_path']),
                               manifest['scores_sha256'], columns)
        inputs['lm_manifest'] = sha256(directory / 'manifest.json')
    else:
        frame = frame.copy()
        frame['feature_status'] = 'ok'
    return frame, inputs


def fit_model(model: str, train: pd.DataFrame, validation: pd.DataFrame,
              test: pd.DataFrame, cfg: dict, seed: int) -> dict:
    if model == GHOST:
        from horus_benchmark.detectors.ghostbuster_adapted import fit_ghostbuster
        return fit_ghostbuster(train, validation, test, cfg['ghostbuster'], seed)
    from horus_benchmark.detectors.classical_adapted import fit_classical
    classical_cfg = yaml.safe_load(Path(cfg['classical_config']).read_text(encoding='utf-8'))
    return fit_classical(model, train, validation, test, classical_cfg, seed)


def read_task(config_path: Path, plan_path: Path, index: int) -> tuple[dict, dict]:
    cfg = config(config_path)
    marker = read_json(plan_path.with_suffix('.json'))
    if marker['sha256'] != sha256(plan_path) or marker['config_sha256'] != sha256(config_path):
        raise ValueError('Evaluation plan or configuration changed')
    plan = pd.read_csv(plan_path, keep_default_na=False)
    if not 0 <= index < len(plan):
        raise IndexError('Task index outside plan')
    task = plan.iloc[index].to_dict()
    if task['task_id'] != task_id(task):
        raise ValueError('Task identity changed')
    if (task['classical_config_sha256'] != sha256(cfg['classical_config']) or
            task['lm_lock_sha256'] != sha256(cfg['lm_lock']) or
            task['implementation_sha256'] != implementation_hash()):
        raise ValueError('Model configuration, implementation or checkpoint lock changed after planning')
    return cfg, task


def run_task(config_path: Path, plan_path: Path, index: int,
             output: Path) -> Path:
    cfg, task = read_task(config_path, plan_path, index)
    prior_path = output / task['task_id'] / 'metrics.json'
    if prior_path.exists():
        prior = read_json(prior_path)
        if prior['task'] != task:
            raise ValueError('Existing detector result has another task identity')
        if prior['status'] == 'complete' and sha256(prior_path.parent / 'predictions.parquet') != prior['predictions_sha256']:
            raise ValueError('Existing detector prediction artifact changed')
        return prior_path
    texts, _ = load_release(cfg)
    if task['text_view'] != 'native':
        view = checked_parquet(task['text_view'], task['text_view_sha256'])
        if set(view.record_id) != set(texts.record_id):
            raise ValueError('Common text view must cover every selected document')
        texts = texts.drop(columns=['content', 'content_hash']).merge(
            view[['record_id', 'content', 'content_hash']], on='record_id', validate='one_to_one')
        cfg = {**cfg, 'lm_cache': str(Path(cfg['output_root']) / 'cache' / 'common_prefix'),
               'linguistic_cache': str(Path(cfg['output_root']) / 'linguistic' / 'common_prefix')}
    frame = partition(cfg, texts, task)
    frame, cache_inputs = enrich(frame, task['model'], cfg)
    all_test = frame.loc[frame.role.isin(['test_target', 'test_control', 'test_unmatched_target'])].sort_values('record_id')
    valid = frame.feature_status.eq('ok')
    train = frame.loc[frame.role.eq('train') & valid].sort_values('record_id')
    validation = frame.loc[frame.role.eq('validation') & valid].sort_values('record_id')
    test = all_test.loc[all_test.feature_status.eq('ok')]
    binary = test.role.isin(['test_target', 'test_control']).to_numpy()
    directory = output / task['task_id']
    if (directory / 'metrics.json').exists():
        raise FileExistsError(f'Task already complete: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    model_config_path = Path(cfg['classical_config'])
    input_hashes = {'config': sha256(config_path), 'model_lock': sha256(cfg['lm_lock']),
                    'classical_config': sha256(model_config_path), **cache_inputs}
    payload = {'kind': 'horus_advanced_result', 'task': task, 'inputs': input_hashes,
               'software': {name: importlib.metadata.version(name) for name in
                            ('numpy', 'pandas', 'scikit-learn', 'pyphen')},
               'coverage': {'train_eligible': len(train), 'validation_eligible': len(validation),
                            'test_total': len(all_test), 'test_eligible': len(test)},
               'counts': {'binary_test': int(binary.sum())},
               'status': 'unsupported', 'reason': '', 'evaluation': None}
    for role, data in [('train', train), ('validation', validation), ('test', test.loc[binary])]:
        if (task['model'] not in ZERO_SHOT or role != 'train') and \
                set(data.label_name) != {'human', 'synthetic'}:
            payload['reason'] = f'{role}_missing_binary_class_after_validity_checks'
            write_json(directory / 'metrics.json', payload)
            return directory / 'metrics.json'
    model_seed = int(hashlib.sha256(
        f"{task['model']}|{task['seed']}|{task['fold']}".encode()).hexdigest()[:8], 16)
    if task['model'] in ZERO_SHOT:
        col = ZERO_SHOT[task['model']][1]
        validation_scores = validation[col].to_numpy(dtype=float)
        scores = test[col].to_numpy(dtype=float)
        model_info = {'training_free': True, 'pair': ZERO_SHOT[task['model']][0]}
    else:
        fitted = fit_model(task['model'], train, validation, test, cfg, model_seed)
        validation_scores, scores = fitted['validation_scores'], fitted['test_scores']
        model_info = fitted['metadata']
        checkpoint = directory / 'model' / 'estimator.joblib'
        checkpoint.parent.mkdir(exist_ok=True)
        joblib.dump(fitted['estimator'], checkpoint, compress=3)
        model_info['estimator_sha256'] = sha256(checkpoint)
    labels = lambda part: part.label_name.eq('synthetic').to_numpy(dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    evaluation = evaluate_binary(labels(validation), np.asarray(validation_scores),
                                  labels(test.loc[binary]), scores[binary],
                                  test.loc[binary, 'probe_group_id'].to_numpy(),
                                  bootstrap_replicates=cfg['bootstrap_replicates'],
                                  bootstrap_seed=model_seed + 1)
    predictions = all_test[META + ['feature_status']].copy()
    scored = pd.DataFrame({'record_id': test.record_id.to_numpy(), 'score_n2': scores})
    predictions = predictions.merge(scored, on='record_id', how='left', validate='one_to_one')
    predictions['status'] = np.where(predictions.feature_status.eq('ok'), 'ok', 'invalid')
    prediction_path = directory / 'predictions.parquet'
    predictions.to_parquet(prediction_path, index=False, compression='zstd')
    payload.update(status='complete', model_seed=model_seed, model_info=model_info,
                   evaluation=evaluation, predictions_sha256=sha256(prediction_path))
    write_json(directory / 'metrics.json', payload)
    return directory / 'metrics.json'


def check_plan(plan_path: Path, output: Path) -> dict:
    marker = read_json(plan_path.with_suffix('.json'))
    if marker['sha256'] != sha256(plan_path):
        raise ValueError('Changed execution plan')
    plan = pd.read_csv(plan_path, keep_default_na=False)
    missing, unsupported, complete = [], [], 0
    for index, task in enumerate(plan.to_dict('records')):
        path = output / task['task_id'] / 'metrics.json'
        if not path.exists():
            missing.append(index)
            continue
        value = read_json(path)
        if value['task']['task_id'] != task['task_id']:
            raise ValueError('Task output identity mismatch')
        if value['status'] == 'unsupported':
            unsupported.append({'index': index, 'reason': value['reason']})
        else:
            if sha256(path.parent / 'predictions.parquet') != value['predictions_sha256']:
                raise ValueError('Corrupt prediction artifact')
            complete += 1
    return {'finished': not missing, 'planned': len(plan), 'complete': complete,
            'unsupported': unsupported, 'missing_indices': missing}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan')
    a.add_argument('--config', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--protocols', nargs='+', required=True)
    a.add_argument('--text-view', type=Path)
    a = sub.add_parser('run-task')
    a.add_argument('--config', type=Path, required=True)
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--index', type=int, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('check')
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('run-bundle')
    a.add_argument('--config', type=Path, required=True)
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--indices', type=int, nargs='+', required=True)
    a.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'plan':
        print(create_plan(args.config, args.output, args.protocols, args.text_view))
    elif args.command == 'run-task':
        print(run_task(args.config, args.plan, args.index, args.output))
    elif args.command == 'run-bundle':
        for index in args.indices:
            print(run_task(args.config, args.plan, index, args.output), flush=True)
    else:
        result = check_plan(args.plan, args.output)
        print(json.dumps(result, indent=2))
        return 0 if result['finished'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
