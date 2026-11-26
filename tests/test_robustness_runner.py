"""Identity/alignment tests for frozen-model robustness scoring."""

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.evaluation.robustness_runner import _joined, _reference, task_id


def _test() -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": ["h", "s"], "label_name": ["human", "synthetic"],
        "probe_group_id": ["gh", "gs"], "broad_area": ["social", "social"],
        "source_name": ["source", None], "generator": [None, "generator"],
    })


def test_reference_requires_exact_frozen_parent_set_and_groups() -> None:
    test = _test()
    previous = test[["record_id", "label_name", "probe_group_id"]].copy()
    previous["score_n2"] = [.1, .9]
    reference = _reference(test, previous)
    assert reference.score_clean.tolist() == [.1, .9]
    previous.loc[0, "probe_group_id"] = "wrong"
    with pytest.raises(ValueError, match="frozen test documents differ"):
        _reference(test, previous)


def test_joined_scores_preserve_no_ops_and_explicit_invalid_scores() -> None:
    test = _test()
    previous = test[["record_id", "label_name", "probe_group_id"]].copy()
    previous["score_n2"] = [.1, .9]
    view = pd.DataFrame({
        "record_id": ["h", "s"], "parent_record_id": ["h", "s"],
        "variant_id": ["vh", "vs"], "content_hash": ["ch", "cs"],
        "parent_content_hash": ["ph", "ps"], "attack": "upper_lower",
        "intensity": .01, "status": ["no_op", "changed"], "reason": ["dose_rounds_to_zero", ""],
        "eligible_positions": [4, 100], "num_edits": [0, 1],
    })
    thresholds = {"youden": .5, "fpr_0.01": .8, "fpr_0.05": .7}
    frame = _joined(test, view, _reference(test, previous), np.asarray([.1, np.nan]),
                    {"model": "stylo_svm", "seed": 20261105, "fold": 0}, thresholds)
    assert frame.status.tolist() == ["no_op", "changed"]
    assert frame.score_status.tolist() == ["ok", "invalid"]
    assert frame["threshold_fpr_0.01"].eq(.8).all()
    assert frame.parent_record_id.tolist() == ["h", "s"]


def test_robustness_task_identity_binds_clean_fit_and_attack_manifest() -> None:
    row = {"family": "advanced", "model": "stylo_svm", "seed": 20261105,
           "fold": 0, "source_task_id": "clean", "source_plan": "iid.csv",
           "source_plan_sha256": "plan", "config_sha256": "config",
           "attack_manifest_sha256": "attack"}
    assert task_id(row) != task_id({**row, "attack_manifest_sha256": "different"})
    assert task_id(row) != task_id({**row, "source_task_id": "another-fit"})
