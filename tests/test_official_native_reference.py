import copy
import json

import pytest

from mjlab_microduck import official_native_reference as reference
from test_integrated_turn_foundation import accepted_rows


def rows_with_positions():
    rows = accepted_rows()
    names = [f"joint-{i}" for i in range(13)] + ["right_hip_yaw"]
    for row in rows:
        row.update(checkpoint_sha256=reference.POLICY_SHA,
            response_diagnostics=dict(protocol="signed-body-response-and-joint-load-v1",
                hard_stop_margin_rad=.05, joint_columns=names, steps=240,
                min_distance_to_hard_stop_rad_by_joint=[.2]*14,
                hard_stop_proximity_fraction_by_joint=[0.]*14))
    return rows


def test_exact_single_ordered_matrix_and_exclusive_outputs(tmp_path):
    rows, calls, checks = rows_with_positions(), [], []
    policy = object()
    def evaluate(path, seed, speed, yaw, health, **kwargs):
        assert path == reference.POLICY and kwargs == dict(diagnostics=True,
            yaw_trace=True, gait_trace=True, onnx_reference=policy)
        assert health is check
        calls.append((seed, speed, yaw))
        return copy.deepcopy(rows[len(calls)-1])
    def check():
        checks.append(True)
    result = reference.evaluate_matrix(policy, check, tmp_path, evaluator=evaluate)
    assert calls == [(s, *c) for s in reference.base.SEEDS for c in reference.base.CASES]
    assert len(checks) == len(list(tmp_path.glob('s*-c*.json'))) == 12
    assert not result['ppo_checkpoint'] and not result['command_remap']
    assert not result['policy_acceptance'] and not result['physical_motion_authorized']
    assert result['numerical_gate']['decision'] == 'foundation-ready-for-next-experiment'
    assert json.loads((tmp_path / 's839-c0.json').read_text()) == rows[0]
    with pytest.raises(FileExistsError):
        reference.evaluate_matrix(policy, check, tmp_path, evaluator=lambda *_a, **_k: rows[0])


@pytest.mark.parametrize('fault', ['hash', 'identity', 'tracking', 'position'])
def test_binding_and_unchanged_motion_and_range_gates(tmp_path, fault):
    rows = rows_with_positions()
    if fault == 'hash':
        rows[0]['checkpoint_sha256'] = '0'*64
    elif fault == 'identity':
        rows[0]['seed'] = 0
    elif fault == 'tracking':
        rows[0]['yaw_mae'][7] = .10001
    else:
        rows[0]['response_diagnostics']['min_distance_to_hard_stop_rad_by_joint'][0] = -.00501
    index = iter(rows)
    evaluate = lambda *_a, **_k: next(index)
    if fault in ('hash', 'identity'):
        with pytest.raises(ValueError):
            reference.evaluate_matrix(object(), lambda: None, tmp_path, evaluator=evaluate)
        assert not list(tmp_path.iterdir())
    else:
        result = reference.evaluate_matrix(object(), lambda: None, tmp_path, evaluator=evaluate)
        assert result['numerical_gate']['decision'] == 'foundation-not-ready'
        assert len(result['numerical_gate']['failures']) == 1
