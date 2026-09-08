import pytest
import torch

from mjlab_microduck.stance_evaluation import score_attempt
from mjlab_microduck.stance_transition import PhysicsState


def evidence(n=2501):
    state = PhysicsState(torch.zeros(n), torch.zeros(n, 3), torch.full((n,), .12),
        torch.ones(n, 2), torch.zeros(n, 14), torch.zeros(n, 14),
        torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))
    return state, torch.zeros(n, 3), torch.zeros(n, 14, dtype=torch.bool), torch.arange(n)


def test_full_synthetic_stance_pass_is_not_model_admission():
    r = score_attempt(*evidence())
    assert r['candidate_pass'] and r['complete_first_attempt']
    assert not r['checkpoint_admitted'] and not r['provenance_validated']
    assert not r['learned_stance_accepted']


def test_short_clean_trace_incomplete_not_success():
    r = score_attempt(*evidence(100))
    assert not r['candidate_pass'] and not r['complete_first_attempt']


def test_first_failure_retained_but_later_recovery_refused():
    e = evidence(100); e[0].tilt[-1] = .4
    r = score_attempt(*e)
    assert not r['candidate_pass'] and r['hard_failure'] and r['complete_first_attempt']
    e[0].tilt[50] = .4
    with pytest.raises(ValueError, match='post-terminal'): score_attempt(*e)


@pytest.mark.parametrize('case', ['tilt', 'speed', 'height', 'displacement', 'soft'])
def test_performance_gates_fail_without_relaxing_hold_stops(case):
    e = evidence(); state, positions, soft, _ = e
    if case == 'tilt': state.tilt[2000:] = .09
    if case == 'speed': state.root_velocity[2000:, 0] = .031
    if case == 'height': state.height[2000:] = .104
    if case == 'displacement': positions[1000:, 0] = .021
    if case == 'soft': soft[1:50] = True
    r = score_attempt(*e)
    assert not r['candidate_pass'] and not r['hard_failure']


def test_unsmoothed_terminal_motor_violation_cannot_hide_under_p95():
    e = evidence(); e[0].torque[-1, 3] = .361
    r = score_attempt(*e)
    assert r['hard_failure'] and not r['candidate_pass']


def test_initial_frame_does_not_inflate_soft_limit_denominator():
    e = evidence(); e[2][0] = True
    assert score_attempt(*e)['metrics']['soft_limit_exposure_fraction'] == 0


def test_finite_positions_that_overflow_displacement_are_rejected():
    e = evidence(); e[1][1, 0] = 1e38
    with pytest.raises(ValueError, match='arithmetic'): score_attempt(*e)


def test_reset_skip_nan_and_fake_rejected_command_are_refused():
    e = evidence(); e[3][1000] = 0
    with pytest.raises(ValueError, match='continuous'): score_attempt(*e)
    e = evidence(); e[1][1, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'): score_attempt(*e)
    with pytest.raises(ValueError, match='excessive'):
        score_attempt(*evidence(1), rejected_proposed_torque=torch.zeros(14))
    r = score_attempt(*evidence(1), rejected_proposed_torque=torch.full((14,), .37))
    assert r['complete_first_attempt'] and r['hard_failure'] and not r['candidate_pass']


def test_retained_native_hold_failure_is_rejected_without_rerunning_physics():
    import hashlib
    import json
    from pathlib import Path
    path = Path(__file__).parents[1]/'artifacts/diagnostics/football-flat-hold-local-a37d00c.json'
    if not path.is_file(): pytest.skip('retained native hold trace not installed')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '6bd04008ac8161f55f7a3c77c45b10ccee58c37cba6d4e8de5c5107ca2ea768f'
    frames = json.loads(raw)['report']['frames']; n = len(frames)
    def floats(key): return torch.tensor([f[key] for f in frames], dtype=torch.float64)
    state = PhysicsState(floats('tilt_rad'), floats('qvel')[:, :3], floats('root_height_m'),
        torch.tensor([[f['support_normal_n'][k] for k in ('left_foot_collision', 'right_foot_collision')]
                      for f in frames], dtype=torch.float64),
        floats('applied_torque_nm'), floats('hinge_speed_rad_s'),
        torch.tensor([f['outside_hard_limit'] for f in frames]),
        torch.tensor([bool(f['forbidden_contacts']) for f in frames]),
        torch.tensor([f['warning_count'] > 0 for f in frames]))
    # Every original per-boundary fraction is zero, so every per-joint bit is zero.
    assert all(f['soft_limit_exposure_fraction'] == 0 for f in frames)
    result = score_attempt(state, floats('qpos')[:, :3], torch.zeros(n, 14, dtype=torch.bool), torch.arange(n))
    assert result['hard_failure'] and result['complete_first_attempt']
    assert result['last_physics_step'] == 474 and not result['candidate_pass']
    assert path.read_bytes() == raw
