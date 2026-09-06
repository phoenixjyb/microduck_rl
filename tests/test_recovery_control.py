"""CPU command-contract tests; no policies, GPU or physical motion are run."""

import copy
import json

import pytest
import torch

from mjlab_microduck.hierarchical_obstacle import (
    apply_bounded_supervisor_command, clone_teacher_state, make_teacher_state,
    reset_teacher_state, teacher_command,
)
from mjlab_microduck.recovery_control import (
    RecoveryAccelerationCfg, ROLLOUT_STAGE, validate_rollout_mode,
)


def state_and_observation():
    state = make_teacher_state(3, device="cpu", nominal_speed_mps=.3)
    state.phase[:] = torch.arange(3)
    obstacle = torch.zeros(3, 7)
    obstacle[:, 6] = 1
    return state, obstacle


def test_only_recovery_positive_acceleration_changes_and_inputs_do_not_mutate():
    state, obstacle = state_and_observation()
    other = clone_teacher_state(state)
    desired = torch.tensor([[.5, .3]] * 3)
    before = desired.clone()
    legacy = apply_bounded_supervisor_command(desired, obstacle, state)
    changed = apply_bounded_supervisor_command(desired, obstacle, other,
                    recovery_cfg=RecoveryAccelerationCfg(), update_dt_s=.1)
    torch.testing.assert_close(legacy[:2], changed[:2], rtol=0, atol=0)
    torch.testing.assert_close(legacy[:, 1], changed[:, 1], rtol=0, atol=0)
    assert changed[2, 0] == pytest.approx(.32)
    assert legacy[2, 0] == pytest.approx(.38)
    assert torch.equal(desired, before) and torch.equal(other.previous_command, changed)
    assert other.phase.tolist() == [0, 1, 2]


@pytest.mark.parametrize("desired_speed", [0., .1, .3])
def test_normal_braking_and_hold_are_bit_exact_to_legacy(desired_speed):
    state, obstacle = state_and_observation()
    other = clone_teacher_state(state)
    command = torch.tensor([[desired_speed, -.5]] * 3)
    a = apply_bounded_supervisor_command(command, obstacle, state)
    b = apply_bounded_supervisor_command(command, obstacle, other,
                    recovery_cfg=RecoveryAccelerationCfg(), update_dt_s=.1)
    assert torch.equal(a, b)


def test_invalid_geometry_stops_immediately_outside_recovery_and_reset_clears_history():
    state, obstacle = state_and_observation()
    obstacle[:, 6] = 0
    result = apply_bounded_supervisor_command(torch.tensor([[.8, .5]] * 3), obstacle, state,
                    recovery_cfg=RecoveryAccelerationCfg(), update_dt_s=.1)
    assert torch.equal(result[:2], torch.zeros(2, 2))
    assert result[2, 0] == pytest.approx(.32)  # preserve positively passed recovery semantics
    cloned = clone_teacher_state(state)
    reset_teacher_state(state, torch.tensor([False, False, True]), nominal_speed_mps=.4)
    assert state.phase[2] == 0 and state.previous_command[2, 0] == pytest.approx(.4)
    assert cloned.phase[2] == 2 and cloned.previous_command[2, 0] == pytest.approx(.32)


@pytest.mark.parametrize("dt,updates", [(.05, 20), (.1, 10), (.2, 5)])
def test_constant_target_is_reached_in_one_second_independent_of_command_cadence(dt, updates):
    state, obstacle = state_and_observation()
    desired = torch.tensor([[.5, 0.]] * 3)
    for _ in range(updates):
        previous = state.previous_command.clone()
        output = apply_bounded_supervisor_command(desired, obstacle, state,
                    recovery_cfg=RecoveryAccelerationCfg(.2), update_dt_s=dt)
        assert output[2, 0] - previous[2, 0] <= .2 * dt + 1e-7
        assert output[2, 0] <= .5 + 1e-7
    assert output[2, 0] == pytest.approx(.5, abs=1e-6)


def test_large_dt_never_relaxes_existing_slew_or_absolute_limits():
    state, obstacle = state_and_observation()
    output = apply_bounded_supervisor_command(torch.tensor([[50., 50.]] * 3), obstacle, state,
                    recovery_cfg=RecoveryAccelerationCfg(.2), update_dt_s=1.)
    torch.testing.assert_close(output, torch.tensor([[.38, .2]] * 3))


def test_teacher_phase_transition_also_uses_recovery_cap_without_observation_changes():
    state = make_teacher_state(1, device="cpu", nominal_speed_mps=.5)
    state.phase[:] = 1
    state.previous_command[0, 0] = .3
    obstacle = torch.tensor([[.2, 0., -1., .2, .1, 0., 1.]])  # valid, behind robot
    before = obstacle.clone()
    output = teacher_command(obstacle, torch.tensor([.5]), torch.tensor([0.]), torch.tensor([0.]), state,
                             recovery_cfg=RecoveryAccelerationCfg(), update_dt_s=.1)
    assert state.phase[0] == 2 and output[0, 0] == pytest.approx(.32)
    assert torch.equal(obstacle, before)


@pytest.mark.parametrize("bad", [0., -.1, float("nan"), float("inf"), True])
def test_invalid_acceleration_is_rejected(bad):
    with pytest.raises(ValueError): RecoveryAccelerationCfg(bad)


@pytest.mark.parametrize("bad", [None, 0., -.1, float("nan"), float("inf"), True])
def test_enabled_cap_requires_real_finite_elapsed_time(bad):
    state, obstacle = state_and_observation()
    before = state.previous_command.clone()
    with pytest.raises(ValueError):
        apply_bounded_supervisor_command(torch.ones(3, 2), obstacle, state,
                    recovery_cfg=RecoveryAccelerationCfg(), update_dt_s=bad)
    assert torch.equal(before, state.previous_command)


@pytest.mark.parametrize("change", [
    lambda s, d: d.fill_(float("nan")),
    lambda s, d: d.fill_(float("inf")),
    lambda s, d: s.previous_command.fill_(float("inf")),
    lambda s, d: s.previous_command[:, 0].fill_(-1),
    lambda s, d: s.phase.fill_(3),
])
def test_bad_state_fails_before_mutating_command_history(change):
    state, obstacle = state_and_observation()
    desired = torch.ones(3, 2)
    change(state, desired)
    before = state.previous_command.clone()
    with pytest.raises(ValueError):
        apply_bounded_supervisor_command(desired, obstacle, state,
                    recovery_cfg=RecoveryAccelerationCfg(), update_dt_s=.1)
    torch.testing.assert_close(before, state.previous_command, rtol=0, atol=0)


@pytest.mark.parametrize("overrides", [dict(first_attempt_only=False), dict(motor_measurement_audit=False),
    dict(collecting_dataset=True), dict(recording=True), dict(range_noise_m=.01)])
def test_experimental_rollout_rejects_unreviewed_combinations(overrides):
    kwargs = dict(first_attempt_only=True, motor_measurement_audit=True,
                  collecting_dataset=False, recording=False, range_noise_m=0.)
    kwargs.update(overrides)
    with pytest.raises(ValueError): validate_rollout_mode(RecoveryAccelerationCfg(), **kwargs)
    validate_rollout_mode(None, **kwargs)


def test_rollout_default_stays_exact_and_opt_in_has_distinct_provenance(tmp_path, monkeypatch):
    from mjlab_microduck import hierarchical_obstacle_rollout as rollout
    from mjlab_microduck.motor_measurement_audit import PROTOCOL as AUDIT
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"synthetic")
    baseline = dict(collision_events=0, clean_pass_events=2, attempt_timeout_events=0,
                    fall_events=0, nan_termination_events=0, nonfinite_steps=0,
                    expected_attempts=2, completed_attempts=2, other_terminal_events=0,
                    motor_torque_utilization_p99=.6125551462173462)
    def case(*args, **kwargs):
        result = copy.deepcopy(baseline)
        if kwargs["motor_measurement_audit"]:
            result["motor_measurement_audit"] = dict(protocol=AUDIT)
        cfg = kwargs.get("recovery_cfg")
        if cfg is not None:
            result["recovery_control"] = {**cfg.provenance(), "update_dt_s": .1}
        return result
    monkeypatch.setattr(rollout, "_run_case", case)
    args = dict(num_envs=2, steps=2, speeds=(.4,), forward_positions=(.9,),
                lateral_positions=(0.,), seeds=(1,), first_attempt_only=True)
    a = json.loads(rollout.run_rollout(actor, tmp_path / "default", **args).read_text())
    b = json.loads(rollout.run_rollout(actor, tmp_path / "off", recovery_cfg=None, **args).read_text())
    assert a == b and "recovery_control" not in a
    cfg = RecoveryAccelerationCfg()
    c = json.loads(rollout.run_rollout(actor, tmp_path / "on", recovery_cfg=cfg,
                                      motor_measurement_audit=True, **args).read_text())
    assert c["stage"] == ROLLOUT_STAGE and c["source_controller_stage"] == a["stage"]
    assert c["recovery_control"] == cfg.provenance() and c["policy_acceptance"] is False
    assert c["cases"][0]["motor_torque_utilization_p99"] > .60  # unchanged legacy number
    assert c["cases"][0]["recovery_control"]["update_dt_s"] == .1
    monkeypatch.setattr(rollout, "_run_case", lambda *a, **k: baseline.copy())
    with pytest.raises(RuntimeError, match="recovery"):
        rollout.run_rollout(actor, tmp_path / "missing", recovery_cfg=cfg, motor_measurement_audit=True, **args)
    with pytest.raises(ValueError, match="requires"):
        rollout.run_rollout(actor, tmp_path / "invalid", recovery_cfg=cfg, **args)
    assert not (tmp_path / "invalid").exists()


def test_new_execution_identity_cannot_pass_the_existing_u4_gate(tmp_path):
    from test_hc4u4_contract import reports
    from mjlab_microduck.hc4u4_gate import compare_hc4u4_prescreen
    paths = reports(tmp_path)
    result = json.loads(paths[0].read_text())
    result["source_controller_stage"] = result["stage"]
    result["stage"] = ROLLOUT_STAGE
    paths[0].write_text(json.dumps(result))
    with pytest.raises(ValueError, match="stage"):
        compare_hc4u4_prescreen(*paths, candidate_sha256="a" * 64, seed=347)
