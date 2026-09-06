"""CPU-only ordering regression using the actual installed mjlab step method."""

from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck.motor_measurement_audit import (
    MotorMeasurementAudit, PROTOCOL, TERM, capture_metric, install_metric,
    motor_layout, validate_mode,
)


def make_audit():
    return MotorMeasurementAudit(2, 3, ("b", "a"), (1, 0), stall_reference_nm=.6)


def make_env(audit, reset_force=9.):
    from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg, NullMetricsManager
    from mjlab.managers.recorder_manager import NullRecorderManager

    data = NS(actuator_force=torch.zeros(2, 2), joint_vel=torch.zeros(2, 2))
    class Scene(dict):
        def write_data_to_sim(self):
            pass
        def update(self, dt):
            pass
    env = NS(cfg=NS(auto_reset=True, decimation=2), num_envs=2, device="cpu",
             _sim_step_counter=0, common_step_counter=0, physics_dt=.005, step_dt=.01,
             episode_length_buf=torch.zeros(2, dtype=torch.long), extras={},
             scene=Scene(robot=NS(data=data)), recorder_manager=NullRecorderManager())
    env._microduck_motor_measurement_audit = audit
    env.action_manager = NS(action=torch.zeros(2, 2), apply_action=lambda: None)
    env.action_manager.process_action = lambda actions: env.action_manager.action.copy_(actions)
    def physics_step():
        data.actuator_force.copy_(torch.tensor([[.3, -.48], [.12, .24]]))
        data.joint_vel.copy_(torch.tensor([[1., 2.], [3., 4.]]))
    env.sim = NS(step=physics_step, forward=lambda: None, sense=lambda: None)
    env.termination_manager = NS(terminated=torch.tensor([True, False]), time_outs=torch.zeros(2, dtype=torch.bool))
    env.termination_manager.compute = lambda: env.termination_manager.terminated.clone()
    env.reward_manager = NS(compute=lambda dt: torch.ones(2))
    env.command_manager = NS(compute=lambda dt: None)
    env.event_manager = NS(available_modes=[])
    env.observation_manager = NS(compute=lambda update_history: {"actor": data.joint_vel.clone()})
    def reset(ids):
        # Mimic reset/forward replacing the terminal motor state and action.
        data.actuator_force[ids] = reset_force
        data.joint_vel[ids] = 0.
        env.action_manager.action[ids] = 0.
        env.metrics_manager.reset(ids)
    env._reset_idx = reset
    env.metrics_manager = (MetricsManager({TERM: MetricsTermCfg(func=capture_metric)}, env)
                           if audit is not None else NullMetricsManager())
    return env


@pytest.mark.parametrize("reset_force", [0., 9.])
def test_actual_mjlab_step_proves_post_reset_contamination_and_preserves_terminal_force(reset_force):
    from mjlab.envs import ManagerBasedRlEnv
    audit = make_audit()
    env = make_env(audit, reset_force)
    active = torch.ones(2, dtype=torch.bool)
    audit.begin(0, active, torch.tensor([0, 1]))
    obs, reward, terminated, timeout, _ = ManagerBasedRlEnv.step(env, torch.ones(2, 2))
    after = env.scene["robot"].data.actuator_force
    assert torch.equal(after[0], torch.full((2,), reset_force))  # legacy sampling site
    assert torch.equal(audit.snapshot["force"][0], torch.tensor([.3, -.48]))
    assert torch.equal(audit.snapshot["speed"][0], torch.tensor([2., 1.]))  # mapped order
    audit.finish(terminated | timeout, after)
    result = audit.report()
    assert result["terminal_environment_steps"] == 1
    assert result["terminal_force_nm"]["abs_max"] == pytest.approx(.48)
    assert result["terminal_post_return_force_nm"]["abs_max"] == reset_force
    assert result["groups"]["all"]["force_nm"]["samples"] == 4
    assert result["groups"]["approach"]["by_joint"]["a"]["abs_p99"] == pytest.approx(.8)
    assert result["groups"]["recovery"]["force_nm"]["samples"] == 0
    assert result["groups"]["recovery"]["force_nm"]["abs_p99"] is None
    assert result["incomplete_first_attempts"] == 1
    assert not result["policy_acceptance"] and not result["runtime_equivalence_validated"]
    assert not result["legacy_metrics_replaced"] and not result["physical_motion_authorized"]
    # Second first attempt terminates; completed env 0 must never be pooled again.
    env.termination_manager.terminated = torch.tensor([False, True])
    audit.begin(1, torch.tensor([False, True]), torch.tensor([0, 2]))
    _, _, terminated, timeout, _ = ManagerBasedRlEnv.step(env, torch.ones(2, 2))
    audit.finish(terminated | timeout, after)
    result = audit.report()
    assert result["terminal_environment_steps"] == 2 and result["incomplete_first_attempts"] == 0
    assert result["groups"]["all"]["environment_steps"] == 3
    assert result["groups"]["all"]["force_nm"]["samples"] == 6


def test_observer_does_not_change_mock_dynamics_actions_observations_or_rng():
    from mjlab.envs import ManagerBasedRlEnv
    audit = make_audit()
    observed, plain = make_env(audit), make_env(None)
    rng = torch.random.get_rng_state().clone()
    audit.begin(0, torch.ones(2, dtype=torch.bool), torch.tensor([0, 1]))
    a = ManagerBasedRlEnv.step(observed, torch.ones(2, 2))
    b = ManagerBasedRlEnv.step(plain, torch.ones(2, 2))
    assert torch.equal(rng, torch.random.get_rng_state())
    for left, right in zip(a[1:4], b[1:4]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(a[0]["actor"], b[0]["actor"], rtol=0, atol=0)
    torch.testing.assert_close(observed.action_manager.action, plain.action_manager.action, rtol=0, atol=0)
    assert observed._sim_step_counter == plain._sim_step_counter == 2


def test_snapshot_and_metadata_are_not_views_into_mutable_reset_buffers():
    audit = make_audit()
    active, phase = torch.ones(2, dtype=torch.bool), torch.tensor([0, 1])
    data = NS(actuator_force=torch.ones(2, 2), joint_vel=torch.ones(2, 2))
    terminal = torch.tensor([True, False])
    audit.begin(0, active, phase)
    audit.capture(data, terminal)
    active[:] = False
    phase[:] = 2
    data.actuator_force[:] = 99.
    data.joint_vel[:] = 99.
    audit.finish(terminal, data.actuator_force)
    data.actuator_force[:] = 0.
    result = audit.report()
    assert result["groups"]["all"]["force_nm"]["abs_max"] == 1.
    assert result["groups"]["all"]["speed_rad_s"]["abs_max"] == 1.
    assert result["groups"]["approach"]["environment_steps"] == 1
    assert result["terminal_post_return_force_nm"]["abs_max"] == 99.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_flagged_without_silently_dropping_or_zeroing(bad):
    audit = make_audit()
    data = NS(actuator_force=torch.tensor([[bad, 0.], [0., 0.]]), joint_vel=torch.zeros(2, 2))
    done = torch.ones(2, dtype=torch.bool)
    audit.begin(0, done, torch.zeros(2, dtype=torch.long))
    audit.capture(data, done)
    audit.finish(done, torch.zeros(2, 2))
    result = audit.report()
    assert not result["finite"]
    assert result["groups"]["all"]["force_nm"]["nonfinite_samples"] == 1
    assert result["groups"]["all"]["force_nm"]["abs_p99"] is None
    import json
    json.dumps(result, allow_nan=False)


def test_finite_large_values_do_not_overflow_rms_or_normalization():
    import json
    audit = make_audit()
    data = NS(actuator_force=torch.full((2, 2), 1.e38), joint_vel=torch.zeros(2, 2))
    done = torch.ones(2, dtype=torch.bool)
    audit.begin(0, done, torch.zeros(2, dtype=torch.long))
    audit.capture(data, done)
    audit.finish(done, data.actuator_force)
    result = audit.report()
    assert result["finite"]
    assert result["groups"]["all"]["force_nm"]["rms"] == pytest.approx(1.e38)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["unarmed", "missing", "duplicate", "phase", "mask", "reentry", "terminal", "shape", "sequence"])
def test_capture_protocol_rejects_missing_misaligned_or_duplicate_samples(fault):
    audit = make_audit()
    active, phase = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long)
    data = NS(actuator_force=torch.zeros(2, 2), joint_vel=torch.zeros(2, 2))
    done = torch.tensor([True, False])
    with pytest.raises(ValueError):
        if fault == "unarmed":
            audit.capture(data, done)
        else:
            audit.begin(5 if fault == "sequence" else 0,
                        torch.tensor([True, False]) if fault == "mask" else active,
                        torch.tensor([0, 3]) if fault == "phase" else phase)
            if fault == "shape":
                data.actuator_force = torch.zeros(2, 3)
            if fault != "missing":
                audit.capture(data, done)
            if fault == "duplicate":
                audit.capture(data, done)
            audit.finish(~done if fault == "terminal" else done, data.actuator_force)
            if fault == "reentry":
                audit.begin(1, active, phase)


@pytest.mark.parametrize("overrides", [dict(first_attempt_only=False), dict(collecting_dataset=True),
    dict(recording=True), dict(num_envs=65), dict(steps=1001), dict(case_count=13)])
def test_audit_is_bounded_and_separate_from_datasets_and_unvalidated_recorder(overrides):
    args = dict(first_attempt_only=True, collecting_dataset=False, recording=False, num_envs=4, steps=700, case_count=1)
    args.update(overrides)
    with pytest.raises(ValueError):
        validate_mode(True, **args)
    validate_mode(False, **args)  # opt-out does not impose new legacy restrictions


def test_real_microduck_mapping_is_named_direct_hinge_and_config_observation_stays_unchanged():
    from mjlab.entity import Entity
    from mjlab_microduck.hierarchical_obstacle_rollout import prepare_rollout_configs
    cfg, _ = prepare_rollout_configs(4, .4, .9, 0.)
    robot = Entity(cfg.scene.entities["robot"])
    names, ids = motor_layout(robot)
    assert len(names) == len(ids) == 14
    assert tuple(robot.joint_names[i] for i in ids) == names
    assert all(not name.startswith("passive_") for name in names)
    actor, rewards, events = cfg.observations["actor"], cfg.rewards.copy(), cfg.events.copy()
    assert cfg.auto_reset and TERM not in cfg.metrics
    install_metric(cfg)
    assert cfg.metrics[TERM].func is capture_metric and not cfg.metrics[TERM].per_substep
    assert cfg.observations["actor"] is actor and cfg.rewards == rewards and cfg.events == events
    assert cfg.auto_reset
    with pytest.raises(ValueError, match="already"):
        install_metric(cfg)


@pytest.mark.parametrize("fault", ["gear", "transmission", "target", "duplicate", "order"])
def test_equal_shapes_are_not_enough_for_force_joint_mapping(fault):
    import mujoco
    joint = NS(name="a", type=mujoco.mjtJoint.mjJNT_HINGE)
    actuator = NS(name="motor_a", target="a", trntype=mujoco.mjtTrn.mjTRN_JOINT, gear=[1., 0., 0., 0., 0., 0.])
    robot = NS(spec=NS(joints=[joint], actuators=[actuator]), joint_names=("a",), actuator_names=("motor_a",))
    if fault == "gear": actuator.gear[0] = 2.
    if fault == "transmission": actuator.trntype = mujoco.mjtTrn.mjTRN_TENDON
    if fault == "target": actuator.target = "missing"
    if fault == "duplicate": robot.spec.actuators.append(actuator)
    if fault == "order": robot.actuator_names = ("other",)
    with pytest.raises(ValueError):
        motor_layout(robot)


def test_opt_in_report_is_separate_and_does_not_replace_old_gate_values(tmp_path, monkeypatch):
    import copy
    import json
    from mjlab_microduck import hierarchical_obstacle_rollout as rollout

    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"synthetic-actor")
    audit = make_audit()
    done = torch.ones(2, dtype=torch.bool)
    data = NS(actuator_force=torch.zeros(2, 2), joint_vel=torch.zeros(2, 2))
    audit.begin(0, done, torch.zeros(2, dtype=torch.long))
    audit.capture(data, done)
    audit.finish(done, data.actuator_force)
    baseline = dict(collision_events=0, clean_pass_events=2, attempt_timeout_events=0,
                    fall_events=0, nan_termination_events=0, nonfinite_steps=0,
                    expected_attempts=2, completed_attempts=2, other_terminal_events=0,
                    motor_torque_utilization_p99=.6125551462173462)
    def case(*args, **kwargs):
        result = copy.deepcopy(baseline)
        if kwargs["motor_measurement_audit"]:
            result["motor_measurement_audit"] = audit.report()
        return result
    monkeypatch.setattr(rollout, "_run_case", case)
    args = dict(num_envs=2, steps=2, speeds=(.4,), forward_positions=(.9,),
                lateral_positions=(0.,), seeds=(1,), first_attempt_only=True)
    legacy = json.loads(rollout.run_rollout(actor, tmp_path / "legacy", **args).read_text())
    explicit_off = json.loads(rollout.run_rollout(actor, tmp_path / "off", motor_measurement_audit=False, **args).read_text())
    observed = json.loads(rollout.run_rollout(actor, tmp_path / "audit", motor_measurement_audit=True, **args).read_text())
    assert legacy == explicit_off
    assert observed.pop("motor_measurement_audit_protocol") == PROTOCOL
    separate = observed["cases"][0].pop("motor_measurement_audit")
    assert separate["groups"]["all"]["stall_reference_utilization"]["abs_p99"] == 0.
    assert observed == legacy
    assert observed["cases"][0]["motor_torque_utilization_p99"] > .60
    assert not separate["policy_acceptance"] and not separate["runtime_equivalence_validated"]
    monkeypatch.setattr(rollout, "_run_case", lambda *a, **k: baseline.copy())
    with pytest.raises(RuntimeError, match="missing"):
        rollout.run_rollout(actor, tmp_path / "missing", motor_measurement_audit=True, **args)
