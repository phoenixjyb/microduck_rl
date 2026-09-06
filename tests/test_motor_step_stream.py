"""CPU-only reset-order, bounded-retention and motor-cost contract regressions."""

from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck.motor_step_stream import (
    MotorStepCostCfg, MotorStepStream, TERM, capture_metric, install_metric,
)


def stream():
    return MotorStepStream(2, ("b", "a"), (1, 0), device="cpu", cost_cfg=MotorStepCostCfg())


def data():
    return NS(actuator_force=torch.tensor([[.3, -.48], [.12, .24]]),
              joint_vel=torch.tensor([[1., 2.], [3., 4.]]))


def test_costs_are_named_pre_reset_detached_and_generation_advances_after_consumption():
    observer, values = stream(), data()
    values.actuator_force.requires_grad_()
    phase, terminal = torch.tensor([2, 1]), torch.tensor([True, False])
    observer.begin(0, phase)
    observer.capture(values, terminal)
    phase.zero_()
    with torch.no_grad(): values.actuator_force.fill_(9)
    values.joint_vel.zero_()
    sample = observer.consume(terminal)
    assert sample.joint_names == ("b", "a") and sample.episode_generation.tolist() == [0, 0]
    assert sample.phase.tolist() == [2, 1] and sample.terminal.tolist() == [True, False]
    torch.testing.assert_close(sample.force_nm, data().actuator_force)
    torch.testing.assert_close(sample.speed_rad_s, torch.tensor([[2., 1.], [4., 3.]]))
    torch.testing.assert_close(sample.joint_cost, torch.tensor([[.25, .68], [.04, .16]], dtype=torch.float64))
    assert sample.mean_cost[0] == pytest.approx(.465)
    assert not sample.force_nm.requires_grad and not sample.joint_cost.requires_grad
    sample.episode_generation.fill_(99)
    sample.terminal.fill_(False)
    observer.begin(1, torch.tensor([0, 2]))
    observer.capture(data(), torch.tensor([False, True]))
    second = observer.consume(torch.tensor([False, True]))
    assert second.episode_generation.tolist() == [1, 0] and second.phase.tolist() == [0, 2]
    assert observer._phase is None and observer._snapshot is None and observer.next_step == 2


def test_stream_is_constant_retention_across_many_completed_episodes():
    observer = stream()
    values = data()
    for step in range(300):
        phase = torch.tensor([step % 3, 2])
        done = torch.tensor([step % 2 == 0, True])
        observer.begin(step, phase)
        observer.capture(values, done)
        observer.consume(done)
        assert observer._snapshot is None and observer._phase is None
        assert not any(isinstance(v, list) for v in vars(observer).values())
    assert observer._generation.tolist() == [150, 300]
    assert observer.next_step == 300 and observer._generation.numel() == 2


@pytest.mark.parametrize("fault", ["skip", "bool-step", "double-begin", "unarmed", "missing", "double-capture",
                                   "terminal-mismatch", "terminal-dtype", "phase", "layout", "double-consume"])
def test_capture_lifecycle_fails_closed(fault):
    observer = stream()
    done, phase = torch.tensor([True, False]), torch.tensor([0, 2])
    with pytest.raises(ValueError):
        if fault == "skip": observer.begin(1, phase)
        elif fault == "bool-step": observer.begin(False, phase)
        elif fault == "unarmed": observer.capture(data(), done)
        elif fault == "phase": observer.begin(0, torch.tensor([0, 3]))
        else:
            observer.begin(0, phase)
            if fault == "double-begin": observer.begin(0, phase)
            elif fault == "missing": observer.consume(done)
            elif fault == "layout": observer.capture(NS(actuator_force=torch.ones(2, 3), joint_vel=torch.ones(2, 3)), done)
            else:
                observer.capture(data(), done)
                if fault == "double-capture": observer.capture(data(), done)
                elif fault == "terminal-mismatch": observer.consume(~done)
                elif fault == "terminal-dtype": observer.consume(done.int())
                else:
                    observer.consume(done)
                    observer.consume(done)


@pytest.mark.parametrize("bad,field", [(float("nan"), "actuator_force"), (float("inf"), "joint_vel"),
                                       (1e300, "actuator_force")])
def test_nonfinite_sample_or_cost_never_advances_or_becomes_a_zero_reward(bad, field):
    observer = stream()
    values = NS(actuator_force=data().actuator_force.double(), joint_vel=data().joint_vel.double())
    getattr(values, field)[0, 0] = bad
    done = torch.tensor([True, False])
    observer.begin(0, torch.tensor([0, 1]))
    observer.capture(values, done)
    with pytest.raises(FloatingPointError): observer.consume(done)
    assert observer.next_step == 0 and observer._generation.tolist() == [0, 0]
    with pytest.raises(ValueError): observer.begin(1, torch.tensor([0, 1]))


@pytest.mark.parametrize("kwargs", [dict(stall_reference_nm=0), dict(stall_reference_nm=True),
    dict(stall_reference_nm=float("nan")), dict(soft_limit_fraction=0), dict(soft_limit_fraction=1.1),
    dict(over_limit_gain=-1), dict(over_limit_gain=float("inf"))])
def test_invalid_cost_models_rejected(kwargs):
    with pytest.raises(ValueError): MotorStepCostCfg(**kwargs)


def test_real_robot_mapping_and_installation_leave_rewards_observations_unchanged():
    from mjlab.entity import Entity
    from mjlab_microduck.hierarchical_obstacle_rollout import prepare_rollout_configs
    cfg, _ = prepare_rollout_configs(2, .4, .9, 0.)
    robot = Entity(cfg.scene.entities["robot"])
    observer = MotorStepStream.from_robot(robot, 2, device="cpu", cost_cfg=MotorStepCostCfg())
    assert len(observer.names) == 14 and tuple(robot.joint_names[i] for i in observer.joint_ids) == observer.names
    rewards, actor = cfg.rewards.copy(), cfg.observations["actor"]
    install_metric(cfg)
    assert cfg.metrics[TERM].func is capture_metric and not cfg.metrics[TERM].per_substep
    assert cfg.rewards == rewards and cfg.observations["actor"] is actor
    with pytest.raises(ValueError, match="already"): install_metric(cfg)
    cfg.metrics.pop(TERM)
    cfg.auto_reset = False
    with pytest.raises(ValueError, match="automatic reset"): install_metric(cfg)
    for key in ("reward_weight_applied", "policy_acceptance", "physical_motion_authorized",
                "runtime_equivalence_validated", "trainer_integration_validated"):
        assert observer.provenance()[key] is False


@pytest.mark.parametrize("reset_force", [0., 9.])
def test_actual_installed_mjlab_hook_over_repeated_autoresets(reset_force):
    # Reuse the existing synthetic physics/reset fixture; execute the actual
    # installed mjlab step method, not a locally reimplemented ordering.
    from test_motor_measurement_audit import make_env
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg
    observer = stream()
    observed, plain = make_env(None, reset_force), make_env(None, reset_force)
    observed._microduck_motor_step_stream = observer
    observed.metrics_manager = MetricsManager({TERM: MetricsTermCfg(func=capture_metric)}, observed)
    rng = torch.random.get_rng_state().clone()
    for step in range(3):
        observer.begin(step, torch.tensor([step % 3, 1]))
        a = ManagerBasedRlEnv.step(observed, torch.ones(2, 2))
        b = ManagerBasedRlEnv.step(plain, torch.ones(2, 2))
        sample = observer.consume(a[2] | a[3])
        assert sample.episode_generation.tolist() == [step, 0]
        assert sample.force_nm[0, 1] == pytest.approx(-.48)
        assert observed.scene["robot"].data.actuator_force[0, 1] == reset_force
        for x, y in zip(a[1:4], b[1:4]): torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(a[0]["actor"], b[0]["actor"], rtol=0, atol=0)
        torch.testing.assert_close(observed.action_manager.action, plain.action_manager.action, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert observed._sim_step_counter == plain._sim_step_counter == 6
