import pytest

from mjlab_microduck import integrated_rollout_batch_refinement as batch
from mjlab_microduck import integrated_training_budget_refinement as duration


def test_declares_only_the_rollout_batch_factor_and_keeps_parent_pipeline():
    assert batch.PROTOCOL == "integrated-rollout-batch-continuation-v1"
    assert batch.ARMS == ("control", "batch")
    assert batch.TRAINING_WORLDS == {"control": 256, "batch": 1024}
    assert batch.MODES == {
        "smoke-control": (64, 5, 1073),
        "smoke-batch": (64, 5, 1073),
        "benchmark-control": (256, 10, 1079),
        "benchmark-batch": (1024, 10, 1079),
        "control": (256, 1000, 1087),
        "batch": (1024, 1000, 1087),
    }
    assert batch.CAP_SECONDS == 2100
    assert batch.TRAINING_CAP_SECONDS == {"control": 2100, "batch": 4500}
    assert batch.PARENT == duration.PARENT
    assert batch.PARENT_SHA == duration.PARENT_SHA
    assert batch.PARENT_LEARNING_RATE == duration.PARENT_LEARNING_RATE
    assert batch.warm_load is duration.warm_load
    assert batch.evaluate_case is duration.evaluate_case
    assert batch.decision is duration.decision
    assert batch.EXPOSURE_BINS is duration.EXPOSURE_BINS


@pytest.mark.parametrize("phase,seed,updates", [
    ("smoke-", 1073, 5), ("benchmark-", 1079, 10), ("", 1087, 1000),
])
def test_arm_configs_match_except_declared_world_count_and_run_name(phase, seed, updates):
    control, control_agent = batch.prepare_config(phase + "control")
    treatment, treatment_agent = batch.prepare_config(phase + "batch")
    expected_control_worlds = 64 if phase == "smoke-" else 256
    expected_batch_worlds = 64 if phase == "smoke-" else 1024
    assert control.seed == treatment.seed == seed
    assert control_agent.seed == treatment_agent.seed == seed
    assert control.scene.num_envs == control.scene.terrain.num_envs == expected_control_worlds
    assert treatment.scene.num_envs == treatment.scene.terrain.num_envs == expected_batch_worlds
    assert control_agent.max_iterations == treatment_agent.max_iterations == updates
    assert control_agent.experiment_name == treatment_agent.experiment_name == batch.PROTOCOL
    assert control_agent.run_name != treatment_agent.run_name
    assert control_agent.save_interval == treatment_agent.save_interval == 250

    # Normalize just the intended world-count difference and run tag, then
    # compare every nested environment/optimizer setting.
    treatment.scene.num_envs = control.scene.num_envs
    treatment.scene.terrain.num_envs = control.scene.terrain.num_envs
    assert repr(control) == repr(treatment)
    treatment_agent.run_name = control_agent.run_name
    assert control_agent == treatment_agent

    assert control.rewards[duration.previous.TERM].weight == 0.
    assert control.rewards["motor_load"].weight == -1.
    assert control.rewards["joint_limit"].weight == -4.
    assert control.rewards["command_error"].weight == -.5
    assert control.actions["joint_pos"].clip is None
    assert not control.curriculum


def test_observer_reuses_duration_telemetry_but_accepts_only_new_modes():
    assert issubclass(batch.MotorObserver, duration.MotorObserver)
    for name in ("before", "step", "update"):
        assert getattr(batch.MotorObserver, name) is getattr(duration.MotorObserver, name)
    for mode in batch.MODES:
        observer = batch.MotorObserver(mode)
        assert observer.pending_exposure is None
        assert observer.exposure_steps == 0
        assert tuple(observer.exposure_counts) == batch.EXPOSURE_BINS
        assert tuple(observer.cumulative_exposure) == batch.EXPOSURE_BINS


@pytest.mark.parametrize("bad_mode", ["unknown", "benchmark", "smoke", "batch-control"])
def test_unknown_mode_rejected_by_config_and_observer(bad_mode):
    with pytest.raises(ValueError):
        batch.prepare_config(bad_mode)
    with pytest.raises(ValueError):
        batch.MotorObserver(bad_mode)

