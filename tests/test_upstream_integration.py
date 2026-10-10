"""Do not redefine the historical athletics recipe during upstream sync."""

from mjlab.tasks.registry import load_env_cfg, list_tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.athletics_velocity_env_cfg import (
    make_microduck_velocity_env_cfg as athletics,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg as walking,
)
from mjlab_microduck.tasks.hop import make_hop_variant


def test_walking_and_athletics_have_separate_curricula_and_rewards():
    old, new = athletics(), walking()
    assert "velocity_command_ranges" in old.curriculum
    assert "velocity_command_ranges" not in new.curriculum
    assert old.rewards["com_height_target"].params == {
        "target_height_min": 0.11, "target_height_max": 0.14,
    }
    assert "com_height_target" not in new.rewards
    assert "stillness_at_zero_command" in old.rewards
    assert "stillness_at_zero_command" not in new.rewards
    assert old.rewards["upright"].weight == 1.0
    assert new.rewards["upright"].weight == 2.0


def test_registry_keeps_upstream_and_custom_task_families():
    tasks = set(list_tasks())
    assert {
        "Mjlab-Velocity-Flat-MicroDuck",
        "Mjlab-VelStand-Rough-Backlash-MicroDuck",
        "Mjlab-Run-MotorAware-Flat-MicroDuck",
        "Mjlab-Run-Obstacle-Assisted-PhaseSpeed-Flat-MicroDuck",
        "Mjlab-Hop-H1T-Flat-Sprung-K3900-MicroDuck",
    } <= tasks
    run = load_env_cfg("Mjlab-Run-MotorAware-Flat-MicroDuck")
    assert run.rewards["upright"].weight == 1.0
    assert "motor_envelope_monitor" in run.rewards
    assert "velocity_command_ranges" in run.curriculum


def test_phase_command_drops_only_inert_turn_sampling_knobs():
    cfg = athletics()
    cfg.commands["twist"].turn_in_place_min_frac = 0.75
    cfg.commands["twist"].rel_turn_in_place_envs = 0.5
    hopped = make_hop_variant(cfg)
    assert hopped.commands["twist"].class_type is mdp.GroundPickPhaseCommand
    assert not hasattr(hopped.commands["twist"], "turn_in_place_min_frac")


def test_unknown_command_field_still_fails_closed():
    import pytest

    cfg = athletics()
    cfg.commands["twist"].unreviewed_sampling_knob = 0.5
    with pytest.raises(ValueError, match="unreviewed_sampling_knob"):
        make_hop_variant(cfg)


def test_smoke_finite_check_covers_saved_models_and_adam():
    import importlib.util
    from pathlib import Path
    import torch

    path = Path(__file__).resolve().parents[1] / "scripts/smoke_upstream_integration.py"
    spec = importlib.util.spec_from_file_location("integration_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.finite_tree({"actor": torch.ones(2), "adam": [{"step": 1.0}]})
    assert not module.finite_tree({"adam": [{"exp_avg": torch.tensor(float("nan"))}]})
    assert not module.finite_tree({"loss": float("inf")})
    from tensordict import TensorDict
    assert not module.finite_tree(TensorDict(
        {"actor": torch.full((1, 61), float("nan"))}, batch_size=[1],
    ))
    assert module.TASKS == (
        "Mjlab-Velocity-Flat-MicroDuck", "Mjlab-Run-MotorAware-Flat-MicroDuck",
    )
