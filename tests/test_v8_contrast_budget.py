"""Contracts for explicit per-arm V8 contrast training budgets."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from mjlab_microduck import v8_gait_scale_contrast as contrast


MODES = {
    "smoke": (64, 5, 947),
    "benchmark": (256, 10, 953),
    "foundation": (256, 1000, 967),
}
STAGE_SPECS = {
    "control": {**MODES},
    "gait-scale": {**MODES, "foundation": (256, 2000, 967)},
}


def _experiment(stage_specs=None, arms=("control", "gait-scale")):
    values = SimpleNamespace(ARMS=arms, MODES=copy.deepcopy(MODES))
    if stage_specs is not None:
        values.STAGE_SPECS = copy.deepcopy(stage_specs)
    return values


@pytest.mark.parametrize("mode,spec", MODES.items())
@pytest.mark.parametrize("arm", ("control", "gait-scale"))
def test_absent_stage_specs_preserve_shared_mode_defaults(mode, spec, arm):
    assert contrast.stage_spec(_experiment(), mode, arm) == spec


@pytest.mark.parametrize("arm,updates", (("control", 1000), ("gait-scale", 2000)))
def test_explicit_budget_matrix_selects_only_foundation_iteration_count(arm, updates):
    recipe = _experiment(STAGE_SPECS)
    assert contrast.stage_spec(recipe, "foundation", arm) == (256, updates, 967)
    assert contrast.stage_spec(recipe, "smoke", arm) == MODES["smoke"]
    assert contrast.stage_spec(recipe, "benchmark", arm) == MODES["benchmark"]
    assert contrast.final_iteration(recipe, arm) == updates - 1


def test_explicit_matrix_uses_declared_treatment_arm_name():
    specs = {"control": copy.deepcopy(MODES),
        "extended": {**MODES, "foundation": (256, 2000, 967)}}
    recipe = _experiment(specs, arms=("control", "extended"))
    assert contrast.stage_spec(recipe, "foundation", "extended") == (256, 2000, 967)
    assert contrast.final_iteration(recipe, "extended") == 1999


def test_final_evaluation_and_pairing_resolve_arm_specific_foundation_checkpoints(tmp_path):
    recipe = _experiment(STAGE_SPECS)
    assert contrast.foundation_checkpoint(tmp_path, recipe, "control") == (
        tmp_path / "control/foundation/model_999.pt")
    assert contrast.foundation_checkpoint(tmp_path, recipe, "gait-scale") == (
        tmp_path / "gait-scale/foundation/model_1999.pt")


@pytest.mark.parametrize("mutation", ("missing-arm", "missing-mode", "wrong-phase", "bool-budget", "other-budget"))
def test_explicit_budget_matrix_rejects_every_unapproved_shape(mutation):
    specs = copy.deepcopy(STAGE_SPECS)
    if mutation == "missing-arm":
        del specs["gait-scale"]
    elif mutation == "missing-mode":
        del specs["control"]["benchmark"]
    elif mutation == "wrong-phase":
        specs["gait-scale"]["foundation"] = (256, 2000, 953)
    elif mutation == "bool-budget":
        specs["gait-scale"]["foundation"] = (256, True, 967)
    else:
        specs["gait-scale"]["foundation"] = (256, 1999, 967)
    with pytest.raises(ValueError, match="exact supported explicit stage-spec"):
        contrast.stage_spec(_experiment(specs), "foundation", "gait-scale")


@pytest.mark.parametrize("updates", (0, -1, 2001, True, 1.5))
def test_projection_rejects_invalid_foundation_iteration_counts(updates):
    with pytest.raises(ValueError, match="integer foundation budget"):
        contrast.projection_seconds(10., updates)


def test_projection_scales_benchmark_to_selected_foundation_budget():
    assert contrast.projection_seconds(10.) == 1560.
    assert contrast.projection_seconds(10., 1000) == 1560.
    assert contrast.projection_seconds(10., 2000) == 3060.


@pytest.mark.parametrize("arm,updates", (("control", 1000), ("gait-scale", 2000)))
def test_config_admission_checks_arm_selected_worlds_seeds_and_iterations(arm, updates):
    spec = contrast.stage_spec(_experiment(STAGE_SPECS), "foundation", arm)
    cfg = SimpleNamespace(seed=967, scene=SimpleNamespace(num_envs=256,
        terrain=SimpleNamespace(num_envs=256)))
    agent = SimpleNamespace(seed=967, max_iterations=updates)
    contrast._verify_config_budget(cfg, agent, spec)
    agent.max_iterations += 1
    with pytest.raises(ValueError, match="match declared stage budget"):
        contrast._verify_config_budget(cfg, agent, spec)
