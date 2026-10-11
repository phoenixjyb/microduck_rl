from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microduck import integrated_motor_refinement as refinement


def budget_recipe(*, control_updates=1000, budget_updates=4000,
                  training_caps=None, arms=("control", "budget")):
    return SimpleNamespace(
        ARMS=arms,
        MODES={
            "control": (256, control_updates, 1039),
            "budget": (256, budget_updates, 1039),
        },
        CAP_SECONDS=2100,
        TRAINING_CAP_SECONDS=training_caps or {"control": 2100, "budget": 7200},
    )


def test_legacy_recipe_defaults_to_1000_updates_and_selection_wording():
    control = refinement.training_contract(refinement, "control")
    motor = refinement.training_contract(refinement, "motor")
    assert control == dict(arm="control", worlds=256, updates=1000, seed=863,
                          cap_seconds=2100, final_iteration=999)
    assert motor["updates"] == 1000 and motor["final_iteration"] == 999
    plan = refinement.evaluation_plan(refinement)
    assert plan["selection"] == "both final999 only; unchanged held-out protocol"
    assert "final_iterations" not in plan


def test_heterogeneous_budget_selects_only_declared_final_checkpoint():
    recipe = budget_recipe()
    plan = refinement.evaluation_plan(recipe)
    budget = plan["contracts"]["budget"]
    assert budget["updates"] == 4000 and budget["final_iteration"] == 3999
    assert refinement.checkpoint_path(Path("run"), "budget", budget) == Path("run/budget/model_3999.pt")
    assert plan["selection"] == "declared final checkpoints only; unchanged held-out protocol"
    assert plan["final_iterations"] == {"control": 999, "budget": 3999}


@pytest.mark.parametrize("arm", ["unknown", True, None])
def test_training_contract_rejects_undeclared_arm(arm):
    with pytest.raises(ValueError, match="declared continuation arm"):
        refinement.training_contract(budget_recipe(), arm)


@pytest.mark.parametrize("mode", [
    (64, 1000, 7),
    (True, 1000, 7),
    (256, False, 7),
    (256, 0, 7),
    (256, 10001, 7),
    (256, 1, True),
    (256, 1, 7, 9),
    [256, 1, 7],
])
def test_training_contract_rejects_malformed_or_unbounded_mode(mode):
    recipe = budget_recipe()
    recipe.MODES["budget"] = mode
    with pytest.raises(ValueError):
        refinement.training_contract(recipe, "budget")


@pytest.mark.parametrize("arms,caps", [
    (("control", "budget"), {"control": 2100}),
    (("control", "budget"), {"control": 2100, "budget": 0}),
    (("control", "budget"), {"control": 2100, "budget": True}),
    (("control", "control"), {"control": 2100}),
    (("treatment", "budget"), {"treatment": 2100, "budget": 7200}),
])
def test_training_contract_rejects_invalid_arm_or_cap_declarations(arms, caps):
    recipe = budget_recipe(arms=arms, training_caps=caps)
    with pytest.raises(ValueError):
        refinement.training_contract(recipe, arms[-1])


@pytest.mark.parametrize("elapsed", [float("nan"), float("inf"), float("-inf"), True, 0., "10"])
def test_benchmark_projection_rejects_nonfinite_or_invalid_elapsed(elapsed):
    contract = refinement.training_contract(budget_recipe(), "budget")
    with pytest.raises(ValueError, match="finite positive benchmark elapsed time"):
        refinement.benchmark_projection_seconds(elapsed, contract)


def test_benchmark_projection_scales_to_declared_arm_budget():
    recipe = budget_recipe()
    control = refinement.training_contract(recipe, "control")
    budget = refinement.training_contract(recipe, "budget")
    assert refinement.benchmark_projection_seconds(8., control) == 1200.
    assert refinement.benchmark_projection_seconds(8., budget) == 4800.


def test_training_caps_must_have_exact_arm_keys():
    recipe = budget_recipe(training_caps={"control": 2100, "budget": 7200, "extra": 1})
    with pytest.raises(ValueError, match="exactly match"):
        refinement.training_contract(recipe, "control")


def test_explicit_none_training_caps_do_not_fall_back_to_global_cap():
    recipe = budget_recipe()
    recipe.TRAINING_CAP_SECONDS = None
    with pytest.raises(ValueError, match="exactly match"):
        refinement.training_contract(recipe, "control")


@pytest.mark.parametrize("fault", [None, "source", "protocol", "mode", "updates", "worlds", "seed",
                                   "steps", "transitions", "status", "parent", "checkpoint"])
def test_receipt_requires_exact_declared_training_identity(monkeypatch, fault):
    recipe = budget_recipe()
    recipe.PROTOCOL, recipe.PARENT_SHA = "duration-test", "parent"
    contract = refinement.training_contract(recipe, "budget")
    receipt = dict(source="source", protocol=recipe.PROTOCOL, mode="budget",
                   status="training-complete-not-accepted", updates=4000, worlds=256,
                   seed=1039, steps=96000, transitions=24576000,
                   warm_start=dict(parent_sha256="parent"), checkpoint_sha256="weights")
    monkeypatch.setattr(refinement.base, "sha256", lambda path: "weights")
    if fault == "parent": receipt["warm_start"]["parent_sha256"] = "other"
    elif fault == "checkpoint": receipt["checkpoint_sha256"] = "other"
    elif fault is not None:
        receipt[fault] = receipt[fault]+1 if type(receipt[fault]) is int else "other"
    if fault is None:
        refinement.validate_training_receipt(receipt, recipe, "source", contract, Path("model_3999.pt"))
    else:
        with pytest.raises(ValueError, match="exact declared"):
            refinement.validate_training_receipt(receipt, recipe, "source", contract, Path("model_3999.pt"))


@pytest.mark.parametrize("phase,worlds,updates,seed", [("smoke", 64, 5, 1049), ("benchmark", 256, 10, 1051)])
@pytest.mark.parametrize("fault", [None, "mode", "seed", "worlds", "updates", "steps", "transitions"])
def test_preflight_identity_is_not_inferred_from_receipt_location(monkeypatch, phase, worlds, updates, seed, fault):
    from mjlab_microduck import integrated_training_budget_refinement as recipe
    contract = refinement.preflight_contract(recipe, phase, "budget")
    assert contract == dict(arm=f"{phase}-budget", worlds=worlds, updates=updates,
                            seed=seed, final_iteration=updates-1)
    receipt = dict(source="source", protocol=recipe.PROTOCOL, mode=f"{phase}-budget",
                   status="training-complete-not-accepted", updates=updates, worlds=worlds,
                   seed=seed, steps=updates*24, transitions=updates*24*worlds,
                   warm_start=dict(parent_sha256=recipe.PARENT_SHA), checkpoint_sha256="weights")
    monkeypatch.setattr(refinement.base, "sha256", lambda path: "weights")
    if fault is not None: receipt[fault] = receipt[fault]+1 if type(receipt[fault]) is int else "other"
    checkpoint = refinement.checkpoint_path(Path("run"), contract["arm"], contract)
    if fault is None: refinement.validate_training_receipt(receipt, recipe, "source", contract, checkpoint)
    else:
        with pytest.raises(ValueError):
            refinement.validate_training_receipt(receipt, recipe, "source", contract, checkpoint)


@pytest.mark.parametrize("spec", [None, (64, 5), (64, 5, True), (64, 6, 1), (256, 5, 1)])
def test_preflight_contract_rejects_wrong_declaration(spec):
    recipe = budget_recipe()
    recipe.MODES["smoke-budget"] = spec
    with pytest.raises(ValueError): refinement.preflight_contract(recipe, "smoke", "budget")


def batch_recipe():
    recipe = budget_recipe(budget_updates=1000)
    recipe.TRAINING_WORLDS = {"control": 256, "budget": 1024}
    recipe.MODES["budget"] = (1024, 1000, 1039)
    recipe.MODES.update({"smoke-budget": (64, 5, 1049),
                         "benchmark-budget": (1024, 10, 1051)})
    return recipe


def test_explicit_world_comparison_requires_arm_matched_benchmark():
    recipe = batch_recipe()
    assert refinement.training_contract(recipe, "control")["worlds"] == 256
    assert refinement.training_contract(recipe, "budget")["worlds"] == 1024
    assert refinement.preflight_contract(recipe, "smoke", "budget")["worlds"] == 64
    assert refinement.preflight_contract(recipe, "benchmark", "budget")["worlds"] == 1024
    assert refinement.evaluation_plan(recipe)["selection"] == "both final999 only; unchanged held-out protocol"
    recipe.MODES["benchmark-budget"] = (256, 10, 1051)
    with pytest.raises(ValueError, match="preflight mode"):
        refinement.preflight_contract(recipe, "benchmark", "budget")


@pytest.mark.parametrize("declaration", [None, True, {}, {"control": 256},
    {"control": 256, "budget": 1024, "extra": 256},
    {"control": 1024, "budget": 256}, {"control": 256, "budget": 512},
    {"control": 256, "budget": 4096}, {"control": 256, "budget": 1024.0},
    {"control": 256, "budget": True}])
def test_world_comparison_rejects_ambiguous_or_unbounded_declarations(declaration):
    recipe = batch_recipe()
    recipe.TRAINING_WORLDS = declaration
    with pytest.raises(ValueError, match="world declaration"):
        refinement.training_contract(recipe, "budget")


@pytest.mark.parametrize("worlds", [256, 64, 4096, True, 1024.0])
def test_full_world_count_cannot_differ_from_declared_arm(worlds):
    recipe = batch_recipe()
    recipe.MODES["budget"] = (worlds, 1000, 1039)
    with pytest.raises(ValueError, match="world count matches"):
        refinement.training_contract(recipe, "budget")


def test_legacy_recipe_cannot_silently_enable_larger_world_count():
    recipe = budget_recipe()
    recipe.MODES["budget"] = (1024, 1000, 1039)
    with pytest.raises(ValueError, match="exactly 256"):
        refinement.training_contract(recipe, "budget")

