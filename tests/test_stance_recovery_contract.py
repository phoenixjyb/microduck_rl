"""Pure contract checks for the frozen-policy D1 recovery declaration."""

import copy

import pytest

from mjlab_microduck import stance_cpu_replay_profile
from mjlab_microduck import stance_disturbance_contract as force
from mjlab_microduck import stance_recovery_contract as recovery


SOURCE = "a" * 40
PLANT = {
    "nbody": 4,
    "body_names": ["world", "pelvis", "trunk_base", "left_foot"],
    "body_name": "trunk_base",
    "body_id": 2,
    "selected_fields_sha256": "b" * 64,
}
_DEFAULT_PROFILE = object()


def checkpoint_identity(**changes):
    identity = {
        "source": recovery.TRAINING_SOURCE,
        "training_seed": recovery.TRAINING_SEED,
        "iteration": recovery.ITERATION,
        "purpose": "lean-replication",
    }
    identity.update(changes)
    return identity


def make_binding(*, source=SOURCE, plant=None, case="+x", device="cpu",
                 identity=None, profile=_DEFAULT_PROFILE):
    return recovery.binding(
        source,
        PLANT if plant is None else plant,
        case,
        device,
        checkpoint_identity() if identity is None else identity,
        stance_cpu_replay_profile.expected_receipt() if profile is _DEFAULT_PROFILE else profile,
    )


def test_declaration_matches_the_frozen_policy_and_ten_step_pulse_predeclaration():
    result = recovery.declaration(SOURCE, PLANT)

    assert result["protocol"] == "football-b1d-frozen-recovery-v1"
    assert result["source"] == SOURCE
    assert result["training_source"] == "be2d59661af293b0d67ae20d2e16db50514cce14"
    assert (result["training_seed"], result["checkpoint_iteration"],
            result["checkpoint_file"], result["checkpoint_sha256"]) == (
                577, 255, "model_255.pt",
                "2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5")
    assert (result["evaluation_seed"], result["actor_dim"], result["action_dim"]) == (619, 44, 10)
    assert (result["worlds_per_case"], result["actor_device"], result["dt"]) == (1, "cpu", .002)
    assert result["cases"] == list(force_name for force_name, _ in force.CASES)
    assert result["cases"] == ["zero-wrench", "+x", "-x", "+y", "-y"]
    assert (result["onset_physics_step"], result["pulse_physics_steps"],
            result["total_physics_steps"], result["policy_ticks"]) == (500, 10, 2500, 250)
    assert result["pulse_newtons"] == 2.
    assert result["impulse_newton_seconds"] == pytest.approx(2. * 10 * .002)
    assert (result["frame"], result["wrench_order"], result["application"]) == (
        "world", "force-then-torque", "body-inertial-com")
    assert result["pre_forward_pulse_only"] is True
    assert result["clear_before_unforced_post"] is True
    assert result["optimizer_steps"] == 0
    assert result["first_attempt_only"] is True and result["auto_reset"] is False
    assert result["actor_observation_expanded"] is False and result["raw_perception"] is False
    assert all(value is False for value in recovery.FALSE_FLAGS.values())
    assert all(result[key] is False for key in recovery.FALSE_FLAGS)


@pytest.mark.parametrize("change", [
    {"nbody": True},
    {"body_id": True},
    {"body_id": 0},
    {"body_id": 4},
    {"body_names": ["world", "trunk_base", "trunk_base", "foot"]},
    {"body_names": ["world", "pelvis", "other", "left_foot"]},
    {"body_name": "pelvis"},
    {"selected_fields_sha256": float("nan")},
])
def test_declaration_delegates_compiled_body_table_and_finite_binding_checks(change):
    plant = copy.deepcopy(PLANT)
    plant.update(change)
    with pytest.raises(ValueError):
        recovery.declaration(SOURCE, plant)


@pytest.mark.parametrize("step,expected_x", [
    (499, [0, 0, 0, 0, 0, 0]),
    (500, [2, 0, 0, 0, 0, 0]),
    (509, [2, 0, 0, 0, 0, 0]),
    (510, [0, 0, 0, 0, 0, 0]),
    (2500, [0, 0, 0, 0, 0, 0]),
])
def test_scheduler_edges_are_exact_pre_euler_counters(step, expected_x):
    xfrc, qfrc = recovery.expected_wrenches(["+x"], [step], [True], 4, 2)
    assert xfrc == [[ [0] * 6, [0] * 6, expected_x, [0] * 6 ]]
    assert qfrc == [[0] * force.NV]
    assert recovery.validate_wrenches(xfrc, qfrc, ["+x"], [step], [True], 4, 2)


def test_scheduler_builds_exact_full_two_world_force_matrices_and_masks_closed_rows():
    cases = ["+x", "-y"]
    steps = [500, 509]
    accepted = [True, True]
    xfrc, qfrc = recovery.expected_wrenches(cases, steps, accepted, 4, 2)
    assert xfrc == [
        [[0] * 6, [0] * 6, [2, 0, 0, 0, 0, 0], [0] * 6],
        [[0] * 6, [0] * 6, [0, -2, 0, 0, 0, 0], [0] * 6],
    ]
    assert qfrc == [[0] * force.NV, [0] * force.NV]
    assert recovery.validate_wrenches(xfrc, qfrc, cases, steps, accepted, 4, 2)

    masked_xfrc, masked_qfrc = recovery.expected_wrenches(
        cases, steps, [False, True], 4, 2)
    assert masked_xfrc[0] == [[0] * 6 for _ in range(4)]
    assert masked_xfrc[1] == xfrc[1]
    assert masked_qfrc == qfrc

    zero_xfrc, zero_qfrc = recovery.expected_wrenches(
        ["zero-wrench", "+x"], steps, [True, True], 4, 2)
    assert zero_xfrc[0] == [[0] * 6 for _ in range(4)]
    assert zero_xfrc[1] == xfrc[0]
    assert zero_qfrc == qfrc


@pytest.mark.parametrize("cases", [[], ["diagonal"], ["+x", "+x"], [True]])
def test_unknown_duplicate_boolean_or_empty_case_sets_are_rejected(cases):
    with pytest.raises(ValueError):
        recovery.cases_checked(cases)


@pytest.mark.parametrize("steps", [
    [True], [500.0], [-1], [2501], [], [500, 500],
])
def test_scheduler_rejects_boolean_float_out_of_range_or_misaligned_counters(steps):
    with pytest.raises(ValueError):
        recovery.expected_wrenches(["+x"], steps, [True], 4, 2)


@pytest.mark.parametrize("accepted", [[1], [False, True], [None], ["true"]])
def test_scheduler_requires_exact_boolean_accepted_mask(accepted):
    with pytest.raises(ValueError):
        recovery.expected_wrenches(["+x"], [500], accepted, 4, 2)


def _scheduled_two_worlds():
    cases = ["+x", "-y"]
    steps = [500, 509]
    accepted = [True, True]
    xfrc, qfrc = recovery.expected_wrenches(cases, steps, accepted, 4, 2)
    return xfrc, qfrc, cases, steps, accepted


@pytest.mark.parametrize("damage", [
    "malformed-shape", "off-body", "wrong-direction", "wrong-magnitude",
    "torque", "nonfinite", "boolean", "generalized-force",
])
def test_wrench_validator_refuses_malformed_or_undeclared_force(damage):
    xfrc, qfrc, cases, steps, accepted = _scheduled_two_worlds()
    if damage == "malformed-shape":
        xfrc[0].pop()
    elif damage == "off-body":
        xfrc[0][1][0] = 2
    elif damage == "wrong-direction":
        xfrc[0][2][1] = 2
    elif damage == "wrong-magnitude":
        xfrc[0][2][0] = 3
    elif damage == "torque":
        xfrc[0][2][3] = 1
    elif damage == "nonfinite":
        xfrc[0][2][0] = float("inf")
    elif damage == "boolean":
        xfrc[0][0][0] = False
    elif damage == "generalized-force":
        qfrc[1][0] = .001

    with pytest.raises(ValueError):
        recovery.validate_wrenches(xfrc, qfrc, cases, steps, accepted, 4, 2)


@pytest.mark.parametrize("field,value", [
    ("source", "A" * 40),
    ("source", "g" * 40),
    ("source", "a" * 39),
    ("device", "cuda:1"),
    ("device", True),
    ("identity", {"source": "f" * 40}),
    ("identity", checkpoint_identity(source="f" * 40)),
    ("identity", checkpoint_identity(training_seed=True)),
    ("identity", checkpoint_identity(training_seed=578)),
    ("identity", checkpoint_identity(iteration=True)),
    ("identity", checkpoint_identity(iteration=254)),
    ("identity", checkpoint_identity(purpose="pilot")),
    ("profile", None),
])
def test_binding_rejects_wrong_source_checkpoint_policy_backend_or_profile(field, value):
    args = dict(source=SOURCE, plant=PLANT, case="+x", device="cpu",
                identity=checkpoint_identity(),
                profile=stance_cpu_replay_profile.expected_receipt())
    if field in ("source", "device"):
        args[field] = value
    elif field == "identity":
        args["identity"] = value
    else:
        args["profile"] = value
    with pytest.raises(ValueError):
        make_binding(**args)


def test_declaration_and_binding_return_owned_plant_checkpoint_and_profile_copies():
    plant = copy.deepcopy(PLANT)
    declaration = recovery.declaration(SOURCE, plant)
    plant["body_names"][2] = "changed"
    assert declaration["plant"]["body_names"][2] == "trunk_base"

    identity = checkpoint_identity()
    profile = stance_cpu_replay_profile.expected_receipt()
    bound = make_binding(identity=identity, profile=profile)
    identity["training_seed"] = 578
    profile["settings"]["OMP_NUM_THREADS"] = "99"
    assert bound["checkpoint_identity"]["training_seed"] == 577
    assert bound["cpu_math_profile"]["settings"]["OMP_NUM_THREADS"] == "1"
    assert bound["case"] == "+x" and bound["capture_device"] == "cpu"
