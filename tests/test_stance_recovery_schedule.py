"""Pure finite declaration/math checks for the proposed per-row adapter."""

from copy import deepcopy

import pytest

from mjlab_microduck import stance_recovery_schedule as schedule


SOURCE = "a" * 40


def plan(ids, *, stage="dose", split="training"):
    return schedule.declaration(SOURCE, stage, split, ids)


def test_schedule_binds_source_split_stage_rows_and_all_nonadmission_flags():
    ids = ["+x-2n-10steps-t500", "+x-2n-10steps-t500", "zero-wrench"]
    value = plan(ids)
    assert value["protocol"] == schedule.PROTOCOL and value["source"] == SOURCE
    assert value["stage"] == "dose" and value["split"] == "training"
    assert value["worlds"] == 3 and value["cell_ids"] == ids
    assert value["row_cells"] == [value["row_cells"][0], value["row_cells"][0], value["row_cells"][2]]
    assert value["parent_checkpoint_sha256"]
    assert value["actor_dim"] == 44 and value["critic_dim"] == 50 and value["action_dim"] == 10
    assert value["total_steps"] == 2500 and value["dt"] == .002
    assert value["auto_reset"] is False and value["first_attempt_only"] is True
    assert value["execution_admitted"] is False
    assert all(value[key] is False for key in schedule.baseline.FALSE_FLAGS)


@pytest.mark.parametrize("source", [None, 3, "A"*40, "a"*39, "g"*40])
def test_schedule_rejects_invalid_source_hash(source):
    with pytest.raises(ValueError, match="exact schedule source revision"):
        schedule.declaration(source, "dose", "training", ["zero-wrench"])


@pytest.mark.parametrize("stage,split,ids", [
    ("unknown", "training", ["zero-wrench"]),
    ("gentle", "other", ["zero-wrench"]),
    ("gentle", "training", []),
    ("gentle", "training", ["zero-wrench"]*65),
    ("gentle", "training", ["diagonal-++-2n-10steps-t375"]),
    ("gentle", "held-out", ["not-a-cell"]),
    ("gentle", "training", [True]),
])
def test_schedule_rejects_invalid_stage_split_or_row_catalog(stage, split, ids):
    with pytest.raises(ValueError):
        schedule.declaration(SOURCE, stage, split, ids)


def test_heldout_diagonal_rows_are_available_only_on_heldout_split():
    diagonal = "diagonal-++-2n-10steps-t375"
    held = plan([diagonal], stage="gentle", split="held-out")
    assert held["row_cells"][0]["onset_step"] == 375
    with pytest.raises(ValueError, match="only declared split cells"):
        plan([diagonal], stage="gentle", split="training")


def test_checked_schedule_owns_exact_typed_rows_and_rejects_any_mutation():
    declaration = plan(["+x-2n-10steps-t500", "zero-wrench"])
    checked = schedule.checked(declaration)
    assert checked == declaration and checked is not declaration
    assert checked["row_cells"] is not declaration["row_cells"]
    for mutation in (
        lambda x: x.update(worlds=1),
        lambda x: x["row_cells"][0]["force_world_newtons"].__setitem__(0, 2.1),
        lambda x: x.update(training_admitted=True),
        lambda x: x.update(cell_ids=["+x-2n-10steps-t500"]),
        lambda x: x.update(worlds=True),
    ):
        changed = deepcopy(declaration); mutation(changed)
        with pytest.raises(ValueError, match="unchanged exact typed schedule declaration"):
            schedule.checked(changed)


def test_heterogeneous_row_clocks_masks_force_duration_and_zero_control_capture_window():
    value = plan(["+x-2n-10steps-t500", "+y-4n-10steps-t750", "zero-wrench"])
    steps = [500, 750, 500]
    accepted = [True, True, True]
    xfrc, qfrc = schedule.expected_wrenches(value, steps, accepted, 4, 2)
    assert schedule.window_mask(value, steps, accepted) == [True, True, True]
    assert len(xfrc) == 3 and all(len(world) == 4 for world in xfrc)
    assert xfrc[0][2] == [2., 0., 0., 0., 0., 0.]
    assert xfrc[1][2] == [0., 4., 0., 0., 0., 0.]
    assert xfrc[2] == [[0.]*6 for _ in range(4)]
    assert qfrc == [[0.]*20 for _ in range(3)]
    assert schedule.validate_wrenches(xfrc, qfrc, value, steps, accepted, 4, 2)

    rejected = [False, True, False]
    x_inactive, q_inactive = schedule.expected_wrenches(value, steps, rejected, 4, 2)
    assert schedule.window_mask(value, steps, rejected) == [False, True, False]
    assert x_inactive[0] == x_inactive[2] == [[0.]*6 for _ in range(4)]
    assert x_inactive[1][2][:3] == [0., 4., 0.] and q_inactive == [[0.]*20]*3


@pytest.mark.parametrize("step,active", [(499, False), (500, True), (509, True), (510, False), (520, False)])
def test_exact_ten_step_window_edges(step, active):
    value = plan(["+x-2n-10steps-t500"])
    assert schedule.window_mask(value, [step], [True]) == [active]
    xfrc, qfrc = schedule.expected_wrenches(value, [step], [True], 3, 1)
    assert xfrc[0][1][0] == (2. if active else 0.)
    assert qfrc == [[0.]*20]


def test_twenty_step_duration_and_four_newton_magnitude_are_exact():
    value = plan(["-y-2n-20steps-t500", "+y-4n-10steps-t750"])
    for step, active in ((519, True), (520, False)):
        xfrc, qfrc = schedule.expected_wrenches(value, [step, 750], [True, False], 3, 1)
        assert xfrc[0][1] == ([0., -2., 0., 0., 0., 0.] if active else [0.]*6)
        assert xfrc[1][1] == [0.]*6 and qfrc == [[0.]*20]*2


@pytest.mark.parametrize("steps,accepted,nbody,body", [
    ([True], [True], 3, 1), ([500.], [True], 3, 1), ([-1], [True], 3, 1),
    ([2501], [True], 3, 1), ([500], [1], 3, 1), ([500], [True, False], 3, 1),
    ([500], [True], True, 1), ([500], [True], 3., 1), ([500], [True], 3, True),
    ([500], [True], 3, 3), ([500], [True], 1, 0),
])
def test_clock_mask_and_compiled_body_binding_are_exactly_typed(steps, accepted, nbody, body):
    value = plan(["+x-2n-10steps-t500"])
    with pytest.raises(ValueError):
        schedule.expected_wrenches(value, steps, accepted, nbody, body)


@pytest.mark.parametrize("damage", ["shape", "wrong-force", "torque", "nonzero-q", "nonfinite", "boolean"])
def test_complete_force_array_validation_rejects_malformed_or_undeclared_arrays(damage):
    value = plan(["+x-2n-10steps-t500"])
    xfrc, qfrc = schedule.expected_wrenches(value, [500], [True], 3, 1)
    if damage == "shape": xfrc[0].pop()
    elif damage == "wrong-force": xfrc[0][1][0] = 4.
    elif damage == "torque": xfrc[0][1][3] = .1
    elif damage == "nonzero-q": qfrc[0][0] = .1
    elif damage == "nonfinite": xfrc[0][1][0] = float("inf")
    else: xfrc[0][1][0] = True
    with pytest.raises(ValueError):
        schedule.validate_wrenches(xfrc, qfrc, value, [500], [True], 3, 1)
