"""CPU-hidden schema tests: a source-only proposal, never a learner/admission."""

from copy import deepcopy

import pytest

from mjlab_microduck import stance_disturbance_contract as force
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_lesson_plan as lesson


SOURCE = "a" * 40
PLANT = dict(nbody=3, body_names=["world", "trunk_base", "foot"],
             body_name="trunk_base", body_id=1)


def rowset(stage="gentle", *, checkpoints=(64, 128)):
    rows = []
    for evaluation, checkpoint in enumerate(checkpoints):
        for training_seed in lesson.TRAINING_SEEDS:
            for seed in lesson.HELD_OUT_SEEDS:
                for cell in lesson.cells(stage, held_out=True):
                    rows.append(dict(evaluation=evaluation, training_seed=training_seed,
                        seed=seed, cell=cell["id"], checkpoint=checkpoint,
                        first_attempts=128, attempts_passed=128,
                        complete_force_checks=True, raw_evidence_verified=True))
    return rows


def test_plan_binds_frozen_parent_architecture_and_is_never_an_admission():
    plan = lesson.plan(SOURCE)
    assert plan["protocol"] == lesson.PROTOCOL
    assert plan["source"] == SOURCE and plan["status"] == "source-plan-only"
    assert plan["parent"] == dict(source=baseline.TRAINING_SOURCE,
        training_seed=baseline.TRAINING_SEED, iteration=baseline.ITERATION,
        sha256=baseline.CHECKPOINT_SHA256, actor_dim=44, critic_dim=50,
        action_dim=10, restore="weights-only-fresh-optimizer")
    assert (plan["parent"]["training_seed"], plan["parent"]["iteration"],
            plan["parent"]["sha256"]) == (577, 255, "2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5")
    assert plan["training_seeds"] == [653, 659]
    assert plan["held_out_seeds"] == [671, 677, 683]
    assert set(plan["training_seeds"]).isdisjoint(plan["held_out_seeds"])
    assert plan["common_checkpoints"] == [64, 128, 192, 255]
    assert plan["learner_available"] is False and plan["execution_admitted"] is False
    assert plan["gpu_job_predeclared"] is False
    assert all(plan[flag] is False for flag in baseline.FALSE_FLAGS)


@pytest.mark.parametrize("source", [None, 42, "A" * 40, "a" * 39, "g" * 40])
def test_plan_rejects_invalid_source_revision(source):
    with pytest.raises(ValueError, match="exact proposed source"):
        lesson.plan(source)


@pytest.mark.parametrize("stage,training_count,held_count", [
    ("gentle", 5, 13), ("timing", 13, 21), ("dose", 37, 45),
])
def test_stages_have_exact_progressive_cell_counts_and_diagonals_heldout_only(stage, training_count, held_count):
    training = lesson.cells(stage)
    held = lesson.cells(stage, held_out=True)
    assert len(training) == training_count and len(held) == held_count
    training_ids = {cell["id"] for cell in training}
    held_ids = {cell["id"] for cell in held}
    assert "zero-wrench" in training_ids
    assert training_ids <= held_ids
    diagonal_ids = {cell["id"] for cell in held if cell["id"].startswith("diagonal-")}
    assert len(diagonal_ids) == 8
    assert not training_ids.intersection(diagonal_ids)
    diagonal_onsets = {cell["onset_step"] for cell in held if cell["id"] in diagonal_ids}
    assert diagonal_onsets == {375, 625}


def test_stage_cells_are_typed_finite_and_bind_one_world_frame_body_contract():
    plan = lesson.plan(SOURCE)
    stage = plan["stages"][0]
    assert stage["name"] == "gentle"
    for cell in stage["training_cells"] + stage["held_out_cells"]:
        assert cell["total_steps"] == 2500 and cell["dt"] == .002
        assert cell["application"] == "trunk-inertial-com"
        assert cell["first_attempt_only"] is True and cell["auto_reset"] is False
        assert cell["torque_world_nm"] == [0., 0., 0.]
        assert all(type(value) in (int, float) for value in cell["force_world_newtons"])
        assert len(cell["force_world_newtons"]) == len(cell["torque_world_nm"]) == 3
        assert cell["onset_step"] + cell["duration_steps"] <= cell["total_steps"]


def test_expected_wrench_has_exact_window_mask_body_force_and_zero_generalized_force():
    cell = next(c for c in lesson.cells("dose") if c["id"] == "+x-2n-10steps-t500")
    for step in (499, 510, 511, 2500):
        xfrc, qfrc = lesson.expected_wrench(cell, step, True, 4, 2)
        assert xfrc == [[0.]*6 for _ in range(4)]
        assert qfrc == [0.]*20
        assert len(xfrc) == 4 and all(len(body_row) == 6 for body_row in xfrc)
    for accepted in (False, True):
        for step in range(500, 510):
            xfrc, qfrc = lesson.expected_wrench(cell, step, accepted, 4, 2)
            expected = [0.]*6
            if accepted:
                expected[:3] = [2., 0., 0.]
            assert xfrc[2] == expected
            assert all(row == [0.]*6 for index, row in enumerate(xfrc) if index != 2)
            assert qfrc == [0.]*20
            assert xfrc[2][3:] == [0., 0., 0.]


@pytest.mark.parametrize("case,expected", [
    ("+x-2n-10steps-t500", [2., 0., 0.]), ("-x-2n-10steps-t500", [-2., 0., 0.]),
    ("+y-4n-10steps-t750", [0., 4., 0.]), ("-y-4n-10steps-t750", [0., -4., 0.]),
])
def test_expected_wrench_preserves_exact_2n_and_4n_cardinal_components(case, expected):
    cell = next(c for c in lesson.cells("dose") if c["id"] == case)
    xfrc, qfrc = lesson.expected_wrench(cell, cell["onset_step"], True, 3, 1)
    assert len(xfrc) == 3 and all(len(body_row) == 6 for body_row in xfrc)
    assert xfrc[1][:3] == cell["force_world_newtons"] == expected
    assert xfrc[1][3:] == [0., 0., 0.] and qfrc == [0.]*20


@pytest.mark.parametrize("bad", [
    None, "not-a-cell", {"id": "alien"},
    {"id": "+x-2n-10steps-t500", "force_world_newtons": [2., 0., 0.]},
])
def test_expected_wrench_rejects_missing_untyped_or_incomplete_cell(bad):
    with pytest.raises(ValueError, match="exact finite typed declared lesson cell"):
        lesson.expected_wrench(bad, 500, True, 3, 1)


def test_expected_wrench_rejects_mutated_and_nonfinite_cells():
    cell = next(c for c in lesson.cells("gentle") if c["id"] == "+x-2n-10steps-t500")
    changed = deepcopy(cell); changed["force_world_newtons"][0] = 2.0001
    with pytest.raises(ValueError, match="exact finite typed declared lesson cell"):
        lesson.expected_wrench(changed, 500, True, 3, 1)
    changed = deepcopy(cell); changed["force_world_newtons"][0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        lesson.expected_wrench(changed, 500, True, 3, 1)


@pytest.mark.parametrize("step", [True, False, 499.0, -1, 2501, "500"])
def test_expected_wrench_rejects_untyped_or_out_of_range_physics_clock(step):
    cell = lesson.cells("gentle")[1]
    with pytest.raises(ValueError, match="bounded planned pulse clock/mask"):
        lesson.expected_wrench(cell, step, True, 3, 1)


@pytest.mark.parametrize("accepted", [1, 0, None, "true"])
def test_expected_wrench_rejects_non_boolean_accepted_mask(accepted):
    cell = lesson.cells("gentle")[1]
    with pytest.raises(ValueError, match="bounded planned pulse clock/mask"):
        lesson.expected_wrench(cell, 500, accepted, 3, 1)


@pytest.mark.parametrize("nbody,body", [(True, 1), (3.0, 1), (1, 0), (3, True), (3, 3), (3, -1), (65, 1)])
def test_expected_wrench_rejects_invalid_compiled_body_binding(nbody, body):
    cell = lesson.cells("gentle")[1]
    with pytest.raises(ValueError, match="caller must bind actual ordered trunk body"):
        lesson.expected_wrench(cell, 500, True, nbody, body)


def test_expected_wrench_returns_exact_body_rows_with_no_torque_or_generalized_force():
    cell = lesson.cells("gentle")[1]
    xfrc, qfrc = lesson.expected_wrench(cell, 500, True, 3, 1)
    assert len(xfrc) == 3 and all(type(row) is list and len(row) == 6 for row in xfrc)
    assert xfrc[0] == xfrc[2] == [0.]*6
    assert xfrc[1] == [2., 0., 0., 0., 0., 0.]
    assert qfrc == [0.]*20


def test_promotion_pass_requires_all_seeds_cells_and_adjacent_distinct_snapshot_checkpoints():
    rows = rowset("gentle", checkpoints=(64, 128))
    result = lesson.promotion(rows, stage="gentle")
    assert result["numerical_proposal_passed"] is True
    assert result["common_checkpoints"] == [64, 128]
    assert result["rows_checked"] == len(rows)
    assert result["raw_provenance_enforced_by_this_pure_screen"] is False
    assert result["execution_admitted"] is False
    assert all(result[flag] is False for flag in baseline.FALSE_FLAGS)


@pytest.mark.parametrize("checkpoints", [(128, 192), (192, 255)])
def test_promotion_accepts_only_the_other_declared_adjacent_checkpoint_pairs(checkpoints):
    result = lesson.promotion(rowset("timing", checkpoints=checkpoints), stage="timing")
    assert result["numerical_proposal_passed"] is True
    assert result["common_checkpoints"] == list(checkpoints)


@pytest.mark.parametrize("checkpoints", [(64, 192), (128, 128), (255, 255), (64, 255)])
def test_promotion_rejects_nonadjacent_or_repeated_evaluation_snapshots(checkpoints):
    with pytest.raises(ValueError, match="two adjacent common declared checkpoints"):
        lesson.promotion(rowset("gentle", checkpoints=checkpoints), stage="gentle")


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong-seed", "wrong-training-seed",
                                     "wrong-cell", "wrong-evaluation", "bool-evaluation", "bool-seed",
                                     "float-checkpoint", "window-drift"])
def test_promotion_fails_closed_for_missing_duplicate_or_wrong_typed_identity(damage):
    rows = rowset("gentle")
    if damage == "missing": rows.pop()
    elif damage == "duplicate": rows[-1] = deepcopy(rows[0])
    elif damage == "wrong-seed": rows[0]["seed"] = 999
    elif damage == "wrong-training-seed": rows[0]["training_seed"] = 577
    elif damage == "wrong-cell": rows[0]["cell"] = "unknown"
    elif damage == "wrong-evaluation": rows[0]["evaluation"] = 2
    elif damage == "bool-evaluation": rows[0]["evaluation"] = True
    elif damage == "bool-seed": rows[0]["seed"] = True
    elif damage == "float-checkpoint": rows[0]["checkpoint"] = 64.0
    else: rows[-1]["checkpoint"] = 192
    with pytest.raises(ValueError):
        lesson.promotion(rows, stage="gentle")


@pytest.mark.parametrize("field,value", [
    ("first_attempts", 127), ("first_attempts", True), ("attempts_passed", 129),
    ("attempts_passed", False), ("complete_force_checks", 1), ("raw_evidence_verified", 1),
])
def test_promotion_rejects_incomplete_or_untyped_first_attempt_force_and_evidence_receipts(field, value):
    rows = rowset("gentle"); rows[0][field] = value
    with pytest.raises(ValueError, match="typed complete first-attempt and force receipts"):
        lesson.promotion(rows, stage="gentle")


@pytest.mark.parametrize("field", ["attempts_passed", "complete_force_checks", "raw_evidence_verified"])
def test_numeric_proposal_failure_does_not_admit_execution_or_claim_raw_provenance(field):
    rows = rowset("gentle")
    if field == "attempts_passed": rows[0][field] = 127
    else: rows[0][field] = False
    result = lesson.promotion(rows, stage="gentle")
    assert result["numerical_proposal_passed"] is False
    assert result["raw_provenance_enforced_by_this_pure_screen"] is False
    assert result["execution_admitted"] is False
    assert all(result[flag] is False for flag in baseline.FALSE_FLAGS)


@pytest.mark.parametrize("stage", [None, "unknown", True, 1])
def test_cells_and_promotion_reject_invalid_stage(stage):
    with pytest.raises(ValueError):
        lesson.cells(stage)
    with pytest.raises(ValueError):
        lesson.promotion([], stage=stage)
