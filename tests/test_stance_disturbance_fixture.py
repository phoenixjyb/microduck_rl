"""CPU full-robot physics evidence for the separate one-step force fixture."""

from __future__ import annotations

import copy

import pytest
import torch

from mjlab_microduck import stance_disturbance_contract as contract
from mjlab_microduck import stance_disturbance_fixture as fixture


SOURCE = "c" * 40


@pytest.fixture(scope="module")
def captured():
    assert not torch.cuda.is_initialized()
    result = fixture.capture_cases(SOURCE, device="cpu")
    assert not torch.cuda.is_initialized()
    return result


def assert_finite_tree(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_finite_tree(child)


def test_real_five_case_pipeline_is_one_cpu_euler_with_cleared_forces(captured):
    plan, rows = captured["plan"], captured["rows"]
    assert plan["protocol"] == contract.PROTOCOL
    assert captured["backend"] == {"torch_device": "cpu", "warp_is_cuda": False}
    assert [row["case"] for row in rows] == [name for name, _ in contract.CASES]
    assert len(rows) == 5
    assert all(captured[key] is False for key in contract.NO_ADMISSION)
    for row in rows:
        for phase in (row["before"], row["forced_pre"], row["integrated"], row["unforced_post"]):
            assert_finite_tree(phase["solved"])

    baseline = rows[0]["before"]
    for row, (case, _) in zip(rows, contract.CASES):
        assert fixture.exact_tree(row["before"], baseline)
        for phase_name in fixture.PHASES:
            phase = row[phase_name]
            expected_case = case if phase_name in ("forced_pre", "integrated") else "zero-wrench"
            contract.validate_applied(
                phase["inputs"]["xfrc_applied"].tolist(),
                phase["inputs"]["qfrc_applied"].tolist(),
                expected_case,
                plan["plant"]["nbody"],
                plan["plant"]["body_id"],
            )
            assert torch.equal(phase["inputs"]["ctrl"], baseline["inputs"]["ctrl"])
            assert all(torch.equal(phase["motor_fields"][k], baseline["motor_fields"][k])
                       for k in baseline["motor_fields"])

        start = row["before"]["solved"]["kinematics"]["time"]
        pre_time = row["forced_pre"]["solved"]["kinematics"]["time"]
        integrated = row["integrated"]["solved"]["kinematics"]
        post = row["unforced_post"]["solved"]["kinematics"]
        assert torch.equal(start, pre_time)
        assert torch.equal(integrated["time"], start + contract.DT)
        assert torch.equal(post["time"], integrated["time"])
        for name in ("qpos", "qvel", "time", "qacc_warmstart"):
            assert torch.equal(integrated[name], post[name])

    zero_acc = rows[0]["forced_pre"]["solved"]["dynamics"]["qacc"]
    assert all(float((row["forced_pre"]["solved"]["dynamics"]["qacc"] - zero_acc)
                     .abs().max()) > 1e-5 for row in rows[1:])


def test_prepared_ctrl_friction_damping_and_capture_storage_are_owned(captured):
    rows = captured["rows"]
    baseline = rows[0]["before"]
    assert not baseline["inputs"]["ctrl"].any()
    assert set(baseline["motor_fields"]) == {"dof_frictionloss", "dof_damping"}
    assert baseline["motor_fields"]["dof_frictionloss"].shape == (1, contract.NV)
    assert baseline["motor_fields"]["dof_damping"].shape == (1, contract.NV)
    assert (baseline["motor_fields"]["dof_frictionloss"] > 0).any()
    assert (baseline["motor_fields"]["dof_damping"] > 0).any()
    assert (rows[0]["before"]["solved"]["solver"]["nf"] == 14).all()

    snapshots = [row[phase] for row in rows for phase in fixture.PHASES]
    for field in fixture.INPUT_FIELDS:
        pointers = [snapshot["inputs"][field].untyped_storage().data_ptr() for snapshot in snapshots]
        assert len(pointers) == len(set(pointers)), f"{field} captures must own distinct copies"
    for field in baseline["motor_fields"]:
        pointers = [snapshot["motor_fields"][field].untyped_storage().data_ptr() for snapshot in snapshots]
        assert len(pointers) == len(set(pointers)), f"{field} captures must own distinct copies"


def test_fresh_cpu_replay_reexecutes_all_five_force_cases(captured):
    replay = fixture.replay(captured)
    assert replay["protocol"] == contract.PROTOCOL
    assert replay["force_path_replayed"] is True
    assert (replay["cases"], replay["euler_steps"]) == (5, 5)
    assert replay["cuda_initialized"] is False
    assert replay["maximum_solved_field_error"] <= max(contract.CHECK_TOL_ATOL, contract.CHECK_TOL_RTOL)
    assert all(replay[key] is False for key in contract.NO_ADMISSION)


def _tamper_force(value):
    value["rows"][1]["integrated"]["inputs"]["xfrc_applied"][0, 0, 0] = 1


def _tamper_generalized_force(value):
    value["rows"][1]["forced_pre"]["inputs"]["qfrc_applied"][0, 0] = 1


def _tamper_state(value):
    value["rows"][1]["forced_pre"]["solved"]["dynamics"]["qacc"][0, 0] += 1


def _remove_case(value):
    value["rows"].pop()


def _wrong_case(value):
    value["rows"][1]["case"] = "-y"


def _wrong_plant_binding(value):
    value["plan"]["plant"]["selected_plant"]["protocol"] = "mutated-plant"


def _wrong_order(value):
    value["rows"].reverse()


@pytest.mark.parametrize("mutate", [
    _tamper_force,
    _tamper_generalized_force,
    _tamper_state,
    _remove_case,
    _wrong_case,
    _wrong_plant_binding,
    _wrong_order,
])
def test_fresh_replay_rejects_mutated_force_state_and_binding(captured, monkeypatch, mutate):
    candidate = copy.deepcopy(captured)
    mutate(candidate)
    monkeypatch.setattr(fixture, "capture_cases", lambda *_args, **_kwargs: captured)
    with pytest.raises(ValueError):
        fixture.replay(candidate)


def test_undeclared_case_faults_before_euler_and_cannot_be_reused():
    env = fixture.SingleSubstepFixture(device="cpu")
    before_time = env._view("time").detach().cpu().clone()
    with pytest.raises(ValueError, match="undeclared force case"):
        env.run("diagonal")
    assert env.faulted is True and env.used is True
    assert torch.equal(env._view("time").detach().cpu(), before_time)
    with pytest.raises(ValueError, match="fresh nonfaulted one-use force fixture"):
        env.run("zero-wrench")


def test_successfully_used_fixture_cannot_run_a_second_substep():
    env = fixture.SingleSubstepFixture(device="cpu")
    env.run("zero-wrench")
    assert env.used is True and env.faulted is False
    with pytest.raises(ValueError, match="fresh nonfaulted one-use force fixture"):
        env.run("+x")


def test_unexpected_applied_force_faults_before_euler_and_cannot_be_reused():
    env = fixture.SingleSubstepFixture(device="cpu")
    before_time = env._view("time").detach().cpu().clone()
    env._view("qfrc_applied")[0, 0] = 1
    with pytest.raises(ValueError, match="qfrc_applied"):
        env.run("zero-wrench")
    assert env.faulted is True and env.used is True
    assert torch.equal(env._view("time").detach().cpu(), before_time)
    with pytest.raises(ValueError, match="fresh nonfaulted one-use force fixture"):
        env.run("zero-wrench")
