"""CPU integration/control-flow checks; counters are synthetic pulse selectors only."""

import gc
from hashlib import sha256
import inspect

import pytest
import torch

from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.first_attempt_smoke import canonical
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


SOURCE = "a" * 40


@pytest.fixture(autouse=True)
def collect_only_finished_native_fixtures():
    yield
    gc.collect()


def supported_refresh(env):
    original = env._refresh
    def refresh(rows):
        original(rows)
        env.state.support[rows] = 1.
    env._refresh = refresh


def fixture_runtime(cell_ids):
    declaration = schedule.declaration(SOURCE, "dose", "training", cell_ids)
    env = ScheduledRecoveryRuntime(declaration, device="cpu")
    # A real short nominal tick builds/refreshes owned controls. Subsequent
    # counter selection targets the pulse window; it is not a long rollout.
    env.step(torch.zeros(len(cell_ids), 10))
    supported_refresh(env)
    steps = torch.tensor([cell["onset_step"] for cell in declaration["row_cells"]],
                         dtype=torch.long)
    env.steps.copy_(steps)
    env._view("time").copy_(steps.to(torch.float32)*declaration["dt"])
    env.state.support.fill_(1.)
    return env


def controls_equal(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def test_runtime_binds_actual_compiled_body_and_fixed_owned_schedule():
    ids = ["+x-2n-10steps-t500", "+y-4n-10steps-t750", "zero-wrench"]
    declaration = schedule.declaration(SOURCE, "dose", "training", ids)
    env = ScheduledRecoveryRuntime(declaration, device="cpu")
    assert env.n == 3 and env.schedule_declaration == declaration
    assert env.binding["body_names"][env.binding["body_id"]] == "trunk_base"
    assert env.binding["body_id"] == env.plant_binding["body_id"]
    assert env._schedule_sha256 == schedule.binding_sha256(declaration)
    assert env._binding_sha256 == sha256(canonical(env.binding).encode()).hexdigest()
    # Reset is inherited; no row rescheduling or schedule argument is introduced.
    before = env.schedule_declaration.copy()
    env.reset(torch.ones(3, dtype=torch.bool))
    assert env.schedule_declaration == before
    assert "schedule" not in inspect.signature(env.reset).parameters


def test_heterogeneous_rows_capture_full_three_phases_including_active_zero_control(monkeypatch):
    ids = ["+x-2n-10steps-t500", "+y-4n-10steps-t750", "zero-wrench", "zero-wrench"]
    env = fixture_runtime(ids)
    calls = []
    original_expected = schedule.expected_wrenches
    original_validate = schedule.validate_wrenches
    def expected(value, steps, accepted, nbody, body_id):
        calls.append((list(steps), list(accepted)))
        return original_expected(value, steps, accepted, nbody, body_id)
    validate_calls = []
    def validate(*args, **kwargs):
        validate_calls.append(1)
        return original_validate(*args, **kwargs)
    monkeypatch.setattr(schedule, "expected_wrenches", expected)
    monkeypatch.setattr(schedule, "validate_wrenches", validate)

    # First-terminal behavior is inherited: close the zero-control row before
    # its substeps while both scheduled pulse rows continue.
    env.state.tilt[3] = .4
    result = env.step_with_schedule(torch.zeros(4, 10), capture_control=True)
    assert result["executed_steps"].tolist() == [10, 10, 10, 0]
    assert len(result["boundaries"]) == 11
    assert "scheduled_pulse_evidence" in result and "pulse_evidence" not in result
    assert len(result["scheduled_pulse_evidence"]) == 10
    distinct_clocks = list(dict.fromkeys(tuple(steps) for steps, _mask in calls))
    assert distinct_clocks == [(500+i, 750+i, 500+i, 500) for i in range(10)]
    assert all(mask == [True, True, True, False] for _steps, mask in calls)
    assert len(calls) == 50  # install, phase window validation, and force checks at the solver boundary
    assert len(validate_calls) == 20  # exact pre/post solver validation per accepted substep
    for i, event in enumerate(result["scheduled_pulse_evidence"]):
        assert event["before_steps"].tolist() == [500+i, 750+i, 500+i, 500]
        assert event["accepted"].tolist() == [True, True, True, False]
        assert event["pre_xfrc"].shape == (4, env.native.nbody, 6)
        assert event["pre_xfrc"][0, env.binding["body_id"]].tolist() == [2., 0., 0., 0., 0., 0.]
        assert event["pre_xfrc"][1, env.binding["body_id"]].tolist() == [0., 4., 0., 0., 0., 0.]
        assert not event["pre_xfrc"][2].any() and not event["pre_xfrc"][3].any()
        assert not event["pre_qfrc"].any()
        assert not event["post_xfrc"].any() and not event["post_qfrc"].any()
        assert event["phases"] is not None
        assert set(event["phases"]) == {"forced_pre", "integrated", "unforced_post"}
        for phase in event["phases"].values():
            assert phase["inputs"]["xfrc_applied"].shape == (4, env.native.nbody, 6)
            assert not phase["inputs"]["qfrc_applied"].any()
            assert set(phase["motor_fields"]) == {"dof_frictionloss", "dof_damping"}
        assert event["phases"]["forced_pre"]["inputs"]["xfrc_applied"][0,
            env.binding["body_id"], 0] == 2.
        assert event["phases"]["forced_pre"]["inputs"]["xfrc_applied"][1,
            env.binding["body_id"], 1] == 4.
        assert not event["phases"]["forced_pre"]["inputs"]["xfrc_applied"][2].any()
        assert not event["phases"]["forced_pre"]["inputs"]["xfrc_applied"][3].any()
        assert not event["phases"]["unforced_post"]["inputs"]["xfrc_applied"].any()
    assert result["terminal_records"][3]["physics_step"] == 500
    assert result["control_evidence"]
    assert not env._view("xfrc_applied").any() and not env._view("qfrc_applied").any()


@pytest.mark.parametrize("failure", ["forced-forward", "forced-phase", "integrated-phase", "euler"])
def test_any_owned_operation_failure_clears_full_force_arrays_and_faults_runtime(monkeypatch, failure):
    env = fixture_runtime(["+x-2n-10steps-t500"])
    if failure == "forced-forward":
        monkeypatch.setattr(env, "_scheduled_forward", lambda *_a: (_ for _ in ()).throw(
            RuntimeError("synthetic forced-forward failure")))
    elif failure.endswith("phase"):
        original = env._phase_capture
        count = 0
        def capture():
            nonlocal count
            count += 1
            if count == (1 if failure == "forced-phase" else 2):
                raise RuntimeError("synthetic "+failure)
            return original()
        monkeypatch.setattr(env, "_phase_capture", capture)
    else:
        monkeypatch.setattr(env.integrator, "integrate", lambda *_a: (_ for _ in ()).throw(
            RuntimeError("synthetic Euler failure")))
    with pytest.raises(RuntimeError, match="synthetic"):
        env.step_with_schedule(torch.zeros(1, 10))
    assert env.faulted
    assert not env._view("xfrc_applied").any() and not env._view("qfrc_applied").any()
    with pytest.raises(RuntimeError, match="job closeout"):
        env.reset(torch.ones(1, dtype=torch.bool))


@pytest.mark.parametrize("field", ["xfrc_applied", "qfrc_applied"])
def test_external_force_refusal_precedes_any_bam_fifo_or_counter_mutation(field):
    env = fixture_runtime(["+x-2n-10steps-t500"])
    env._view(field)[0].reshape(-1)[0] = .001
    before = env._control_snapshot()
    steps = env.steps.clone()
    queue = env.delay.queue.clone()
    with pytest.raises(ValueError, match="zero applied-force arrays at entry"):
        env.step_with_schedule(torch.ones(1, 10))
    assert env.faulted and controls_equal(before, env._control_snapshot())
    assert torch.equal(queue, env.delay.queue) and torch.equal(steps, env.steps)


@pytest.mark.parametrize("damage", ["schedule", "binding", "plant-binding", "graph"])
def test_changed_schedule_body_binding_or_graph_refuses_before_controls(damage):
    env = fixture_runtime(["+x-2n-10steps-t500"])
    if damage == "schedule":
        env.schedule_declaration["row_cells"][0]["force_world_newtons"][0] = 3.
    elif damage == "binding":
        env.binding["body_id"] += 1
    elif damage == "plant-binding":
        env.plant_binding["body_names"][env.plant_binding["body_id"]] = "changed"
    else:
        env.forward_graph = object()
    before = env._control_snapshot()
    with pytest.raises(ValueError, match="unchanged exact typed schedule|plant/body binding|declared eager"):
        env.step_with_schedule(torch.ones(1, 10))
    assert env.faulted and controls_equal(before, env._control_snapshot())


def test_schedule_pulse_is_cleared_before_inherited_nominal_forward(monkeypatch):
    env = fixture_runtime(["+x-2n-10steps-t500"])
    original = env._forward
    observations = []
    def guarded_forward():
        observations.append((bool(env._view("xfrc_applied").any()), bool(env._view("qfrc_applied").any())))
        assert not observations[-1][0] and not observations[-1][1]
        return original()
    monkeypatch.setattr(env, "_forward", guarded_forward)
    result = env.step_with_schedule(torch.zeros(1, 10))
    assert len(observations) == 10 and result["executed_steps"].tolist() == [10]
    assert all(pair == (False, False) for pair in observations)


def test_accepted_zero_wrench_window_alone_captures_all_three_full_phases():
    env = fixture_runtime(["zero-wrench"])
    result = env.step_with_schedule(torch.zeros(1, 10))
    rows = result["scheduled_pulse_evidence"]
    assert len(rows) == 10
    for row in rows:
        assert row["accepted"].tolist() == [True]
        assert row["pre_xfrc"].shape == (1, env.native.nbody, 6)
        assert not row["pre_xfrc"].any() and not row["pre_qfrc"].any()
        assert set(row["phases"]) == {"forced_pre", "integrated", "unforced_post"}
        for phase in row["phases"].values():
            assert not phase["inputs"]["xfrc_applied"].any()
            assert not phase["inputs"]["qfrc_applied"].any()


def test_inherited_nominal_step_is_unchanged_and_keeps_forceguard():
    assert ScheduledRecoveryRuntime.step is WarpStanceRuntime.step
    env = fixture_runtime(["+x-2n-10steps-t500"])
    env._view("xfrc_applied")[0, env.binding["body_id"], 0] = 1.
    with pytest.raises(ValueError, match="does not permit external assistance or pushes"):
        env.step(torch.zeros(1, 10))
    assert env.faulted


@pytest.mark.parametrize("field", ["xfrc_applied", "qfrc_applied"])
def test_partial_installation_failure_still_clears_both_force_arrays(monkeypatch, field):
    env = fixture_runtime(["+x-2n-10steps-t500"])
    ptr = env._view(field).data_ptr()
    original = torch.Tensor.copy_
    def fail_after_copy(destination, source, *args, **kwargs):
        result = original(destination, source, *args, **kwargs)
        if destination.data_ptr() == ptr:
            raise RuntimeError("synthetic partial install error")
        return result
    monkeypatch.setattr(torch.Tensor, "copy_", fail_after_copy)
    with pytest.raises(RuntimeError, match="synthetic partial install error"):
        env.step_with_schedule(torch.zeros(1, 10))
    assert env.faulted
    assert not env._view("xfrc_applied").any() and not env._view("qfrc_applied").any()
