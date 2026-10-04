"""Synthetic CPU consistency checks; none of these fixtures attest CUDA."""

from copy import deepcopy

import pytest
import torch

from mjlab_microduck import stance_attempt_trace as legacy_trace
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_forward_probe as forward
from mjlab_microduck import stance_recovery_cuda_record_archive as archive
from mjlab_microduck import stance_recovery_cuda_record_replay as replay
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_recovery_schedule as schedule

# Reuse the repository's carefully typed 64-world archive fixture. Its phase
# payloads are deliberately placeholders, so phase checks below build their
# own complete synthetic solved-output values.
from test_stance_recovery_cuda_record_archive import _plant, _record


SOURCE = "a" * 40
N = replay.N
BODY_ID = 1
N_BODY = 2
CPU_PROFILE = profile.expected_receipt()


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _zero_archive():
    declaration = schedule.declaration(SOURCE, "dose", "training", ["zero-wrench"] * N)
    record = _record(0, declaration)
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": [record],
        **transition.FALSE_FLAGS,
    }
    initial = deepcopy(record["runtime_result_before_reset"]["boundaries"][0])
    return value, initial, declaration, _plant()


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _committed():
    return {
        "correction": torch.zeros((N, 10)),
        "target": torch.zeros((N, 14)),
        "queue": torch.zeros((N, 3, 14)),
        "previous": torch.zeros((N, 14)),
        "voltage": torch.full((N, 1), 7.5),
        "kp": torch.full((N, 1), 200.0),
        "friction": torch.zeros((N, 20)),
        "damping": torch.zeros((N, 20)),
        "ctrl": torch.zeros((N, 14)),
    }


def _solved(steps, *, time_values=None, qpos=None, qvel=None):
    count = 14
    if qpos is None:
        qpos = torch.zeros((N, 21), dtype=torch.float32)
        qpos[:, 2] = 0.12
        qpos[:, 3] = 1.0
    if qvel is None:
        qvel = torch.zeros((N, 20), dtype=torch.float32)
    kin = {
        "qpos": qpos.clone(),
        "qvel": qvel.clone(),
        "time": torch.tensor(steps, dtype=torch.float32) * 0.002
        if time_values is None
        else time_values.clone(),
        "qacc_warmstart": torch.zeros((N, 20), dtype=torch.float32),
    }
    dynamics = {
        name: torch.zeros((N, 20), dtype=torch.float32) for name in forward.DYNAMICS
    }
    solver = {
        "nefc": torch.full((N,), count, dtype=torch.int32),
        "nf": torch.full((N,), 14, dtype=torch.int32),
        "solver_niter": torch.ones((N,), dtype=torch.int32),
    }
    rows = []
    for _ in range(N):
        rows.append(
            {
                "type": torch.zeros(count, dtype=torch.int32),
                "id": torch.zeros(count, dtype=torch.int32),
                "J": torch.zeros((count, 20), dtype=torch.float32),
                "D": torch.zeros(count, dtype=torch.float32),
                "aref": torch.zeros(count, dtype=torch.float32),
                "force": torch.zeros(count, dtype=torch.float32),
                "state": torch.zeros(count, dtype=torch.int32),
            }
        )
    contacts = {
        "worldid": torch.empty((0,), dtype=torch.int64),
        "geom": torch.empty((0, 2), dtype=torch.int64),
        "dist": torch.empty((0,), dtype=torch.float32),
        "pos": torch.empty((0, 3), dtype=torch.float32),
        "frame": torch.empty((0, 3, 3), dtype=torch.float32),
        "friction": torch.empty((0, 5), dtype=torch.float32),
        "dim": torch.empty((0,), dtype=torch.int64),
        "efc_address": torch.empty((0, 1), dtype=torch.int64),
        "force": torch.empty((0, 6), dtype=torch.float32),
    }
    return {
        "kinematics": kin,
        "dynamics": dynamics,
        "solver": solver,
        "contacts": contacts,
        "constraints": rows,
    }


def _single_contact(*, world=0, world_dtype=torch.int64, geom_dtype=torch.int64):
    return {
        "worldid": torch.tensor([world], dtype=world_dtype),
        "geom": torch.zeros((1, 2), dtype=geom_dtype),
        "dist": torch.zeros((1,), dtype=torch.float32),
        "pos": torch.zeros((1, 3), dtype=torch.float32),
        "frame": torch.zeros((1, 3, 3), dtype=torch.float32),
        "friction": torch.zeros((1, 5), dtype=torch.float32),
        "dim": torch.ones((1,), dtype=torch.int64),
        "efc_address": torch.zeros((1, 1), dtype=torch.int64),
        "force": torch.zeros((1, 6), dtype=torch.float32),
    }


def _phase(declaration, steps, accepted, committed, *, phase_name):
    xfrc, qfrc = schedule.expected_wrenches(
        declaration, steps, accepted, N_BODY, BODY_ID
    )
    if phase_name == "unforced_post":
        xfrc, qfrc = schedule.expected_wrenches(
            declaration, steps, [False] * N, N_BODY, BODY_ID
        )
    return {
        "solved": _solved(steps),
        "inputs": {
            "ctrl": committed["ctrl"].clone(),
            "xfrc_applied": torch.tensor(xfrc, dtype=torch.float32),
            "qfrc_applied": torch.tensor(qfrc, dtype=torch.float32),
        },
        "motor_fields": {
            "dof_frictionloss": committed["friction"].clone(),
            "dof_damping": committed["damping"].clone(),
        },
    }


def _phase_case():
    cell = next(
        item["id"]
        for item in replay.schedule.lesson.cells("dose", held_out=False)
        if item["onset_step"] == 250
        and item["duration_steps"] == 20
        and item["force_world_newtons"] == [2.0, 0.0, 0.0]
    )
    declaration = schedule.declaration(SOURCE, "dose", "training", [cell] * N)
    steps = [249 + (row % 13) for row in range(N)]
    accepted = [row % 3 != 0 for row in range(N)]
    committed = _committed()
    compiled = _plant()
    pre = _phase(declaration, steps, accepted, committed, phase_name="forced_pre")
    integrated = deepcopy(pre)
    post = deepcopy(pre)
    # The test spans pulse-onset and pulse-end clocks, with rejected rows held.
    after_steps = [step + int(live) for step, live in zip(steps, accepted)]
    integrated_time = pre["solved"]["kinematics"]["time"] + (
        torch.tensor(accepted, dtype=torch.float32) * 0.002
    )
    for phase in (integrated, post):
        phase["solved"]["kinematics"]["time"] = integrated_time.clone()
    post["inputs"]["xfrc_applied"].zero_()
    post["inputs"]["qfrc_applied"].zero_()
    phases = {"forced_pre": pre, "integrated": integrated, "unforced_post": post}
    before = {
        "physics_steps": torch.tensor(steps, dtype=torch.int64),
        "qpos": pre["solved"]["kinematics"]["qpos"].clone(),
        "qvel": pre["solved"]["kinematics"]["qvel"].clone(),
    }
    after = {
        "physics_steps": torch.tensor(after_steps, dtype=torch.int64),
        "qpos": before["qpos"].clone(),
        "qvel": before["qvel"].clone(),
    }
    proposal = {"accepted": torch.tensor(accepted), "committed": committed}
    entry = {
        "before_steps": before["physics_steps"].clone(),
        "accepted": proposal["accepted"].clone(),
        "phases": phases,
    }
    record = {
        "runtime_result_before_reset": {
            "scheduled_pulse_evidence": [entry],
            "boundaries": [before, after],
            "control_evidence": {"proposals": [proposal]},
        }
    }
    return record, declaration, compiled, steps, accepted, committed


def _install_synthetic_check_seams(monkeypatch, value, compiled, events):
    monkeypatch.setattr(
        replay,
        "_compiled_reference",
        lambda: (events.append("cpu-compile"), deepcopy(compiled))[1],
    )
    monkeypatch.setattr(
        replay.profile,
        "check_recorded",
        lambda got: (
            (events.append("cpu-profile"), got == CPU_PROFILE)[1]
            or (_ for _ in ()).throw(ValueError("synthetic profile mismatch"))
        ),
    )
    monkeypatch.setattr(
        replay.plant,
        "check_trace",
        lambda payload, plant: (events.append("plant-check"), {"synthetic": True})[1],
    )
    monkeypatch.setattr(
        replay.control,
        "replay",
        lambda evidence, payload, plant: (
            events.append("control-check"),
            {"synthetic": True},
        )[1],
    )


def test_protocol_is_separate_and_existing_iteration_allowlists_stay_unchanged():
    assert replay.PROTOCOL not in legacy_trace.ITERATIONS
    assert legacy_trace.ITERATIONS[legacy_trace.PROTOCOL] == (128, 256, 384, 511)
    assert legacy_trace.ITERATIONS[legacy_trace.LEAN_REPLICATION_PROTOCOL] == (
        64,
        128,
        192,
        255,
    )
    assert replay.RecordTrace is not legacy_trace.FirstAttemptTrace
    assert replay.RecordTrace.__bases__ == (legacy_trace.FirstAttemptTrace,)


def test_compiled_reference_uses_a_real_cpu_model_without_initializing_cuda():
    compiled = replay._compiled_reference()
    assert compiled["nbody"] == 16
    assert compiled["body_name"] == "trunk_base"
    assert compiled["body_id"] > 0
    assert torch.cuda.is_initialized() is False


def test_public_check_orders_archive_profile_and_independent_cpu_compilation(
    monkeypatch,
):
    value, initial, declaration, compiled = _zero_archive()
    events = []
    _install_synthetic_check_seams(monkeypatch, value, compiled, events)
    original_archive_check = replay.archive.check

    def archive_first(*args, **kwargs):
        events.append("archive-check")
        return original_archive_check(*args, **kwargs)

    monkeypatch.setattr(replay.archive, "check", archive_first)
    result = replay.check(
        value,
        initial,
        declaration,
        compiled,
        seed=653,
        cpu_profile=CPU_PROFILE,
    )
    assert events == [
        "archive-check",
        "cpu-profile",
        "cpu-compile",
        "plant-check",
        "control-check",
    ]
    assert result["archive"]["records"] == 1
    assert result["trajectory_continuity_checked"] is True
    assert result["whole_trajectory_physics_resimulated"] is False
    assert result["native_transition_qualified"] is False
    assert result["actor_independently_replayed"] is False
    assert all(result[key] is False for key in transition.FALSE_FLAGS)


def test_public_check_does_not_mutate_archived_records_or_retained_initial(monkeypatch):
    value, initial, declaration, compiled = _zero_archive()
    before_value, before_initial = deepcopy(value), deepcopy(initial)
    events = []
    _install_synthetic_check_seams(monkeypatch, value, compiled, events)
    replay.check(
        value, initial, declaration, compiled, seed=653, cpu_profile=CPU_PROFILE
    )
    assert _tree_equal(value, before_value)
    assert _tree_equal(initial, before_initial)


def test_public_check_rejects_counterfeit_initial_before_plant_or_control(monkeypatch):
    value, initial, declaration, compiled = _zero_archive()
    events = []
    _install_synthetic_check_seams(monkeypatch, value, compiled, events)
    initial["qpos"][0, 0] += 0.01
    with pytest.raises(ValueError):
        replay.check(
            value, initial, declaration, compiled, seed=653, cpu_profile=CPU_PROFILE
        )
    assert events == ["cpu-profile", "cpu-compile"]
    assert "control-check" not in events


def test_public_check_rejects_compiled_plant_mismatch_after_archive_and_profile(
    monkeypatch,
):
    value, initial, declaration, compiled = _zero_archive()
    events = []
    mismatched = deepcopy(compiled)
    mismatched["body_id"] += 1
    _install_synthetic_check_seams(monkeypatch, value, mismatched, events)
    with pytest.raises(
        ValueError, match="independently CPU-compiled selected plant binding"
    ):
        replay.check(
            value, initial, declaration, compiled, seed=653, cpu_profile=CPU_PROFILE
        )
    assert events == ["cpu-profile", "cpu-compile"]


def test_cuda_visible_or_initialized_process_refuses_before_archive_check(monkeypatch):
    value, initial, declaration, compiled = _zero_archive()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        replay.archive,
        "check",
        lambda *args, **kwargs: pytest.fail("archive must not run"),
    )
    with pytest.raises(ValueError, match="CUDA-hidden record consistency replay"):
        replay.check(
            value, initial, declaration, compiled, seed=653, cpu_profile=CPU_PROFILE
        )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(replay.torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="CUDA-hidden record consistency replay"):
        replay.check(
            value, initial, declaration, compiled, seed=653, cpu_profile=CPU_PROFILE
        )


def test_phase_helper_accepts_full_typed_n64_mixed_clock_and_acceptance_matrix():
    record, declaration, compiled, steps, accepted, committed = _phase_case()
    for name in replay.PHASES:
        phase = record["runtime_result_before_reset"]["scheduled_pulse_evidence"][0][
            "phases"
        ][name]
        mask = accepted if name != "unforced_post" else [False] * N
        replay._phase(phase, compiled, declaration, steps, mask, committed)


@pytest.mark.parametrize("damage", ["force", "motor", "control", "shape", "dtype"])
def test_phase_helper_rejects_corrupt_force_motor_control_or_tensor_layout(damage):
    record, declaration, compiled, steps, accepted, committed = _phase_case()
    phase = record["runtime_result_before_reset"]["scheduled_pulse_evidence"][0][
        "phases"
    ]["forced_pre"]
    if damage == "force":
        phase["inputs"]["xfrc_applied"][1, BODY_ID, 0] += 1
    elif damage == "motor":
        phase["motor_fields"]["dof_damping"][2, 0] = 1
    elif damage == "control":
        phase["inputs"]["ctrl"][3, 0] = 0.1
    elif damage == "shape":
        phase["inputs"]["qfrc_applied"] = torch.zeros((N, 19))
    else:
        phase["inputs"]["xfrc_applied"] = phase["inputs"]["xfrc_applied"].double()
    with pytest.raises(ValueError):
        replay._phase(phase, compiled, declaration, steps, accepted, committed)


@pytest.mark.parametrize(
    "damage",
    [
        "nonfinite-dynamics",
        "constraint-width",
        "missing-constraint-field",
        "solver-nf",
        "contact-world",
        "contact-dtype",
    ],
)
def test_phase_helper_runs_full_forward_output_validation_for_corrupt_solved_values(
    damage,
):
    record, declaration, compiled, steps, accepted, committed = _phase_case()
    phase = record["runtime_result_before_reset"]["scheduled_pulse_evidence"][0][
        "phases"
    ]["forced_pre"]
    solved = phase["solved"]
    if damage == "nonfinite-dynamics":
        solved["dynamics"]["qacc"][0, 0] = float("nan")
    elif damage == "constraint-width":
        solved["constraints"][0]["J"] = torch.zeros((14, 19), dtype=torch.float32)
    elif damage == "missing-constraint-field":
        del solved["constraints"][0]["D"]
    elif damage == "solver-nf":
        solved["solver"]["nf"][4] = 13
    elif damage == "contact-world":
        solved["contacts"] = _single_contact(world=N)
    else:
        solved["contacts"] = _single_contact(world=0, geom_dtype=torch.float32)
    with pytest.raises(ValueError):
        replay._phase(phase, compiled, declaration, steps, accepted, committed)


def test_full_phase_checker_accepts_consistent_substep_and_preserves_solved_state():
    record, declaration, compiled, _, _, _ = _phase_case()
    result = replay._check_phases([record], declaration, compiled)
    assert result["phase_entries"] == 1
    assert result["full_phase_consistency_checked"] is True
    assert result["solver_independently_reexecuted"] is False
    assert result["bam_independently_recomputed"] is False


@pytest.mark.parametrize(
    "damage", ["coverage", "clock", "clock-offset", "integration", "post", "dynamics"]
)
def test_full_phase_checker_rejects_missing_or_inconsistent_replay(damage):
    record, declaration, compiled, _, _, _ = _phase_case()
    raw = record["runtime_result_before_reset"]
    entry = raw["scheduled_pulse_evidence"][0]
    phases = entry["phases"]
    if damage == "coverage":
        entry["phases"] = None
    elif damage == "clock":
        phases["integrated"]["solved"]["kinematics"]["time"][1] += 0.01
    elif damage == "clock-offset":
        for phase in phases.values():
            phase["solved"]["kinematics"]["time"] += 0.01
    elif damage == "integration":
        phases["integrated"]["solved"]["kinematics"]["qpos"][1, 0] += 0.01
    elif damage == "post":
        phases["unforced_post"]["solved"]["kinematics"]["qvel"][1, 0] += 0.01
    else:
        phases["integrated"]["solved"]["dynamics"]["qacc"][1, 0] += 0.01
    with pytest.raises(ValueError):
        replay._check_phases([record], declaration, compiled)


def test_missing_archive_controls_are_rejected_by_archive_before_replay(monkeypatch):
    value, initial, declaration, compiled = _zero_archive()
    del value["records"][0]["runtime_result_before_reset"]["control_evidence"]
    with pytest.raises(ValueError):
        replay.check(
            value, initial, declaration, compiled, seed=653, cpu_profile=CPU_PROFILE
        )
