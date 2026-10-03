"""Deterministic synthetic CPU checks for the CUDA64 archive source contract.

No tensor in these fixtures came from CUDA or MuJoCo-Warp execution.
"""

from copy import deepcopy
from hashlib import sha256

import pytest
import torch

from mjlab_microduck import stance_recovery_cuda_record_archive as archive
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_recovery_lesson_plan as lesson
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_cuda_rng_scope as rng_scope
from mjlab_microduck import stance_recovery_cuda_shadow_sampling as sampling


SOURCE = "a" * 40
LEASE_FD = 19
N = transition.WORLDS
N_BODY = 2
BODY_ID = 1


def _plant():
    return {
        "selected_plant": {"synthetic_cpu_fixture": True},
        "nbody": N_BODY,
        "body_names": ["world", "trunk_base"],
        "body_name": "trunk_base",
        "body_id": BODY_ID,
    }


def _declaration():
    cells = lesson.cells("dose", held_out=False)
    chosen = {
        step: next(
            cell
            for cell in cells
            if cell["onset_step"] == step
            and cell["duration_steps"] == duration
            and any(cell["force_world_newtons"])
        )["id"]
        for step, duration in ((250, 20), (500, 10), (750, 10))
    }
    ids = [chosen[250]] * 16 + [chosen[500]] * 16 + [chosen[750]] * 16
    ids += ["zero-wrench"] * (N - len(ids))
    return schedule.declaration(SOURCE, "dose", "training", ids)


def _obs():
    actor = torch.zeros((N, 44), dtype=torch.float32)
    actor[:, 2] = -1.0
    critic = torch.zeros((N, 50), dtype=torch.float32)
    critic[:, :44] = actor
    critic[:, 48:] = 0.1
    return {"actor": actor, "critic": critic}


def _state():
    return {
        "tilt": torch.zeros(N),
        "root_velocity": torch.zeros(N, 3),
        "height": torch.full((N,), 0.12),
        "support": torch.ones(N, 2),
        "torque": torch.zeros(N, 14),
        "joint_velocity": torch.zeros(N, 14),
        "hard_limit": torch.zeros(N, dtype=torch.bool),
        "forbidden_contact": torch.zeros(N, dtype=torch.bool),
        "warning": torch.zeros(N, dtype=torch.bool),
    }


def _frame(clocks):
    qpos = torch.zeros((N, 21), dtype=torch.float32)
    qpos[:, 2] = 0.12
    qpos[:, 3] = 1.0
    return {
        "physics_steps": torch.tensor(clocks, dtype=torch.int64),
        "qpos": qpos,
        "qvel": torch.zeros((N, 20), dtype=torch.float32),
        "soft_limit_mask": torch.zeros((N, 14), dtype=torch.bool),
        "state": _state(),
        "observation": _obs(),
    }


def _control_state():
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


def _private_state(start):
    return torch.arange(start, start + 8, dtype=torch.uint8)


def _record(index, declaration):
    entry_steps = torch.full((N,), index * 10, dtype=torch.long)
    end_steps = entry_steps + 10
    boundaries = [_frame(entry_steps.tolist())]
    boundaries.extend(
        _frame((entry_steps + substep).tolist()) for substep in range(1, 11)
    )
    accepted = torch.ones(N, dtype=torch.bool)
    schedule_sha = schedule.binding_sha256(declaration)
    plant = _plant()
    plant_sha = sha256(archive.canonical(plant).encode()).hexdigest()
    pulses = []
    proposals = []
    controls = _control_state()
    for substep in range(10):
        clocks = entry_steps + substep
        clock_list = clocks.tolist()
        accepted_list = accepted.tolist()
        xfrc, qfrc = schedule.expected_wrenches(
            declaration, clock_list, accepted_list, N_BODY, BODY_ID
        )
        zero_x = torch.zeros((N, N_BODY, 6), dtype=torch.float32)
        zero_q = torch.zeros((N, 20), dtype=torch.float32)
        pre_x = torch.tensor(xfrc, dtype=torch.float32)
        pre_q = torch.tensor(qfrc, dtype=torch.float32)
        active = any(schedule.window_mask(declaration, clock_list, accepted_list))
        phases = (
            {name: {} for name in ("forced_pre", "integrated", "unforced_post")}
            if active
            else None
        )
        pulses.append(
            {
                "before_steps": clocks,
                "accepted": accepted.clone(),
                "pre_xfrc": pre_x,
                "pre_qfrc": pre_q,
                "post_xfrc": zero_x,
                "post_qfrc": zero_q,
                "phases": phases,
                "schedule_sha256": schedule_sha,
                "binding_sha256": plant_sha,
            }
        )
        proposals.append(
            {
                "before_steps": clocks.clone(),
                "live": accepted.clone(),
                "command": {
                    name: torch.zeros((N, 14), dtype=torch.float32)
                    for name in (
                        "position_target",
                        "velocity_target",
                        "effort_target",
                        "pos",
                        "vel",
                    )
                },
                "torque_nm": torch.zeros((N, 14), dtype=torch.float32),
                "accepted": accepted.clone(),
                "rejected": torch.zeros(N, dtype=torch.bool),
                "committed": deepcopy(controls),
            }
        )

    observations = _obs()
    actions = torch.zeros((N, 10), dtype=torch.float32)
    mean = torch.zeros_like(actions)
    std = torch.ones_like(actions)
    zero_values = torch.zeros((N, 1), dtype=torch.float32)
    reward = torch.full((N,), 0.09, dtype=torch.float32)
    done = torch.zeros(N, dtype=torch.bool)
    times = end_steps.to(torch.float32) * 0.002
    pre_kinematics = {
        "qpos": boundaries[-1]["qpos"].clone(),
        "qvel": boundaries[-1]["qvel"].clone(),
        "time": times.clone(),
    }
    runtime_result = {
        "reward": reward.clone(),
        "terminated": done.clone(),
        "timed_out": done.clone(),
        "episode_steps": end_steps.clone(),
        "executed_steps": torch.full((N,), 10, dtype=torch.long),
        "live": torch.ones(N, dtype=torch.bool),
        "term_sums": {
            "upright": torch.full((N,), 0.04),
            "stillness": torch.full((N,), 0.02),
            "height": torch.full((N,), 0.02),
            "support": torch.full((N,), 0.01),
            "motor": torch.zeros(N),
            "joint_speed": torch.zeros(N),
            "correction": torch.zeros(N),
            "correction_change": torch.zeros(N),
        },
        "observation": deepcopy(observations),
        "boundaries": boundaries,
        "terminal_records": [None] * N,
        "optimizer_launched": False,
        "scheduled_pulse_evidence": pulses,
        "control_evidence": {
            "initial": deepcopy(controls),
            "after_action": deepcopy(controls),
            "proposals": proposals,
            "final": deepcopy(controls),
        },
    }
    before = _private_state(index * 8 + 1)
    after = _private_state((index + 1) * 8 + 1)
    initial_rng = _private_state(1)
    return {
        "receipt": {
            "protocol": transition.PROTOCOL,
            "source": SOURCE,
            "learner_seed": 653,
            "storage_step": index,
            "worlds": N,
            "device": "cuda:0",
            "schedule_sha256": schedule_sha,
            "optimizer_steps": 0,
            "transition_bridge_qualified": False,
            "native_terminal_qualified": False,
            "native_selective_reset_qualified": False,
            **transition.FALSE_FLAGS,
        },
        "pre_action_observations": deepcopy(observations),
        "raw_actions": actions,
        "pre_action_values": zero_values.clone(),
        "raw_actions_log_prob": torch.full((N,), -10 * 0.9189385332046727),
        "distribution_params": (mean, std),
        "terminal_observations": deepcopy(observations),
        "timeout_terminal_values": zero_values.clone(),
        "environment_reward": reward.clone(),
        "learner_reward": reward.clone(),
        "terminated": done.clone(),
        "timed_out": done.clone(),
        "pre_reset_steps": end_steps.clone(),
        "post_reset_steps": end_steps.clone(),
        "pre_reset_kinematics": pre_kinematics,
        "post_reset_kinematics": deepcopy(pre_kinematics),
        "pre_reset_controls": deepcopy(controls),
        "post_reset_controls": deepcopy(controls),
        "initial_reset_controls": deepcopy(controls),
        "terminal_records": [None] * N,
        "reset_records": [None] * N,
        "next_observations": deepcopy(observations),
        "runtime_result_before_reset": runtime_result,
        "private_rng_state_before": before,
        "private_rng_state": after,
        "rng_scope_receipt": {
            "protocol": rng_scope.PROTOCOL,
            "source": SOURCE,
            "seed": 653,
            "device": "cuda:0",
            "scope_count": index + 1,
            "private_state_sha256": sampling._state_digest(after),
            "initial_state_sha256": sampling._state_digest(initial_rng),
            "state_bytes": after.numel(),
            "scope_active": False,
            "faulted": False,
            "caller_cpu_rng_preserved": True,
            "caller_cuda_rng_preserved": True,
            "native_cuda_rng_scope_qualified": False,
            "native_cuda_sampling_qualified": False,
            **transition.FALSE_FLAGS,
        },
    }


def _records(count, declaration):
    return [_record(index, declaration) for index in range(count)]


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_retain_record_owns_nested_cpu_tensors_and_preserves_tuple_layout(monkeypatch):
    record = _record(0, _declaration())
    original = record["distribution_params"][0]
    original_action = record["raw_actions"]
    lease_calls = []
    monkeypatch.setattr(
        archive.training, "inherited_lease", lambda fd: lease_calls.append(fd)
    )

    owned = archive.retain_record_cpu(record, lease_fd=LEASE_FD)

    assert lease_calls == [LEASE_FD]
    assert set(owned) == archive.RECORD_KEYS
    assert type(owned["distribution_params"]) is tuple
    assert owned["distribution_params"][0].device.type == "cpu"
    assert owned["distribution_params"][0].data_ptr() != original.data_ptr()
    owned["distribution_params"][0].fill_(4)
    owned["raw_actions"].fill_(3)
    assert not original.any()
    assert not original_action.any()


def test_capture_lease_is_checked_before_record_traversal(monkeypatch):
    events = []
    monkeypatch.setattr(
        archive.training, "inherited_lease", lambda fd: events.append(fd)
    )
    with pytest.raises(ValueError, match="exact collector record"):
        archive.retain_record_cpu(object(), lease_fd=LEASE_FD)
    assert events == [LEASE_FD]


def test_owned_tree_rejects_tensor_bytes_nodes_depth_and_non_cpu_source():
    over_budget = torch.tensor([0.0]).as_strided(
        (archive.TENSOR_BUDGET // 4 + 1,), (0,)
    )
    with pytest.raises(ValueError, match="bounded archive tensor bytes"):
        archive._owned_tree(over_budget, allow_cuda=False)

    with pytest.raises(ValueError, match="bounded archive tree"):
        archive._owned_tree([None] * archive.NODE_BUDGET, allow_cuda=False)

    nested = None
    for _ in range(34):
        nested = [nested]
    with pytest.raises(ValueError, match="bounded archive tree"):
        archive._owned_tree(nested, allow_cuda=False)

    with pytest.raises(ValueError, match="CPU archive or leased CUDA0 capture tensor"):
        archive._owned_tree(torch.empty(1, device="meta"), allow_cuda=False)


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(1, dtype=torch.complex64),
        torch.zeros(1, dtype=torch.float16),
        torch.sparse_coo_tensor(
            torch.empty((1, 0), dtype=torch.long), torch.empty(0), (1,)
        ),
    ],
    ids=["complex", "half", "sparse"],
)
def test_owned_tree_rejects_unsupported_tensor_layouts_and_dtypes(value):
    with pytest.raises(ValueError, match="dense supported archive tensor"):
        archive._owned_tree(value, allow_cuda=False)


def test_synthetic_28_tick_64_row_encode_verify_round_trip():
    declaration = _declaration()
    records = _records(28, declaration)

    raw, receipt = archive.encode(records, declaration, _plant(), seed=653)
    score = archive.verify(raw, receipt["sha256"], declaration, _plant(), seed=653)

    assert receipt["bytes"] == len(raw)
    assert score["records"] == 28
    assert score["final_steps"] == [280] * N
    assert score["complete_short_window"] is True
    assert score["pulse"]["solver_phases_replayed"] is False
    assert score["pulse"]["solved_phase_payloads_replayed"] is False
    assert score["native_transition_qualified"] is False
    assert score["natural_timeout_qualified"] is False
    assert score["selective_reset_qualified"] is False
    assert score["cuda_rng_math_replayed"] is False
    assert score["caller_rng_states_independently_checked"] is False
    assert score["model_and_optimizer_states_independently_checked"] is False

    rows = score["pulse"]["rows"]
    assert all(row["window_status"] == "complete" for row in rows[:16])
    assert all(row["delivered_nonzero_steps"] == 20 for row in rows[:16])
    assert all(row["window_status"] == "not-reached" for row in rows[16:])
    assert all(row["delivered_nonzero_steps"] == 0 for row in rows[16:48])
    assert score["pulse"]["complete_nonzero_pulse_delivery"] is False


def test_positive_seed_659_archive_requires_consistent_record_and_scope_seed():
    declaration = _declaration()
    record = _record(0, declaration)
    record["receipt"]["learner_seed"] = 659
    record["rng_scope_receipt"]["seed"] = 659

    raw, receipt = archive.encode([record], declaration, _plant(), seed=659)
    score = archive.verify(raw, receipt["sha256"], declaration, _plant(), seed=659)

    assert score["records"] == 1
    assert score["native_transition_qualified"] is False


@pytest.mark.parametrize(
    ("count", "expected_status", "expected_steps"),
    [(25, "not-reached", 0), (26, "partial", 10), (28, "complete", 20)],
)
def test_row_reachability_distinguishes_before_partial_and_complete(
    count, expected_status, expected_steps
):
    all_cells = lesson.cells("dose", held_out=False)
    cell = next(
        candidate
        for candidate in all_cells
        if candidate["onset_step"] == 250
        and candidate["duration_steps"] == 20
        and any(candidate["force_world_newtons"])
    )
    declaration = schedule.declaration(SOURCE, "dose", "training", [cell["id"]] * N)
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": _records(count, declaration),
        **transition.FALSE_FLAGS,
    }

    score = archive.check(value, declaration, _plant(), seed=653)
    row = score["pulse"]["rows"][0]

    assert row["window_status"] == expected_status
    assert row["delivered_nonzero_steps"] == expected_steps
    assert score["pulse"]["complete_nonzero_pulse_delivery"] == (
        expected_status == "complete"
    )


def test_zero_wrench_rows_do_not_claim_a_window_when_onset_is_unreachable():
    declaration = schedule.declaration(SOURCE, "dose", "training", ["zero-wrench"] * N)
    score = archive.check(
        {
            "protocol": archive.PROTOCOL,
            "declaration": declaration,
            "compiled_plant": _plant(),
            "learner_seed": 653,
            "records": _records(28, declaration),
            **transition.FALSE_FLAGS,
        },
        declaration,
        _plant(),
        seed=653,
    )

    assert all(row["window_status"] == "not-reached" for row in score["pulse"]["rows"])
    assert score["pulse"]["phase_entries"] == 0
    assert score["pulse"]["complete_nonzero_pulse_delivery"] is False


def test_verify_checks_hash_before_tensor_load(monkeypatch):
    called = []
    monkeypatch.setattr(
        archive.torch, "load", lambda *args, **kwargs: called.append(True)
    )
    with pytest.raises(ValueError, match="whole archive byte hash"):
        archive.verify(
            b"not the declared archive", "0" * 64, _declaration(), _plant(), seed=653
        )
    assert called == []


def test_archive_byte_limit_is_enforced_during_encode(monkeypatch):
    monkeypatch.setattr(archive, "LIMIT", 128)
    declaration = _declaration()
    with pytest.raises(ValueError, match="bounded serialized archive"):
        archive.encode(_records(1, declaration), declaration, _plant(), seed=653)


def test_verify_refuses_oversized_bytes_before_load(monkeypatch):
    monkeypatch.setattr(archive, "LIMIT", 8)
    called = []
    monkeypatch.setattr(
        archive.torch, "load", lambda *args, **kwargs: called.append(True)
    )
    with pytest.raises(ValueError, match="bounded serialized archive"):
        archive.verify(b"123456789", "0" * 64, _declaration(), _plant(), seed=653)
    assert called == []


@pytest.mark.parametrize(
    "damage",
    [
        "missing-control",
        "missing-control-field",
        "missing-pulse",
        "missing-pulse-field",
        "missing-field",
    ],
)
def test_archive_refuses_incomplete_control_pulse_or_record(damage):
    declaration = _declaration()
    records = _records(1, declaration)
    if damage == "missing-control":
        del records[0]["runtime_result_before_reset"]["control_evidence"]
    elif damage == "missing-control-field":
        del records[0]["runtime_result_before_reset"]["control_evidence"]["initial"][
            "voltage"
        ]
    elif damage == "missing-pulse":
        records[0]["runtime_result_before_reset"]["scheduled_pulse_evidence"] = []
    elif damage == "missing-pulse-field":
        del records[0]["runtime_result_before_reset"]["scheduled_pulse_evidence"][0][
            "schedule_sha256"
        ]
    else:
        del records[0]["pre_action_values"]

    with pytest.raises(ValueError):
        archive.encode(records, declaration, _plant(), seed=653)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("seed", "fixed archive seed"),
        ("source", "source-bound ordered no-update receipt"),
        ("cursor", "source-bound ordered no-update receipt"),
        ("flag", "fixed archive seed/protocol and false capability flags"),
        ("nan", "finite archive tensor"),
        ("rng-no-advance", "continuous advancing private state"),
        ("rng-discontinuous", "continuous advancing private state"),
        ("rng-digest", "non-admitting private scope receipt"),
        ("rng-initial-digest", "non-admitting private scope receipt"),
        ("rng-state-bytes", "non-admitting private scope receipt"),
        ("rng-extra-key", "non-admitting private scope receipt"),
        ("logprob", "raw action Gaussian log-probability consistency"),
        ("timeout", "natural timeout unreachable"),
        ("timeout-clock", "bounded actual row clocks"),
        ("clock-injection", "bounded actual row clocks"),
        ("hidden-nan", "finite archive tensor"),
        ("proposal-clock", "proposal current physical boundary"),
        ("observation-gap", "continuous observation and control records"),
        ("control-gap", "continuous observation and control records"),
        ("pre-reset-copy", "pre-reset last boundary qpos"),
    ],
)
def test_archive_rejects_mutated_consistency_boundaries(damage, message):
    declaration = _declaration()
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": _records(2, declaration),
        **transition.FALSE_FLAGS,
    }
    first, second = value["records"]
    if damage == "seed":
        value["learner_seed"] = 659
    elif damage == "source":
        first["receipt"]["source"] = "b" * 40
    elif damage == "cursor":
        second["receipt"]["storage_step"] = 0
    elif damage == "flag":
        value[next(iter(transition.FALSE_FLAGS))] = True
    elif damage == "nan":
        first["raw_actions"][0, 0] = float("nan")
    elif damage == "rng-no-advance":
        first["private_rng_state"] = first["private_rng_state_before"].clone()
        first["rng_scope_receipt"]["private_state_sha256"] = sampling._state_digest(
            first["private_rng_state"]
        )
    elif damage == "rng-discontinuous":
        second["private_rng_state_before"][0] ^= 1
    elif damage == "rng-digest":
        first["rng_scope_receipt"]["private_state_sha256"] = "0" * 64
    elif damage == "rng-initial-digest":
        first["rng_scope_receipt"]["initial_state_sha256"] = "0" * 64
    elif damage == "rng-state-bytes":
        first["rng_scope_receipt"]["state_bytes"] = True
    elif damage == "rng-extra-key":
        first["rng_scope_receipt"]["unexpected"] = False
    elif damage == "logprob":
        first["raw_actions_log_prob"][0] += 1
    elif damage == "timeout":
        first["timed_out"][0] = True
        first["runtime_result_before_reset"]["timed_out"][0] = True
    elif damage == "clock-injection":
        first["pre_reset_steps"][0] = 281
        first["runtime_result_before_reset"]["episode_steps"][0] = 281
    elif damage == "timeout-clock":
        first["pre_reset_steps"][0] = 2500
        first["runtime_result_before_reset"]["episode_steps"][0] = 2500
    elif damage == "hidden-nan":
        first["runtime_result_before_reset"]["term_sums"]["hidden"] = torch.tensor(
            [float("nan")]
        )
    elif damage == "proposal-clock":
        first["runtime_result_before_reset"]["control_evidence"]["proposals"][0][
            "before_steps"
        ][0] = 1
    elif damage == "observation-gap":
        second["pre_action_observations"]["actor"][0, 0] = 1
        second["pre_action_observations"]["critic"][0, 0] = 1
    elif damage == "control-gap":
        second["runtime_result_before_reset"]["control_evidence"]["initial"]["voltage"][
            0, 0
        ] = 7.0
    elif damage == "pre-reset-copy":
        first["pre_reset_kinematics"]["qpos"][0, 0] = 1

    with pytest.raises(ValueError, match=message):
        archive.check(value, declaration, _plant(), seed=653)


def _terminalize_first_row(record, declaration, *, failure_step=1):
    raw = record["runtime_result_before_reset"]
    frames = raw["boundaries"]
    # Row zero fails naturally at the selected substep. Earlier boundaries
    # advance normally; later boundaries keep the failed row frozen.
    for step, frame in enumerate(frames[1:], start=1):
        frame["physics_steps"][0] = min(step, failure_step)
        if step >= failure_step:
            frame["state"]["forbidden_contact"][0] = True

    record["terminated"][0] = True
    raw["terminated"][0] = True
    raw["live"][0] = False
    raw["executed_steps"][0] = failure_step
    raw["episode_steps"][0] = failure_step
    record["pre_reset_steps"][0] = failure_step
    record["post_reset_steps"][0] = 0
    for field in ("environment_reward", "learner_reward"):
        record[field][0] = -2 + failure_step * 0.009
    raw["reward"][0] = -2 + failure_step * 0.009
    record["pre_reset_kinematics"]["qpos"][0] = frames[-1]["qpos"][0]
    record["pre_reset_kinematics"]["qvel"][0] = frames[-1]["qvel"][0]
    record["pre_reset_kinematics"]["time"][0] = failure_step * 0.002
    record["post_reset_kinematics"]["qpos"][0] = frames[0]["qpos"][0]
    record["post_reset_kinematics"]["qvel"][0].zero_()
    record["post_reset_kinematics"]["time"][0] = 0.0

    final = frames[-1]
    terminal = {
        "world_id": 0,
        "physics_step": failure_step,
        "terminated": True,
        "timed_out": False,
        "checkpoint_admitted": False,
        "trajectory_continuity_validated": False,
        "qpos": final["qpos"][0].tolist(),
        "qvel": final["qvel"][0].tolist(),
        "state": {key: value[0].tolist() for key, value in final["state"].items()},
        "observation": {
            key: value[0].tolist() for key, value in final["observation"].items()
        },
        "contacts": {"worldid": [0]},
        "rejected_proposed_torque_nm": None,
    }
    record["terminal_records"][0] = terminal
    record["reset_records"][0] = deepcopy(terminal)
    raw["terminal_records"][0] = deepcopy(terminal)

    for substep, proposal in enumerate(raw["control_evidence"]["proposals"]):
        live = torch.ones(N, dtype=torch.bool)
        if substep >= failure_step:
            live[0] = False
        proposal["before_steps"] = frames[substep]["physics_steps"].clone()
        proposal["live"] = live
        proposal["accepted"] = live.clone()
        proposal["rejected"] = torch.zeros(N, dtype=torch.bool)

        pulse = raw["scheduled_pulse_evidence"][substep]
        clocks = proposal["before_steps"].tolist()
        accepted = proposal["accepted"].tolist()
        pre_x, pre_q = schedule.expected_wrenches(
            declaration, clocks, accepted, N_BODY, BODY_ID
        )
        pulse["before_steps"] = proposal["before_steps"].clone()
        pulse["accepted"] = proposal["accepted"].clone()
        pulse["pre_xfrc"] = torch.tensor(pre_x, dtype=torch.float32)
        pulse["pre_qfrc"] = torch.tensor(pre_q, dtype=torch.float32)
        pulse["phases"] = (
            {name: {} for name in ("forced_pre", "integrated", "unforced_post")}
            if any(schedule.window_mask(declaration, clocks, accepted))
            else None
        )


def test_synthetic_physical_failure_can_be_final_record_but_not_followed():
    declaration = _declaration()
    records = _records(2, declaration)
    _terminalize_first_row(records[0], declaration)
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": records[:1],
        **transition.FALSE_FLAGS,
    }

    score = archive.check(value, declaration, _plant(), seed=653)
    assert score["complete_short_window"] is False
    assert score["natural_timeout_qualified"] is False
    assert score["selective_reset_qualified"] is False

    value["records"] = records
    with pytest.raises(ValueError, match="archive stops at first observed terminal"):
        archive.check(value, declaration, _plant(), seed=653)


def test_archive_rejects_omitted_termination_for_recorded_final_boundary_fall():
    declaration = _declaration()
    record = _record(0, declaration)
    _terminalize_first_row(record, declaration, failure_step=10)
    record["terminated"][0] = False
    record["runtime_result_before_reset"]["terminated"][0] = False
    record["runtime_result_before_reset"]["live"][0] = True
    record["post_reset_steps"][0] = 10
    record["post_reset_kinematics"]["qpos"][0] = record["pre_reset_kinematics"]["qpos"][
        0
    ]
    record["post_reset_kinematics"]["qvel"][0] = record["pre_reset_kinematics"]["qvel"][
        0
    ]
    record["post_reset_kinematics"]["time"][0] = record["pre_reset_kinematics"]["time"][
        0
    ]
    record["terminal_records"][0] = None
    record["reset_records"][0] = None
    record["runtime_result_before_reset"]["terminal_records"][0] = None
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": [record],
        **transition.FALSE_FLAGS,
    }

    with pytest.raises(
        ValueError, match="termination matches retained physical/proposed failures"
    ):
        archive.check(value, declaration, _plant(), seed=653)


def test_archive_rejects_fabricated_termination_without_failure():
    declaration = _declaration()
    record = _record(0, declaration)
    record["terminated"][0] = True
    record["runtime_result_before_reset"]["terminated"][0] = True
    record["runtime_result_before_reset"]["live"][0] = False
    record["post_reset_steps"][0] = 0
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": [record],
        **transition.FALSE_FLAGS,
    }

    with pytest.raises(
        ValueError, match="termination matches retained physical/proposed failures"
    ):
        archive.check(value, declaration, _plant(), seed=653)


def test_archive_rejects_inconsistent_torque_rejection_mask():
    declaration = _declaration()
    record = _record(0, declaration)
    proposal = record["runtime_result_before_reset"]["control_evidence"]["proposals"][0]
    proposal["accepted"][0] = False
    proposal["rejected"][0] = True
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": [record],
        **transition.FALSE_FLAGS,
    }

    with pytest.raises(ValueError, match="proposal recorded torque rejection mask"):
        archive.check(value, declaration, _plant(), seed=653)


def test_archive_rejects_acceptance_after_a_row_has_physically_failed():
    declaration = _declaration()
    record = _record(0, declaration)
    _terminalize_first_row(record, declaration, failure_step=1)
    proposal = record["runtime_result_before_reset"]["control_evidence"]["proposals"][1]
    proposal["live"][0] = True
    proposal["accepted"][0] = True
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": [record],
        **transition.FALSE_FLAGS,
    }

    with pytest.raises(
        ValueError, match="proposal excludes already failed physical rows"
    ):
        archive.check(value, declaration, _plant(), seed=653)


def test_archive_rejects_physical_mutation_on_nonaccepted_row():
    declaration = _declaration()
    record = _record(0, declaration)
    _terminalize_first_row(record, declaration, failure_step=1)
    frames = record["runtime_result_before_reset"]["boundaries"]
    frames[2]["qpos"][0, 0] = 0.1
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": [record],
        **transition.FALSE_FLAGS,
    }

    with pytest.raises(ValueError, match="unchanged physical boundary: qpos"):
        archive.check(value, declaration, _plant(), seed=653)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("reset-kinematics", "reset physical rows"),
        ("reset-controls", "retained reset control rows target"),
    ],
)
def test_archive_rejects_reset_row_copy_drift(mutation, message):
    declaration = _declaration()
    record = _record(0, declaration)
    _terminalize_first_row(record, declaration)
    if mutation == "reset-kinematics":
        record["post_reset_kinematics"]["qpos"][0, 0] = 1.0
    else:
        record["post_reset_controls"]["target"][0, 0] = 0.1
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": [record],
        **transition.FALSE_FLAGS,
    }

    with pytest.raises(ValueError, match=message):
        archive.check(value, declaration, _plant(), seed=653)


def test_check_requires_cuda_hidden_cpu_array(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    declaration = _declaration()
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": _records(1, declaration),
        **transition.FALSE_FLAGS,
    }

    with pytest.raises(ValueError, match="CUDA-hidden archive checker"):
        archive.check(value, declaration, _plant(), seed=653)


def test_archive_retains_only_nonadmitting_native_physics_claims():
    declaration = _declaration()
    value = {
        "protocol": archive.PROTOCOL,
        "declaration": declaration,
        "compiled_plant": _plant(),
        "learner_seed": 653,
        "records": _records(28, declaration),
        **transition.FALSE_FLAGS,
    }

    score = archive.check(value, declaration, _plant(), seed=653)

    for key in (
        "whole_trajectory_physics_resimulated",
        "physics_control_solver_replayed",
        "cuda_rng_math_replayed",
        "native_transition_qualified",
        "natural_timeout_qualified",
        "selective_reset_qualified",
        "caller_rng_states_independently_checked",
        "model_and_optimizer_states_independently_checked",
    ):
        assert score[key] is False
