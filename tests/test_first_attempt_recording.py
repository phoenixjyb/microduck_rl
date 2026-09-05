import copy
import hashlib
import itertools
import json

import pytest
import torch

from mjlab_microduck import hierarchical_obstacle_rollout as rollout
from mjlab_microduck.first_attempt_recording import (
    FRAME_FIELDS, RAW_FLAGS, RECORDING_PROTOCOL, FirstAttemptRecorder,
)


def fields(n=3, offset=0.0):
    return {k: torch.arange(n, dtype=torch.float32) + offset for k in FRAME_FIELDS}


def flags(n=3, **enabled):
    result = {k: torch.zeros(n, dtype=torch.bool) for k in RAW_FLAGS}
    for key, indices in enabled.items():
        result[key][indices] = True
    return result


def resolved(active, done, raw):
    result = rollout.first_terminal_outcomes(active, done, *(raw[k] for k in RAW_FLAGS))
    classified = torch.stack(list(raw.values())).any(0)
    return {
        "hard_failure": active & done & (raw["fall"] | raw["nan"]),
        "collision": result["collision"], "timeout": result["timeout"],
        "pass": result["pass"], "other_terminal": active & done & ~classified,
    }


def test_nonzero_collision_and_exact_preterminal_frame_survive_reset():
    recorder = FirstAttemptRecorder(3, 10)
    active = torch.ones(3, dtype=torch.bool)
    for step in range(3):
        state = fields(offset=100 * step)
        recorder.capture_pre_step(step, step * .02, active, state)
        done = torch.tensor([step == 0, step == 2, False])
        raw = flags(**({"pass": [0]} if step == 0 else {"collision": [1]} if step == 2 else {}))
        recorder.finish_step((step + 1) * .02, done, raw, resolved(active, done, raw))
        active &= ~done
        for value in state.values():
            value.fill_(-999)  # The simulator has auto-reset; snapshots must own copies.
    report = recorder.report()
    first, collision, incomplete = report["attempts"]
    assert first["terminal"]["outcome"] == "pass" and len(first["frames"]) == 1
    assert collision["environment_id"] == 1 and collision["terminal"]["outcome"] == "collision"
    assert [f["step"] for f in collision["frames"]] == [0, 2]
    assert collision["frames"][-1]["route_progress_m"] == 201.0
    assert collision["frames"][-1]["time_s"] == .04
    assert collision["terminal"]["time_s"] == .06
    assert "route_progress_m" not in collision["terminal"]
    assert incomplete["status"] == "incomplete" and incomplete["terminal"] is None
    report["attempts"][1]["frames"].clear()
    assert len(recorder.report()["attempts"][1]["frames"]) == 2


@pytest.mark.parametrize("bits", tuple(itertools.product((False, True), repeat=5)))
def test_all_raw_combinations_preserve_frozen_outcome_priority(bits):
    recorder = FirstAttemptRecorder(1, 1)
    active = torch.ones(1, dtype=torch.bool)
    raw = {k: torch.tensor([b]) for k, b in zip(RAW_FLAGS, bits, strict=True)}
    recorder.capture_pre_step(0, 0.0, active, fields(1))
    recorder.finish_step(.02, active, raw, resolved(active, active, raw))
    terminal = recorder.report()["attempts"][0]["terminal"]
    expected = "hard_failure" if bits[3] or bits[4] else "collision" if bits[0] else "timeout" if bits[2] else "pass" if bits[1] else "other_terminal"
    assert terminal["outcome"] == expected
    assert terminal["raw_flags"] == dict(zip(RAW_FLAGS, bits, strict=True))
    assert terminal["overlap"] == (sum(bits) > 1)


def test_capture_does_not_mutate_inputs_gradients_or_rng():
    recorder = FirstAttemptRecorder(3, 1)
    state = {k: v.requires_grad_() for k, v in fields().items()}
    before = {k: v.detach().clone() for k, v in state.items()}
    active = torch.ones(3, dtype=torch.bool)
    raw = flags(**{"pass": [0, 1, 2]})
    outcome = resolved(active, active, raw)
    raw_before, outcome_before = copy.deepcopy(raw), copy.deepcopy(outcome)
    rng = torch.get_rng_state().clone()
    recorder.capture_pre_step(0, 0.0, active, state)
    recorder.finish_step(.02, active, raw, outcome)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert value.grad is None
    assert active.tolist() == [True, True, True]
    for original, saved in ((raw, raw_before), (outcome, outcome_before)):
        for key in original:
            torch.testing.assert_close(original[key], saved[key])
            original[key].zero_()
    assert all(a["terminal"]["outcome"] == "pass" for a in recorder.report()["attempts"])


def test_storage_bound_and_closed_attempt_flags_are_ignored():
    recorder = FirstAttemptRecorder(2, 10)
    active = torch.ones(2, dtype=torch.bool)
    for step in range(10):
        recorder.capture_pre_step(step, step * .02, active, fields(2))
        # Env zero repeatedly auto-resets, but must have only its first trace.
        done = torch.tensor([True, step == 9])
        raw = flags(2, **{"pass": [0, 1] if step == 9 else [0]})
        recorder.finish_step((step + 1) * .02, done, raw, resolved(active, done, raw))
        active &= ~done
    report = recorder.report()
    assert [len(a["frames"]) for a in report["attempts"]] == [1, 3]
    assert report["max_frames_per_environment"] == 3
    with pytest.raises(ValueError, match="bounded"):
        recorder.capture_pre_step(10, .2, active, fields(2))


def test_nonfinite_state_is_explicit_and_json_safe():
    recorder = FirstAttemptRecorder(1, 1)
    state = fields(1)
    state["route_speed_mps"][0] = float("nan")
    state["obstacle_clearance_m"][0] = float("inf")
    active = torch.ones(1, dtype=torch.bool)
    raw = flags(1, nan=[0])
    recorder.capture_pre_step(0, 0.0, active, state)
    recorder.finish_step(.02, active, raw, resolved(active, active, raw))
    report = json.loads(json.dumps(recorder.report(), allow_nan=False))
    frame = report["attempts"][0]["frames"][0]
    assert frame["route_speed_mps"] is None
    assert set(frame["nonfinite_fields"]) == {"route_speed_mps", "obstacle_clearance_m"}
    assert report["attempts"][0]["terminal"]["outcome"] == "hard_failure"


@pytest.mark.parametrize("n,steps", [(65, 10), (1, 1001), (0, 1), (1, 0), (True, 1)])
def test_invalid_storage_bounds_rejected(n, steps):
    with pytest.raises(ValueError):
        FirstAttemptRecorder(n, steps)


def test_unpaired_steps_and_bad_resolution_fail_closed():
    recorder = FirstAttemptRecorder(1, 3)
    active = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match="sequential"):
        recorder.capture_pre_step(1, .02, active, fields(1))
    recorder.capture_pre_step(0, 0., active, fields(1))
    with pytest.raises(ValueError, match="unfinished"):
        recorder.report()
    raw = flags(1, collision=[0], **{"pass": [0]})
    outcome = resolved(active, active, raw)
    outcome["pass"][:] = True
    with pytest.raises(ValueError, match="partition"):
        recorder.finish_step(.02, active, raw, outcome)
    outcome["collision"][:] = False
    with pytest.raises(ValueError, match="failure-priority"):
        recorder.finish_step(.02, active, raw, outcome)


def test_closed_attempt_cannot_be_reactivated():
    recorder = FirstAttemptRecorder(2, 3)
    active = torch.ones(2, dtype=torch.bool)
    done = torch.tensor([False, True])
    raw = flags(2, collision=[1])
    recorder.capture_pre_step(0, 0., active, fields(2))
    recorder.finish_step(.02, done, raw, resolved(active, done, raw))
    with pytest.raises(ValueError, match="unfinished first attempts"):
        recorder.capture_pre_step(1, .02, active, fields(2))


def test_nonterminal_raw_flags_cannot_be_silently_discarded():
    recorder = FirstAttemptRecorder(1, 2)
    active, done = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    recorder.capture_pre_step(0, 0., active, fields(1))
    empty = resolved(active, done, flags(1))
    with pytest.raises(ValueError, match="raw terminal flag"):
        recorder.finish_step(.02, done, flags(1, collision=[0]), empty)


def test_bad_shapes_and_time_are_rejected_before_capture():
    recorder = FirstAttemptRecorder(1, 2)
    active = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match="finite"):
        recorder.capture_pre_step(0, float("nan"), active, fields(1))
    with pytest.raises(ValueError, match="boolean vectors"):
        recorder.capture_pre_step(0, 0., active.float(), fields(1))
    malformed = fields(1)
    malformed["phase"] = torch.zeros(2)
    with pytest.raises(ValueError, match="compact-state vectors"):
        recorder.capture_pre_step(0, 0., active, malformed)


@pytest.mark.parametrize("changes", [
    {"first_attempt_only": False}, {"num_envs": 65},
    {"case_count": 13}, {"collecting_dataset": True},
])
def test_opt_in_limits_do_not_change_disabled_mode(changes):
    args = dict(first_attempt_only=True, num_envs=64, case_count=12, collecting_dataset=False)
    args.update(changes)
    with pytest.raises(ValueError, match="first-attempt recording"):
        rollout.validate_first_attempt_recording_mode(True, **args)
    rollout.validate_first_attempt_recording_mode(False, **args)


def test_opt_in_report_sidecar_binding_and_disabled_report_unchanged(tmp_path, monkeypatch):
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"synthetic-locomotion")
    fake_case = dict(nominal_speed_mps=.3, obstacle_forward_m=.9, obstacle_lateral_m=0.,
                     seed=359, num_envs=2, steps=2, steps_executed=1,
                     collision_events=0, clean_pass_events=2, attempt_timeout_events=0,
                     fall_events=0, nan_termination_events=0, nonfinite_steps=0,
                     expected_attempts=2, completed_attempts=2, other_terminal_events=0)
    recorder = FirstAttemptRecorder(2, 2)
    active = torch.ones(2, dtype=torch.bool)
    raw = flags(2, **{"pass": [0, 1]})
    recorder.capture_pre_step(0, 0., active, fields(2))
    recorder.finish_step(.02, active, raw, resolved(active, active, raw))

    def run_case(*args, **kwargs):
        case = copy.deepcopy(fake_case)
        if kwargs["record_first_attempts"]:
            case["_first_attempt_recording"] = recorder.report()
        return case

    monkeypatch.setattr(rollout, "_run_case", run_case)
    kwargs = dict(num_envs=2, steps=2, speeds=(.3,), forward_positions=(.9,),
                  lateral_positions=(0.,), seeds=(359,), first_attempt_only=True)
    off = json.loads(rollout.run_rollout(actor, tmp_path / "off", **kwargs).read_text())
    explicit_off = json.loads(rollout.run_rollout(actor, tmp_path / "explicit-off", record_first_attempts=False, **kwargs).read_text())
    on = json.loads(rollout.run_rollout(actor, tmp_path / "on", record_first_attempts=True, **kwargs).read_text())
    assert off == explicit_off
    assert not (tmp_path / "off" / "first-attempt-traces").exists()
    assert on.pop("first_attempt_recording_protocol") == RECORDING_PROTOCOL
    sidecar = on["cases"][0].pop("first_attempt_recording")
    from pathlib import Path
    path = Path(sidecar["path"])
    assert sidecar["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    saved = json.loads(path.read_text())
    assert saved["case"]["seed"] == 359 and saved["case_index"] == 0
    assert saved["checkpoint_sha256"] == hashlib.sha256(actor.read_bytes()).hexdigest()
    assert saved["supervisor_checkpoint_sha256"] is None
    assert on == off
