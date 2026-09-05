import datetime as dt
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mjlab_microduck import first_attempt_smoke as smoke
from mjlab_microduck.first_attempt_recording import FRAME_FIELDS, FirstAttemptRecorder


@pytest.fixture
def pair(tmp_path):
    """Independent serialized fixture, including a nonzero collision and overlap."""
    active = torch.ones(4, dtype=torch.bool)
    recorder = FirstAttemptRecorder(4, 700)
    recorder.capture_pre_step(0, 0., active, {k: torch.zeros(4) for k in FRAME_FIELDS})
    raw = {k: torch.zeros(4, dtype=torch.bool) for k in smoke.RAW}
    raw["pass"][:] = True
    raw["collision"][1] = True
    resolved = {k: torch.zeros(4, dtype=torch.bool) for k in
                ("hard_failure", "collision", "timeout", "pass", "other_terminal")}
    resolved["collision"][1] = True
    resolved["pass"][:] = True
    resolved["pass"][1] = False
    recorder.finish_step(.02, active, raw, resolved)
    counts = dict(expected_attempts=4, completed_attempts=4, unresolved_attempts=0,
                  collision_events=1, clean_pass_events=3, attempt_timeout_events=0,
                  fall_events=0, nan_termination_events=0, hard_failure_events=0,
                  other_terminal_events=0, nonfinite_steps=0)
    case = {**smoke.CASE, **counts, "steps_executed": 1,
            "terminal_outcome_protocol": smoke.OUTCOME,
            "raw_terminal_events": {k: int(v.sum()) for k, v in raw.items()},
            "terminal_overlap_events": 1, "motor_torque_utilization_p99": .5,
            "motor_speed_rated_exceed_fraction": 0., "approach_route_speed_mps": .399,
            "representative_first_attempt_trace": [{"time_s": 0., "speed": .1}]}
    report = dict(stage="HC4U4-near-state-correction-phase-BC-rollout", decision="diagnostic-only",
                  physical_motion_authorized=False, checkpoint_sha256=smoke.ACTOR_SHA256,
                  supervisor_checkpoint_sha256=smoke.SUPERVISOR_SHA256,
                  terminal_outcome_protocol=smoke.OUTCOME,
                  evaluation_window="first-terminal-attempt-per-environment-v1",
                  attempt_timeout_s=12., obstacle_sensor_model=dict(range_noise_m=0., bearing_noise_rad=0.,
                  width_noise_m=0., height_noise_m=0., closing_rate_noise_mps=0., dropout_probability=0.),
                  cases=[case], totals=counts)
    off = tmp_path / "disabled" / "hierarchical-teacher-evaluation.json"
    on = tmp_path / "enabled" / "hierarchical-teacher-evaluation.json"
    trace = on.parent / "first-attempt-traces" / "case-000.json"
    off.parent.mkdir()
    trace.parent.mkdir(parents=True)
    off.write_text(json.dumps(report))
    recording = recorder.report()
    recording.update(case={**smoke.CASE, "steps_executed": 1}, case_index=0,
                     checkpoint_sha256=smoke.ACTOR_SHA256,
                     supervisor_checkpoint_sha256=smoke.SUPERVISOR_SHA256)
    trace.write_text(json.dumps(recording))
    report["first_attempt_recording_protocol"] = smoke.RECORDING
    case["first_attempt_recording"] = dict(path=str(trace), sha256=smoke.sha256(trace), protocol=smoke.RECORDING)
    on.write_text(json.dumps(report))
    return off, on, trace


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_valid_pair_is_instrumentation_only_and_does_not_mutate_artifacts(pair):
    before = [p.read_bytes() for p in pair]
    result = smoke.evaluate_smoke(*pair[:2])
    assert result["decision"] == "recorder-validated"
    assert result["controller_failures_observed"] is True
    assert result["may_predeclare_fresh_diagnostic"] is True
    assert not result["policy_acceptance"] and not result["training_data_admitted"]
    assert result["recording"]["outcomes"] == {"pass": 3, "collision": 1}
    assert before == [p.read_bytes() for p in pair]


@pytest.mark.parametrize("change", [
    lambda r: r["cases"][0].update(approach_route_speed_mps=.3990000001),
    lambda r: r["cases"][0]["representative_first_attempt_trace"][0].update(speed=.2),
    lambda r: r.update(checkpoint_sha256="0" * 64),
    lambda r: r["cases"][0].update(seed=347),
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=float("nan")),
    lambda r: r["cases"][0].update(completed_attempts=3, unresolved_attempts=1),
    lambda r: r["cases"][0].update(raw_terminal_events=dict.fromkeys(smoke.RAW, 0)),
    lambda r: r["cases"][0]["first_attempt_recording"].update(sha256="0" * 64),
    lambda r: r["cases"][0].pop("motor_torque_utilization_p99"),
])
def test_changed_metric_identity_outcome_or_hash_stops(pair, change):
    edit(pair[1], change)
    result = smoke.evaluate_smoke(*pair[:2])
    assert result["decision"] == "stop-recorder-diagnosis"
    assert result["errors"] and not result["may_predeclare_fresh_diagnostic"]


@pytest.mark.parametrize("change", [
    lambda r: r["attempts"][1].update(environment_id=0),
    lambda r: r["attempts"][1]["terminal"].update(outcome="pass"),
    lambda r: r["attempts"][1]["terminal"].update(route_progress_m=999.),
    lambda r: r["attempts"][1]["frames"].clear(),
    lambda r: r["attempts"][1]["frames"][0].update(time_s=.02),
    lambda r: r["attempts"][1]["frames"][0].update(phase=3),
    lambda r: r["attempts"][1]["frames"][0].update(route_speed_mps=None),
    lambda r: r["attempts"][1].update(status="incomplete"),
    lambda r: r.update(supervisor_checkpoint_sha256="0" * 64),
])
def test_malformed_trace_stops_even_with_recomputed_hash(pair, change):
    edit(pair[2], change)
    edit(pair[1], lambda r: r["cases"][0]["first_attempt_recording"].update(sha256=smoke.sha256(pair[2])))
    result = smoke.evaluate_smoke(*pair[:2])
    assert result["decision"] == "stop-recorder-diagnosis"
    assert result["errors"] and not result["may_predeclare_fresh_diagnostic"]


def test_matching_motor_failure_does_not_open_next_runtime_gate(pair):
    for path in pair[:2]:
        edit(path, lambda r: r["cases"][0].update(motor_torque_utilization_p99=.61))
    result = smoke.evaluate_smoke(*pair[:2])
    assert result["decision"] == "recorder-validated"
    assert not result["runtime_numerical_checks_passed"]
    assert not result["may_predeclare_fresh_diagnostic"]


def test_missing_trace_fails_closed(pair):
    pair[2].unlink()
    assert smoke.evaluate_smoke(*pair[:2])["decision"] == "stop-recorder-diagnosis"


@pytest.mark.parametrize("failure", [None, "exit", "timeout", "existing", "dirty"])
def test_runner_is_sequential_bounded_preserve_first_and_stops_on_failure(tmp_path, monkeypatch, failure):
    spec = importlib.util.spec_from_file_location("recorder_runner", Path(__file__).parents[1] / "scripts/check_first_attempt_recorder.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "evidence"
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "LAST_START", dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(sys, "argv", ["smoke", "--source", "a" * 40])
    answers = {("git", "branch", "--show-current"): "feat/athletics-obstacle-curriculum",
               ("git", "status", "--porcelain"): " M user.py" if failure == "dirty" else "",
               ("git", "rev-parse", "HEAD"): "a" * 40}
    monkeypatch.setattr(runner, "read", lambda *args: answers[args])
    monkeypatch.setattr(runner, "sha256", lambda p: smoke.ACTOR_SHA256 if p == runner.ACTOR else smoke.SUPERVISOR_SHA256)
    monkeypatch.setattr(runner, "check_host", lambda: {"mocked": True})
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "evaluate_smoke", lambda *args: {"may_predeclare_fresh_diagnostic": True})
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 240 and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 240)
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    if failure == "existing":
        output.mkdir()
    if failure in ("existing", "dirty"):
        with pytest.raises((ValueError, FileExistsError)):
            runner.main()
        assert not calls
        return
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == (0 if failure is None else 2)
    assert len(calls) == (2 if failure is None else 1)
    assert "--record-first-attempts" not in calls[0]
    if len(calls) == 2:
        assert calls[1][-1] == "--record-first-attempts"
    decision = json.loads((output / "decision.json").read_text())
    assert decision["may_predeclare_fresh_diagnostic"] == (failure is None)
    assert (output / "launch.json").exists() and (output / "runtime.json").exists()
