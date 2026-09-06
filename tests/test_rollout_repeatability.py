"""CPU-only control tests; all live commands and GPU child launches are mocked."""

import datetime as dt
import json
import subprocess
import sys

import pytest

from mjlab_microduck import rollout_repeatability as control


def test_start_deadline_is_the_explicit_single_control_amendment():
    assert control.LAST_START == dt.datetime(2026, 9, 6, 1, 5, tzinfo=dt.timezone.utc)


@pytest.fixture
def report():
    counts = dict.fromkeys(control.COUNT_KEYS, 0)
    counts.update(expected_attempts=4, completed_attempts=4,
                  collision_events=1, clean_pass_events=2, attempt_timeout_events=1)
    case = {**control.CASE, **counts, "steps_executed": 600,
            "terminal_outcome_protocol": control.OUTCOME,
            "motor_speed_rated_exceed_fraction": 0., "motor_torque_utilization_p99": .5,
            "representative_first_attempt_trace": [{"time_s": 0., "speed": .1}]}
    return dict(stage="HC4U4-near-state-correction-phase-BC-rollout", decision="diagnostic-only",
                physical_motion_authorized=False, checkpoint_sha256=control.ACTOR_SHA256,
                supervisor_checkpoint_sha256=control.SUPERVISOR_SHA256,
                terminal_outcome_protocol=control.OUTCOME,
                evaluation_window="first-terminal-attempt-per-environment-v1",
                attempt_timeout_s=12., obstacle_sensor_model=dict(range_noise_m=0., bearing_noise_rad=0.,
                width_noise_m=0., height_noise_m=0., closing_rate_noise_mps=0., dropout_probability=0.),
                cases=[case], totals=counts)


@pytest.fixture
def pair(tmp_path, report):
    paths = (tmp_path / "first.json", tmp_path / "second.json")
    for path in paths:
        path.write_text(json.dumps(report))
    return paths


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def assert_closed(result):
    for key in ("policy_acceptance", "recorder_validated", "training_data_admitted",
                "physical_motion_authorized", "further_gpu_job_admitted"):
        assert result[key] is False


def test_matching_reports_are_descriptive_only_and_immutable(pair):
    before = [path.read_bytes() for path in pair]
    result = control.evaluate_pair(*pair)
    assert result["decision"] == "same-seed-reports-match-in-control"
    assert result["exact_reports_equal"] and result["differences"] == []
    assert result["outcomes"][0]["collision_events"] == 1
    assert result["outcomes"][0]["attempt_timeout_events"] == 1
    assert result["report_sha256"] == dict(first=control.sha256(pair[0]), second=control.sha256(pair[1]))
    assert before == [path.read_bytes() for path in pair]
    assert_closed(result)


@pytest.mark.parametrize("change", [
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=.5000000001),
    lambda r: r["cases"][0]["representative_first_attempt_trace"][0].update(speed=.1000000001),
    lambda r: r["cases"][0]["representative_first_attempt_trace"][0].update(time_s=-0.),
    lambda r: r["cases"][0]["representative_first_attempt_trace"].append({"time_s": .1}),
    lambda r: r.update(extra_metric=0.),
])
def test_every_valid_difference_is_retained_without_tolerance(pair, change):
    edit(pair[1], change)
    result = control.evaluate_pair(*pair)
    assert result["decision"] == "same-seed-reports-diverge-with-recording-disabled"
    assert result["exact_reports_equal"] is False
    assert result["difference_count"] == len(result["differences"]) > 0
    assert result["errors"] == []
    assert_closed(result)


@pytest.mark.parametrize("change", [
    lambda r: r.update(stage="wrong"),
    lambda r: r.update(decision="accepted"),
    lambda r: r.update(physical_motion_authorized=True),
    lambda r: r.update(checkpoint_sha256="0" * 64),
    lambda r: r.update(supervisor_checkpoint_sha256="0" * 64),
    lambda r: r.update(evaluation_window="all-episodes"),
    lambda r: r.update(attempt_timeout_s=13.),
    lambda r: r["obstacle_sensor_model"].update(range_noise_m=.01),
    lambda r: r.update(first_attempt_recording_protocol="recording"),
    lambda r: r.update(dataset_path="not-allowed"),
    lambda r: r["cases"].clear(),
    lambda r: r["cases"][0].update(seed=359),
    lambda r: r["cases"][0].update(num_envs=8),
    lambda r: r["cases"][0].update(first_attempt_recording={}),
    lambda r: r["cases"][0].update(dataset_path="not-allowed"),
    lambda r: r["cases"][0].update(steps_executed=701),
    lambda r: r["cases"][0].update(steps_executed=True),
    lambda r: r["cases"][0].update(terminal_outcome_protocol="wrong"),
    lambda r: r["cases"][0].update(representative_first_attempt_trace=[]),
    lambda r: r["cases"][0].update(representative_first_attempt_trace="not-a-list"),
    lambda r: r["cases"][0].update(completed_attempts=3, unresolved_attempts=1),
    lambda r: r["cases"][0].update(collision_events=True),
    lambda r: r["cases"][0].update(clean_pass_events=-1),
    lambda r: r["cases"][0].update(fall_events=1),
    lambda r: r["cases"][0].update(nan_termination_events=1),
    lambda r: r["cases"][0].update(nonfinite_steps=1),
    lambda r: r["cases"][0].update(hard_failure_events=1),
    lambda r: r["cases"][0].update(other_terminal_events=1),
    lambda r: r["cases"][0].update(clean_pass_events=3),
    lambda r: r["totals"].update(collision_events=2, clean_pass_events=1),
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=.60001),
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=.6125551462173462),
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=-.01),
    lambda r: r["cases"][0].update(motor_speed_rated_exceed_fraction=.00001),
    lambda r: r.update(unused_metric=float("nan")),
    lambda r: r["cases"][0]["representative_first_attempt_trace"][0].update(speed=float("inf")),
    lambda r: r.pop("totals"),
])
def test_invalid_reports_stop_even_if_both_match(pair, change):
    for path in pair:
        edit(path, change)
    result = control.evaluate_pair(*pair)
    assert result["decision"] == "invalid-control-stop" and result["errors"]
    assert_closed(result)


@pytest.mark.parametrize("mode", ["same-path", "missing", "bad-json"])
def test_distinct_readable_reports_required(pair, mode):
    if mode == "same-path":
        pair = (pair[0], pair[0])
    elif mode == "missing":
        pair[1].unlink()
    else:
        pair[1].write_text("{")
    result = control.evaluate_pair(*pair)
    assert result["decision"] == "invalid-control-stop" and result["errors"]
    assert_closed(result)


def test_recursive_diff_is_typed_signed_zero_aware_and_ordered():
    left = {"z": [1, False, 0.], "gone": None}
    right = {"z": [1., 0, -0., 2], "new": None}
    diff = control.differences(left, right)
    assert [row["path"] for row in diff] == ["/gone", "/new", "/z/length", "/z/0", "/z/1", "/z/2"]
    assert diff[0]["missing"] == "second" and diff[1]["missing"] == "first"
    assert control.differences({"b": 2, "a": 1}, {"a": 1, "b": 2}) == []


@pytest.mark.parametrize("failure", [None, "exit", "timeout", "invalid-first", "motor-first", "existing",
                                   "dirty", "branch", "source", "expired", "occupied-second", "diverge"])
def test_runner_is_bounded_sequential_preserve_first_and_fail_closed(tmp_path, monkeypatch, report, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path)
    output = tmp_path / "evidence"
    monkeypatch.setattr(control, "OUTPUT", output)
    monkeypatch.setattr(control, "LAST_START", (dt.datetime.min if failure == "expired" else
                                              dt.datetime.max).replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(sys, "argv", ["control", "--source", "a" * 40])
    answers = {("git", "branch", "--show-current"): "wrong" if failure == "branch" else "feat/athletics-obstacle-curriculum",
               ("git", "status", "--porcelain"): " M user.py" if failure == "dirty" else "",
               ("git", "rev-parse", "HEAD"): "b" * 40 if failure == "source" else "a" * 40}
    monkeypatch.setattr(control, "read", lambda *args: answers[args])
    monkeypatch.setattr(control, "sha256", lambda p: control.ACTOR_SHA256 if p == control.ACTOR else control.SUPERVISOR_SHA256)
    monkeypatch.setattr(control.importlib.metadata, "version", lambda _: "mocked")
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    calls, checks = [], []

    def host():
        checks.append(True)
        if failure == "occupied-second" and len(checks) == 2:
            raise ValueError("GPU compute workload")
        return {"mocked": True}

    def child(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 240 and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert "--record-first-attempts" not in command and not any("dataset" in arg for arg in command)
        assert command[command.index("--seeds") + 1] == "367"
        assert command[command.index("--steps") + 1] == "700" and "--first-attempt-only" in command
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 240)
        case_dir = output / ("first" if len(calls) == 1 else "second")
        case_dir.mkdir()
        if failure == "invalid-first":
            report["cases"][0]["fall_events"] = 1
        if failure == "motor-first":
            report["cases"][0]["motor_torque_utilization_p99"] = .6125551462173462
        if failure == "diverge" and len(calls) == 2:
            report["cases"][0]["steps_executed"] = 601
        (case_dir / "hierarchical-teacher-evaluation.json").write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0)

    monkeypatch.setattr(control, "check_host", host)
    monkeypatch.setattr(control.subprocess, "run", child)
    if failure == "existing":
        output.mkdir()
        (output / "user.txt").write_text("preserve")
    if failure in ("existing", "dirty", "branch", "source", "expired"):
        with pytest.raises((ValueError, FileExistsError)):
            control.main()
        assert not calls
        assert sorted(p.name for p in output.iterdir()) == ["user.txt"] if output.exists() else True
        return
    with pytest.raises(SystemExit) as stopped:
        control.main()
    assert stopped.value.code == (0 if failure is None else 2)
    assert len(calls) == (2 if failure in (None, "diverge") else 1)
    decision = json.loads((output / "decision.json").read_text())
    assert_closed(decision)
    runtime = json.loads((output / "runtime.json").read_text())
    assert len(runtime["runs"]) == len(calls)
    assert all(run["wall_seconds"] >= 0 for run in runtime["runs"])
    assert (output / "launch.json").exists()
    if len(calls) == 2:
        assert calls[0] == [arg.replace(str(output / "second"), str(output / "first")) for arg in calls[1]]


@pytest.mark.parametrize("failure", [None, "service", "compute", "busy", "hot", "memory"])
def test_host_gate_never_changes_services_or_processes(monkeypatch, failure):
    calls = []

    def read(*args):
        calls.append(args)
        if args[0] == "systemctl":
            assert args[1] == "show"
            return "active" if failure == "service" else "inactive"
        if "--query-compute-apps=pid" in args:
            return "123" if failure == "compute" else ""
        return {"busy": "1, 45, 12", "hot": "0, 80, 12", "memory": "0, 45, 100"}.get(failure, "0, 45, 12")

    monkeypatch.setattr(control, "read", read)
    if failure:
        with pytest.raises(ValueError):
            control.check_host()
    else:
        assert control.check_host() == dict(utilization_percent=0, temperature_c=45, memory_mib=12)
    assert all(call[0] in ("systemctl", "nvidia-smi") for call in calls)
