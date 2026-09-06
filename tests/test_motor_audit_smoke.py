"""CPU adversarial validation; live commands and the sole GPU child are mocked."""

import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import motor_audit_smoke as smoke
from mjlab_microduck.motor_measurement_audit import MotorMeasurementAudit


@pytest.fixture
def report():
    audit = MotorMeasurementAudit(4, 700, smoke.JOINTS, tuple(range(14)), stall_reference_nm=.6)
    for step in range(3):
        done = torch.full((4,), step == 2)
        force = torch.arange(56).reshape(4, 14).float() / 1000
        data = NS(actuator_force=force, joint_vel=force * 2)
        audit.begin(step, torch.ones(4, dtype=torch.bool), torch.full((4,), step))
        audit.capture(data, done)
        audit.finish(done, force + 1)
    counts = dict.fromkeys(smoke.COUNT_KEYS, 0)
    counts.update(expected_attempts=4, completed_attempts=4, clean_pass_events=4)
    case = {**smoke.CASE, **counts, "steps_executed": 3, "terminal_outcome_protocol": smoke.OUTCOME,
            "motor_speed_rated_exceed_fraction": 0., "motor_torque_utilization_p99": .5,
            "motor_measurement_audit": audit.report()}
    return dict(stage="HC4U4-near-state-correction-phase-BC-rollout", decision="diagnostic-only",
                physical_motion_authorized=False, checkpoint_sha256=smoke.ACTOR_SHA256,
                supervisor_checkpoint_sha256=smoke.SUPERVISOR_SHA256,
                terminal_outcome_protocol=smoke.OUTCOME, evaluation_window="first-terminal-attempt-per-environment-v1",
                attempt_timeout_s=12., obstacle_sensor_model=dict(range_noise_m=0., bearing_noise_rad=0.,
                width_noise_m=0., height_noise_m=0., closing_rate_noise_mps=0., dropout_probability=0.),
                cases=[case], totals=counts, motor_measurement_audit_protocol=smoke.AUDIT)


def decide(tmp_path, report):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    decision = smoke.evaluate_report(path)
    assert path.read_bytes() == before and decision["report_sha256"] == smoke.sha256(path)
    for key in ("policy_acceptance", "runtime_equivalence_validated", "training_data_admitted",
                "physical_motion_authorized", "further_gpu_job_admitted"):
        assert decision[key] is False
    return decision


def test_real_collector_report_validates_without_admission(tmp_path, report):
    result = decide(tmp_path, report)
    assert result["decision"] == "measurement-smoke-validated-not-admission"
    assert result["measurement_structure_validated"] and result["legacy_runtime_gate_passed"]


@pytest.mark.parametrize("change", [
    lambda c: c.update(motor_torque_utilization_p99=.6125551462173462),
    lambda c: c.update(motor_speed_rated_exceed_fraction=.001),
    lambda c: c.update(fall_events=1),
    lambda c: c.update(nonfinite_steps=1),
])
def test_corrected_lower_torque_cannot_reopen_legacy_gate(tmp_path, report, change):
    change(report["cases"][0])
    report["totals"] = {k: report["cases"][0][k] for k in smoke.COUNT_KEYS}
    result = decide(tmp_path, report)
    assert result["measurement_structure_validated"] is True
    assert result["legacy_runtime_gate_passed"] is False
    assert result["decision"] == "legacy-runtime-gate-stop"


@pytest.mark.parametrize("change", [
    lambda r: r.update(checkpoint_sha256="0" * 64),
    lambda r: r.update(physical_motion_authorized=True),
    lambda r: r.update(motor_measurement_audit_protocol="wrong"),
    lambda r: r.update(first_attempt_recording_protocol="forbidden"),
    lambda r: r.update(dataset_path="forbidden"),
    lambda r: r["obstacle_sensor_model"].update(dropout_probability=.1),
    lambda r: r["cases"][0].update(seed=367),
    lambda r: r["cases"][0].update(num_envs=4.),
    lambda r: r["cases"][0].update(steps_executed=True),
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=True),
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=float("nan")),
    lambda r: r["totals"].update(completed_attempts=3),
])
def test_wrong_provenance_or_invalid_base_report_fails(tmp_path, report, change):
    change(report)
    assert decide(tmp_path, report)["decision"] == "invalid-audit-stop"


@pytest.mark.parametrize("change", [
    lambda a: a.update(policy_acceptance=True),
    lambda a: a.update(legacy_metrics_replaced=True),
    lambda a: a.update(finite=False),
    lambda a: a.update(sampling="post-return"),
    lambda a: a.update(force_timing="all-substep-peak"),
    lambda a: a.update(stall_reference_nm=.7),
    lambda a: a["joint_columns"].reverse(),
    lambda a: a.update(steps_captured=4),
    lambda a: a.update(terminal_environment_steps=3),
    lambda a: a.update(incomplete_first_attempts=1),
    lambda a: a["groups"]["all"].update(environment_steps=13),
    lambda a: a["groups"]["approach"].update(environment_steps=0),
    lambda a: a["groups"]["all"]["force_nm"].update(nonfinite_samples=1),
    lambda a: a["groups"]["all"]["force_nm"].update(samples=0),
    lambda a: a["groups"]["all"]["force_nm"].update(abs_p99=None),
    lambda a: a["groups"]["all"]["force_nm"].update(abs_max=float("inf")),
    lambda a: a["groups"]["all"]["force_nm"].update(rms=-1),
    lambda a: a["groups"]["all"]["force_nm"].update(abs_p99=10),
    lambda a: a["groups"]["all"]["stall_reference_utilization"].update(abs_p99=0),
    lambda a: a["groups"]["recovery"]["by_joint"].pop("head_roll"),
    lambda a: a["groups"]["all"]["by_joint"]["head_roll"].update(samples=1),
    lambda a: a["terminal_force_nm"].update(samples=55),
    lambda a: a["terminal_post_return_minus_pre_reset_nm"].update(nonfinite_samples=1),
])
def test_adversarial_audit_summaries_fail_closed(tmp_path, report, change):
    change(report["cases"][0]["motor_measurement_audit"])
    result = decide(tmp_path, report)
    assert result["decision"] == "invalid-audit-stop" and result["errors"]


@pytest.mark.parametrize("failure", [None, "existing", "dirty", "branch", "source", "expired", "version",
                                   "dependency", "busy", "exit", "timeout", "audit", "legacy"])
def test_one_bounded_child_retained_and_no_retry(tmp_path, monkeypatch, report, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    output = tmp_path / "evidence"
    monkeypatch.setattr(smoke, "OUTPUT", output)
    monkeypatch.setattr(smoke, "LAST_START", (dt.datetime.min if failure == "expired" else
                      dt.datetime.max).replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(sys, "argv", ["smoke", "--source", "a" * 40])
    answers = {("git", "branch", "--show-current"): "wrong" if failure == "branch" else "feat/athletics-obstacle-curriculum",
               ("git", "status", "--porcelain"): " M user.py" if failure == "dirty" else "",
               ("git", "rev-parse", "HEAD"): "b" * 40 if failure == "source" else "a" * 40}
    monkeypatch.setattr(smoke, "read", lambda *args: answers[args])
    original_hash = smoke.sha256
    def hashed(path):
        if path == smoke.ACTOR: return smoke.ACTOR_SHA256
        if path == smoke.SUPERVISOR: return smoke.SUPERVISOR_SHA256
        for name, expected in smoke.DEPENDENCIES.items():
            if str(path).endswith(name): return "wrong" if failure == "dependency" else expected
        return original_hash(path)
    monkeypatch.setattr(smoke, "sha256", hashed)
    monkeypatch.setattr(smoke.importlib.metadata, "version", lambda name:
                       "wrong" if failure == "version" else smoke.VERSIONS[name])
    def host():
        if failure == "busy": raise ValueError("GPU occupied")
        return dict(utilization_percent=0, temperature_c=44, memory_mib=12)
    monkeypatch.setattr(smoke, "check_host", host)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 240 and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert "--motor-measurement-audit" in command and "--first-attempt-only" in command
        assert "--record-first-attempts" not in command and not any("dataset" in c for c in command)
        assert command[command.index("--seeds") + 1] == "373"
        assert command[command.index("--steps") + 1] == "700"
        if failure == "timeout": raise subprocess.TimeoutExpired(command, 240)
        if failure == "audit": report["cases"][0]["motor_measurement_audit"]["finite"] = False
        if failure == "legacy": report["cases"][0]["motor_torque_utilization_p99"] = .6125551462173462
        (output / "rollout").mkdir()
        (output / "rollout/hierarchical-teacher-evaluation.json").write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0)
    monkeypatch.setattr(smoke.subprocess, "run", child)
    if failure == "existing":
        output.mkdir()
        (output / "user.txt").write_text("preserve")
    if failure in ("existing", "dirty", "branch", "source", "expired", "version", "dependency", "busy"):
        with pytest.raises((ValueError, FileExistsError)): smoke.main()
        assert not calls
        assert not output.exists() or list(output.iterdir()) == [output / "user.txt"]
        return
    with pytest.raises(SystemExit) as ended: smoke.main()
    assert len(calls) == 1 and ended.value.code == (0 if failure is None else 2)
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["wall_seconds"] >= 0 and runtime["child_timeout_seconds"] == 240
    assert (output / "launch.json").exists() and (output / "rollout.log").exists()
    decision = json.loads((output / "decision.json").read_text())
    assert decision["further_gpu_job_admitted"] is False and decision["physical_motion_authorized"] is False


def test_predeclaration_binds_launch_cutoff_case_and_retention():
    doc = (Path(__file__).resolve().parents[1] / "docs/experiments/2026-09-06-motor-audit-smoke.md").read_text()
    assert smoke.LAST_START == dt.datetime(2026, 9, 6, 9, 35, tzinfo=dt.timezone.utc)
    for value in (smoke.PROTOCOL, smoke.ACTOR_SHA256, smoke.SUPERVISOR_SHA256, "17:35", "240", "600", "373"):
        assert value in doc
