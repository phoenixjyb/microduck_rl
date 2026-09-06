"""Synthetic protocol and mocked orchestration checks; no GPU jobs."""

import copy
from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import recovery_ab as ab
from mjlab_microduck.motor_measurement_audit import MotorMeasurementAudit
from mjlab_microduck.recovery_measurement import RecoveryMeasurement


def make_report(index=0):
    cell, mode = divmod(index, 3)
    speed, forward = ab.CELLS[cell]
    stage, supervisor = (ab.NEAR_STAGE, ab.NEAR_SHA256) if forward == .9 else (ab.FAR_STAGE, ab.FAR_SHA256)
    audit = MotorMeasurementAudit(8, 700, ab.JOINTS, tuple(range(14)), stall_reference_nm=.6)
    observer = RecoveryMeasurement(8, speed, .02)
    for step in range(28):
        phase = torch.full((8,), min(2, step))
        done = torch.full((8,), step == 27)
        force = torch.full((8, 14), .27 if mode == 2 else .3)
        audit.begin(step, torch.ones(8, dtype=torch.bool), phase)
        audit.capture(NS(actuator_force=force, joint_vel=force * 2), done)
        audit.finish(done, force)
        observer.begin(step, phase, torch.full((8,), speed))
        observer.finish(done)
    counts = dict.fromkeys(ab.COUNTS, 0)
    counts.update(expected_attempts=8, completed_attempts=8, resolved_attempts=8, clean_pass_events=8)
    case = dict(nominal_speed_mps=speed, obstacle_forward_m=forward, obstacle_lateral_m=0.,
                seed=379, num_envs=8, steps=700, steps_executed=28, **counts,
                terminal_outcome_protocol=ab.OUTCOME, evaluation_window="first-terminal-attempt-per-environment-v1",
                motor_torque_utilization_p99=.5, motor_speed_rated_exceed_fraction=0.,
                motor_measurement_audit=audit.report(), recovery_speed_measurement=observer.report())
    for p in ab.PHASES:
        case[f"{p}_samples"] = case["motor_measurement_audit"]["groups"][p]["environment_steps"]
        case[f"{p}_route_speed_mps"] = speed
    report = dict(stage=stage, decision="diagnostic-only", physical_motion_authorized=False,
                  checkpoint_sha256=ab.ACTOR_SHA256, supervisor_checkpoint_sha256=supervisor,
                  teacher_config=asdict(ab.ObstacleTeacherCfg()),
                  perception="exact structured geometry; no raw camera perception",
                  obstacle_sensor_model=dict(range_noise_m=0., bearing_noise_rad=0., width_noise_m=0.,
                    height_noise_m=0., closing_rate_noise_mps=0., dropout_probability=0.),
                  evaluation_window=case["evaluation_window"], terminal_outcome_protocol=ab.OUTCOME,
                  attempt_timeout_s=12., cases=[case], totals=counts, motor_measurement_audit_protocol=ab.AUDIT)
    if mode == 2:
        report.update(stage=ab.ROLLOUT_STAGE, source_controller_stage=stage, policy_acceptance=False,
                      recovery_control=ab.RecoveryAccelerationCfg(.2).provenance())
        case["recovery_control"] = {**report["recovery_control"], "update_dt_s": .1}
    return report


def retained(tmp_path, count=12):
    paths = []
    for i in range(count):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(make_report(i)))
        paths.append(path)
    return paths


def mutate(path, fn):
    report = json.loads(path.read_text())
    fn(report)
    path.write_text(json.dumps(report))


def test_complete_supported_but_no_policy_or_historical_admission(tmp_path):
    paths = retained(tmp_path)
    before = [p.read_bytes() for p in paths]
    result = ab.evaluate_paths(paths)
    assert result["decision"] == "recovery-cap-diagnostic-supports-pilot", result
    assert result == ab.evaluate_paths(paths) and before == [p.read_bytes() for p in paths]
    assert result["supports_predeclared_pilot"]
    assert not any(result[k] for k in ("policy_acceptance", "physical_motion_authorized",
                                      "training_data_admitted", "historical_repeatability_validated"))
    assert [r["sha256"] for r in result["reports"]] == [ab.sha256(p) for p in paths]


@pytest.mark.parametrize("count", [1, 2, 3, 6, 11])
def test_valid_prefix_never_supports_pilot(tmp_path, count):
    decision = ab.evaluate_paths(retained(tmp_path, count))
    assert decision["decision"] == "valid-prefix-not-complete" and not decision["supports_predeclared_pilot"]


@pytest.mark.parametrize("change", [
    lambda r: r.update(checkpoint_sha256="0" * 64),
    lambda r: r.update(supervisor_checkpoint_sha256="0" * 64),
    lambda r: r.update(physical_motion_authorized=True),
    lambda r: r.update(first_attempt_recording_protocol="bad"),
    lambda r: r.update(dataset_path="bad"),
    lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
    lambda r: r["cases"][0].update(seed=373),
    lambda r: r["cases"][0].update(num_envs=8.),
    lambda r: r["totals"].update(clean_pass_events=True),
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=float("nan")),
    lambda r: r["cases"][0].update(motor_torque_utilization_p99=True),
    lambda r: r["cases"][0].update(recovery_samples=100),
    lambda r: r["cases"][0]["motor_measurement_audit"].update(finite=False),
    lambda r: r["cases"][0]["motor_measurement_audit"].update(sampling="post-return"),
    lambda r: r["cases"][0]["motor_measurement_audit"]["groups"]["recovery"]["by_joint"].pop("head_roll"),
    lambda r: r["cases"][0]["recovery_speed_measurement"]["environments"][1].update(environment=0),
    lambda r: r["cases"][0]["recovery_speed_measurement"]["environments"][0].update(stable_recovery_latency_s=.48),
    lambda r: r["cases"][0]["recovery_speed_measurement"]["counts"].update(**{"recovered-in-window": 7}),
    lambda r: r["cases"][0]["recovery_speed_measurement"].update(deadline_s=3.),
])
def test_structural_counterexamples_fail_closed(tmp_path, change):
    paths = retained(tmp_path, 1)
    mutate(paths[0], change)
    assert ab.evaluate_paths(paths)["decision"] == "invalid-evidence-stop"


@pytest.mark.parametrize("change", [
    lambda r: r.update(source_controller_stage=ab.FAR_STAGE),
    lambda r: r["recovery_control"]["config"].update(max_acceleration_mps2=.4),
    lambda r: r["cases"][0]["recovery_control"].update(update_dt_s=.02),
])
def test_changed_candidate_runtime_cannot_pass(tmp_path, change):
    paths = retained(tmp_path, 3)
    mutate(paths[2], change)
    assert ab.evaluate_paths(paths)["decision"] == "invalid-evidence-stop"


def test_censored_baseline_stops_not_omitted(tmp_path):
    paths = retained(tmp_path, 1)
    def change(r):
        m = r["cases"][0]["recovery_speed_measurement"]
        m["environments"][0].update(stable_recovery_latency_s=None, status="censored-before-window")
        m["counts"].update(**{"recovered-in-window": 7, "censored-before-window": 1})
    mutate(paths[0], change)
    result = ab.evaluate_paths(paths)
    assert result["decision"] == "numerical-gate-stop" and result["failures"] == ["recovery-window"]


@pytest.mark.parametrize("count,index,key,value,failure", [
    (1, 0, "motor_torque_utilization_p99", .6126, "legacy-torque-p99"),
    (2, 1, "motor_torque_utilization_p99", .53, "repeat-torque"),
    (2, 1, "approach_route_speed_mps", .27, "repeat-approach-speed"),
    (3, 2, "recovery_route_speed_mps", .26, "b-v-a1-recovery-speed"),
])
def test_local_numerical_failure_stops_immediately(tmp_path, count, index, key, value, failure):
    paths = retained(tmp_path, count)
    mutate(paths[index], lambda r: r["cases"][0].update({key: value}))
    result = ab.evaluate_paths(paths)
    assert result["decision"] == "numerical-gate-stop" and failure in result["failures"]


def test_duplicate_missing_swapped_and_post_failure_evidence_refused(tmp_path):
    paths = retained(tmp_path, 3)
    assert ab.evaluate_paths([])["decision"] == "invalid-evidence-stop"
    assert ab.evaluate_paths([paths[0], paths[0]])["decision"] == "invalid-evidence-stop"
    assert ab.evaluate_paths([tmp_path / "absent"])["decision"] == "invalid-evidence-stop"
    assert ab.evaluate_paths([paths[2]])["decision"] == "invalid-evidence-stop"
    mutate(paths[0], lambda r: r["cases"][0].update(motor_torque_utilization_p99=.7))
    assert ab.evaluate_paths(paths)["decision"] == "invalid-evidence-stop"


def test_cellwise_speed_pass_does_not_bypass_pooled_gate(tmp_path):
    paths = retained(tmp_path)
    for p in paths[2::3]:
        mutate(p, lambda r: r["cases"][0].update(recovery_route_speed_mps=r["cases"][0]["nominal_speed_mps"] - .02))
    result = ab.evaluate_paths(paths)
    assert "pooled-v-a1-recovery-speed" in result["failures"]
    assert result["decision"] == "numerical-gate-stop"


def test_no_load_improvement_cannot_support_pilot(tmp_path):
    paths = retained(tmp_path)
    for i in range(2, 12, 3):
        group = json.loads(paths[i - 2].read_text())["cases"][0]["motor_measurement_audit"]["groups"]["recovery"]
        mutate(paths[i], lambda r: r["cases"][0]["motor_measurement_audit"]["groups"].update(recovery=group))
    result = ab.evaluate_paths(paths)
    assert result["decision"] == "numerical-gate-stop"
    assert result["failures"] == ["recovery-cell-mean-v-a1-five-percent", "recovery-cell-mean-v-a2-five-percent"]


def test_single_joint_regression_stops_even_if_pooled_load_improves(tmp_path):
    paths = retained(tmp_path, 3)
    def change(r):
        joint = r["cases"][0]["motor_measurement_audit"]["groups"]["recovery"]["by_joint"]["right_hip_pitch"]
        joint.update(abs_p99=.6, abs_max=.6)
    mutate(paths[2], change)
    assert "b-v-a1-recovery-load-1" in ab.evaluate_paths(paths)["failures"]


@pytest.mark.parametrize("failure", ["cwd", "branch", "head", "dirty", "short"])
def test_exact_source_preflight_is_not_only_mocked(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ab, "ROOT", tmp_path / "wrong" if failure == "cwd" else tmp_path)
    answers = {("git", "branch", "--show-current"): "main" if failure == "branch" else "feat/athletics-obstacle-curriculum",
               ("git", "rev-parse", "HEAD"): "b" * 40 if failure == "head" else "a" * 40,
               ("git", "status", "--porcelain"): " M user" if failure == "dirty" else ""}
    monkeypatch.setattr(ab, "read", lambda *args: answers[args])
    with pytest.raises(ValueError): ab.verify_source("a" * (7 if failure == "short" else 40))


def test_finite_inputs_with_overflow_cannot_clear_pooled_gate(tmp_path):
    paths = retained(tmp_path)
    for p in paths: mutate(p, lambda r: r["cases"][0].update(recovery_route_speed_mps=1e308))
    assert ab.evaluate_paths(paths)["decision"] == "invalid-evidence-stop"


def test_command_matrix_is_frozen_and_only_b_enables_cap(tmp_path):
    for i in range(12):
        cmd = ab.command_for(i, tmp_path / str(i))
        assert ("--recovery-acceleration-mps2" in cmd) == (i % 3 == 2)
        assert "--motor-measurement-audit" in cmd and "--first-attempt-only" in cmd
        assert "--record-first-attempts" not in cmd
        assert cmd[cmd.index("--seeds") + 1] == "379"
        assert cmd[-1] == "0.2" if i % 3 == 2 else cmd[-1] == str(ab.NEAR if i < 6 else ab.FAR)


@pytest.mark.parametrize("failure,children", [(None, 12), ("existing", 0), ("expired", 0),
    ("source", 0), ("hash", 0), ("version", 0), ("dependency", 0), ("busy", 0),
    ("timeout", 1), ("exit", 1), ("gate", 1), ("laterbusy", 1), ("laterdirty", 1)])
def test_runner_stops_and_retains_receipts_without_retry(tmp_path, monkeypatch, failure, children):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ab, "ROOT", tmp_path)
    monkeypatch.setattr(ab, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(ab, "DEADLINE", (dt.datetime.min if failure == "expired" else
                        dt.datetime.max).replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(sys, "argv", ["ab", "--source", "a" * 40])
    calls = []
    def source(value):
        if failure == "source" or (calls and failure == "laterdirty"): raise ValueError("dirty")
    monkeypatch.setattr(ab, "verify_source", source)
    original_hash = ab.sha256
    def hashed(path):
        for p, h in zip((ab.ACTOR, ab.NEAR, ab.FAR), (ab.ACTOR_SHA256, ab.NEAR_SHA256, ab.FAR_SHA256)):
            if path == p: return "bad" if failure == "hash" else h
        for name, h in ab.DEPENDENCIES.items():
            if str(path).endswith(name): return "bad" if failure == "dependency" else h
        return original_hash(path)
    monkeypatch.setattr(ab, "sha256", hashed)
    monkeypatch.setattr(ab.importlib.metadata, "version", lambda name: "bad" if failure == "version" else ab.VERSIONS[name])
    def host():
        if failure == "busy" or (calls and failure == "laterbusy"): raise ValueError("busy")
        return dict(utilization_percent=0, temperature_c=44, memory_mib=12)
    monkeypatch.setattr(ab, "check_host", host)
    monkeypatch.setattr(ab.time, "sleep", lambda _: None)
    def child(command, **kwargs):
        index = len(calls)
        calls.append(command)
        assert kwargs["timeout"] == 180 and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        if failure == "timeout": raise subprocess.TimeoutExpired(command, 180)
        dest = Path(command[command.index("--output-dir") + 1])
        dest.mkdir()
        report = make_report(index)
        if failure == "gate": report["cases"][0]["motor_torque_utilization_p99"] = .7
        (dest / "hierarchical-teacher-evaluation.json").write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0)
    monkeypatch.setattr(ab.subprocess, "run", child)
    if failure == "existing":
        ab.OUTPUT.mkdir()
        (ab.OUTPUT / "keep").write_text("user")
    if children == 0:
        with pytest.raises((ValueError, FileExistsError)): ab.main()
        assert not calls
        if failure == "existing": assert (ab.OUTPUT / "keep").read_text() == "user"
        else: assert not ab.OUTPUT.exists()
        return
    with pytest.raises(SystemExit) as ended: ab.main()
    assert len(calls) == children and ended.value.code == (0 if failure is None else 2)
    result = json.loads((ab.OUTPUT / "decision.json").read_text())
    assert result["supports_predeclared_pilot"] == (failure is None)
    assert not result["physical_motion_authorized"]
    assert len(json.loads((ab.OUTPUT / "runtime.json").read_text())["runs"]) == children
    assert (ab.OUTPUT / "runtime-00.json").exists()


def test_deadline_predeclaration():
    assert ab.DEADLINE == dt.datetime(2026, 9, 6, 23, tzinfo=dt.timezone.utc)
    assert 12 * ab.CHILD_SECONDS + 60 < ab.SERVICE_SECONDS
    doc = (Path(__file__).resolve().parents[1] / "docs/experiments/2026-09-06-overnight-recovery-curriculum.md").read_text()
    for literal in (ab.PROTOCOL, ab.ACTOR_SHA256, ab.NEAR_SHA256, ab.FAR_SHA256, "2400", "180", "07:00"):
        assert literal in doc


def test_retained_seed379_stop_replays_when_evidence_is_available():
    root = Path(__file__).resolve().parents[1]
    options = [root / "artifacts" / kind / ab.PROTOCOL for kind in ("diagnostics", "evaluations")]
    retained_root = next((p for p in options if (p / "decision.json").exists()), None)
    if retained_root is None:
        pytest.skip("retained simulation evidence not distributed in Git")
    report = retained_root / "00-cell0-a1/hierarchical-teacher-evaluation.json"
    decision_path = retained_root / "decision.json"
    assert ab.sha256(report) == "b6bf4a7baa4e16e47a1e672f382cca80e1e9b46dbe70b5bd9da3481062a73828"
    assert ab.sha256(decision_path) == "8b34b5186a45dda26aad43cb71b095367c765a4b3ed090c997179d416422a0a2"
    result = ab.evaluate_paths([report])
    # The public artifact is JSON; tuple/list representation is normalized here.
    assert ab.canonical(result) == ab.canonical(json.loads(decision_path.read_text()))
    assert result["decision"] == "numerical-gate-stop" and result["failures"] == ["recovery-window"]
    assert not result["supports_predeclared_pilot"]
    assert len(json.loads((retained_root / "runtime.json").read_text())["runs"]) == 1
