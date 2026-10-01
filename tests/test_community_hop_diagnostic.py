"""Synthetic orchestration/ONNX fixtures only, not public-policy attempts."""

import copy
import json
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from mjlab_microduck import community_hop_diagnostic as d
from mjlab_microduck import community_hop_rehearsal as c1
from mjlab_microduck import community_policy_inspection as inspect


class FixturePlant:
    def __init__(self, *, unstable_entry=False, fatal_at=None):
        self.index = -1
        self.steps = 0
        self.unstable_entry = unstable_entry
        self.fatal_at = fatal_at

    def prepare(self):
        self.index += 1

    def unsafe_for_inference(self):
        return self.index == self.fatal_at

    def observation_state(self):
        return dict(angular_velocity=[0.] * 3, projected_gravity=[0., 0., -1.],
            servo_position=[0.1] * 14, backlash_position=[0.01] * 14,
            servo_velocity=[self.index / 10.] * 14, backlash_velocity=[0.5] * 14,
            home_position=[0.1] * 14)

    def advance_and_measure(self, applied, *, terminal):
        self.steps += not terminal
        t = self.index * c1.PHYSICS_DT_S
        return dict(base_xy_m=[0., 0.],
            base_velocity_world_m_s=[0.2 if c1.phase_at(t) in ("walk", "resume_walk") else
                                     (0.06 if self.unstable_entry and 3.5 <= t < 4 else 0.), 0., 0.],
            base_roll_pitch_yaw_rad=[0.] * 3, base_angular_velocity_rad_s=[0.] * 3,
            foot_clearance_m=[0.] * 2, foot_contact=[True] * 2, foot_normal_force_n=[5.] * 2,
            motor_current_a=[0.3] * 14, motor_torque_nm=[0.1] * 14,
            motor_velocity_rad_s=[0.2] * 14, soft_limit_exposed=[False] * 14,
            body_contact=self.index == self.fatal_at, reset_event=False)


class FixturePolicy:
    def __init__(self, value):
        self.value = value
        self.observations = []

    def predict(self, obs):
        self.observations.append(copy.deepcopy(obs))
        return [self.value] * 14


def collect(plant, case):
    walk, hop = FixturePolicy(2.), FixturePolicy(-3.)
    events = []
    result = d.run_attempt(plant, {"walk": walk, "hop": hop}, case,
                           lambda kind, value: events.append((kind, copy.deepcopy(value))))
    samples = [value for kind, value in events if kind == "sample"]
    controls = {value["t_s"]: value for kind, value in events if kind == "control"}
    return result, {"schema": c1.TRACE_SCHEMA, "samples": samples}, controls, walk, hop


def test_fixed_timing_raw_action_and_velocity_histories_carry_across_switches():
    plant = FixturePlant()
    result, trace, ticks, walk, hop = collect(plant, "walk-hop-transition")
    assert result == dict(first_runtime_failure=None, policy_inferences=600, physics_steps=2400)
    assert len(trace["samples"]) == 2401 and plant.steps == 2400
    assert len(walk.observations) == 450 and len(hop.observations) == 150
    assert ticks[0.]["obs"][20:48] == [0.] * 28
    assert ticks[4.]["policy"] == "hop"
    assert ticks[4.]["obs"][34:48] == [2.] * 14
    assert ticks[4.]["obs"][20:34] == pytest.approx([80.1] * 14)  # index796/10 +.5
    assert ticks[4.]["action_applied"] == [2.] * 14
    assert ticks[7.]["policy"] == "walk"
    assert ticks[7.]["obs"][34:48] == [-3.] * 14
    assert ticks[7.]["action_applied"] == [-3.] * 14
    assert ticks[1404 * c1.PHYSICS_DT_S]["action_applied"] == [2.] * 14
    assert 12. not in ticks  # Inclusive terminal state never infers a new action.
    scored = c1.score_trace(trace)
    assert scored["gates"]["complete_first_attempt"] and scored["gates"]["no_fatal_event"]


def test_no_hop_inference_at_unstable_entry_and_no_reset_retry():
    plant = FixturePlant(unstable_entry=True)
    result, trace, _, walk, hop = collect(plant, "walk-hop-transition")
    assert not hop.observations
    assert len(walk.observations) == 200
    assert plant.steps == 800 and len(trace["samples"]) == 801
    assert result["first_runtime_failure"]["reason"] == "unstable-hop-entry"
    assert c1.score_trace(trace)["first_failure"] == result["first_runtime_failure"]


def test_first_body_fatal_stops_without_policy_inference_or_extra_physics():
    plant = FixturePlant(fatal_at=100)
    result, trace, _, walk, hop = collect(plant, "walk-only-control")
    assert plant.steps == 100 and len(walk.observations) == 25 and not hop.observations
    assert len(trace["samples"]) == 101
    assert result["first_runtime_failure"] == c1.score_trace(trace)["first_failure"]


def test_invalid_policy_action_cannot_modify_history_or_retry():
    plant = FixturePlant()
    policy = FixturePolicy(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        d.run_attempt(plant, {"walk": policy}, "walk-only-control", lambda *args: None)
    assert plant.steps == 0 and len(policy.observations) == 1


def test_synthetic_onnx_cpu_session_reuses_validated_bytes_without_file_io():
    graph = helper.make_graph([helper.make_node("Constant", [], ["actions"], value=helper.make_tensor(
        "constant", TensorProto.FLOAT, [1, 14], [0.125] * 14))], "fixture-not-public-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    payload = model.SerializeToString()
    import hashlib
    inspected = inspect.inspect_policy_payload(payload, hashlib.sha256(payload).hexdigest())
    assert inspected["interface_classification"] == "feedforward-microduck-api1"
    policy = d.CpuPolicy(payload)
    assert policy.predict([0.] * 61) == [0.125] * 14
    assert policy.session.get_providers() == ["CPUExecutionProvider"]
    assert policy.session.get_session_options().intra_op_num_threads == 1
    assert policy.session.get_session_options().inter_op_num_threads == 1


def test_hard_timeout_kills_only_owned_child_and_retains_fsynced_prefix(tmp_path):
    journal = tmp_path / "journal.jsonl"
    command = [sys.executable, "-c",
        "import os,time; f=open('" + str(journal) + "','x'); f.write('prefix\\n'); f.flush(); os.fsync(f.fileno()); time.sleep(30)"]
    start = time.monotonic()
    child = d.bound_child(command, tmp_path, 0.25)
    assert child["timed_out"] and child["returncode"] < 0
    assert time.monotonic() - start < 5
    assert journal.read_text() == "prefix\n"


def test_normal_exit_and_no_remaining_budget_do_not_retry(tmp_path):
    result = d.bound_child([sys.executable, "-c", "print('fixture')"], tmp_path, 5)
    assert result["returncode"] == 0 and not result["timed_out"]
    assert (tmp_path / "worker.stdout.log").read_text() == "fixture\n"
    assert not d.bound_child(["must-not-launch"], tmp_path, 0)["launched"]


def test_tagged_nonfinite_and_killed_journal_prefix_are_recovered(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = d.Journal(path)
    journal.write("sample", {"t_s": 0., "value": float("nan")})
    journal.close()
    with path.open("ab") as stream:
        stream.write(b'{"kind":"sample"')
    trace, recovered = d.read_journal(path)
    assert np.isnan(trace["samples"][0]["value"])
    assert recovered["partial_lines_dropped"] == 1
    json.loads(path.read_text().splitlines()[0])  # Standard JSON, no bare NaN.


def test_evidence_is_exclusive_and_hash_manifest_is_exact(tmp_path):
    d.save_json(tmp_path / "evidence.json", {"fixture": True})
    with pytest.raises(FileExistsError):
        d.save_json(tmp_path / "evidence.json", {})
    manifest = d.artifact_manifest(tmp_path)
    assert set(manifest["artifacts"]) == {"evidence.json"}
    assert manifest["artifacts"]["evidence.json"]["bytes"] == (tmp_path / "evidence.json").stat().st_size


@pytest.mark.parametrize("control,transfer,expected", [
    (None, None, "preflight-only-no-policy-execution"),
    ({"decision": "rejected", "gates": {"complete": False}}, None, "walk-only-rejected-transfer-not-run"),
    ({"decision": "walk-only-indicators-pass", "gates": dict.fromkeys(d.BASELINE_GATES, True)}, None, "walk-only-passed-transfer-not-run"),
    ({"decision": "walk-only-indicators-pass", "gates": dict.fromkeys(d.BASELINE_GATES, True)},
     {"decision": "declared-trace-rejected", "gates": {"flight": False}}, "substituted-entry-transfer-rejected"),
    ({"decision": "walk-only-indicators-pass", "gates": dict.fromkeys(d.BASELINE_GATES, True)},
     {"decision": "declared-trace-indicators-pass", "gates": dict.fromkeys(d.TRANSFER_GATES, True)}, "nominal-substituted-entry-diagnostic-candidate-only"),
])
def test_sequential_decision_never_promotes_or_runs_hop_without_control(control, transfer, expected):
    assert d.sequential_decision(control, transfer) == expected


@pytest.mark.parametrize("gates", [{}, {"complete_first_attempt": True}, dict.fromkeys(d.BASELINE_GATES, 1)])
def test_empty_missing_or_nonboolean_gates_cannot_admit_transfer(gates):
    assert d.sequential_decision({"decision": "walk-only-indicators-pass", "gates": gates}, None) == \
        "walk-only-rejected-transfer-not-run"


def test_worker_without_separate_go_ahead_fails_before_native_or_policy_loading(tmp_path):
    with pytest.raises(ValueError, match="go-ahead"):
        d.worker(tmp_path, tmp_path, "walk-only-control", execute_approved=False)


def fixture_receipt():
    return {"experiment_id": "community-hop-c1s-velstand-substitution-v1",
            "measurement_protocol": c1.PROTOCOL, "measurement_scorer_sha256": "s" * 64,
            "predeclaration_sha256": "d" * 64, "runner_source_commit": "a" * 40,
            "runner_source_sha256": {"fixture": "f" * 64},
            **{key: {} for key in d.BINDING_KEYS if key not in (
                "runner_source_commit", "runner_source_sha256", "predeclaration_sha256")}}


def test_malformed_complete_sample_preserves_rejected_report_and_manifest(tmp_path):
    _, trace, _, _, _ = collect(FixturePlant(), "walk-only-control")
    receipt = fixture_receipt()
    d.save_json(tmp_path / "preflight.json", receipt)
    journal = d.Journal(tmp_path / "journal.jsonl")
    for row in trace["samples"][:8]:
        journal.write("sample", row)
    journal.write("sample", {"malformed": True})
    journal.close()
    result = d.finish_case(tmp_path, "walk-only-control", receipt,
                           {"returncode": 0, "timed_out": False})
    assert result["decision"] == "runtime-incomplete-or-failed-rejected"
    assert result["first_failure"]["sample"] == 8
    assert result["validation_error"]["reason"] == "invalid-trace-sample"
    assert result["scored_samples"] == 8
    assert (tmp_path / "trace.json").exists() and (tmp_path / "report.json").exists()
    d.save_json(tmp_path / "manifest.json", d.artifact_manifest(tmp_path))
    assert "report.json" in d.read_record(tmp_path / "manifest.json")["artifacts"]


def test_attempt_identity_drift_is_rejected_and_actual_identity_is_reported(tmp_path):
    _, trace, _, _, _ = collect(FixturePlant(), "walk-only-control")
    setup, attempt = fixture_receipt(), fixture_receipt()
    attempt["runner_source_commit"] = "b" * 40
    d.save_json(tmp_path / "preflight.json", attempt)
    journal = d.Journal(tmp_path / "journal.jsonl")
    for row in trace["samples"]:
        journal.write("sample", row)
    journal.close()
    d.save_json(tmp_path / "worker-result.json", {
        "first_runtime_failure": None, "policy_inferences": 600, "physics_steps": 2400})
    result = d.finish_case(tmp_path, "walk-only-control", setup, {"returncode": 0, "timed_out": False})
    assert result["binding_error"] == "attempt binding differs from campaign setup"
    assert result["runner_source_commit"] == "b" * 40
    assert result["setup_runner_source_commit"] == "a" * 40
    assert not result["gates"]["runtime_completed_without_failure"]


@pytest.mark.parametrize("record", [[], None, {"policy_inferences": float("nan")}])
def test_invalid_worker_record_cannot_be_accepted_or_crash_finisher(tmp_path, record):
    receipt = fixture_receipt()
    d.save_json(tmp_path / "preflight.json", receipt)
    # Nonfinite is encoded by evidence writer; even a numeric tag is not a
    # matching count, so it cannot satisfy the runtime-completion gate.
    d.save_json(tmp_path / "worker-result.json", record)
    result = d.finish_case(tmp_path, "walk-only-control", receipt, {"returncode": 0, "timed_out": False})
    assert result["decision"] == "runtime-incomplete-or-failed-rejected"
    assert (tmp_path / "report.json").exists()


@pytest.mark.parametrize("mode,expected,stages", [
    ("default", "preflight-only-no-policy-execution", ["preflight"]),
    ("rejected-control", "walk-only-rejected-transfer-not-run", ["preflight", "walk-only-control"]),
    ("malformed-control", "walk-only-rejected-transfer-not-run", ["preflight", "walk-only-control"]),
    ("both-pass", "nominal-substituted-entry-diagnostic-candidate-only",
     ["preflight", "walk-only-control", "walk-hop-transition"]),
])
def test_fixture_campaign_default_guard_sequence_and_final_evidence(tmp_path, monkeypatch, mode, expected, stages):
    from mjlab_microduck import community_hop_surrogate as surrogate
    from pathlib import Path

    real_root = Path(__file__).resolve().parents[1]
    plan, declaration = surrogate.load_declaration(real_root)
    receipt = fixture_receipt()
    monkeypatch.setattr(surrogate, "load_declaration", lambda root: (plan, declaration))
    monkeypatch.setattr(surrogate, "source_binding", lambda *a, **kw: {
        "runner_source_commit": receipt["runner_source_commit"],
        "runner_source_sha256": receipt["runner_source_sha256"], "repository_clean": True})
    launched = []
    def fixture_child(command, directory, timeout):
        case = command[command.index("--_worker") + 1]
        launched.append(case)
        d.save_json(directory / "preflight.json", receipt)
        if case != "preflight":
            _, trace, _, _, _ = collect(FixturePlant(), case)
            if mode == "rejected-control":
                trace["samples"][0]["body_contact"] = True
            if mode == "malformed-control":
                trace["samples"][0] = []
            if mode == "both-pass" and case == "walk-hop-transition":
                for row in trace["samples"]:
                    if 4.5 <= row["t_s"] < 4.6:
                        row.update(foot_contact=[False, False], foot_clearance_m=[0.035, 0.035],
                                   foot_normal_force_n=[0., 0.])
            journal = d.Journal(directory / "journal.jsonl")
            for row in trace["samples"]:
                journal.write("sample", row)
            journal.close()
            d.save_json(directory / "worker-result.json", {
                "first_runtime_failure": None, "policy_inferences": 600, "physics_steps": 2400})
        return {"returncode": 0, "timed_out": False, "wall_s": 0., "launched": True}
    monkeypatch.setattr(d, "bound_child", fixture_child)
    output = d.campaign(tmp_path, execute_approved=mode != "default")
    decision = d.read_record(output / "decision.json")
    assert launched == stages
    assert decision["decision"] == expected
    assert all(decision[key] is False for key in plan["not_claimed"])
    manifest = d.read_record(output / "manifest.json")
    assert manifest == d.artifact_manifest(output)
    assert "decision.json" in manifest["artifacts"]
