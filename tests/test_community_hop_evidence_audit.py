"""Synthetic evidence fixtures for the offline C1-S campaign auditor."""

import copy
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

import pytest

from mjlab_microduck import community_hop_baseline as baseline
from mjlab_microduck import community_hop_diagnostic as runner
from mjlab_microduck import community_hop_evidence_audit as audit
from mjlab_microduck import community_hop_rehearsal as c1


class Plant:
    def __init__(self, speed=0.2):
        self.index = -1
        self.speed = speed

    def prepare(self):
        self.index += 1

    def unsafe_for_inference(self):
        return False

    def observation_state(self):
        return dict(angular_velocity=[0.0] * 3, projected_gravity=[0.0, 0.0, -1.0],
                    servo_position=[0.0] * 14, backlash_position=[0.0] * 14,
                    servo_velocity=[self.index / 1000.0] * 14, backlash_velocity=[0.0] * 14,
                    home_position=[0.0] * 14)

    def advance_and_measure(self, applied, *, terminal):
        t = self.index * c1.PHYSICS_DT_S
        vx = self.speed if c1.phase_at(t) in ("walk", "resume_walk") else 0.0
        return dict(base_xy_m=[t * vx, 0.0], base_velocity_world_m_s=[vx, 0.0, 0.0],
                    base_roll_pitch_yaw_rad=[0.0] * 3, base_angular_velocity_rad_s=[0.0] * 3,
                    foot_clearance_m=[0.0, 0.0], foot_contact=[True, True],
                    foot_normal_force_n=[5.0, 5.0], motor_current_a=[0.2] * 14,
                    motor_torque_nm=[0.1] * 14, motor_velocity_rad_s=[0.0] * 14,
                    soft_limit_exposed=[False] * 14, body_contact=False, reset_event=False)


class Policy:
    def predict(self, observation):
        return [0.01] * 14


def _json(path, value):
    runner.save_json(path, value)


def _write_manifest(root):
    _json(root / "manifest.json", runner.artifact_manifest(root))


def make_campaign(root: Path, *, speed=0.2):
    declaration_path = Path(__file__).resolve().parents[1] / audit.DECLARATION_PATH
    declaration = declaration_path.read_bytes()
    declaration_obj = json.loads(declaration)
    receipt = {
        "experiment_id": "community-hop-c1s-velstand-substitution-v1",
        "measurement_protocol": c1.PROTOCOL,
        "measurement_scorer_sha256": declaration_obj["measurement"]["scorer_sha256"],
        "predeclaration_sha256": audit.DECLARATION_SHA256,
        "runner_source_commit": "a" * 40,
        "runner_source_sha256": {"fixture.py": "b" * 64},
        "policies": {name: {"artifact": {"bytes": record["bytes"], "sha256": record["sha256"]}}
                    for name, record in declaration_obj["policies"].items()},
        "asset_closure": {"asset_tree_sha1": declaration_obj["plant"]["asset_tree_sha1"]},
        "plant": {"compiled_mjb_sha256": hashlib.sha256(b"fixture compiled plant").hexdigest()},
        "bam_parameters": {"actuator": "fixture"},
        "bam_settings": {key: declaration_obj["plant"][key] for key in (
            "vin_v", "kp_fw", "max_current_a", "voltage_drop_gain",
            "firmware_position_view", "back_emf_and_friction_velocity_view")},
        **declaration_obj["not_claimed"],
    }
    (root / "preflight").mkdir(parents=True)
    _json(root / "preflight" / "preflight.json", receipt)
    (root / "preflight" / "predeclaration.json").write_bytes(declaration)
    (root / "preflight" / "compiled-plant.mjb").write_bytes(b"fixture compiled plant")
    _json(root / "preflight" / "child-exit.json", {"returncode": 0, "timed_out": False})
    for log in ("worker.stdout.log", "worker.stderr.log"):
        (root / "preflight" / log).write_bytes(b"")

    case_id = "walk-only-control"
    case = root / case_id
    case.mkdir()
    _json(case / "preflight.json", {**receipt, "case_id": case_id})
    (case / "predeclaration.json").write_bytes(declaration)
    (case / "compiled-plant.mjb").write_bytes(b"fixture compiled plant")
    child_record = {"returncode": 0, "timed_out": False}
    _json(case / "child-exit.json", child_record)
    for log in ("worker.stdout.log", "worker.stderr.log"):
        (case / log).write_bytes(b"")
    events = []
    runtime = runner.run_attempt(Plant(speed), {"walk": Policy()}, case_id,
                                 lambda kind, value: events.append({"kind": kind, "value": copy.deepcopy(value)}))
    with (case / "journal.jsonl").open("x") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    _json(case / "worker-result.json", {**runtime, "provider": "CPUExecutionProvider",
                                          "intra_threads": 1, "inter_threads": 1,
                                          "first_runtime_error": None})
    report = runner.finish_case(case, case_id, receipt, child_record)
    _json(root / "request.json", {"fixture_only": True})
    _json(root / "decision.json", {
        **runner.identity(receipt, "campaign"),
        "decision": runner.sequential_decision(report, None),
        "case_decisions": {case_id: report["decision"]},
        **declaration_obj["not_claimed"],
    })
    _write_manifest(root)
    return case


def _refresh_manifest(root):
    (root / "manifest.json").unlink()
    _write_manifest(root)


def _events(case):
    path = case / "journal.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _store_events(case, events):
    (case / "journal.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events))


def test_complete_fixture_audits_consistently_but_claims_no_authenticity(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    make_campaign(root)
    result = audit.audit_campaign(root)
    assert result["audit_passed"] is True
    assert result["status"] == "evidence-consistent"
    assert result["cases"]["walk-only-control"]["replay"]["control_count"] == 600
    assert result["cases"]["walk-only-control"]["phase_descriptives"]["walk"]["forward_velocity_m_s"]["mean"] == pytest.approx(0.2)
    assert all(result[name] is False for name in audit.FALSE_FLAGS)


def test_failed_speed_is_consistent_evidence_and_stays_rejected(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    make_campaign(root, speed=0.0)
    result = audit.audit_campaign(root)
    assert result["audit_passed"] is True
    assert result["decision"] == "walk-only-rejected-transfer-not-run"
    assert result["cases"]["walk-only-control"]["scored_decision"] == baseline.REJECTED_DECISION


@pytest.mark.parametrize("mutation", ["missing", "hash", "absolute", "parent", "extra"])
def test_manifest_integrity_fails_closed(tmp_path, mutation):
    root = tmp_path / "campaign"
    root.mkdir()
    make_campaign(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "missing":
        (root / "request.json").unlink()
    elif mutation == "hash":
        (root / "request.json").write_text("{}"); _refresh_manifest(root)
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["request.json"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "absolute":
        manifest["artifacts"]["/tmp/escape"] = {"bytes": 0, "sha256": "0" * 64}
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "parent":
        manifest["artifacts"]["../escape"] = {"bytes": 0, "sha256": "0" * 64}
        manifest_path.write_text(json.dumps(manifest))
    else:
        (root / "unlisted.txt").write_text("extra")
    result = audit.audit_campaign(root)
    assert result["audit_passed"] is False
    assert result["errors"]


def test_symlink_file_or_directory_is_rejected(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    make_campaign(root)
    target = tmp_path / "outside"
    target.write_text("outside")
    (root / "link").symlink_to(target)
    assert not audit.audit_campaign(root)["audit_passed"]
    (root / "link").unlink()
    (root / "subdir").mkdir()
    (root / "subdir" / "file").write_text("x")
    _refresh_manifest(root)
    (root / "linked-dir").symlink_to(root / "subdir", target_is_directory=True)
    assert not audit.audit_campaign(root)["audit_passed"]


def test_duplicate_manifest_json_keys_are_rejected(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    make_campaign(root)
    path = root / "manifest.json"
    payload = path.read_text()
    path.write_text(payload.replace('"schema":', '"schema":"other", "schema":', 1))
    assert not audit.audit_campaign(root)["audit_passed"]


@pytest.mark.parametrize("mutation", ["command", "history", "delay", "policy", "time", "order",
                                      "duplicate", "terminal", "controlobs", "sampleaction", "truncated"])
def test_journal_protocol_mutations_fail_closed(tmp_path, mutation):
    root = tmp_path / "campaign"
    root.mkdir()
    case = make_campaign(root)
    events = _events(case)
    if mutation == "command":
        events[0]["value"]["obs"][48] = 0.0
    elif mutation == "history":
        events[0]["value"]["obs"][34] = 0.4
    elif mutation == "delay":
        events[1]["value"]["action_applied"][0] = 0.5
    elif mutation == "policy":
        events[0]["value"]["policy"] = "hop"
    elif mutation == "time":
        events[0]["value"]["t_s"] = 0.005
    elif mutation == "order":
        events[0], events[1] = events[1], events[0]
    elif mutation == "duplicate":
        events.insert(0, copy.deepcopy(events[0]))
    elif mutation == "terminal":
        # Append a terminal inference-intent/control pair before terminal sample.
        terminal_sample_index = len(events) - 1
        last_control = next(e for e in reversed(events) if e["kind"] == "control")
        extra = copy.deepcopy(last_control)
        extra["value"]["t_s"] = c1.TOTAL_TIME_S
        events[terminal_sample_index:terminal_sample_index] = [
            {"kind": "inference-intent", "value": copy.deepcopy(extra["value"])}, extra]
    elif mutation == "controlobs":
        events[1]["value"]["obs"][0] = 1.0
    elif mutation == "sampleaction":
        events[2]["value"]["action_raw"][0] = 0.5
    else:
        events.pop()
        trace_path = case / "trace.json"
        trace = json.loads(trace_path.read_text())
        trace["samples"].pop()
        trace_path.write_text(json.dumps(trace))
    _store_events(case, events)
    _refresh_manifest(root)
    result = audit.audit_campaign(root)
    assert result["audit_passed"] is False


def test_setup_case_identity_drift_fails_closed(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    case = make_campaign(root)
    path = case / "preflight.json"
    receipt = json.loads(path.read_text())
    receipt["runner_source_commit"] = "f" * 40
    path.write_text(json.dumps(receipt))
    _refresh_manifest(root)
    assert not audit.audit_campaign(root)["audit_passed"]


@pytest.mark.parametrize("mutation", ["count", "child", "validation"])
def test_runtime_result_and_report_chain_must_agree(tmp_path, mutation):
    root = tmp_path / "campaign"
    root.mkdir()
    case = make_campaign(root)
    worker_path = case / "worker-result.json"
    report_path = case / "report.json"
    if mutation == "count":
        worker = json.loads(worker_path.read_text())
        worker["policy_inferences"] = 599
        worker_path.write_text(json.dumps(worker))
    elif mutation == "child":
        report = json.loads(report_path.read_text())
        report["child"]["timed_out"] = True
        report_path.write_text(json.dumps(report))
    else:
        report = json.loads(report_path.read_text())
        report["validation_error"] = {"reason": "synthetic"}
        report_path.write_text(json.dumps(report))
    _refresh_manifest(root)
    assert not audit.audit_campaign(root)["audit_passed"]


def test_score_or_sequential_decision_drift_rejected(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    make_campaign(root)
    report_path = root / "walk-only-control" / "report.json"
    report = json.loads(report_path.read_text())
    report["metrics"]["approach"]["forward_mae_m_s"] = 0.123
    report_path.write_text(json.dumps(report))
    _refresh_manifest(root)
    assert not audit.audit_campaign(root)["audit_passed"]

    root2 = tmp_path / "campaign2"
    root2.mkdir()
    make_campaign(root2)
    decision_path = root2 / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["decision"] = "substituted-entry-transfer-rejected"
    decision_path.write_text(json.dumps(decision))
    _refresh_manifest(root2)
    assert not audit.audit_campaign(root2)["audit_passed"]


def test_output_is_exclusive_and_cannot_be_inside_campaign(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    make_campaign(root)
    target = tmp_path / "audit.json"
    assert audit.main([str(root), "--output", str(target)]) == 0
    with pytest.raises(FileExistsError):
        audit.main([str(root), "--output", str(target)])
    with pytest.raises(SystemExit):
        audit.main([str(root), "--output", str(root / "audit.json")])


def test_isolated_module_import_does_not_load_heavy_runtime_stack():
    code = ("import sys; import mjlab_microduck.community_hop_evidence_audit; "
            "assert not (set(sys.modules) & {'numpy','torch','onnxruntime','mujoco','warp','bam'}); "
            "print('lightweight')")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parents[1], check=True)
    assert completed.stdout.strip() == "lightweight"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes unavailable")
@pytest.mark.parametrize("read_hash", [False, True])
def test_special_file_reader_cannot_block_before_regular_file_check(tmp_path, read_hash):
    fifo = tmp_path / "fixture.fifo"
    os.mkfifo(fifo)
    def timeout(*_):
        raise TimeoutError("FIFO audit open blocked")
    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(2)
    try:
        with pytest.raises(audit.AuditError, match="regular"):
            if read_hash:
                audit._hash_regular(fifo, "fixture", max_bytes=1024)
            else:
                audit._regular_bytes(fifo, "fixture")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.mark.parametrize("field", ["intra_threads", "returncode"])
def test_boolean_runtime_integer_receipts_fail_closed(tmp_path, field):
    root = tmp_path / "campaign"
    root.mkdir()
    case = make_campaign(root)
    path = case / ("worker-result.json" if field == "intra_threads" else "child-exit.json")
    value = json.loads(path.read_text())
    value[field] = field == "intra_threads"
    path.write_text(json.dumps(value))
    report_path = case / "report.json"
    report = json.loads(report_path.read_text())
    report["runtime" if field == "intra_threads" else "child"] = value
    report_path.write_text(json.dumps(report))
    _refresh_manifest(root)
    assert audit.audit_campaign(root)["audit_passed"] is False
