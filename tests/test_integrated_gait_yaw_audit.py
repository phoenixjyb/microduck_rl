import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from mjlab_microduck import integrated_gait_yaw_audit as audit
from mjlab_microduck import integrated_training_budget_refinement as recipe


AUDIT_SOURCE = "a" * 40
RUNTIME = {name: "test-version" for name in audit.RUNTIME_PACKAGES}
EVALUATION_SHA = "b" * 64
DEFAULT_INPUT = audit.INPUT


@pytest.fixture(autouse=True)
def reset_input(monkeypatch):
    monkeypatch.setattr(audit, "INPUT", DEFAULT_INPUT)


class Lease:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def case_row(seed, speed, yaw, digest):
    joints = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
              "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
              "right_ankle"]
    return dict(seed=seed, speed=speed, yaw=yaw, worlds=8, steps=240,
                complete=True, terminal_worlds=[], checkpoint_sha256=digest,
                speed_mae=[0.] * 8, yaw_mae=[0.] * 8, torque_p99=.2,
                rated_speed_exceed_fraction=0., response_diagnostics=dict(
                    protocol="signed-body-response-and-joint-load-v1", steps=240,
                    hard_stop_margin_rad=.05, joint_columns=joints,
                    min_distance_to_hard_stop_rad_by_joint=[.1] * 14,
                    hard_stop_proximity_fraction_by_joint=[0.] * 14))


def bundle(tmp_path, monkeypatch):
    cases, decisions, records, hashes = {}, {}, {}, {}
    for arm in audit.ARMS:
        contract = audit.motor.training_contract(recipe, arm)
        arm_dir = tmp_path / "input" / arm
        arm_dir.mkdir(parents=True)
        checkpoint = arm_dir / f"model_{contract['final_iteration']}.pt"
        checkpoint.write_bytes(f"checkpoint-{arm}".encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        hashes[arm] = digest
        rows = [case_row(seed, speed, yaw, digest)
                for seed in audit.base.SEEDS for speed, yaw in audit.base.CASES]
        cases[arm], decisions[arm] = rows, recipe.decision(rows)
        records[arm] = dict(
            launch=dict(source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL, mode=arm,
                        parent_sha256=recipe.PARENT_SHA, runtime=dict(RUNTIME)),
            result=dict(source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL,
                        status="training-complete-not-accepted", mode=arm,
                        updates=contract["updates"], worlds=contract["worlds"],
                        seed=contract["seed"], steps=contract["updates"] * 24,
                        transitions=contract["updates"] * 24 * contract["worlds"],
                        warm_start={"parent_sha256": recipe.PARENT_SHA},
                        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest),
            actual_checkpoint_sha256=digest)
    prior = dict(source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL,
                 selection=audit.EVALUATION_SELECTION,
                 final_iterations={arm: audit.motor.training_contract(recipe, arm)["final_iteration"]
                                   for arm in audit.ARMS},
                 checkpoint_sha256=hashes, cases=cases, decisions=decisions)
    launch = dict(source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL, mode="evaluate",
                  parent_sha256=recipe.PARENT_SHA, runtime=dict(RUNTIME))
    monkeypatch.setattr(audit, "INPUT", tmp_path / "input")
    return prior, records, launch


def test_invocation_rejects_malformed_paths_hash_and_source_runtime_guards():
    audit.validate_invocation(AUDIT_SOURCE, "gait-run-01", EVALUATION_SHA,
                              head=AUDIT_SOURCE, status="", cuda_visible_devices="0")
    for mutation in (
        dict(source="A" * 40), dict(run_id="../escape"),
        dict(evaluation_sha256="B" * 64), dict(head="c" * 40),
        dict(status=" M dirty.py"), dict(cuda_visible_devices="0,1"),
    ):
        values = dict(source=AUDIT_SOURCE, run_id="gait-run-01",
                      evaluation_sha256=EVALUATION_SHA, head=AUDIT_SOURCE,
                      status="", cuda_visible_devices="0")
        values.update(mutation)
        with pytest.raises(ValueError):
            audit.validate_invocation(**values)


def test_evaluation_hash_binds_exact_retained_aggregate_bytes():
    raw = b'{"protocol":"integrated-training-duration-continuation-v1"}\n'
    digest = hashlib.sha256(raw).hexdigest()
    assert audit.prior_audit.validate_evaluation_bytes(raw, digest) == digest
    with pytest.raises(ValueError, match="aggregate SHA"):
        audit.prior_audit.validate_evaluation_bytes(raw + b" ", digest)


def test_input_bundle_pins_final_duration_receipts_cases_and_seed839_jobs(tmp_path, monkeypatch):
    prior, records, launch = bundle(tmp_path, monkeypatch)
    validated = audit.validate_input_bundle(prior, EVALUATION_SHA, source=audit.INPUT_SOURCE,
                                            runtime=RUNTIME, arm_records=records,
                                            evaluation_launch=launch)
    assert validated["decisions"] == prior["decisions"]
    assert validated["final_iterations"] == {"control": 999, "budget": 3999}
    jobs = list(audit.case_jobs(validated))
    assert [(arm, index) for arm, index, _ in jobs] == [
        (arm, index) for arm in audit.ARMS for index in range(len(audit.base.CASES))]
    assert all(row["seed"] == 839 for _, _, row in jobs)


@pytest.mark.parametrize("mutation", [
    "source", "protocol", "selection", "iterations", "eval_runtime", "arm_runtime",
    "arm_mode", "parent", "receipt_seed", "receipt_transitions", "receipt_checkpoint",
    "aggregate_hash", "case_hash", "case_order", "decision",
])
def test_input_bundle_rejects_malformed_identity_and_reordered_cases(tmp_path, monkeypatch, mutation):
    prior, records, launch = bundle(tmp_path, monkeypatch)
    if mutation == "source": prior["source"] = "wrong"
    elif mutation == "protocol": prior["protocol"] = "wrong"
    elif mutation == "selection": prior["selection"] = "intermediate checkpoint"
    elif mutation == "iterations": prior["final_iterations"]["budget"] = 999
    elif mutation == "eval_runtime": launch["runtime"]["torch"] = "wrong"
    elif mutation == "case_order": prior["cases"]["control"][0], prior["cases"]["control"][1] = (
        prior["cases"]["control"][1], prior["cases"]["control"][0])
    elif mutation == "decision": prior["decisions"]["budget"] = {"decision": "changed"}
    else:
        record = records["budget"]
        if mutation == "arm_runtime": record["launch"]["runtime"]["torch"] = "wrong"
        elif mutation == "arm_mode": record["launch"]["mode"] = "control"
        elif mutation == "parent": record["launch"]["parent_sha256"] = "f" * 64
        elif mutation == "receipt_seed": record["result"]["seed"] += 1
        elif mutation == "receipt_transitions": record["result"]["transitions"] -= 1
        elif mutation == "receipt_checkpoint": record["result"]["checkpoint"] = "model_1000.pt"
        elif mutation == "aggregate_hash": prior["checkpoint_sha256"]["budget"] = "f" * 64
        else: prior["cases"]["budget"][0]["checkpoint_sha256"] = "f" * 64
    with pytest.raises((ValueError, KeyError)):
        audit.validate_input_bundle(prior, EVALUATION_SHA, source=audit.INPUT_SOURCE,
                                    runtime=RUNTIME, arm_records=records,
                                    evaluation_launch=launch)


def test_raw_observation_is_written_before_partial_terminal_summary_rejection(tmp_path):
    original = case_row(839, *audit.base.CASES[0], "c" * 64)
    short = dict(original, complete=False, steps=61, terminal_worlds=[0])
    trace = dict(steps=61, **{key: [] for key in
        ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w")})
    observed = {**short, "body_yaw_trace": dict(trace), "gait_yaw_trace": dict(trace)}
    with pytest.raises(ValueError, match="complete"):
        audit.retain_case(tmp_path, "control", 0, original, observed, "c" * 64)
    raw_path = tmp_path / "control-s839-c0.json"
    assert raw_path.is_file()
    raw = json.loads(raw_path.read_text())
    assert raw["observed_case"]["steps"] == 61
    assert raw["body_yaw_trace"]["steps"] == raw["gait_yaw_trace"]["steps"] == 61
    assert not (tmp_path / "control-s839-c0-summary.json").exists()


def test_entrypoint_sequence_uses_exact_final_cases_and_persists_raw_before_both_summaries(
        tmp_path, monkeypatch):
    prior, records, eval_launch = bundle(tmp_path, monkeypatch)
    input_root = audit.INPUT
    evaluate_dir = input_root / "evaluate"
    evaluate_dir.mkdir()
    eval_bytes = json.dumps(prior, separators=(",", ":")).encode()
    (evaluate_dir / "result.json").write_bytes(eval_bytes)
    (evaluate_dir / "launch.json").write_text(json.dumps(eval_launch))
    for arm in audit.ARMS:
        arm_dir = input_root / arm
        (arm_dir / "launch.json").write_text(json.dumps(records[arm]["launch"]))
        (arm_dir / "result.json").write_text(json.dumps(records[arm]["result"]))

    monkeypatch.setattr(audit.base, "ROOT", tmp_path)
    monkeypatch.setattr(audit.prior_audit, "_git_identity", lambda source: (AUDIT_SOURCE, ""))
    lease = Lease()
    monkeypatch.setattr(audit.prior_audit, "_validate_gpu_lease", lambda: lease)
    monkeypatch.setattr(audit.prior_audit, "shared_host", lambda: {"host": "unit-test"})
    monkeypatch.setattr(audit.prior_audit, "_runtime", lambda: dict(RUNTIME))
    monkeypatch.setattr(audit.prior_audit, "_health", lambda output: {"ok": True})
    monkeypatch.setattr(audit.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(audit.torch.cuda, "set_per_process_memory_fraction", lambda *a, **kw: None)
    fake_torch_utils = types.ModuleType("mjlab.utils.torch")
    fake_torch_utils.configure_torch_backends = lambda: None
    monkeypatch.setitem(sys.modules, "mjlab.utils.torch", fake_torch_utils)
    calls = []
    trace_stub = dict(timing="pre-action", **{key: [] for key in
        ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w")})

    def evaluate(checkpoint, seed, speed, yaw, health, **kwargs):
        arm = Path(checkpoint).parent.name
        index = audit.base.CASES.index((speed, yaw))
        calls.append((arm, seed, speed, yaw, kwargs))
        return {**prior["cases"][arm][index],
                "body_yaw_trace": dict(trace_stub), "gait_yaw_trace": dict(trace_stub)}

    monkeypatch.setattr(audit.base, "evaluate_case", evaluate)

    def body_summary(case, trace):
        assert trace == trace_stub
        assert list((tmp_path / "artifacts/evaluations/gait-run-02").glob("*-s839-c*.json"))
        return {"kind": "body"}

    def gait_summary(case, trace):
        assert trace == trace_stub
        assert list((tmp_path / "artifacts/evaluations/gait-run-02").glob("*-s839-c*.json"))
        return {"kind": "gait"}

    monkeypatch.setattr(audit.body_yaw_trace, "summarize_trace", body_summary)
    monkeypatch.setattr(audit.gait_yaw_trace, "summarize_trace", gait_summary)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    digest = hashlib.sha256(eval_bytes).hexdigest()
    assert audit.main(["--source", AUDIT_SOURCE, "--run-id", "gait-run-02",
                       "--evaluation-sha256", digest]) == 0
    assert lease.closed
    assert [(arm, seed, speed, yaw) for arm, seed, speed, yaw, _ in calls] == [
        (arm, 839, speed, yaw) for arm in audit.ARMS for speed, yaw in audit.base.CASES]
    assert all(kwargs == {"diagnostics": True, "yaw_trace": True, "gait_trace": True}
               for *_, kwargs in calls)
    output = tmp_path / "artifacts/evaluations/gait-run-02"
    result = json.loads((output / "result.json").read_text())
    assert result["final_iterations"] == {"control": 999, "budget": 3999}
    assert result["original_decisions"] == result["recomputed_decisions"]
    assert result["decision"] == "diagnostic-only-not-admission"
    assert result["policy_acceptance"] is result["simulator_qualified"] is False
    assert len(result["cases"]) == 8
    for row in result["cases"]:
        raw_path = output / row["raw_file"]
        summary_path = output / row["summary_file"]
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == row["raw_sha256"]
        assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == row["summary_sha256"]


@pytest.mark.parametrize("key", ["command", "actor_command", "quat_wxyz", "angular_b", "angular_w"])
def test_shared_trace_channel_mismatch_retains_raw_before_rejection(tmp_path, key):
    original = case_row(839, *audit.base.CASES[0], "c"*64)
    trace = {name: [[[0.]]] for name in
             ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w")}
    gait = {**trace, key: [[[1.]]]}
    observed = {**original, "body_yaw_trace": trace, "gait_yaw_trace": gait}
    with pytest.raises(ValueError, match="identical aligned shared"):
        audit.retain_case(tmp_path, "control", 0, original, observed, "c"*64)
    raw = json.loads((tmp_path/"control-s839-c0.json").read_text())
    assert raw["body_yaw_trace"][key] != raw["gait_yaw_trace"][key]
    assert not (tmp_path/"control-s839-c0-summary.json").exists()

