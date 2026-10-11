import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from mjlab_microduck import integrated_action_response_audit as audit
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_rollout_batch_refinement as recipe


AUDIT_SOURCE = "a" * 40
RUNTIME = {name: "test-version" for name in audit.RUNTIME_PACKAGES}
PINNED_EVALUATION_SHA = "9a3ab60c7a4f7613bfd371015ce9ab9a59eb15921d5bc56a04d0ba2ea97067e4"
PINNED_CHECKPOINTS = {
    "control": "bdd2f66a64f86fdab6622ec6f945322688d92516fccd759f1b89d8db5378d4ca",
    "batch": "26dd8ace189b2ff0c15e252093672986587107fb935e2795ca90f8002373f1b7",
}
DEFAULT_INPUT = audit.INPUT


@pytest.fixture(autouse=True)
def reset_input(monkeypatch):
    monkeypatch.setattr(audit, "INPUT", DEFAULT_INPUT)


class Lease:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def case_row(seed, speed, yaw, digest, arm):
    offset = 0.01 if arm == "batch" else 0.
    joints = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
              "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
              "right_ankle"]
    return dict(
        seed=seed, speed=speed, yaw=yaw, worlds=8, steps=240,
        complete=True, terminal_worlds=[], checkpoint_sha256=digest,
        speed_mae=[.01 + offset] * 8, yaw_mae=[.02 + offset] * 8,
        torque_p99=.20 + offset, rated_speed_exceed_fraction=0.,
        soft_limit_fraction=.10 + offset,
        mean_abs_mechanical_power_w=1. + offset,
        thermal_load_proxy=.5 + offset,
        response_diagnostics=dict(
            protocol="signed-body-response-and-joint-load-v1", steps=240,
            hard_stop_margin_rad=.05, joint_columns=joints,
            min_distance_to_hard_stop_rad_by_joint=[.1] * 14,
            hard_stop_proximity_fraction_by_joint=[0.] * 14),
    )


def bundle(tmp_path, monkeypatch):
    """Build a full retained evaluation plus real temporary final checkpoint bytes."""
    input_root = tmp_path / "input"
    cases, decisions, records, hashes = {}, {}, {}, {}
    for arm in audit.ARMS:
        contract = motor.training_contract(recipe, arm)
        arm_dir = input_root / arm
        arm_dir.mkdir(parents=True)
        checkpoint = arm_dir / f"model_{contract['final_iteration']}.pt"
        checkpoint.write_bytes(f"temporary checkpoint bytes for {arm}".encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        hashes[arm] = digest
        rows = [case_row(seed, speed, yaw, digest, arm)
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

    evaluation_plan = motor.evaluation_plan(recipe)
    prior = dict(
        source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL,
        selection=evaluation_plan["selection"],
        checkpoint_sha256=hashes, cases=cases, decisions=decisions,
        paired_differences=motor.paired_differences(
            cases["control"], cases["batch"], label="batch_minus_control"),
    )
    launch = dict(source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL, mode="evaluate",
                  parent_sha256=recipe.PARENT_SHA, runtime=dict(RUNTIME))
    aggregate_bytes = json.dumps(prior, separators=(",", ":")).encode()
    aggregate_sha = hashlib.sha256(aggregate_bytes).hexdigest()
    monkeypatch.setattr(audit, "INPUT", input_root)
    # The explicit-pin test below verifies production constants. Here the complete
    # bundle is temporary, so point the pin checks at these fixture bytes.
    monkeypatch.setattr(audit, "EVALUATION_SHA256", aggregate_sha)
    monkeypatch.setattr(audit, "CHECKPOINT_SHA256", hashes)
    return prior, records, launch, aggregate_bytes, aggregate_sha, hashes


def test_invocation_requires_exact_pinned_hash_clean_source_and_literal_cuda0():
    assert audit.EVALUATION_SHA256 == PINNED_EVALUATION_SHA
    assert audit.CHECKPOINT_SHA256 == PINNED_CHECKPOINTS
    audit.validate_invocation(AUDIT_SOURCE, "action-run-01", PINNED_EVALUATION_SHA,
                              head=AUDIT_SOURCE, status="", cuda_visible_devices="0")
    for mutation in (
        dict(source="A" * 40), dict(run_id="../escape"),
        dict(evaluation_sha256="B" * 64), dict(evaluation_sha256="c" * 64),
        dict(head="d" * 40), dict(status=" M dirty.py"),
        dict(cuda_visible_devices="0,1"), dict(cuda_visible_devices=None),
    ):
        values = dict(source=AUDIT_SOURCE, run_id="action-run-01",
                      evaluation_sha256=PINNED_EVALUATION_SHA, head=AUDIT_SOURCE,
                      status="", cuda_visible_devices="0")
        values.update(mutation)
        with pytest.raises(ValueError):
            audit.validate_invocation(**values)


def test_input_bundle_pins_aggregate_runtime_receipts_parent_cases_and_seed839_jobs(
        tmp_path, monkeypatch):
    prior, records, launch, _, aggregate_sha, hashes = bundle(tmp_path, monkeypatch)
    validated = audit.validate_input_bundle(prior, aggregate_sha, source=audit.INPUT_SOURCE,
                                            runtime=RUNTIME, arm_records=records,
                                            evaluation_launch=launch)
    assert validated["decisions"] == prior["decisions"]
    assert validated["checkpoint_sha256"] == hashes
    assert validated["final_iterations"] == {"control": 999, "batch": 999}
    assert "final_iterations" not in prior
    assert len(prior["cases"]["control"]) == len(prior["cases"]["batch"]) == 12
    assert sum(len(rows) for rows in prior["cases"].values()) == 24
    assert prior["paired_differences"] == motor.paired_differences(
        prior["cases"]["control"], prior["cases"]["batch"], label="batch_minus_control")
    jobs = list(audit.case_jobs(validated))
    assert [(arm, index) for arm, index, _ in jobs] == [
        (arm, index) for arm in audit.ARMS for index in range(4)]
    assert all(row["seed"] == 839 for _, _, row in jobs)
    assert [row["speed"] for _, _, row in jobs[:4]] == [speed for speed, _ in audit.base.CASES]
    assert [row["yaw"] for _, _, row in jobs[:4]] == [yaw for _, yaw in audit.base.CASES]


def test_selection_label_matches_the_production_legacy_final999_plan():
    plan = motor.evaluation_plan(recipe)
    assert audit.EVALUATION_SELECTION == plan["selection"]
    assert "final_iterations" not in plan
    assert {arm: contract["final_iteration"] for arm, contract in plan["contracts"].items()} == {
        "control": 999, "batch": 999}


@pytest.mark.parametrize("mutation", [
    "source", "protocol", "selection", "iterations", "eval_runtime", "eval_parent",
    "arm_runtime", "arm_mode", "arm_parent", "receipt_seed", "receipt_updates",
    "receipt_worlds", "receipt_transitions", "receipt_checkpoint", "aggregate_hash",
    "case_hash", "case_order", "decision", "paired_differences", "extra_arm",
])
def test_input_bundle_rejects_changed_identity_or_retained_evidence(
        tmp_path, monkeypatch, mutation):
    prior, records, launch, _, aggregate_sha, _ = bundle(tmp_path, monkeypatch)
    if mutation == "source":
        prior["source"] = "wrong"
    elif mutation == "protocol":
        prior["protocol"] = "wrong"
    elif mutation == "selection":
        prior["selection"] = "intermediate checkpoint"
    elif mutation == "iterations":
        prior["final_iterations"] = {"control": 999, "batch": 999}
    elif mutation == "eval_runtime":
        launch["runtime"]["torch"] = "wrong"
    elif mutation == "eval_parent":
        launch["parent_sha256"] = "f" * 64
    elif mutation == "extra_arm":
        records["other"] = {}
    elif mutation == "decision":
        prior["decisions"]["batch"] = {"decision": "changed"}
    elif mutation == "paired_differences":
        prior["paired_differences"][0]["batch_minus_control"]["torque_p99"] += 1
    elif mutation == "case_order":
        prior["cases"]["control"][0], prior["cases"]["control"][1] = (
            prior["cases"]["control"][1], prior["cases"]["control"][0])
    else:
        record = records["batch"]
        if mutation == "arm_runtime":
            record["launch"]["runtime"]["torch"] = "wrong"
        elif mutation == "arm_mode":
            record["launch"]["mode"] = "control"
        elif mutation == "arm_parent":
            record["launch"]["parent_sha256"] = "f" * 64
        elif mutation == "receipt_seed":
            record["result"]["seed"] += 1
        elif mutation == "receipt_updates":
            record["result"]["updates"] -= 1
        elif mutation == "receipt_worlds":
            record["result"]["worlds"] = 256
        elif mutation == "receipt_transitions":
            record["result"]["transitions"] -= 1
        elif mutation == "receipt_checkpoint":
            record["result"]["checkpoint"] = "model_998.pt"
        elif mutation == "aggregate_hash":
            prior["checkpoint_sha256"]["batch"] = "f" * 64
        else:
            prior["cases"]["batch"][0]["checkpoint_sha256"] = "f" * 64
    with pytest.raises((ValueError, KeyError)):
        audit.validate_input_bundle(prior, aggregate_sha, source=audit.INPUT_SOURCE,
                                    runtime=RUNTIME, arm_records=records,
                                    evaluation_launch=launch)


def test_aggregate_hash_binds_the_exact_temporary_json_bytes(tmp_path, monkeypatch):
    prior, records, launch, raw, digest, _ = bundle(tmp_path, monkeypatch)
    assert audit.prior_audit.validate_evaluation_bytes(raw, digest) == digest
    with pytest.raises(ValueError, match="aggregate SHA"):
        audit.prior_audit.validate_evaluation_bytes(raw + b" ", digest)
    with pytest.raises(ValueError, match="pinned rollout-batch aggregate"):
        audit.validate_input_bundle(prior, "e" * 64, source=audit.INPUT_SOURCE,
                                    runtime=RUNTIME, arm_records=records,
                                    evaluation_launch=launch)


def trace_stub():
    return dict(timing="pre-action", **{
        key: [[[0.0]]] for key in
        ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w")
    })


def test_raw_payload_is_retained_before_shared_trace_validation_fails(tmp_path):
    digest = "c" * 64
    speed, yaw = audit.base.CASES[0]
    original = case_row(839, speed, yaw, digest, "control")
    body, action = trace_stub(), trace_stub()
    action["angular_b"] = [[[1.0]]]
    observed = {**original, "joint_target_diagnostics": {"captured": True},
                "body_yaw_trace": body, "action_response_trace": action}
    with pytest.raises(ValueError, match="identical aligned shared"):
        audit.retain_case(tmp_path, "control", 0, original, observed, digest)
    raw = json.loads((tmp_path / "control-s839-c0.json").read_text())
    assert raw["observed_case"]["joint_target_diagnostics"] == {"captured": True}
    assert raw["body_yaw_trace"]["angular_b"] != raw["action_response_trace"]["angular_b"]
    assert not (tmp_path / "control-s839-c0-summary.json").exists()


def test_declared_target_diagnostics_are_the_only_omitted_replay_field(tmp_path):
    digest = "c" * 64
    speed, yaw = audit.base.CASES[0]
    original = case_row(839, speed, yaw, digest, "control")
    observed = {**original, "joint_target_diagnostics": {"captured": True},
                "unexpected_observed_channel": {"value": 7}}
    trace = trace_stub()
    # Keep this assertion focused on the runner's replay-field selection; the
    # trace summary implementations have their own shape and timing contracts.
    original_body_summary = audit.body_yaw_trace.summarize_trace
    original_action_summary = audit.action_response_trace.summarize_trace
    audit.body_yaw_trace.summarize_trace = lambda case, sample: {"body": True}
    audit.action_response_trace.summarize_trace = lambda case, sample: {"action": True}
    try:
        result = audit.retain_case(tmp_path, "control", 0, original,
                                   {**observed, "body_yaw_trace": trace,
                                    "action_response_trace": trace}, digest)
    finally:
        audit.body_yaw_trace.summarize_trace = original_body_summary
        audit.action_response_trace.summarize_trace = original_action_summary
    summary = json.loads((tmp_path / "control-s839-c0-summary.json").read_text())
    assert result["differences"]
    assert "joint_target_diagnostics" in repr(result["differences"])
    assert "unexpected_observed_channel" in repr(result["differences"])
    assert result["retained_metric_differences"]
    assert "joint_target_diagnostics" not in repr(result["retained_metric_differences"])
    assert "unexpected_observed_channel" in repr(result["retained_metric_differences"])
    assert summary["expected_augmented_fields"] == ["joint_target_diagnostics"]
    assert summary["replay_verified"] is False
    assert summary["policy_acceptance"] is summary["simulator_qualified"] is False
    raw = json.loads((tmp_path / "control-s839-c0.json").read_text())
    assert raw["observed_case"]["unexpected_observed_channel"] == {"value": 7}


def test_main_runs_eight_serial_final999_cases_with_diagnostic_capture_and_no_acceptance(
        tmp_path, monkeypatch):
    prior, records, eval_launch, eval_bytes, aggregate_sha, hashes = bundle(tmp_path, monkeypatch)
    evaluate_dir = audit.INPUT / "evaluate"
    evaluate_dir.mkdir(parents=True)
    (evaluate_dir / "result.json").write_bytes(eval_bytes)
    (evaluate_dir / "launch.json").write_text(json.dumps(eval_launch))
    for arm in audit.ARMS:
        arm_dir = audit.INPUT / arm
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
    shared = trace_stub()

    def evaluate(checkpoint, seed, speed, yaw, health, **kwargs):
        arm = Path(checkpoint).parent.name
        index = audit.base.CASES.index((speed, yaw))
        calls.append((arm, Path(checkpoint).name, seed, speed, yaw, kwargs))
        old = prior["cases"][arm][index]
        return {**old, "joint_target_diagnostics": {"observed": True},
                "body_yaw_trace": dict(shared), "action_response_trace": dict(shared),
                "unexpected_observed_channel": {"case": index}}

    monkeypatch.setattr(audit.base, "evaluate_case", evaluate)
    summaries = []

    def summarize(case, trace):
        raw_files = list((tmp_path / "artifacts/evaluations/action-run-02").glob("*-s839-c[0-3].json"))
        assert raw_files
        summaries.append((case["seed"], trace))
        return {"summarized": True}

    monkeypatch.setattr(audit.body_yaw_trace, "summarize_trace", summarize)
    monkeypatch.setattr(audit.action_response_trace, "summarize_trace", summarize)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert audit.main(["--source", AUDIT_SOURCE, "--run-id", "action-run-02",
                       "--evaluation-sha256", aggregate_sha]) == 0
    assert lease.closed
    expected_calls = [(arm, 839, speed, yaw) for arm in audit.ARMS
                      for speed, yaw in audit.base.CASES]
    assert [(arm, seed, speed, yaw) for arm, _, seed, speed, yaw, _ in calls] == expected_calls
    assert all(checkpoint == "model_999.pt" for _, checkpoint, _, _, _, _ in calls)
    assert all(kwargs == {"diagnostics": True, "target_diagnostics": True,
                          "yaw_trace": True, "action_trace": True}
               for *_, kwargs in calls)
    assert len(summaries) == 16
    for _, trace in summaries:
        for key in ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w"):
            assert trace[key] == shared[key]

    output = tmp_path / "artifacts/evaluations/action-run-02"
    result = json.loads((output / "result.json").read_text())
    assert result["final_iterations"] == {"control": 999, "batch": 999}
    assert result["original_decisions"] == result["recomputed_decisions"] == prior["decisions"]
    assert result["decision"] == "diagnostic-only-not-admission"
    assert result["policy_acceptance"] is result["simulator_qualified"] is False
    assert result["physical_motion_authorized"] is False
    assert result["diagnostic_case_count"] == len(result["cases"]) == 8
    for row in result["cases"]:
        assert row["retained_metric_differences"]
        assert "joint_target_diagnostics" not in repr(row["retained_metric_differences"])
        assert "unexpected_observed_channel" in repr(row["retained_metric_differences"])
        assert row["differences"]
        assert "joint_target_diagnostics" in repr(row["differences"])
        assert "unexpected_observed_channel" in repr(row["differences"])
        assert row["replay_verified"] is False
        raw_path = output / row["raw_file"]
        summary_path = output / row["summary_file"]
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == row["raw_sha256"]
        assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == row["summary_sha256"]
        raw = json.loads(raw_path.read_text())
        assert raw["observed_case"]["joint_target_diagnostics"] == {"observed": True}
        assert raw["observed_case"]["unexpected_observed_channel"] == {"case": row["case_index"]}
