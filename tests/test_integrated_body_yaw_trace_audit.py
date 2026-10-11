import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from mjlab_microduck import integrated_body_yaw_trace_audit as audit
from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as recipe


RUNTIME = {name: "1.0-test" for name in audit.RUNTIME_PACKAGES}
CHECKPOINTS = dict(audit.FINAL_SHA256)
EVALUATION_SHA = audit.EVALUATION_SHA256
RUN_SOURCE = "d" * 40


def case_row(seed, speed, yaw, digest):
    return {
        "seed": seed,
        "speed": speed,
        "yaw": yaw,
        "worlds": 8,
        "steps": 240,
        "complete": True,
        "terminal_worlds": [],
        "checkpoint_sha256": digest,
        "speed_mae": [0.] * 8,
        "yaw_mae": [0.] * 8,
        "torque_p99": .2,
        "rated_speed_exceed_fraction": 0.,
        "response_diagnostics": {
            "protocol": "signed-body-response-and-joint-load-v1",
            "steps": 240,
            "hard_stop_margin_rad": .05,
            "joint_columns": [
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
                "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
                "right_ankle",
            ],
            "min_distance_to_hard_stop_rad_by_joint": [.1] * 14,
            "hard_stop_proximity_fraction_by_joint": [0.] * 14,
        },
    }


def bundle():
    cases = {}
    decisions = {}
    records = {}
    for arm in audit.ARMS:
        rows = [case_row(seed, speed, yaw, CHECKPOINTS[arm])
                for seed in (839, 853, 857) for speed, yaw in audit.base.CASES]
        cases[arm] = rows
        decisions[arm] = recipe.decision(rows)
        launch = {
            "source": audit.INPUT_SOURCE,
            "protocol": audit.INPUT_PROTOCOL,
            "mode": arm,
            "parent_sha256": audit.PARENT_SHA,
            "runtime": dict(RUNTIME),
        }
        result = {
            "source": audit.INPUT_SOURCE,
            "protocol": audit.INPUT_PROTOCOL,
            "mode": arm,
            "status": "training-complete-not-accepted",
            "updates": 1000,
            "worlds": 256,
            "seed": 1027,
            "warm_start": {"parent_sha256": audit.PARENT_SHA},
            "checkpoint": f"/native/{arm}/model_999.pt",
            "checkpoint_sha256": CHECKPOINTS[arm],
        }
        records[arm] = {
            "launch": launch,
            "result": result,
            "actual_checkpoint_sha256": CHECKPOINTS[arm],
        }
    prior = {
        "source": audit.INPUT_SOURCE,
        "protocol": audit.INPUT_PROTOCOL,
        "selection": "both final999 only; unchanged held-out protocol",
        "checkpoint_sha256": dict(CHECKPOINTS),
        "cases": cases,
        "decisions": decisions,
    }
    eval_launch = {
        "source": audit.INPUT_SOURCE,
        "protocol": audit.INPUT_PROTOCOL,
        "mode": "evaluate",
        "runtime": dict(RUNTIME),
    }
    return prior, records, eval_launch


def test_invocation_requires_clean_pinned_source_safe_id_explicit_hash_and_gpu0():
    audit.validate_invocation(RUN_SOURCE, "trace-run-01", EVALUATION_SHA,
                              head=RUN_SOURCE, status="", cuda_visible_devices="0")
    for kwargs in (
        dict(source="A" * 40),
        dict(run_id="../unsafe"),
        dict(evaluation_sha256="C" * 64),
        dict(head="e" * 40),
        dict(status=" M dirty.py"),
        dict(cuda_visible_devices="0,1"),
    ):
        args = dict(source=RUN_SOURCE, run_id="trace-run-01",
                    evaluation_sha256=EVALUATION_SHA, head=RUN_SOURCE,
                    status="", cuda_visible_devices="0")
        args.update(kwargs)
        with pytest.raises(ValueError):
            audit.validate_invocation(**args)


def test_evaluation_aggregate_sha_binds_exact_raw_bytes():
    raw = b'{"protocol":"integrated-decoupled-yaw-precision-v1"}\n'
    digest = hashlib.sha256(raw).hexdigest()
    assert audit.validate_evaluation_bytes(raw, digest) == digest
    with pytest.raises(ValueError, match="aggregate SHA"):
        audit.validate_evaluation_bytes(raw + b" ", digest)


def test_validates_exact_input_runtime_training_parent_final_checkpoint_and_cases():
    prior, records, launch = bundle()
    validated = audit.validate_input_bundle(prior, EVALUATION_SHA, source=audit.INPUT_SOURCE,
        runtime=RUNTIME, arm_records=records, evaluation_launch=launch)
    assert validated["decisions"] == prior["decisions"]
    assert [row["yaw"] for row in validated["cases"]["precision"]] == [yaw for _, yaw in audit.base.CASES]
    jobs = list(audit.case_jobs(prior))
    assert [(arm, index) for arm, index, _ in jobs] == [
        (arm, index) for arm in audit.ARMS for index in range(4)
    ]
    assert all(row["seed"] == 839 for _, _, row in jobs)


@pytest.mark.parametrize("mutation", [
    "input_source", "input_protocol", "input_selection", "eval_runtime", "arm_source",
    "arm_protocol", "arm_runtime", "parent", "updates", "worlds", "seed", "status",
    "checkpoint_name", "receipt_hash", "aggregate_hash", "actual_hash",
])
def test_rejects_wrong_training_or_input_identity(mutation):
    prior, records, launch = bundle()
    if mutation == "input_source": prior["source"] = "x"
    elif mutation == "input_protocol": prior["protocol"] = "other"
    elif mutation == "input_selection": prior["selection"] = "not final999"
    elif mutation == "eval_runtime": launch["runtime"]["torch"] = "other"
    else:
        arm = "control"
        record = records[arm]
        if mutation == "arm_source": record["launch"]["source"] = "other"
        elif mutation == "arm_protocol": record["launch"]["protocol"] = "other"
        elif mutation == "arm_runtime": record["launch"]["runtime"]["torch"] = "other"
        elif mutation == "parent": record["result"]["warm_start"]["parent_sha256"] = "f" * 64
        elif mutation == "updates": record["result"]["updates"] = 999
        elif mutation == "worlds": record["result"]["worlds"] = 64
        elif mutation == "seed": record["result"]["seed"] = 1021
        elif mutation == "status": record["result"]["status"] = "training-failed"
        elif mutation == "checkpoint_name": record["result"]["checkpoint"] = "/native/model_750.pt"
        elif mutation == "receipt_hash": record["result"]["checkpoint_sha256"] = "f" * 64
        elif mutation == "aggregate_hash": prior["checkpoint_sha256"][arm] = "f" * 64
        else: record["actual_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        audit.validate_input_bundle(prior, EVALUATION_SHA, source=audit.INPUT_SOURCE,
            runtime=RUNTIME, arm_records=records, evaluation_launch=launch)


def test_rejects_wrong_source_or_malformed_evaluation_hash():
    prior, records, launch = bundle()
    with pytest.raises(ValueError, match="original training source"):
        audit.validate_input_bundle(prior, EVALUATION_SHA, source=RUN_SOURCE,
            runtime=RUNTIME, arm_records=records, evaluation_launch=launch)
    with pytest.raises(ValueError, match="aggregate SHA"):
        audit.validate_input_bundle(prior, "not-a-sha", source=audit.INPUT_SOURCE,
            runtime=RUNTIME, arm_records=records, evaluation_launch=launch)


def test_rejects_reordered_original_case_matrix():
    prior, records, launch = bundle()
    prior["cases"]["control"][0], prior["cases"]["control"][1] = (
        prior["cases"]["control"][1], prior["cases"]["control"][0])
    with pytest.raises(ValueError, match="ordering"):
        audit.validate_input_bundle(prior, EVALUATION_SHA, source=audit.INPUT_SOURCE,
            runtime=RUNTIME, arm_records=records, evaluation_launch=launch)


def make_trace_case(*, torque=.2, arm="control", index=0):
    speed, yaw = audit.base.CASES[index]
    seed, digest = 839, CHECKPOINTS[arm]
    case = case_row(seed, speed, yaw, digest)
    case["yaw_mae"] = [.05] * 8
    case["torque_p99"] = torque
    expected = [speed, 0., yaw]
    command = [[expected[:] for _ in range(8)] for _ in range(240)]
    actor = [[expected[:] for _ in range(8)] for _ in range(240)]
    quats = [[[1., 0., 0., 0.] for _ in range(8)] for _ in range(240)]
    body = [[[0., 0., yaw + .05] for _ in range(8)] for _ in range(240)]
    world = [[[0., 0., yaw + .05] for _ in range(8)] for _ in range(240)]
    trace = {
        "protocol": "body-yaw-orientation-trace-v1",
        "step_dt": .02,
        "startup_steps": 60,
        "steps": 240,
        "worlds": 8,
        "checkpoint_sha256": digest,
        "timing": "pre-action state before policy/step",
        "command": command,
        "actor_command": actor,
        "quat_wxyz": quats,
        "angular_b": body,
        "angular_w": world,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
    }
    return case, trace


def test_retain_case_binds_raw_trace_and_hash_bound_summary_without_acceptance(tmp_path):
    case, trace = make_trace_case()
    original = dict(case)
    observed = {**case, "body_yaw_trace": trace}
    summary = audit.retain_case(tmp_path, "control", 0, original, observed)
    raw_path = tmp_path / summary["case_file"]
    summary_path = tmp_path / summary["summary_file"]
    assert summary["case_trace_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert summary["summary_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    report = json.loads(summary_path.read_text())
    assert report["replay_verified"] is True
    assert report["trace_summary"]["policy_acceptance"] is False
    assert report["physical_motion_authorized"] is False
    assert set(path.name for path in tmp_path.iterdir()) == {raw_path.name, summary_path.name}


def test_retains_differences_and_false_replay_without_promoting_gate(tmp_path):
    case, trace = make_trace_case(torque=.24, arm="precision", index=3)
    original = dict(case, torque_p99=.2)
    observed = {**case, "body_yaw_trace": trace}
    report = audit.retain_case(tmp_path, "precision", 3, original, observed)
    summary = json.loads((tmp_path / report["summary_file"]).read_text())
    assert summary["replay_verified"] is False
    assert summary["original_case_differences"][0]["field"] == "torque_p99"
    assert report["replay_verified"] is False
    assert report["differences"]
    assert summary["policy_acceptance"] is False


@pytest.mark.parametrize("field,value", [
    ("seed", 853), ("speed", .3), ("yaw", .2),
    ("checkpoint_sha256", CHECKPOINTS["precision"]),
])
def test_wrong_observed_identity_retains_raw_but_rejects_summary(tmp_path, field, value):
    case, trace = make_trace_case()
    observed = {**case, field: value, "body_yaw_trace": trace}
    with pytest.raises(ValueError, match="observed declared"):
        audit.retain_case(tmp_path, "control", 0, case, observed)
    assert (tmp_path / "control-s839-c0.json").is_file()
    assert not (tmp_path / "control-s839-c0-summary.json").exists()


def test_extra_observed_field_is_not_hidden_from_replay_comparison(tmp_path):
    case, trace = make_trace_case()
    report = audit.retain_case(tmp_path, "control", 0, case,
        {**case, "unexpected_result_field": 1, "body_yaw_trace": trace})
    assert report["replay_verified"] is False
    assert report["differences"]


def test_raw_trace_is_written_before_summary_audit_failure(tmp_path, monkeypatch):
    case, trace = make_trace_case()
    observed = {**case, "body_yaw_trace": trace}
    def fail(*args):
        raise ValueError("intentional audit failure")
    monkeypatch.setattr(audit, "summarize_trace", fail)
    with pytest.raises(ValueError, match="intentional"):
        audit.retain_case(tmp_path, "control", 0, case, observed)
    raw = tmp_path / "control-s839-c0.json"
    assert raw.is_file()
    payload = json.loads(raw.read_text())
    assert payload["case"]["checkpoint_sha256"] == CHECKPOINTS["control"]
    assert payload["trace"]["protocol"] == "body-yaw-orientation-trace-v1"


def test_new_case_artifact_is_never_overwritten(tmp_path):
    case, trace = make_trace_case()
    observed = {**case, "body_yaw_trace": trace}
    audit.retain_case(tmp_path, "control", 0, case, observed)
    with pytest.raises(FileExistsError):
        audit.retain_case(tmp_path, "control", 0, case, observed)


def _trace_for_case(case):
    command_value = [case["speed"], 0., case["yaw"]]
    command = [[command_value[:] for _ in range(8)] for _ in range(240)]
    actor = [[command_value[:] for _ in range(8)] for _ in range(240)]
    quats = [[[1., 0., 0., 0.] for _ in range(8)] for _ in range(240)]
    body = [[[0., 0., case["yaw"] + case["yaw_mae"][world]] for world in range(8)]
            for _ in range(240)]
    return {
        "protocol": "body-yaw-orientation-trace-v1",
        "step_dt": .02,
        "startup_steps": 60,
        "steps": 240,
        "worlds": 8,
        "checkpoint_sha256": case["checkpoint_sha256"],
        "timing": "pre-action state before policy/step",
        "command": command,
        "actor_command": actor,
        "quat_wxyz": quats,
        "angular_b": body,
        "angular_w": body,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
    }


def test_mocked_main_writes_eight_diagnostic_only_hash_bound_replays(tmp_path, monkeypatch):
    prior, records, eval_launch = bundle()
    input_root = tmp_path / "input"
    for arm in audit.ARMS:
        arm_dir = input_root / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "launch.json").write_text(json.dumps(records[arm]["launch"]))
        model = f"mock model bytes {arm}".encode()
        (arm_dir / "model_999.pt").write_bytes(model)
        records[arm]["actual_checkpoint_sha256"] = hashlib.sha256(model).hexdigest()
        monkeypatch.setitem(audit.FINAL_SHA256, arm, records[arm]["actual_checkpoint_sha256"])
        records[arm]["result"]["checkpoint_sha256"] = records[arm]["actual_checkpoint_sha256"]
        prior["checkpoint_sha256"][arm] = records[arm]["actual_checkpoint_sha256"]
        for row in prior["cases"][arm]:
            row["checkpoint_sha256"] = records[arm]["actual_checkpoint_sha256"]
        records[arm]["launch"]["runtime"] = dict(RUNTIME)
        records[arm]["launch"]["source"] = audit.INPUT_SOURCE
        records[arm]["launch"]["protocol"] = audit.INPUT_PROTOCOL
    prior["decisions"] = {arm: recipe.decision(prior["cases"][arm]) for arm in audit.ARMS}
    for arm in audit.ARMS:
        records[arm]["result"]["checkpoint"] = f"/native/{arm}/model_999.pt"
    evaluate_dir = input_root / "evaluate"
    evaluate_dir.mkdir()
    eval_bytes = json.dumps(prior, sort_keys=True).encode()
    monkeypatch.setattr(audit, "EVALUATION_SHA256", hashlib.sha256(eval_bytes).hexdigest())
    (evaluate_dir / "result.json").write_bytes(eval_bytes)
    (evaluate_dir / "launch.json").write_text(json.dumps(eval_launch))
    for arm in audit.ARMS:
        (input_root / arm / "result.json").write_text(json.dumps(records[arm]["result"]))

    monkeypatch.setattr(audit, "INPUT", input_root)
    monkeypatch.setattr(audit.base, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "_git_identity", lambda source: (source, ""))
    monkeypatch.setattr(audit, "_runtime", lambda: dict(RUNTIME))
    monkeypatch.setattr(audit, "shared_host", lambda: {"gpu": "mock-guarded"})
    lease_path = tmp_path / "gpu.lock"
    lease_path.write_text("owned test lease")
    monkeypatch.setattr(audit, "LOCK", lease_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(audit.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(audit.torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: None)
    mock_torch = types.ModuleType("mjlab.utils.torch")
    mock_torch.configure_torch_backends = lambda: None
    monkeypatch.setitem(sys.modules, "mjlab.utils.torch", mock_torch)

    def evaluate(checkpoint, seed, speed, yaw, health, *, diagnostics, yaw_trace):
        arm = Path(checkpoint).parent.name
        index = list(audit.base.CASES).index((speed, yaw))
        row = dict(prior["cases"][arm][index])
        assert seed == audit.SEED and diagnostics is True and yaw_trace is True
        return {**row, "body_yaw_trace": _trace_for_case(row)}

    monkeypatch.setattr(audit.base, "evaluate_case", evaluate)
    run_id = "mock-yaw-trace-audit"
    assert audit.main(["--source", RUN_SOURCE, "--run-id", run_id,
                       "--evaluation-sha256", hashlib.sha256(eval_bytes).hexdigest()]) == 0
    output = tmp_path / "artifacts/evaluations" / run_id
    result = json.loads((output / "result.json").read_text())
    assert result["diagnostic_case_count"] == 8
    assert result["replay_verified"] is True
    assert result["decision"] == "diagnostic-only-not-admission"
    assert result["policy_acceptance"] is False
    assert result["simulator_qualified"] is False
    assert result["physical_motion_authorized"] is False
    assert len(list(output.glob("*-s839-c[0-3].json"))) == 8
    assert len(list(output.glob("*-summary.json"))) == 8
    assert (output / "launch.json").is_file()
    assert (output / "health.jsonl").is_file()


def test_main_records_failure_without_overwriting_or_claiming_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.base, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "_git_identity", lambda source: (source, " M source.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    run_id = "mock-yaw-trace-failure"
    with pytest.raises(ValueError, match="clean audit source"):
        audit.main(["--source", RUN_SOURCE, "--run-id", run_id,
                    "--evaluation-sha256", EVALUATION_SHA])
    output = tmp_path / "artifacts/evaluations" / run_id
    failure = json.loads((output / "failure.json").read_text())
    assert failure["type"] == "ValueError"
    assert failure["policy_acceptance"] is False
    assert failure["physical_motion_authorized"] is False

