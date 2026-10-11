from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_head_yaw_audit as audit


def bundle(tmp_path, monkeypatch):
    from test_integrated_gait_yaw_audit import case_row

    root = tmp_path / "input"
    (root / "evaluate").mkdir(parents=True)
    monkeypatch.setattr(audit, "INPUT", root)
    cases, hashes, decisions = {}, {}, {}
    for arm in audit.ARMS:
        contract = audit.motor.training_contract(audit.recipe, arm)
        folder = root / arm
        folder.mkdir()
        checkpoint = folder / "model_999.pt"
        checkpoint.write_bytes(arm.encode())
        digest = audit.base.sha256(checkpoint)
        hashes[arm] = digest
        rows = [dict(case_row(seed, speed, yaw, digest),
                     soft_limit_fraction=0., mean_abs_mechanical_power_w=.2,
                     thermal_load_proxy=.03)
                for seed in audit.base.SEEDS for speed, yaw in audit.base.CASES]
        cases[arm], decisions[arm] = rows, audit.recipe.decision(rows)
        launch = dict(source=audit.INPUT_SOURCE, protocol=audit.recipe.PROTOCOL,
                      mode=arm, runtime=audit.RUNTIME, parent_sha256=audit.recipe.PARENT_SHA)
        receipt = dict(source=audit.INPUT_SOURCE, protocol=audit.recipe.PROTOCOL,
                       status="training-complete-not-accepted", mode=arm,
                       updates=contract["updates"], worlds=contract["worlds"],
                       seed=contract["seed"], steps=24000, transitions=6144000,
                       warm_start=dict(parent_sha256=audit.recipe.PARENT_SHA),
                       checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest)
        for name, value in (("launch", launch), ("result", receipt)):
            (folder / f"{name}.json").write_text(json.dumps(value))
        for index, row in enumerate(rows):
            (root / "evaluate" / f"{arm}-s{row['seed']}-c{index % 4}.json").write_text(json.dumps(row))
    prior = dict(source=audit.INPUT_SOURCE, protocol=audit.recipe.PROTOCOL,
                 selection=audit.motor.evaluation_plan(audit.recipe)["selection"],
                 cases=cases, decisions=decisions, checkpoint_sha256=hashes,
                 paired_differences=audit.motor.paired_differences(
                     cases["control"], cases["consolidate"], label="consolidate_minus_control"),
                 policy_acceptance=False, simulator_qualified=False,
                 physical_motion_authorized=False)
    launch = dict(source=audit.INPUT_SOURCE, protocol=audit.recipe.PROTOCOL,
                  mode="evaluate", runtime=audit.RUNTIME, parent_sha256=audit.recipe.PARENT_SHA)
    (root / "evaluate" / "launch.json").write_text(json.dumps(launch))
    (root / "evaluate" / "result.json").write_text(json.dumps(prior))
    return prior, audit.base.sha256(root / "evaluate" / "result.json")


def test_exact_complete_original_matrix_authenticated(tmp_path, monkeypatch):
    prior, digest = bundle(tmp_path, monkeypatch)
    assert audit.validated_input(digest, dict(audit.RUNTIME)) == prior
    assert audit.ARMS == ("control", "consolidate")
    assert (audit.SEED, audit.SPEED, audit.BODY_YAW, audit.HEAD_YAWS) == (839, .3, 0., (-.2, .2))


@pytest.mark.parametrize("fault", ["hash", "runtime", "source", "selection", "decision",
                                  "pairing", "case-file", "receipt", "checkpoint", "symlink"])
def test_input_identity_drift_is_rejected(tmp_path, monkeypatch, fault):
    prior, digest = bundle(tmp_path, monkeypatch)
    runtime = dict(audit.RUNTIME)
    if fault == "hash": digest = "f" * 64
    elif fault == "runtime": runtime["torch"] = "wrong"
    elif fault in ("source", "selection", "decision", "pairing"):
        if fault == "source": prior["source"] = "f" * 40
        elif fault == "selection": prior["selection"] = "best intermediate"
        elif fault == "decision": prior["decisions"]["control"] = {}
        else: prior["paired_differences"] = []
        path = audit.INPUT / "evaluate/result.json"
        path.write_text(json.dumps(prior))
        digest = audit.base.sha256(path)
    elif fault == "case-file":
        (audit.INPUT / "evaluate/control-s839-c1.json").write_text("{}")
    elif fault == "receipt":
        path = audit.INPUT / "control/result.json"
        receipt = json.loads(path.read_text())
        receipt["seed"] += 1
        path.write_text(json.dumps(receipt))
    elif fault == "checkpoint":
        (audit.INPUT / "control/model_999.pt").write_bytes(b"changed")
    else:
        path = audit.INPUT / "evaluate/control-s839-c1.json"
        moved = path.with_suffix(".backup")
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises((ValueError, KeyError)):
        audit.validated_input(digest, runtime)


def observed_case(head_yaw=-.2):
    from test_gait_yaw_trace import make_case_trace

    case, gait = make_case_trace()
    case.update(speed=.3, yaw=0., yaw_mae=[.01] * 8,
                head_yaw_command=head_yaw, head_measurement_view=audit.VIEW)
    head = torch.tensor([0., 0., head_yaw, 0.], dtype=torch.float32).double().tolist()
    for step in range(240):
        for world in range(8):
            gait["command"][step][world] = [.3, 0., 0.]
            gait["actor_command"][step][world] = [.3, 0., 0.]
            gait["angular_b"][step][world] = [0., 0., .01]
            gait["angular_w"][step][world] = [0., 0., .01]
            gait["head_command"][step][world] = head[:]
            gait["actor_head_command"][step][world] = head[:]
            gait["head_position_offset"][step][world] = [v + .02 for v in head]
    body = dict(protocol=audit.body_yaw_trace.PROTOCOL,
                timing=audit.body_yaw_trace.TIMING, step_dt=.02, startup_steps=60,
                steps=240, worlds=8, checkpoint_sha256=case["checkpoint_sha256"],
                policy_acceptance=False, physical_motion_authorized=False,
                **{key: deepcopy(gait[key]) for key in
                   ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w")})
    case.update(gait_yaw_trace=gait, body_yaw_trace=body)
    return case


@pytest.mark.parametrize("head_yaw", [-.2, .2])
def test_named_home_relative_head_summary_uses_float32_commands_and_startup(head_yaw):
    case = observed_case(head_yaw)
    before = deepcopy(case)
    for worlds in case["gait_yaw_trace"]["head_position_offset"][:60]:
        for row in worlds: row[:] = [100.] * 4
    summary = audit.head_summary(case, case["gait_yaw_trace"], head_yaw)
    assert summary["head_joints"] == list(audit.gait_yaw_trace.HEAD_JOINTS)
    assert summary["pooled_mae_rad_by_joint"] == pytest.approx([.02] * 4)
    assert summary["policy_acceptance"] is summary["physical_motion_authorized"] is False
    assert before["gait_yaw_trace"]["head_command"] == case["gait_yaw_trace"]["head_command"]


def test_full_retention_binds_both_summaries_and_raw_without_new_admission(tmp_path):
    case = observed_case()
    original = {key: value for key, value in case.items()
                if key not in ("head_yaw_command", "head_measurement_view", "body_yaw_trace", "gait_yaw_trace")}
    row = audit.retain_case(tmp_path, "control", 0, case, original, case["checkpoint_sha256"])
    raw = tmp_path / row["raw_file"]
    summary = json.loads((tmp_path / row["summary_file"]).read_text())
    assert row["raw_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert row["status"] == "complete-descriptive-only"
    assert summary["simulator_qualified"] is summary["policy_acceptance"] is False
    assert json.loads(raw.read_text())["baseline_is_not_rng_identical_replay"] is True


def test_partial_first_episode_is_saved_without_reset_or_summary_acceptance(tmp_path):
    case = observed_case()
    original = dict(case)
    case.update(complete=False, steps=61, terminal_worlds=[3])
    for trace in (case["body_yaw_trace"], case["gait_yaw_trace"]):
        trace["steps"] = 61
        for key, values in trace.items():
            if isinstance(values, list) and len(values) == 240:
                trace[key] = values[:61]
    row = audit.retain_case(tmp_path, "control", 0, case, original, case["checkpoint_sha256"])
    summary = json.loads((tmp_path / row["summary_file"]).read_text())
    assert summary["status"] == "partial-first-episode-terminal"
    assert summary["head_tracking"] is None and summary["steps"] == 61
    assert json.loads((tmp_path / row["raw_file"]).read_text())["observed_case"]["terminal_worlds"] == [3]


def test_trace_mismatch_or_wrong_head_command_retains_raw_before_rejection(tmp_path):
    case = observed_case()
    original = dict(case)
    case["body_yaw_trace"]["angular_w"][100][0][2] += .1
    with pytest.raises(ValueError, match="aligned shared"):
        audit.retain_case(tmp_path, "control", 0, case, original, case["checkpoint_sha256"])
    assert (tmp_path / "control-s839-h0.json").is_file()
    assert not (tmp_path / "control-s839-h0-summary.json").exists()
    case = observed_case()
    case["gait_yaw_trace"]["head_command"][100][0][2] = .1
    case["gait_yaw_trace"]["actor_head_command"][100][0][2] = .1
    with pytest.raises(ValueError, match="constant requested"):
        audit.head_summary(case, case["gait_yaw_trace"], -.2)


def test_entrypoint_runs_four_exact_jobs_and_closes_lease(tmp_path, monkeypatch):
    import importlib.metadata
    import mjlab.utils.torch

    prior, digest = bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(audit.base, "ROOT", tmp_path)
    source = "a" * 40
    monkeypatch.setattr(audit.retained, "_git_identity", lambda value: (source, ""))
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **kw: NS(returncode=0))
    lease = NS(closed=False)
    lease.close = lambda: setattr(lease, "closed", True)
    monkeypatch.setattr(audit.retained, "_validate_gpu_lease", lambda: lease)
    monkeypatch.setattr(audit.retained, "_runtime", lambda: dict(audit.RUNTIME))
    monkeypatch.setattr(audit.retained, "shared_host", lambda: {"unit-test": True})
    monkeypatch.setattr(audit.retained, "_health", lambda output: {"unit-test": True})
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "5.0.1")
    monkeypatch.setattr(torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a, **kw: None)
    monkeypatch.setattr(mjlab.utils.torch, "configure_torch_backends", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = []

    def evaluate(checkpoint, seed, speed, yaw, health, **kwargs):
        arm = checkpoint.parent.name
        calls.append((arm, seed, speed, yaw, kwargs))
        case = observed_case(kwargs["head_yaw_command"])
        case["checkpoint_sha256"] = prior["checkpoint_sha256"][arm]
        for trace in (case["body_yaw_trace"], case["gait_yaw_trace"]):
            trace["checkpoint_sha256"] = case["checkpoint_sha256"]
        return case

    monkeypatch.setattr(audit.base, "evaluate_case", evaluate)
    assert audit.main(["--source", source, "--run-id", "head-probe-test",
                       "--evaluation-sha256", digest]) == 0
    assert lease.closed
    assert calls == [(arm, 839, .3, 0., dict(diagnostics=True, yaw_trace=True,
                                          gait_trace=True, head_yaw_command=head))
                     for arm in audit.ARMS for head in audit.HEAD_YAWS]
    result = json.loads((tmp_path / "artifacts/evaluations/head-probe-test/result.json").read_text())
    assert result["diagnostic_case_count"] == len(result["cases"]) == 4
    assert result["decision"] == "diagnostic-only-not-admission"
    assert result["original_decisions"] == prior["decisions"]
    assert result["policy_acceptance"] is result["simulator_qualified"] is False
