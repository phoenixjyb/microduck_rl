import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_observation_replay_probe as probe


CHECKPOINT = "a" * 64


def replay_case(*, speed_delta=0., yaw_delta=0., torque=.2, trace=False, **changes):
    row = dict(
        seed=probe.timing.SEED, speed=probe.CASE[0], yaw=probe.CASE[1], worlds=8,
        steps=240, complete=True, terminal_worlds=[], checkpoint_sha256=CHECKPOINT,
        motor_stream="named-14-joint-stream", speed_mae=[.1 + speed_delta] * 8,
        yaw_mae=[.02 + yaw_delta] * 8, torque_p99=torque, soft_limit_fraction=.01,
        rated_speed_exceed_fraction=0., mean_abs_mechanical_power_w=1.5,
        thermal_load_proxy=.1,
    )
    row.update(changes)
    if trace:
        row["observation_timing_trace"] = {"protocol": "trace-only-payload", "steps": 240}
    return row


def test_compare_uses_signed_right_minus_left_and_excludes_only_timing_trace():
    left = replay_case()
    right = replay_case(speed_delta=.02, yaw_delta=-.01, torque=.24, trace=True)

    result = probe.compare(left, right)

    assert result["scalar_delta"]["torque_p99"] == pytest.approx(.04)
    assert result["world_metrics"]["speed_mae"]["signed_delta_per_world"] == pytest.approx([.02] * 8)
    assert result["world_metrics"]["yaw_mae"]["signed_delta_per_world"] == pytest.approx([-.01] * 8)
    assert result["world_metrics"]["speed_mae"]["max_abs_delta"] == pytest.approx(.02)
    assert result["common_fields_exact"] is False  # metrics differ, despite trace-only metadata
    same = probe.compare(left, replay_case(trace=True))
    assert same["common_fields_exact"] is True
    assert same["observation_trace_exact_if_both"] is None
    same_trace = replay_case(trace=True)
    assert probe.compare(same_trace, dict(same_trace))["observation_trace_exact_if_both"] is True
    changed_trace = replay_case(trace=True)
    changed_trace["observation_timing_trace"]["steps"] = 239
    assert probe.compare(same_trace, changed_trace)["observation_trace_exact_if_both"] is False
    assert same["policy_acceptance"] is same["physical_motion_authorized"] is False
    assert "causal" in same["interpretation"]


@pytest.mark.parametrize("fault", ["identity", "nonfinite", "world-shape", "scalar-type"])
def test_compare_rejects_changed_identity_nonfinite_or_malformed_metrics(fault):
    left, right = replay_case(), replay_case()
    if fault == "identity":
        right["motor_stream"] = "different-stream"
    elif fault == "nonfinite":
        right["yaw_mae"][2] = float("nan")
    elif fault == "world-shape":
        right["speed_mae"] = [0.] * 7
    else:
        right["torque_p99"] = True
    with pytest.raises(ValueError):
        probe.compare(left, right)


def test_comparisons_requires_and_names_exact_abba_pairs():
    rows = [dict(mode=mode, case=replay_case(torque=.2 + index * .01))
            for index, mode in enumerate(probe.ORDER)]
    result = probe.comparisons(rows)
    assert tuple(result) == ("baseline_repeat", "trace_repeat", "first_trace_minus_baseline",
                             "second_trace_minus_baseline")
    assert result["baseline_repeat"]["scalar_delta"]["torque_p99"] == pytest.approx(.03)
    assert result["trace_repeat"]["scalar_delta"]["torque_p99"] == pytest.approx(.01)
    assert result["first_trace_minus_baseline"]["scalar_delta"]["torque_p99"] == pytest.approx(.01)
    assert result["second_trace_minus_baseline"]["scalar_delta"]["torque_p99"] == pytest.approx(-.01)
    rows[2]["mode"] = "baseline"
    with pytest.raises(ValueError, match="ABBA"):
        probe.comparisons(rows)


def timing_fixture(tmp_path, monkeypatch):
    timing = probe.timing
    input_root = tmp_path / "training-input"
    eval_path = input_root / "evaluate" / "result.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text('{"retained":true}\n')
    monkeypatch.setattr(timing, "INPUT", input_root)
    prior = {"checkpoint_sha256": {"control": CHECKPOINT}, "decisions": {"control": {"ok": True}}}
    raw_dir = tmp_path / "evaluation"
    raw_dir.mkdir(exist_ok=True)
    raw_cases = []
    for index in range(8):
        name = f"{probe.timing.ARMS[index // 4]}-s{probe.timing.SEED}-c{index % 4}.json"
        path = raw_dir / name
        path.write_text(json.dumps({"case": index}))
        raw_cases.append({"complete": True, "raw_json": name,
                          "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    evidence = dict(
        source=probe.TIMING_SOURCE, protocol=timing.PROTOCOL, input_source=timing.INPUT_SOURCE,
        checkpoint_sha256=prior["checkpoint_sha256"], original_decisions=prior["decisions"],
        evaluation_sha256=timing.base.sha256(eval_path), diagnostic_case_count=8,
        cases=raw_cases,
        decision="diagnostic-only-not-admission", policy_acceptance=False,
        simulator_qualified=False, physical_motion_authorized=False,
    )
    aggregate_path = tmp_path / "evaluation" / "result.json"
    aggregate_path.write_text(json.dumps(evidence))
    launch_path = aggregate_path.parent / "launch.json"
    launch_path.write_text(json.dumps(dict(source=probe.TIMING_SOURCE, protocol=timing.PROTOCOL,
                                           runtime=timing.RUNTIME)))
    monkeypatch.setattr(probe, "TIMING_INPUT", aggregate_path)
    monkeypatch.setattr(probe, "TIMING_SHA", hashlib.sha256(aggregate_path.read_bytes()).hexdigest())
    monkeypatch.setattr(probe, "TIMING_LAUNCH_SHA", hashlib.sha256(launch_path.read_bytes()).hexdigest())
    return prior, evidence, aggregate_path


def test_validate_timing_pins_full_completed_nonadmitting_timing_identity(tmp_path, monkeypatch):
    prior, _, _ = timing_fixture(tmp_path, monkeypatch)
    evidence = probe.validate_timing(prior)
    assert evidence["source"] == probe.TIMING_SOURCE
    launch = json.loads((Path(probe.TIMING_INPUT).parent / "launch.json").read_text())
    assert launch["runtime"] == probe.timing.RUNTIME
    assert evidence["diagnostic_case_count"] == 8
    assert all(row["complete"] for row in evidence["cases"])
    assert evidence["decision"] == "diagnostic-only-not-admission"
    assert evidence["policy_acceptance"] is evidence["simulator_qualified"] is False
    assert evidence["physical_motion_authorized"] is False


@pytest.mark.parametrize("field,value", [
    ("source", "b" * 40), ("protocol", "wrong-protocol"),
    ("input_source", "c" * 40), ("checkpoint_sha256", {"control": "d" * 64}),
    ("original_decisions", {"control": {"ok": False}}), ("decision", "admitted"),
    ("policy_acceptance", True), ("simulator_qualified", True),
    ("physical_motion_authorized", True),
])
def test_validate_timing_rejects_identity_decision_admission_or_runtime_drift(
    tmp_path, monkeypatch, field, value
):
    prior, evidence, path = timing_fixture(tmp_path, monkeypatch)
    evidence[field] = value
    path.write_text(json.dumps(evidence))
    monkeypatch.setattr(probe, "TIMING_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        probe.validate_timing(prior)


def test_validate_timing_rejects_launch_runtime_and_raw_case_hash_drift(tmp_path, monkeypatch):
    prior, _, path = timing_fixture(tmp_path, monkeypatch)
    launch_path = path.parent / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch["runtime"] = {"mjlab": "wrong"}
    launch_path.write_text(json.dumps(launch))
    monkeypatch.setattr(probe, "TIMING_LAUNCH_SHA", hashlib.sha256(launch_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        probe.validate_timing(prior)

    prior, evidence, path = timing_fixture(tmp_path, monkeypatch)
    raw = path.parent / evidence["cases"][0]["raw_json"]
    raw.write_text("mutated")
    with pytest.raises(ValueError):
        probe.validate_timing(prior)


def fake_main_dependencies(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_499.pt"
    checkpoint.write_bytes(b"weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(probe.base, "ROOT", tmp_path)
    monkeypatch.setattr(probe.retained, "_git_identity", lambda source: (source, ""))
    monkeypatch.setattr(probe.timing.audit, "validate_invocation", lambda *args, **kwargs: None)
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: NS(returncode=0))
    monkeypatch.setattr(probe.retained, "_validate_gpu_lease", lambda: lease)
    monkeypatch.setattr(probe.retained, "_runtime", lambda: probe.timing.RUNTIME)
    monkeypatch.setattr(probe, "version", lambda _: "5.0.1")
    monkeypatch.setattr(probe.retained, "shared_host", lambda: "cpu-test-host")
    monkeypatch.setattr(probe.retained, "_health", lambda _path: None)
    monkeypatch.setattr(probe.timing, "validated_input", lambda *_args: {
        "checkpoint_sha256": {"control": digest}, "decisions": {"control": {"foundation": "same"}},
    })
    monkeypatch.setattr(probe, "validate_timing", lambda _prior: {"ok": True})
    monkeypatch.setattr(probe.motor, "checkpoint_path", lambda *_args: checkpoint)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *_args, **_kwargs: None)
    from mjlab.utils import torch as mjlab_torch
    monkeypatch.setattr(mjlab_torch, "configure_torch_backends", lambda: None)
    lease.closed = False
    return checkpoint, digest


class FakeLease:
    closed = False

    def close(self):
        self.closed = True


lease = FakeLease()


def test_main_runs_exact_four_ABBA_cases_with_diagnostics_and_trace_flags(tmp_path, monkeypatch):
    fake_main_dependencies(monkeypatch, tmp_path)
    calls = []

    def evaluate(checkpoint, seed, speed, yaw, health, **kwargs):
        mode_trace = kwargs["observation_trace"]
        calls.append((seed, speed, yaw, kwargs))
        return replay_case(torque=.2 + len(calls) * .01, trace=mode_trace)

    monkeypatch.setattr(probe.base, "evaluate_case", evaluate)
    monkeypatch.setattr(probe.timing.capture, "summarize", lambda trace: {"settled_steps": trace["steps"]})
    assert probe.main(["--source", "1" * 40, "--run-id", "cpu-probe", "--evaluation-sha256", "2" * 64]) == 0

    assert [(speed, yaw) for _, speed, yaw, _ in calls] == [probe.CASE] * 4
    assert [kwargs["observation_trace"] for _, _, _, kwargs in calls] == [False, True, True, False]
    assert all(kwargs["diagnostics"] is kwargs["angular_diagnostics"] is True
               for _, _, _, kwargs in calls)
    assert all(seed == probe.timing.SEED for seed, _, _, _ in calls)
    output = tmp_path / "artifacts" / "evaluations" / "cpu-probe"
    assert json.loads((output / "launch.json").read_text())["order"] == list(probe.ORDER)
    report = json.loads((output / "result.json").read_text())
    assert [row["mode"] for row in report["cases"]] == list(probe.ORDER)
    assert len(report["comparisons"]) == 4
    assert report["policy_acceptance"] is report["physical_motion_authorized"] is False
    assert lease.closed


def test_main_retains_trace_json_before_failure_and_stops_remaining_ABBA_runs(tmp_path, monkeypatch):
    fake_main_dependencies(monkeypatch, tmp_path)
    calls = []

    def evaluate(checkpoint, seed, speed, yaw, health, **kwargs):
        calls.append(kwargs["observation_trace"])
        return replay_case(trace=kwargs["observation_trace"])

    monkeypatch.setattr(probe.base, "evaluate_case", evaluate)

    def fail_summary(_trace):
        raise ValueError("trace validation failed")

    monkeypatch.setattr(probe.timing.capture, "summarize", fail_summary)
    with pytest.raises(ValueError, match="trace validation failed"):
        probe.main(["--source", "1" * 40, "--run-id", "cpu-probe-failure", "--evaluation-sha256", "2" * 64])

    output = tmp_path / "artifacts" / "evaluations" / "cpu-probe-failure"
    raw_path = output / "1-trace.json"
    assert raw_path.is_file()
    raw = json.loads(raw_path.read_text())
    assert raw["case"]["observation_timing_trace"] == {"protocol": "trace-only-payload", "steps": 240}
    assert calls == [False, True]
    assert not (output / "2-trace.json").exists()
    assert json.loads((output / "failure.json").read_text())["error"] == "trace validation failed"
    assert lease.closed
