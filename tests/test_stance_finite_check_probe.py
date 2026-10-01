"""Source-bound checker probe contracts; CPU-only inert fixtures, no runtime."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import stance_finite_check_probe as probe
from mjlab_microduck import stance_solved_field_check as checker


SOURCE = "1" * 40
RUNNER_SHA = "2" * 64
CHECKER_SHA = hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest()
LAUNCH_SHA = "3" * 64


def make_rows():
    seconds = [1.0, 0.8, 0.9, 1.1, 1.2, 0.7]
    return [dict(block=i, variant=variant, warmup_checks=probe.WARMUP,
                 measured_checks=probe.ITERATIONS, seconds=float(seconds[i]))
            for i, variant in enumerate(probe.ORDER)]


def tensor_values():
    return {name: torch.arange(1, 65, dtype=torch.float32).reshape(64, 1)
            for name in checker.FIELDS}


def data_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def fake_host_io(monkeypatch):
    monkeypatch.setattr(probe.host, "digest", data_sha)
    monkeypatch.setattr(probe.host.supervisor, "file_bytes",
                        lambda path, limit=64 * 1024 * 1024, **_: Path(path).read_bytes())
    monkeypatch.setattr(probe.host.supervisor, "parse", lambda raw: json.loads(raw))
    monkeypatch.setattr(probe.host.supervisor, "hex_id",
                        lambda value, size: (_ for _ in ()).throw(ValueError("bad hex id"))
                        if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{size}}}", value) is None
                        else None)


def fake_plan(monkeypatch, deadline=10_000, runner_sha=RUNNER_SHA, checker_sha=CHECKER_SHA):
    monkeypatch.setattr(probe.execution, "PROFILE", {"name": probe.execution.WSL})
    monkeypatch.setattr(probe.host, "identity", lambda source: {"source": probe.RUNTIME_SOURCE,
                        "host": "fixture", "cuda_initialized": False})
    monkeypatch.setattr(probe, "tool_identity", lambda source, runner, checker_hash: {
        "source": source, "runtime_source": probe.RUNTIME_SOURCE,
        "files": {"runner.py": {"path": probe.TOOL_PATH, "sha256": runner, "bytes": 12},
                  "checker.py": {"path": probe.CHECK_PATH, "sha256": checker_hash, "bytes": 13}},
    })
    checker_bytes = Path(checker.__file__).read_bytes()
    monkeypatch.setattr(probe.subprocess, "check_output", lambda argv, **kwargs: checker_bytes)
    return probe.plan(SOURCE, runner_sha, checker_sha, deadline)


def test_fixed_declaration_is_cpu_checker_only_and_non_admitting(monkeypatch):
    plan = fake_plan(monkeypatch)
    assert plan["protocol"] == "football-b1n-wsl-finite-check-probe-v1"
    assert plan["tool"]["source"] == SOURCE
    assert plan["tool"]["runtime_source"] == plan["inputs"]["source"] == probe.RUNTIME_SOURCE
    assert plan["order"] == list(probe.ORDER) == ["legacy", "packed", "packed", "legacy", "legacy", "packed"]
    assert (plan["child_seconds"], plan["service_seconds"], plan["closeout_seconds"]) == (120, 180, 600)
    assert plan["warmup_checks"] == 100 and plan["measured_checks"] == 1000 and plan["worlds"] == 64
    assert plan["policy_inferences"] == plan["physics_steps"] == plan["optimizer_updates"] == 0
    assert plan["forward_graph"] is False
    assert all(value is False for key, value in plan.items() if key in probe.NO_ADMISSION)
    assert plan["cpu_preflight"]["fault_count"] == len(checker.FIELDS) * 3 == 33
    assert plan["cpu_preflight"]["physics_executed"] is False
    assert plan["physics_step_meaning"] == "Euler-integration-ticks-not-initialization-forward"


def test_tool_identity_requires_exact_git_bytes_and_executing_runner(monkeypatch, tmp_path):
    runner = b"pinned runner source"
    archived_checker = b"pinned checker source"
    runner_sha = hashlib.sha256(runner).hexdigest()
    checker_sha = hashlib.sha256(archived_checker).hexdigest()
    monkeypatch.setattr(probe.host, "ROOT", tmp_path)
    monkeypatch.setattr(probe.host.supervisor, "hex_id", lambda *_: None)
    monkeypatch.setattr(subprocess, "check_output", lambda argv, **_: runner if argv[-1].endswith(probe.TOOL_PATH)
                        else archived_checker)
    monkeypatch.setattr(probe.host, "digest", lambda _path: runner_sha)
    result = probe.tool_identity(SOURCE, runner_sha, checker_sha)
    assert result["files"]["runner.py"]["sha256"] == runner_sha
    assert result["files"]["checker.py"]["sha256"] == checker_sha
    with pytest.raises(ValueError, match="independently pinned tool Git bytes"):
        probe.tool_identity(SOURCE, "0" * 64, checker_sha)
    monkeypatch.setattr(probe.host, "digest", lambda _path: "f" * 64)
    with pytest.raises(ValueError, match="executing pinned runner bytes"):
        probe.tool_identity(SOURCE, runner_sha, checker_sha)


@pytest.mark.parametrize("delta,valid", [(781, True), (780, False), (3600, True), (3601, False)])
def test_window_requires_bounded_service_and_closeout(delta, valid):
    if valid:
        probe.check_window(20_000 + delta, now=20_000)
    else:
        with pytest.raises(ValueError, match="bounded service and closeout"):
            probe.check_window(20_000 + delta, now=20_000)


@pytest.mark.parametrize("deadline,now", [(True, 0), (1000.0, 0), (1000, True), (float("nan"), 0)])
def test_window_rejects_boolean_and_nonintegral_deadlines(deadline, now):
    with pytest.raises(ValueError):
        probe.check_window(deadline, now=now)


def test_prepare_archives_only_mocked_bytes_without_native_compilation(monkeypatch, tmp_path):
    runner, archived_checker = b"archived runner", Path(checker.__file__).read_bytes()
    runner_sha = hashlib.sha256(runner).hexdigest()
    checker_sha = CHECKER_SHA
    plan = fake_plan(monkeypatch, deadline=50_000, runner_sha=runner_sha, checker_sha=checker_sha)
    output = tmp_path / "fresh-output"
    monkeypatch.setattr(probe, "check_window", lambda *_: None)
    monkeypatch.setattr(probe, "plan", lambda *args: plan)
    monkeypatch.setattr(probe.subprocess, "check_output", lambda argv, **kwargs:
                        runner if argv[-1].endswith(probe.TOOL_PATH) else archived_checker)
    monkeypatch.setattr(probe, "output_path", lambda _source: output)
    compiled_sentinel = object()
    monkeypatch.setattr(probe.plant, "build_entity", lambda: SimpleNamespace(compile=lambda: compiled_sentinel))
    monkeypatch.setattr(probe.plant, "runtime_bytes", lambda source, model:
                        (source.encode() + b":" + str(model is compiled_sentinel).encode()))
    monkeypatch.setattr(probe.smoke, "write_bytes", lambda path, data: Path(path).write_bytes(data))
    monkeypatch.setattr(probe.host.supervisor, "write_json", write_json)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    # Preparing the archival runner may compile source into a receipt, but this
    # fixture ensures no real build_entity/compile implementation is reachable.
    result = probe.prepare(SOURCE, runner_sha, checker_sha, 50_000)
    assert Path(result["output"]) == output
    assert {item.name for item in output.iterdir()} == {"runner.py", "checker.py", "runtime.json", "launch.json"}
    assert result["launch_sha256"] == data_sha(output / "launch.json")


def test_checked_refuses_tampered_archived_tool_before_runtime_replay(monkeypatch, tmp_path):
    fake_host_io(monkeypatch)
    launch = fake_plan(monkeypatch)
    output = tmp_path / "attempt"
    output.mkdir()
    runner, archived_checker = b"runner", b"checker"
    (output / "runner.py").write_bytes(runner)
    (output / "checker.py").write_bytes(archived_checker)
    runtime = b'{"source":"' + probe.RUNTIME_SOURCE.encode() + b'","plant":{}}\n'
    (output / "runtime.json").write_bytes(runtime)
    runner_hash, checker_hash = data_sha(output / "runner.py"), data_sha(output / "checker.py")
    launch["tool"]["files"]["runner.py"].update(sha256=runner_hash, bytes=len(runner))
    launch["tool"]["files"]["checker.py"].update(sha256=checker_hash, bytes=len(archived_checker))
    launch["runtime_sha256"] = hashlib.sha256(runtime).hexdigest()
    write_json(output / "launch.json", launch)
    launch_sha = data_sha(output / "launch.json")
    monkeypatch.setattr(probe, "output_path", lambda _source: output)
    monkeypatch.setattr(probe, "plan", lambda *_: launch)
    monkeypatch.setattr(probe.plant, "checked_runtime", lambda *_: {"fixture": True})
    assert probe.checked(SOURCE, RUNNER_SHA, CHECKER_SHA, launch_sha) == launch
    (output / "checker.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="retained tool hash"):
        probe.checked(SOURCE, RUNNER_SHA, CHECKER_SHA, launch_sha)


def test_errors_injects_all_33_faults_without_mutating_source_tensors():
    values = tensor_values()
    before = {name: tensor.view(torch.uint8).clone() for name, tensor in values.items()}
    faults = probe.errors(checker, values)
    assert len(faults) == len(checker.FIELDS) * 3 == 33
    assert [(row["field"], row["fault"]) for row in faults] == [
        (name, label) for name in checker.FIELDS
        for label in ("not-a-number", "positive-infinity", "negative-infinity")]
    assert all(row["error"] == "nonfinite solved stance field: " + row["field"] for row in faults)
    assert all(torch.equal(before[name], values[name].view(torch.uint8)) for name in values)


def test_errors_fault_injection_handles_noncontiguous_source_without_mutation():
    values = tensor_values()
    storage = torch.arange(64 * 2, dtype=torch.float32).reshape(64, 2)
    values["qpos"] = storage[:, ::2]
    before = storage.view(torch.uint8).clone()
    faults = probe.errors(checker, values)
    assert len(faults) == 33
    assert torch.equal(storage.view(torch.uint8), before)


def test_decide_reports_three_paired_blocks_and_only_static_scope():
    result = probe.decide(make_rows())
    assert result["decision"] == "finite-check-probe-complete-not-runtime-equivalence"
    assert [row["blocks"] for row in result["paired_blocks"]] == [[0, 1], [2, 3], [4, 5]]
    assert result["packed_faster_in_all_three_pairs"] is True
    assert result["scope"] == "checker-only-including-fresh-packing-not-end-to-end-collection"
    assert all(value is False for key, value in result.items() if key in probe.NO_ADMISSION)


def test_decide_rejects_overflowing_derived_timing_ratio():
    rows = make_rows()
    rows[0]["seconds"] = 1e-320
    rows[1]["seconds"] = 1e308
    with pytest.raises(ValueError, match="finite derived timing ratio"):
        probe.decide(rows)


@pytest.mark.parametrize("damage", ["dropped", "reordered", "bool-block", "bool-warmup",
                                     "float-count", "bool-measured", "integer-seconds",
                                     "nan-seconds", "infinite-seconds", "zero-seconds", "extra-key"])
def test_decide_rejects_malformed_dropped_reordered_or_nonfinite_blocks(damage):
    rows = make_rows()
    if damage == "dropped":
        rows.pop()
    elif damage == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif damage == "bool-block":
        rows[0]["block"] = True
    elif damage == "bool-warmup":
        rows[0]["warmup_checks"] = True
    elif damage == "float-count":
        rows[0]["warmup_checks"] = float(probe.WARMUP)
    elif damage == "bool-measured":
        rows[0]["measured_checks"] = True
    elif damage == "integer-seconds":
        rows[0]["seconds"] = 1
    elif damage == "nan-seconds":
        rows[0]["seconds"] = float("nan")
    elif damage == "infinite-seconds":
        rows[0]["seconds"] = float("inf")
    elif damage == "zero-seconds":
        rows[0]["seconds"] = 0.0
    else:
        rows[0]["extra"] = 1
    with pytest.raises(ValueError):
        probe.decide(rows)


def _evidence_bundle(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    names = {"runner.py", "checker.py", "runtime.json", "launch.json", "inputs.pt",
             "summary.json", "child.log", *("block-" + str(i) + ".json" for i in range(6))}
    runner = b"retained fixture runner"
    checker_bytes = Path(checker.__file__).read_bytes()
    (root / "runner.py").write_bytes(runner)
    (root / "checker.py").write_bytes(checker_bytes)
    runtime = {"source": probe.RUNTIME_SOURCE, "plant": {"fixture": "inert"}}
    write_json(root / "runtime.json", runtime)
    runtime_sha = data_sha(root / "runtime.json")
    launch = {
        "protocol": probe.PROTOCOL,
        "tool": {"source": SOURCE, "runtime_source": probe.RUNTIME_SOURCE,
                 "files": {
                     "runner.py": {"path": probe.TOOL_PATH,
                                   "sha256": data_sha(root / "runner.py"), "bytes": len(runner)},
                     "checker.py": {"path": probe.CHECK_PATH,
                                    "sha256": data_sha(root / "checker.py"), "bytes": len(checker_bytes)},
                 }},
        "inputs": {"source": probe.RUNTIME_SOURCE},
        "deadline_unix": 50_000,
        "runtime_sha256": runtime_sha,
        "child_seconds": probe.CHILD_SECONDS, "service_seconds": probe.SERVICE_SECONDS,
        "closeout_seconds": probe.CLOSEOUT, "memory_max_bytes": 6 * 1024**3,
        "cpu_quota_per_sec_usec": 2_000_000, "nice": 10, "worlds": 64,
        "order": list(probe.ORDER), "warmup_checks": probe.WARMUP,
        "measured_checks": probe.ITERATIONS, "forward_graph": False,
        "policy_inferences": 0, "physics_steps": 0, "optimizer_updates": 0,
        "cpu_preflight": probe.cpu_preflight(checker),
        "physics_step_meaning": "Euler-integration-ticks-not-initialization-forward",
        **probe.NO_ADMISSION,
    }
    write_json(root / "launch.json", launch)
    launch_sha = data_sha(root / "launch.json")
    values = tensor_values()
    input_buffer = io.BytesIO()
    torch.save(values, input_buffer)
    input_raw = input_buffer.getvalue()
    (root / "inputs.pt").write_bytes(input_raw)
    rows = make_rows()
    for index, row in enumerate(rows):
        write_json(root / f"block-{index}.json", row)
    (root / "child.log").write_bytes(b"")
    summary = probe.decide(rows)
    summary.update(faults=probe.errors(checker, values), input_bits_unchanged=True,
                   physics_steps=0, policy_inferences=0, optimizer_updates=0,
                   worlds=64, device="cuda:0", input_sha256=hashlib.sha256(input_raw).hexdigest())
    write_json(root / "summary.json", summary)
    files = {name: data_sha(root / name) for name in names}
    report = {"files": files, "protocol": probe.PROTOCOL, "launch_sha256": launch_sha,
              "decision": "finite-check-probe-complete-not-runtime-equivalence",
              "child": {"returncode": 0, "elapsed_s": 1.25}, **probe.NO_ADMISSION}
    write_json(root / "report.json", report)
    return root, data_sha(root / "report.json"), launch_sha, data_sha(root / "runner.py"), data_sha(root / "checker.py")


def _mock_evidence_host(monkeypatch, root):
    fake_host_io(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(probe.host, "check_log", lambda path: None)
    monkeypatch.setattr(probe.host.supervisor, "file_bytes",
        lambda path, limit=64 * 1024 * 1024, **_: Path(path).read_bytes())


def test_verify_evidence_cpu_replays_inert_tensors_without_authority(tmp_path, monkeypatch):
    root, report_sha, launch_sha, runner_sha, checker_sha = _evidence_bundle(tmp_path)
    _mock_evidence_host(monkeypatch, root)
    result = probe.verify_evidence(root, report_sha, launch_sha, runner_sha, checker_sha)
    assert result["summary"]["decision"] == "finite-check-probe-complete-not-runtime-equivalence"
    assert result["summary"]["policy_inferences"] == result["summary"]["physics_steps"] == 0
    assert result["live_host_rechecked"] is result["cuda_run_authenticated"] is False
    assert all(result[key] is False for key in probe.NO_ADMISSION)


def test_verify_evidence_rejects_nonfinite_cpu_snapshot_after_valid_rehash(tmp_path, monkeypatch):
    root, _, launch_sha, runner_sha, checker_sha = _evidence_bundle(tmp_path)
    _mock_evidence_host(monkeypatch, root)
    values = tensor_values()
    values[checker.FIELDS[3]][0, 0] = float("nan")
    buffer = io.BytesIO()
    torch.save(values, buffer)
    (root / "inputs.pt").write_bytes(buffer.getvalue())
    report = json.loads((root / "report.json").read_text())
    report["files"]["inputs.pt"] = data_sha(root / "inputs.pt")
    write_json(root / "report.json", report)
    with pytest.raises(ValueError, match="nonfinite solved stance field"):
        probe.verify_evidence(root, data_sha(root / "report.json"), launch_sha, runner_sha, checker_sha)


@pytest.mark.parametrize("damage", ["unexpected-file", "bad-tool-hash", "bad-launch-hash",
                                     "reordered-block", "bad-summary", "missing-preflight",
                                     "tampered-preflight"])
def test_verify_evidence_rejects_inventory_hash_and_replay_mismatch(tmp_path, monkeypatch, damage):
    root, _, launch_sha, runner_sha, checker_sha = _evidence_bundle(tmp_path)
    _mock_evidence_host(monkeypatch, root)
    report = json.loads((root / "report.json").read_text())
    if damage == "unexpected-file":
        (root / "extra.json").write_text("{}")
    elif damage == "bad-tool-hash":
        (root / "runner.py").write_bytes(b"changed")
        report["files"]["runner.py"] = data_sha(root / "runner.py")
        write_json(root / "report.json", report)
    elif damage == "bad-launch-hash":
        bad_launch = "4" * 64
        report["launch_sha256"] = bad_launch
        write_json(root / "report.json", report)
    elif damage in ("missing-preflight", "tampered-preflight"):
        launch = json.loads((root / "launch.json").read_text())
        if damage == "missing-preflight":
            del launch["cpu_preflight"]
        else:
            launch["cpu_preflight"]["fault_count"] = 32
        write_json(root / "launch.json", launch)
        launch_sha = data_sha(root / "launch.json")
        report["launch_sha256"] = launch_sha
        report["files"]["launch.json"] = launch_sha
        write_json(root / "report.json", report)
    elif damage == "reordered-block":
        row = json.loads((root / "block-0.json").read_text())
        row["variant"] = "packed"
        write_json(root / "block-0.json", row)
        report["files"]["block-0.json"] = data_sha(root / "block-0.json")
        write_json(root / "report.json", report)
    else:
        summary = json.loads((root / "summary.json").read_text())
        summary["decision"] = "false-pass"
        write_json(root / "summary.json", summary)
        report["files"]["summary.json"] = data_sha(root / "summary.json")
        write_json(root / "report.json", report)
    with pytest.raises((ValueError, KeyError)):
        probe.verify_evidence(root, data_sha(root / "report.json"), launch_sha, runner_sha, checker_sha)


def test_supervisor_replays_cpu_evidence_before_publishing_passed_report(tmp_path, monkeypatch):
    launch = fake_plan(monkeypatch, deadline=50_000)
    launch["runtime_sha256"] = "4" * 64
    root = tmp_path / "supervised"
    root.mkdir()
    for name in ("runner.py", "checker.py", "runtime.json", "launch.json"):
        (root / name).write_bytes(b"fixture")
    write_json(root / "launch.json", launch)
    replay_calls = []
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe, "checked", lambda *_: launch)
    monkeypatch.setattr(probe, "check_window", lambda *_: None)
    monkeypatch.setattr(probe, "service_name", lambda _source: "fixture.service")
    monkeypatch.setattr(probe.host, "read", lambda *args: {
        "MainPID": str(os.getpid()), "RuntimeMaxUSec": "3min", "KillMode": "control-group",
        "ActiveState": "active", "MemoryMax": str(6 * 1024**3),
        "CPUQuotaPerSecUSec": "2s", "Nice": "10",
    }[args[-2]])
    monkeypatch.setattr(probe.host, "wait_idle", lambda: {"idle": True})
    monkeypatch.setattr(probe.host, "check_log", lambda *_: None)
    monkeypatch.setattr(probe, "load_checker", lambda *_: checker)
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    class Lease:
        def __enter__(self):
            return 9

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(probe.host.supervisor, "gpu_lease", lambda: Lease())
    monkeypatch.setattr(probe.host.supervisor, "child_environment", lambda: {"CUDA_VISIBLE_DEVICES": "0"})

    def child(*args, **kwargs):
        for name in ("inputs.pt", "summary.json", "child.log", *(f"block-{i}.json" for i in range(6))):
            (root / name).write_bytes(b"mock child artifact")
        return {"returncode": 0, "elapsed_s": 1.0}

    monkeypatch.setattr(probe.host.supervisor, "supervised_process", child)
    def replay(_root, report, *_args, **kwargs):
        replay_calls.append(report["decision"])
        assert report["decision"] == "finite-check-probe-complete-not-runtime-equivalence"
        assert set(report["files"]) == {"runner.py", "checker.py", "runtime.json", "launch.json",
            "inputs.pt", "summary.json", "child.log", *(f"block-{i}.json" for i in range(6))}

    monkeypatch.setattr(probe, "replay_payload", replay)
    monkeypatch.setattr(probe.host.supervisor, "write_json", write_json)
    probe.supervise(SOURCE, RUNNER_SHA, CHECKER_SHA, "5" * 64)
    assert replay_calls == ["finite-check-probe-complete-not-runtime-equivalence"]
    report = json.loads((root / "report.json").read_text())
    assert report["decision"] == "finite-check-probe-complete-not-runtime-equivalence"


def test_supervisor_overrides_pass_when_cpu_replay_fails(tmp_path, monkeypatch):
    # Reuse the same fully mocked service sequence; the replay failure must
    # write a failed report before propagating the error.
    launch = fake_plan(monkeypatch, deadline=50_000)
    launch["runtime_sha256"] = "4" * 64
    root = tmp_path / "supervised"
    root.mkdir()
    for name in ("runner.py", "checker.py", "runtime.json", "launch.json"):
        (root / name).write_bytes(b"fixture")
    write_json(root / "launch.json", launch)
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe, "checked", lambda *_: launch)
    monkeypatch.setattr(probe, "check_window", lambda *_: None)
    monkeypatch.setattr(probe, "service_name", lambda _source: "fixture.service")
    monkeypatch.setattr(probe.host, "read", lambda *args: {
        "MainPID": str(os.getpid()), "RuntimeMaxUSec": "3min", "KillMode": "control-group",
        "ActiveState": "active", "MemoryMax": str(6 * 1024**3),
        "CPUQuotaPerSecUSec": "2s", "Nice": "10",
    }[args[-2]])
    monkeypatch.setattr(probe.host, "wait_idle", lambda: {"idle": True})
    monkeypatch.setattr(probe.host, "check_log", lambda *_: None)
    monkeypatch.setattr(probe, "load_checker", lambda *_: checker)
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    class Lease:
        def __enter__(self): return 9
        def __exit__(self, *_): return False

    monkeypatch.setattr(probe.host.supervisor, "gpu_lease", lambda: Lease())
    monkeypatch.setattr(probe.host.supervisor, "child_environment", lambda: {"CUDA_VISIBLE_DEVICES": "0"})
    monkeypatch.setattr(probe.host.supervisor, "supervised_process", lambda *a, **k: {
        "returncode": 0, "elapsed_s": 1.0})
    monkeypatch.setattr(probe, "replay_payload", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad CPU replay")))
    monkeypatch.setattr(probe.host.supervisor, "write_json", write_json)
    with pytest.raises(ValueError, match="bad CPU replay"):
        probe.supervise(SOURCE, RUNNER_SHA, CHECKER_SHA, "5" * 64)
    report = json.loads((root / "report.json").read_text())
    assert report["decision"] == "failed"
    assert report["error"] == "bad CPU replay"
