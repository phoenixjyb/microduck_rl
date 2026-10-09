"""Closed measured-profile/receiver mocks; never allocate or execute physics."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from mjlab_microduck import ada_measured_gpu_boundary as p
from test_ada_saved_pose_response import topology
from test_ada_saved_pose_response_receiver import packet, SOURCE


def test_import_and_help_inert():
    code = "import sys;from mjlab_microduck import ada_measured_gpu_boundary;assert not {'numpy','torch','warp','mujoco','mujoco_warp'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)
    subprocess.run([sys.executable, "-m", p.MODULE, "--help"], check=True, timeout=15, capture_output=True)


def test_new_profile_keeps_existing_bounds_flags_and_closed_dispatch():
    assert p.base.selected_profile(p) is p
    assert p.BOUNDS == p.base.BOUNDS and p.FLAGS == p.base.FLAGS
    assert all(v is False for v in p.FLAGS.values())
    value = NS(**vars(p)); value.MODULE = "arbitrary.runner"
    with pytest.raises(ValueError): p.base.selected_profile(value)
    value.MODULE = p.MODULE; value.BOUNDS = dict(p.BOUNDS, child_seconds=121)
    with pytest.raises(ValueError): p.base.selected_profile(value)
    assert p.specification()["fixture_warp_kinematics_calls"] == 1
    assert p.specification()["same_bank_cpu_solver_control_executed"] is False


@pytest.mark.parametrize("mode,cpu,cuda,arch,ordinal,ok", [
    ("cuda", False, True, 89, 0, True), ("cpu", True, False, 89, 0, True),
    ("cuda", True, False, 89, 0, False), ("cpu", False, True, 89, 0, False),
    ("cuda", False, True, 90, 0, False), ("cuda", False, True, 89, 1, False),
    ("cuda", True, True, 89, 0, False), ("cuda", False, False, 89, 0, False),
    ("arbitrary", False, True, 89, 0, False)])
def test_explicit_device_guard_does_not_broaden_cpu_default(monkeypatch, mode, cpu, cuda, arch, ordinal, ok):
    native, model, data = topology()
    data.qpos.device = NS(is_cpu=cpu, is_cuda=cuda, arch=arch, ordinal=ordinal)
    calls = []; monkeypatch.setattr(p.saved, "dispatch_guard", lambda *args: calls.append(True) or {})
    if ok: assert p.response.topology_guard(native, model, data, device_kind=mode) == {} and calls == [True]
    else:
        with pytest.raises(ValueError): p.response.topology_guard(native, model, data, device_kind=mode)
        assert calls == []
    if cuda:
        with pytest.raises(ValueError): p.response.topology_guard(native, model, data)


@pytest.mark.parametrize("damage", ["none", "cpu", "arch", "ordinal", "name", "count"])
def test_all461_arrays_on_actual_selected_device(monkeypatch, damage):
    device = type("Device", (NS,), {"__str__": lambda s: "cuda:0"})(is_cuda=True, is_cpu=False, arch=89, ordinal=0)
    values = {str(i): NS(device=device) for i in range(461)}
    if damage == "cpu": device.is_cpu = True
    if damage == "arch": device.arch = 90
    if damage == "ordinal": device.ordinal = 1
    if damage == "name": device.__class__.__str__ = lambda s: "cuda:1"
    if damage == "count": values.pop("0")
    monkeypatch.setattr(p.saved.p.p, "all_arrays", lambda v: values)
    if damage == "none": p.require_device(NS(), NS())
    else:
        with pytest.raises(ValueError): p.require_device(NS(), NS())


def fixture(tmp_path, monkeypatch):
    arrays, layout, prior, banks, _ = packet()
    # Deliberately synthetic trees; actual frozen schema tested by allocation suite.
    manifest = dict(synthetic="complete-test-tree")
    monkeypatch.setattr(p, "STATIC_SHA", p.packet.digest(manifest))
    monkeypatch.setattr(p.allocation, "frozen_static_schema", lambda v: None)
    prior["child"]["plant"]["options"].update(enableflags=0)
    reference = dict(static_manifest=manifest, model_array_sha256={"/model/array" + str(i): "a" * 64 for i in range(347)},
                     plant=prior["child"]["plant"], options={"synthetic": True})
    monkeypatch.setattr(p, "allocation_reference", lambda root: reference)
    monkeypatch.setattr(p.saved.p.prior, "authenticated_banks", lambda root: (prior, banks))
    audit = {"synthetic": "frozen-audit"}; monkeypatch.setattr(p.response.source, "audit", lambda: audit)
    expansion = {"synthetic": "expansion"}; monkeypatch.setattr(p.allocation, "expansion_source", lambda: expansion)
    module = b"committed source"; monkeypatch.setattr(p.host, "read", lambda *args, **kw: module if kw.get("binary") else "a" * 40)
    analysis = p.packet.analyze_measured(arrays, layout, prior, banks)
    for name, value in zip(p.PAYLOADS, (arrays, manifest, layout, analysis)): p.retain(tmp_path, name, value)
    payloads = [dict(file=name, bytes=(tmp_path / name).stat().st_size, sha256=sha256((tmp_path / name).read_bytes()).hexdigest()) for name in p.PAYLOADS]
    child = dict(specification=deepcopy(p.specification()), gpu_uuid=p.host.GPU, name=p.host.NAME, capability=[8, 9], torch_cuda="12.8", warp_arch=89,
        warp_precompiled_headers=False, plant=reference["plant"], options=reference["options"], model_array_sha256=reference["model_array_sha256"],
        static_manifest_sha256=p.STATIC_SHA, measured_pose_sha256={k: sha256(arrays["before_collision/data/" + k].tobytes()).hexdigest() for k in p.response.POSES},
        in_process_binding_sha256="b" * 64, source_audit_sha256=p.packet.digest(audit), expansion_source=expansion,
        predecessor_report_sha256=p.saved.p.prior.REPORT_SHA256, input_file_sha256=prior["files"], allocation_reference_sha256=p.ALLOCATION_JSON_SHA,
        allocation_payload_sha256=p.ALLOCATION_NPZ_SHA, payloads=payloads, private_cache_was_absent=True,
        private_cache_files={"warp_cache_path/file.ptx": dict(bytes=1, sha256="a" * 64)},
        existing_gpu_executables=[dict(module="synthetic", block_dim=1, opaque_handle="opaque", module_source_options_hash="a" * 64,
            device="cuda:0", meta={}, kernel_hook_count=1, loaded_binary_bytes_bound=False)])
    source = dict(commit=SOURCE, tree="a" * 40, versions=p.host.VERSIONS, module_sha256=sha256(module).hexdigest(),
        lock_sha256=sha256((Path(p.__file__).resolve().parents[2] / "uv.lock").read_bytes()).hexdigest(),
        interpreter="/home/converge/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12",
        prefix=str(p.host.ROOT / ".venv"), diagnostic_sha256=sha256(module).hexdigest(), measured_module_sha256=sha256(module).hexdigest())
    service = dict(unit=p.PREFIX + SOURCE[:12] + ".service", invocation="1" * 32,
        properties=dict(MainPID="123", ActiveState="active", RuntimeMaxUSec="3min", MemoryMax=str(6 * 1024**3), CPUQuotaPerSecUSec="2s", TasksMax="64",
            Nice="10", KillMode="control-group", LimitFSIZE=str(16 * 1024**2), Restart="no", TimeoutStopUSec="10s"))
    baseline = dict(wall_time_unix=1791561000.0, gpu=p.host.parse_gpu(f"{p.host.GPU},{p.host.DRIVER},8.9,16376,961,15232,0,43"))
    launch = dict(specification=p.specification(), source=source, bounds=p.BOUNDS, baseline=baseline, lease=p.LEASE, foreign=p.FOREIGN)
    p.host.write_json(tmp_path / "launch.json", launch); p.host.write_json(tmp_path / "child.json", child)
    (tmp_path / "child.log").write_bytes(b"synthetic")
    report = dict(specification=p.specification(), source=source, service=service, bounds=p.BOUNDS, decision=p.COLLECTION_DECISION,
        telemetry=[baseline, deepcopy(baseline), deepcopy(baseline)], child_exit=0,
        services_before={(scope + ":" + name): "inactive" for scope in ("system", "user") for name in p.host.SERVICES},
        lease=p.LEASE, foreign_before=p.FOREIGN, child=child, services_unchanged=True, foreign_unchanged=True, lease_unchanged=True,
        elapsed_seconds=12.0, files={name: sha256((tmp_path / name).read_bytes()).hexdigest() for name in p.FILES})
    p.host.write_json(tmp_path / "report.json", report)
    return report, child, arrays


def test_complete_portable_receipt_no_runtime(tmp_path, monkeypatch):
    report, child, arrays = fixture(tmp_path, monkeypatch)
    received, actual, layout, _, analysis = p.receive(tmp_path, SOURCE, tmp_path)
    assert received == report and len(actual) == 570 and len(layout) == 114
    assert analysis["same_bank_cpu_solver_control_executed"] is False
    assert all(actual[k].tobytes() == v.tobytes() for k, v in arrays.items())
    monkeypatch.setattr(p.saved, "cache_files", lambda root: child["private_cache_files"])
    p.receive(tmp_path, SOURCE, tmp_path, cache=True)
    monkeypatch.setattr(p.saved, "cache_files", lambda root: {})
    with pytest.raises(ValueError): p.receive(tmp_path, SOURCE, tmp_path, cache=True)


@pytest.mark.parametrize("damage", ["spec", "source", "tree", "versions", "interpreter", "module", "bounds", "service", "pid", "invocation",
    "lease", "foreign", "services", "elapsed", "exit", "files", "flag", "telemetry", "late", "order", "growth", "extra"])
def test_supervisor_receipt_refuses_every_boundary(tmp_path, monkeypatch, damage):
    report, _, _ = fixture(tmp_path, monkeypatch)
    if damage == "spec": report["specification"]["optimizer_steps"] = 1
    if damage == "source": report["source"]["commit"] = "a" * 40
    if damage in ("tree", "versions", "interpreter"): report["source"][damage] = "wrong"
    if damage == "module": report["source"]["measured_module_sha256"] = "c" * 64
    if damage == "bounds": report["bounds"] = dict(report["bounds"], child_seconds=121)
    if damage == "service": report["service"]["properties"]["MemoryMax"] = "0"
    if damage == "pid": report["service"]["properties"]["MainPID"] = "0"
    if damage == "invocation": report["service"]["invocation"] = "foreign"
    if damage == "lease": report["lease"] = dict(p.LEASE, inode=0)
    if damage == "foreign": report["foreign_before"] = []
    if damage == "services": report["services_before"] = {}
    if damage == "elapsed": report["elapsed_seconds"] = 151.0
    if damage == "exit": report["child_exit"] = 1
    if damage == "files": report["files"]["child.log"] = "a" * 64
    if damage == "flag": report["foreign_unchanged"] = 1
    if damage == "telemetry": report["telemetry"][1]["gpu"]["temperature_c"] = 65
    if damage == "late": report["telemetry"][1]["wall_time_unix"] = 1791590000.0
    if damage == "order": report["telemetry"][1]["wall_time_unix"] -= 1
    if damage == "growth": report["telemetry"][1]["gpu"].update(used_mib=3010, free_mib=13000)
    if damage == "extra": report["unknown"] = True
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError): p.receive(tmp_path, SOURCE, tmp_path)


@pytest.mark.parametrize("damage", ["pose", "model", "static", "allocation", "npz", "cache", "cachepath", "device", "binary", "falseflag", "missingpayload", "sourceaudit"])
def test_child_contract_closed(tmp_path, monkeypatch, damage):
    _, child, _ = fixture(tmp_path, monkeypatch)
    if damage == "pose": child["measured_pose_sha256"].pop("site_xpos")
    if damage == "model": child["model_array_sha256"].pop("/model/array0")
    if damage == "static": child["static_manifest_sha256"] = "c" * 64
    if damage == "allocation": child["allocation_payload_sha256"] = "c" * 64
    if damage == "npz": child["payloads"][0]["bytes"] = p.PAYLOAD_MAX + 1
    if damage == "cache": child["private_cache_was_absent"] = 1
    if damage == "cachepath": child["private_cache_files"] = {"../wrong": dict(bytes=1, sha256="a" * 64)}
    if damage == "device": child["existing_gpu_executables"][0]["device"] = "cpu"
    if damage == "binary": child["existing_gpu_executables"][0]["loaded_binary_bytes_bound"] = True
    if damage == "falseflag": child["specification"]["flags"]["training_authorized"] = 0
    if damage == "missingpayload": child["payloads"].pop()
    if damage == "sourceaudit": child["source_audit_sha256"] = "wrong"
    with pytest.raises(ValueError): p.validate_child(child)


@pytest.mark.parametrize("damage", ["array", "state", "pose", "analysis", "layout", "statics", "symlink", "extra"])
def test_payload_semantics_not_just_outer_hash(tmp_path, monkeypatch, damage):
    report, child, arrays = fixture(tmp_path, monkeypatch)
    if damage in ("array", "state", "pose"):
        if damage == "array": arrays.pop("before_solve/data/qacc")
        if damage == "state": arrays["after_solve/data/qpos"][0] ^= 1
        if damage == "pose": arrays["after_collision/data/site_xpos"][0] ^= 1
        with (tmp_path / p.PAYLOADS[0]).open("wb") as stream: np.savez(stream, **arrays)
    if damage in ("analysis", "layout", "statics"):
        name = dict(analysis=p.PAYLOADS[3], layout=p.PAYLOADS[2], statics=p.PAYLOADS[1])[damage]
        (tmp_path / name).write_text("{}")
    if damage == "symlink":
        path = tmp_path / "child.log"; path.rename(tmp_path / "original"); path.symlink_to(tmp_path / "original")
    if damage == "extra": (tmp_path / "unexpected").write_bytes(b"extra")
    for item in child["payloads"]:
        path = tmp_path / item["file"]; item.update(bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest())
    (tmp_path / "child.json").write_text(json.dumps(child)); report["child"] = child
    report["files"] = {name: sha256((tmp_path / name).read_bytes()).hexdigest() for name in p.FILES}
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError): p.receive(tmp_path, SOURCE, tmp_path)


def test_no_hidden_reconstruction_or_training_calls():
    tree = ast.parse(Path(p.__file__).read_text())
    physics = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "physics")
    calls = [ast.unparse(n.func) for n in ast.walk(physics) if isinstance(n, ast.Call)]
    assert calls.count("packet.measured_staged_calls") == 1
    assert calls.count("mjwarp.make_data") == 1 and calls.count("mjwarp.put_model") == 1
    assert calls.count("randomization.expand_model_fields") == 1
    assert not set(calls) & {"mjwarp.forward", "mujoco.mj_forward", "mujoco.mj_kinematics", "saved.supplied_poses", "response.supplied_poses", "packet.restore_complete_data", "solver.solve"}
    assert calls.count("packet.bind_static_inventory") == 2


def test_new_profile_supervisor_uses_own_module_inventory_and_limits(tmp_path, monkeypatch):
    seed = tmp_path / "seed"; seed.mkdir()
    _, child, _ = fixture(seed, monkeypatch)
    output = tmp_path / "output"
    lock = tmp_path / "existing.lock"; lock.touch()
    monkeypatch.setattr(p.host, "LOCK", lock)
    monkeypatch.setattr(p, "directory", lambda source: output)
    monkeypatch.setattr(p, "identity", lambda source: {"commit": source})
    monkeypatch.setattr(p, "owned_service", lambda source: {"unit": p.PREFIX + source[:12] + ".service"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    sample = dict(gpu=p.host.parse_gpu(f"{p.host.GPU},{p.host.DRIVER},8.9,16376,961,15232,0,43"))
    monkeypatch.setattr(p.host, "telemetry", lambda: sample)
    monkeypatch.setattr(p.host, "service_snapshot", lambda: {"mission": "inactive"})
    monkeypatch.setattr(p.host, "foreign_processes", lambda owned_pid=None: [])
    class Child:
        pid = 2345; returncode = 0
        def __init__(self, command, **kwargs):
            assert command[1:3] == ["-m", p.MODULE] and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            assert kwargs["env"]["CUDA_CACHE_DISABLE"] == "1" and len(kwargs["pass_fds"]) == 1
            for key in p.host.CACHES: assert Path(kwargs["env"][key]) == output / "private-cache" / key.lower()
            for name in p.PAYLOADS: os.link(seed / name, output / name)
            p.host.write_json(output / "child.json", child)
        def poll(self): return 0
    monkeypatch.setattr(p.base.subprocess, "Popen", Child)
    p.base.supervise(SOURCE, probe=p)
    report = json.loads((output / "report.json").read_bytes())
    assert report["decision"] == p.COLLECTION_DECISION and report["child_exit"] == 0
    assert set(report["files"]) == set(p.FILES) and report["specification"] == p.specification()
    assert report["bounds"] == p.base.BOUNDS and lock.exists()
