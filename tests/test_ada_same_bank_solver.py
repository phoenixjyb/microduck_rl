"""Synthetic CPU solver-control packets/source checks; no physics execution."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from mjlab_microduck import ada_same_bank_solver as p
from test_ada_saved_pose_response_receiver import packet, service, SOURCE


def test_import_and_help_inert():
    code = "import sys;from mjlab_microduck import ada_same_bank_solver;assert not {'numpy','warp','torch','mujoco','mujoco_warp'}&sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)
    subprocess.run([sys.executable, "-m", p.__name__, "--help"], capture_output=True, check=True, timeout=15)


@pytest.mark.parametrize("damage", ["none", "head", "branch", "dirty", "module", "source"])
def test_portable_receiver_requires_current_clean_execution_revision(monkeypatch, damage):
    def read(*args, binary=False):
        if binary: return b"different module" if damage == "module" else Path(p.__file__).read_bytes()
        if args[-1] == "HEAD": return "2" * 40 if damage == "head" else SOURCE
        if args[-1] == "--show-current": return "main" if damage == "branch" else p.host.BRANCH
        if args[-1] == "--porcelain": return " M helper.py" if damage == "dirty" else ""
        raise AssertionError(args)
    monkeypatch.setattr(p.host, "read", read)
    if damage == "none": p.receiver_source_check(SOURCE)
    else:
        with pytest.raises(ValueError): p.receiver_source_check("invalid" if damage == "source" else SOURCE)


def reference():
    # Source/options metadata fixture is explicitly not compiled code identity.
    row = dict(p.allocation.EXPECTED_REPEAT, opaque_handle="opaque")
    return {name: dict(deepcopy(row), module=name, kernel_hook_count=10) for name in p.ALLOWED_MODULES}


@pytest.mark.parametrize("damage", ["none", "extra", "collision", "hash", "meta", "device", "block", "hooks", "boolhooks", "binary", "missing", "duplicate"])
def test_passive_metadata_is_solver_only_and_bound(damage):
    refs = reference()
    rows = [deepcopy(refs[name]) for name in ("repeat_array_kernel_39317a34", "mujoco_warp._src.solver")]
    if damage == "extra": rows[0]["unknown"] = 0
    if damage == "collision": rows[0]["module"] = "mujoco_warp._src.collision_driver"
    if damage == "hash": rows[0]["module_source_options_hash"] = "c" * 64
    if damage == "meta": rows[0]["meta"] = {}
    if damage == "device": rows[0]["device"] = "cuda:0"
    if damage == "block": rows[0]["block_dim"] = 256
    if damage == "hooks": rows[0]["kernel_hook_count"] = 11
    if damage == "boolhooks": rows[0]["kernel_hook_count"] = True
    if damage == "binary": rows[0]["loaded_binary_bytes_bound"] = True
    if damage == "missing": rows.pop()
    if damage == "duplicate": rows.append(deepcopy(rows[0]))
    if damage == "none": p.validate_executables(rows, refs)
    else:
        with pytest.raises(ValueError): p.validate_executables(rows, refs)


def test_fewer_current_hooks_do_not_invent_a_launch_count():
    refs = reference(); rows = [deepcopy(refs[name]) for name in ("repeat_array_kernel_39317a34", "mujoco_warp._src.solver")]
    rows[1]["kernel_hook_count"] = 1; p.validate_executables(rows, refs)
    assert p.CONTRACT["public_solver_calls"] == 1 and p.CONTRACT["pre_solve_mass_factorization_calls"] == 0
    assert p.CONTRACT["historical_live_gpu_boundary_established"] is False


def control_service():
    row = service(); row["unit"] = p.PREFIX + SOURCE[:8] + ".service"; return row


@pytest.mark.parametrize("field", list(p.response.SERVICE_CAPS) + ["unit", "MainPID", "InvocationID", "Environment", "extra"])
def test_every_control_service_field_refuses(field):
    row = control_service(); p.validate_service(row, SOURCE, pid=123)
    if field == "unit": row["unit"] = "microduck-ada-response-" + SOURCE[:8] + ".service"
    else: row["properties"][field] = "wrong"
    with pytest.raises(ValueError): p.validate_service(row, SOURCE, pid=123)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "receiver_source_check", lambda source: None)
    gpu_arrays, layout, prior, banks, _ = packet()
    manifest = dict(synthetic="complete-test-manifest")
    monkeypatch.setattr(p.measured, "STATIC_SHA", p.packet.digest(manifest))
    monkeypatch.setattr(p.allocation, "frozen_static_schema", lambda v: None)
    child = dict(static_manifest_sha256=p.measured.STATIC_SHA, plant=prior["child"]["plant"], options={"synthetic": True},
                 model_array_sha256={"/model/array" + str(i): "a" * 64 for i in range(347)})
    gpu = dict(child=child, files={"synthetic": "b" * 64})
    monkeypatch.setattr(p, "gpu_packet", lambda root, predecessor: (gpu, gpu_arrays, layout, manifest, {}))
    monkeypatch.setattr(p.saved.p.prior, "authenticated_banks", lambda root: (prior, banks))
    refs = reference(); monkeypatch.setattr(p, "provenance_reference", lambda root: refs)
    audit = {"frozen": True}; monkeypatch.setattr(p.response.source, "audit", lambda: audit)
    expansion = {"frozen": "helper"}; monkeypatch.setattr(p.allocation, "expansion_source", lambda: expansion)
    module = Path(p.__file__).read_bytes(); monkeypatch.setattr(p.host, "read", lambda *args, **kwargs: module)
    before = {k: gpu_arrays["before_solve" + k] for k in layout}
    gpu_after = {k: gpu_arrays["after_solve" + k] for k in layout}
    after = {k: v.copy() for k, v in gpu_after.items()}
    # One permissible complete output differs; input bank stays exact.
    after["/data/qacc"].view(np.float32)[0] = .5
    arrays = {branch + k: v.copy() for branch, bank in zip(p.BRANCHES, (before, after)) for k, v in bank.items()}
    output = tmp_path / "control.json"
    result = dict(deepcopy(p.CONTRACT), protocol=p.PROTOCOL, decision=p.DECISION, source=SOURCE,
        module_sha256=sha256(module).hexdigest(), versions=deepcopy(p.VERSIONS), gpu_source=p.GPU_SOURCE,
        gpu_report_sha256=p.GPU_REPORT_SHA, gpu_file_sha256=gpu["files"], source_audit_sha256=p.packet.digest(audit),
        provenance_reference_sha256=p.CPU_RESPONSE_SHA, plant=child["plant"], options=child["options"], model_array_sha256=child["model_array_sha256"],
        static_manifest=manifest, static_manifest_sha256=p.measured.STATIC_SHA, data_layout=layout,
        in_process_binding_sha256="c" * 64, in_process_binding_convention=p.response.STATIC_CONVENTION,
        restore_sha256={k: sha256(v.tobytes()).hexdigest() for k, v in before.items()},
        comparison=p.packet.compare_solver_banks(before, gpu_after, after, layout, prior), expansion_source=expansion,
        cpu_counters={k: np.frombuffer(after["/data/" + k], np.int32).tolist() for k in p.saved.p.COUNTERS + ("nacon", "ncollision")},
        pre_solve_executables=[dict(deepcopy(p.allocation.EXPECTED_REPEAT), opaque_handle="wp_repeat_array_kernel_39317a34_1")],
        existing_cpu_executables=[deepcopy(refs[name]) for name in ("repeat_array_kernel_39317a34", "mujoco_warp._src.solver")],
        private_cache_was_absent=True, private_cache_files={"module.o": dict(bytes=4, sha256="a" * 64)},
        running_service=control_service(), protected_services={(scope + ":" + name): "inactive" for scope in ("system", "user") for name in p.host.SERVICES},
        foreign_compute_processes=deepcopy(p.measured.FOREIGN))
    def write():
        path = output.with_suffix(".npz")
        if path.exists(): path.unlink()  # Only test-owned temporary output.
        result["payload"] = p.saved.p.retain(path, arrays)
        output.write_text(json.dumps(result, allow_nan=False))
    write()
    return output, result, arrays, write


def test_complete_solver_control_reception(tmp_path, monkeypatch):
    output, result, arrays, _ = fixture(tmp_path, monkeypatch)
    actual, bank = p.receive(output, SOURCE, tmp_path, tmp_path)
    assert actual == result and len(bank) == 228 and len(result["restore_sha256"]) == 114
    assert result["comparison"]["complete_solver_outputs"]["/data/qacc"]["cpu_minus_gpu_max_abs"] == .5
    assert result["restored_before_matches_new_gpu_bank"] is True and result["training_authorized"] is False
    monkeypatch.setattr(p.saved, "cache_files", lambda root: result["private_cache_files"])
    p.receive(output, SOURCE, tmp_path, tmp_path, cache=tmp_path)
    monkeypatch.setattr(p.saved, "cache_files", lambda root: {})
    with pytest.raises(ValueError): p.receive(output, SOURCE, tmp_path, tmp_path, cache=tmp_path)


@pytest.mark.parametrize("damage", ["extra", "source", "module", "version", "gpu", "files", "audit", "expansion", "provenance", "model", "statics",
    "statichash", "layout", "binding", "restore", "protected", "foreign", "flags", "calls", "preload", "metadata", "comparison", "counter",
    "missing", "dtype", "short", "nonfinite", "restore_byte", "non_solver_write", "payloadhash", "leafhash"])
def test_no_rebinding_partial_banks_or_extra_work(tmp_path, monkeypatch, damage):
    output, r, a, write = fixture(tmp_path, monkeypatch)
    if damage == "extra": r["extra"] = True
    if damage == "source": r["source"] = "2" * 40
    if damage == "module": r["module_sha256"] = "0" * 64
    if damage == "version": r["versions"]["mujoco-warp"] = "new"
    if damage == "gpu": r["gpu_report_sha256"] = "0" * 64
    if damage == "files": r["gpu_file_sha256"] = {}
    if damage == "audit": r["source_audit_sha256"] = "0" * 64
    if damage == "expansion": r["expansion_source"] = {}
    if damage == "provenance": r["provenance_reference_sha256"] = "0" * 64
    if damage == "model": r["model_array_sha256"] = {}
    if damage == "statics": r["static_manifest"] = {}
    if damage == "statichash": r["static_manifest_sha256"] = "0" * 64
    if damage == "layout": r["data_layout"] = {}
    if damage == "binding": r["in_process_binding_convention"] = "recovered historical identity"
    if damage == "restore": r["restore_sha256"].pop("/data/qacc")
    if damage == "protected": r["protected_services"] = {}
    if damage == "foreign": r["foreign_compute_processes"] = []
    if damage == "flags": r["flags"]["training_authorized"] = 0
    if damage == "calls": r["collision_calls"] = 1
    if damage == "preload": r["pre_solve_executables"].append(deepcopy(r["existing_cpu_executables"][1]))
    if damage == "metadata": r["existing_cpu_executables"][1]["module"] = "mujoco_warp._src.forward"
    if damage == "comparison": r["comparison"]["complete_solver_outputs"]["/data/qacc"]["bytes_equal"] = True
    if damage == "counter": r["cpu_counters"]["solver_niter"] = [1000, 1000]
    if damage == "missing": a.pop("cpu_after_solve/data/qacc")
    if damage == "dtype": a["cpu_after_solve/data/qacc"] = a["cpu_after_solve/data/qacc"].astype(np.int32)
    if damage == "short": a["cpu_after_solve/data/qacc"] = a["cpu_after_solve/data/qacc"][:-1]
    if damage == "nonfinite": a["cpu_after_solve/data/qacc"].view(np.float32)[0] = np.nan
    if damage == "restore_byte": a["restored_before/data/qacc"][0] ^= 1
    if damage == "non_solver_write": a["cpu_after_solve/data/efc/J"][0] ^= 1
    if damage == "nonfinite":
        path = output.with_suffix(".npz"); path.unlink(); np.savez(path, **a)
        raw = path.read_bytes(); r["payload"].update(bytes=len(raw), sha256=sha256(raw).hexdigest())
        r["payload"]["arrays"]["cpu_after_solve/data/qacc"]["sha256"] = sha256(a["cpu_after_solve/data/qacc"].tobytes()).hexdigest()
        output.write_text(json.dumps(r, allow_nan=False))
    else:
        write()
        if damage == "payloadhash": r["payload"]["sha256"] = "0" * 64
        if damage == "leafhash": r["payload"]["arrays"]["cpu_after_solve/data/qacc"]["sha256"] = "0" * 64
        output.write_text(json.dumps(r, allow_nan=False))
    with pytest.raises(ValueError): p.receive(output, SOURCE, tmp_path, tmp_path)


def test_runtime_source_has_one_public_solve_after_complete_restore():
    tree = ast.parse(Path(p.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    calls = sorted(((n.lineno, ast.unparse(n.func)) for n in ast.walk(run) if isinstance(n, ast.Call)))
    names = [name for _, name in calls]
    assert names.count("solver.solve") == names.count("packet.restore_complete_data") == 1
    assert names.index("packet.restore_complete_data") < names.index("solver.solve")
    assert names.count("mjwarp.put_model") == names.count("mjwarp.make_data") == 1
    assert not set(names) & {"mjwarp.forward", "smooth.kinematics", "smooth.factor_m", "mujoco.mj_forward", "mujoco.mj_kinematics",
        "collision_driver.collision", "constraint.make_constraint", "response.staged_calls", "packet.measured_staged_calls", "solver.create_solver_context"}
    assert names.count("allocation.repeat_executables") == 2 and names.count("packet.bind_static_inventory") == 2
    assert names.count("response.snapshot_data") == 2
