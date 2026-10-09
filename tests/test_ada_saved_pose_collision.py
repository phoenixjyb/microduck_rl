"""Saved-pose byte and call-boundary tests; no qualification or GPU execution."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mjlab_microduck import ada_saved_pose_collision as p


def pose_fixture():
    values, rows, raw, attrs = {}, {}, {}, {}
    for name in p.p.KINEMATIC:
        tail = (3, 3) if name.endswith("mat") else (4,) if name == "xquat" else (3,)
        value = np.arange(2 * 4 * np.prod(tail), dtype=np.float32).reshape((2, 4) + tail)
        values[name] = value
        attrs[name] = SimpleNamespace(dtype="synthetic", shape=(2, 4), strides=(128, 16), numpy=lambda v=value: v)
        key = "/data/" + name
        raw[key] = np.frombuffer(value.tobytes(), np.uint8)
        rows[key] = dict(dtype="synthetic", shape=[2, 4], strides=[128, 16], bytes=value.nbytes,
                         sha256=sha256(value.tobytes()).hexdigest())
    return values, SimpleNamespace(**attrs), {"child": {"input_manifest": rows}}, {"prepared-inputs.npz": raw}


def test_import_is_inert():
    code = "import sys; import mjlab_microduck.ada_saved_pose_collision; assert not {'numpy','torch','warp','mujoco','mujoco_warp'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_exact_saved_pose_bundle_not_new_kinematics():
    values, data, report, banks = pose_fixture()
    actual = p.supplied_poses(data, report, banks)
    assert p.check_pose_bytes(actual, values) == {k: sha256(v.tobytes()).hexdigest() for k, v in values.items()}
    assert all(not np.shares_memory(actual[k], values[k]) for k in actual)


@pytest.mark.parametrize("damage", ["none", "callback", "sdf", "flex", "capacity", "rows", "worlds", "broadphase", "flags"])
def test_dispatch_guard_refuses_other_physics_paths(damage):
    native = SimpleNamespace(nflex=0)
    opt = SimpleNamespace(**p.p.COLLISION_OPTIONS, broadphase=0, broadphase_filter=11)
    opt.ccd_tolerance = SimpleNamespace(numpy=lambda: np.asarray([1e-6], np.float32))
    model = SimpleNamespace(has_sdf_geom=False, callback=SimpleNamespace(contactfilter=None), opt=opt)
    data = SimpleNamespace(naconmax=256, njmax=512, nworld=2)
    if damage == "callback": model.callback.contactfilter = lambda *args: None
    if damage == "sdf": model.has_sdf_geom = True
    if damage == "flex": native.nflex = 1
    if damage == "capacity": data.naconmax = 0
    if damage == "rows": data.njmax = 256
    if damage == "worlds": data.nworld = 1
    if damage == "broadphase": model.opt.broadphase = 1
    if damage == "flags": model.opt.disableflags = 1
    if damage == "none": assert p.dispatch_guard(native, model, data) == dict(p.p.WARP_COLLISION_OPTIONS, broadphase=0, broadphase_filter=11)
    else:
        with pytest.raises(ValueError): p.dispatch_guard(native, model, data)


@pytest.mark.parametrize("name", p.p.KINEMATIC)
@pytest.mark.parametrize("damage", ["dtype", "shape", "stride", "bytes", "hash"])
def test_supplied_pose_layout_and_bytes_refuse(name, damage):
    _, data, report, banks = pose_fixture()
    row = report["child"]["input_manifest"]["/data/" + name]
    if damage == "dtype": row["dtype"] = "changed"
    if damage == "shape": row["shape"] = [8]
    if damage == "stride": row["strides"][0] += 1
    if damage == "bytes": row["bytes"] -= 1
    if damage == "hash": row["sha256"] = "0" * 64
    with pytest.raises(ValueError): p.supplied_poses(data, report, banks)


@pytest.mark.parametrize("name", p.p.KINEMATIC)
def test_every_actual_pose_one_bit_difference_refuses(name):
    values, _, _, _ = pose_fixture()
    actual = {k: v.copy() for k, v in values.items()}
    actual[name].view(np.uint32).flat[0] ^= 1
    with pytest.raises(ValueError): p.check_pose_bytes(actual, values)


def contacts():
    ints = {"dim", "geom", "flex", "vert", "efc_address", "worldid", "type", "geomcollisionid"}
    values = {k: np.zeros((2,) + tail, np.int32 if k in ints else np.float32) for k, tail in p.CONTACT_TAILS.items()}
    values["worldid"][:] = 1; values["efc_address"][:] = -1
    return values


@pytest.mark.parametrize("name", p.CONTACT_TAILS)
def test_complete_candidate_layout_rejects_changed_shape(name):
    values = contacts(); assert p.contact_layout(values) == 2
    values[name] = values[name][:1]
    with pytest.raises(ValueError): p.contact_layout(values)


@pytest.mark.parametrize("damage", ["extra", "missing", "world", "address", "nan", "dtype"])
def test_candidate_schema_stage_and_finiteness_refuse(damage):
    values = contacts()
    if damage == "extra": values["force"] = np.zeros((2, 6), np.float32)
    if damage == "missing": values.pop("solimp")
    if damage == "world": values["worldid"][0] = 0
    if damage == "address": values["efc_address"][0, 0] = 14
    if damage == "nan": values["pos"][0, 0] = np.nan
    if damage == "dtype": values["pos"] = values["pos"].astype(np.float64)
    with pytest.raises(ValueError): p.contact_layout(values)


def test_candidate_type_margin_are_distinct_and_ada_margin_is_not_fabricated():
    values = contacts()
    values["dist"][:] = [-.1, .1]; values["type"][:] = [0, 1]
    stages = p.comparison_stages(values)
    assert stages["cpu_inside_includemargin_raw_slots"] == [0]
    assert stages["cpu_constraint_type_bit_raw_slots"] == [1]
    assert stages["ada_includemargin_retained"] is False
    assert "before EFC" in stages["cpu"] and "after solve" in stages["ada"]


@pytest.mark.parametrize("name", p.p.COUNTERS + ("nacon", "ncollision"))
@pytest.mark.parametrize("damage", ["claim", "shape", "dtype"])
def test_retained_actual_counter_bytes_not_only_scalar_claim(name, damage):
    arrays = {"counter/" + k: np.zeros(2, np.int32) for k in p.p.COUNTERS}
    arrays.update({"counter/nacon": np.array([8], np.int32), "counter/ncollision": np.array([4], np.int32)})
    result = dict(nacon=8, ncollision=4, counters={k: [0, 0] for k in p.p.COUNTERS})
    p.counter_binding(arrays, result)
    if damage == "claim": arrays["counter/" + name][0] += 1
    if damage == "shape": arrays["counter/" + name] = np.zeros(3, np.int32)
    if damage == "dtype": arrays["counter/" + name] = arrays["counter/" + name].astype(np.int64)
    with pytest.raises(ValueError): p.counter_binding(arrays, result)


def test_self_consistent_nonzero_solver_counter_still_refuses():
    arrays = {"counter/" + k: np.zeros(2, np.int32) for k in p.p.COUNTERS}
    arrays.update({"counter/nacon": np.array([8], np.int32), "counter/ncollision": np.array([4], np.int32)})
    result = dict(nacon=8, ncollision=4, counters={k: [0, 0] for k in p.p.COUNTERS})
    arrays["counter/nefc"][1] = 4; result["counters"]["nefc"][1] = 4
    with pytest.raises(ValueError, match="no constructed"): p.counter_binding(arrays, result)


def test_complete_private_cache_hashes_no_loaded_object_claim(tmp_path):
    directory = tmp_path / "module"; directory.mkdir()
    (directory / "kernel.o").write_bytes(b"opaque object")
    (directory / "kernel.meta").write_bytes(b"{}")
    manifest = p.cache_files(tmp_path)
    assert set(manifest) == {"module/kernel.o", "module/kernel.meta"}
    assert manifest["module/kernel.o"]["sha256"] == sha256(b"opaque object").hexdigest()
    (directory / "alias").symlink_to(directory / "kernel.o")
    with pytest.raises(ValueError): p.cache_files(tmp_path)


def test_empty_cache_refuses(tmp_path):
    with pytest.raises(ValueError): p.cache_files(tmp_path)


def provenance():
    return dict(private_cache_was_absent=True, loaded_binary_bytes_bound=False,
                existing_cpu_executables=[dict(module="collision", block_dim=256, opaque_handle="opaque",
                    module_source_options_hash="a" * 64, device="cpu", meta={}, kernel_hook_count=1,
                    loaded_binary_bytes_bound=False)],
                private_cache_files={"module/kernel.o": dict(bytes=12, sha256="b" * 64)})


@pytest.mark.parametrize("damage", ["none", "cacheold", "claim", "device", "duplicate", "hash", "empty", "parent", "absolute", "size"])
def test_passive_provenance_schema_refuses_upgraded_or_invalid_claims(damage):
    result = provenance()
    if damage == "cacheold": result["private_cache_was_absent"] = False
    if damage == "claim": result["existing_cpu_executables"][0]["loaded_binary_bytes_bound"] = True
    if damage == "device": result["existing_cpu_executables"][0]["device"] = "cuda:0"
    if damage == "duplicate": result["existing_cpu_executables"] *= 2
    if damage == "hash": result["existing_cpu_executables"][0]["module_source_options_hash"] = "bad"
    if damage == "empty": result["private_cache_files"] = {}
    if damage == "parent": result["private_cache_files"] = {"../unsafe": dict(bytes=12, sha256="a" * 64)}
    if damage == "absolute": result["private_cache_files"] = {"/unsafe": dict(bytes=12, sha256="a" * 64)}
    if damage == "size": result["private_cache_files"]["module/kernel.o"]["bytes"] = 16 * 1024**2
    if damage == "none": p.provenance_layout(result)
    else:
        with pytest.raises(ValueError): p.provenance_layout(result)


def test_passive_executable_snapshot_does_not_call_build_load_or_hash(monkeypatch):
    import warp._src.context as context
    device = SimpleNamespace(is_cpu=True, is_cuda=False, __str__=lambda self: "cpu")
    executable = SimpleNamespace(device=device, module_hash=b"a" * 32, handle="opaque", meta={"known": True}, kernel_hooks={})
    module = SimpleNamespace(execs={(None, 256): executable})
    monkeypatch.setattr(context, "user_modules", {"synthetic": module})
    rows = p.passive_executables()
    assert rows[0]["module_source_options_hash"] == (b"a" * 32).hex()
    assert rows[0]["loaded_binary_bytes_bound"] is False and rows[0]["meta"] == {"known": True}
    executable.device.is_cuda = True
    with pytest.raises(ValueError): p.passive_executables()


@pytest.mark.parametrize("cuda,source", [("0", "a" * 40), ("", "bad"), ("", "a" * 40)])
def test_cli_refuses_before_cache_or_output(tmp_path, cuda, source):
    path = tmp_path / "result.json"; cache = tmp_path / "cache"
    command = [sys.executable, "-m", p.__name__, "--input", str(tmp_path), "--output", str(path), "--cache", str(cache), "--source", source]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, env=dict(os.environ, CUDA_VISIBLE_DEVICES=cuda))
    assert result.returncode != 0
    assert "explicit CPU-hidden" in result.stderr or "clean exact saved-pose" in result.stderr
    assert not path.exists() and not cache.exists()


def test_source_call_boundary_no_fixture_kinematics_or_forward():
    import ast
    tree = ast.parse(Path(p.__file__).read_text())
    calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ""
             for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert calls.count("collision") == 1
    assert not set(calls) & {"kinematics", "mj_kinematics", "mj_forward", "forward", "fwd_position", "mj_step",
                             "solve", "allclose", "isclose", "get_module_hash", "ModuleBuilder"}
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load"]
    assert len(loads) == 1 and isinstance(loads[0].func.value, ast.Name) and loads[0].func.value.id == "np"


def test_installed_sources_match_predeclared_wheel_record_bytes():
    sources = p.library_sources()
    assert len(sources) == 72
    assert all(sources[k]["sha256"] == v for k, v in p.SOURCE_FILES.items())
    assert all(v["record_sha256_matches"] for v in sources.values())
