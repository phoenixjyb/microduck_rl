"""Synthetic CPU allocation receipt checks; no simulator allocation or physics."""
import ast
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mjlab_microduck import ada_measured_model_probe as p
from test_ada_saved_pose_response_receiver import packet, service, SOURCE


def test_import_inert_and_closed_help():
    code = "import sys;from mjlab_microduck import ada_measured_model_probe;assert not {'numpy','warp','torch','mujoco','mujoco_warp'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)
    subprocess.run([sys.executable, "-m", p.__name__, "--help"], check=True, timeout=15, capture_output=True)


def schema():
    raw = distribution("mujoco-warp").locate_file("mujoco_warp/_src/types.py").read_bytes()
    classes = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef)}
    dc = {name for name, n in classes.items() if any(ast.unparse(d).startswith("dataclasses.dataclass") for d in n.decorator_list)}
    rows, arrays = {"": dict(kind="dict", keys=["data", "model"])}, []
    def walk(path, ann):
        text = ast.unparse(ann)
        if text.startswith("array(") or text.startswith("wp.array"): arrays.append(path)
        elif isinstance(ann, ast.Name) and ann.id in dc:
            fields = [f for f in classes[ann.id].body if isinstance(f, ast.AnnAssign)]
            rows[path] = dict(kind="dataclass", type="mujoco_warp._src.types." + ann.id, fields=[f.target.id for f in fields])
            for f in fields: walk(path + "/" + f.target.id, f.annotation)
        elif text.startswith("tuple["):
            rows[path] = dict(kind="tuple", size=1); walk(path + "/0", ann.slice.elts[0])
        elif text == "int": rows[path] = dict(kind="int", value=0)
        elif text == "bool": rows[path] = dict(kind="bool", value=False)
        elif text == "float": rows[path] = dict(kind="float64", hex=0.0.hex())
        elif text == "Callable | None": rows[path] = dict(kind="NoneType", value=None)
        else: rows[path] = dict(kind="enum", type="mujoco_warp._src.types." + text, value=0)
    walk("/model", ast.Name(id="Model")); walk("/data", ast.Name(id="Data"))
    return dict(convention=p.packet.STATIC_CONVENTION, arrays=sorted(arrays), scalars=rows)


def test_frozen_schema_and_wheel_expansion_source_without_runtime():
    p.frozen_static_schema(schema())
    value = p.expansion_source()
    assert value["sha256"] == p.EXPANSION_SHA and value["record_sha256_matches"] is True


@pytest.mark.parametrize("damage", ["omitted", "reordered", "unknown", "wrong_root"])
def test_schema_rejects_self_consistent_missing_actual_field(damage):
    value = schema(); row = value["scalars"]["/model"]
    if damage == "omitted":
        name = row["fields"].pop(); value["arrays"].remove("/model/" + name)
    if damage == "reordered": row["fields"].reverse()
    if damage == "unknown": row["type"] = "external.Model"
    if damage == "wrong_root": value["scalars"]["/data"]["type"] = "mujoco_warp._src.types.Model"
    with pytest.raises(ValueError): p.frozen_static_schema(value)


@pytest.mark.parametrize("path,kind", [("/model/opt", "dict"), ("/data/contact", "dict"), ("/model/qM_tiles", "list"),
    ("/model/opt/graph_conditional", "int"), ("/model/nv", "bool"), ("/model/opt/broadphase", "int"),
    ("/model/callback/control", "str"), ("/model/block_dim/euler_dense", "numpy-scalar")])
def test_nested_static_annotation_kinds_are_not_downgraded(path, kind):
    value = schema(); rows = value["scalars"]
    if kind == "dict": rows[path] = dict(kind="dict", keys=rows[path]["fields"])
    elif kind == "list": rows[path]["kind"] = "list"
    elif kind == "numpy-scalar": rows[path] = dict(kind=kind, dtype="float64", bytes=np.float64(1).tobytes().hex())
    else: rows[path] = dict(kind=kind, value={"int": 0, "bool": False, "str": "callback"}[kind])
    p.packet.validate_static_structure(value)
    with pytest.raises(ValueError): p.frozen_static_schema(value)


def test_numpy_integer_statics_keep_exact_dtype_and_bytes():
    value = schema()
    value["scalars"]["/model/nv"] = dict(kind="numpy-scalar", dtype="int64", bytes=np.int64(20).tobytes().hex())
    p.frozen_static_schema(value)


def test_imported_expansion_symbol_bound_to_exact_verified_file(monkeypatch, tmp_path):
    import inspect
    path = distribution("mjlab").locate_file("mjlab/sim/randomization.py")
    def helper(): pass
    helper.__module__ = "mjlab.sim.randomization"; helper.__name__ = "expand_model_fields"
    monkeypatch.setattr(inspect, "getsourcefile", lambda f: str(path))
    assert p.expansion_callable(helper, path)["sha256"] == p.EXPANSION_SHA
    helper.__module__ = "foreign"
    with pytest.raises(ValueError): p.expansion_callable(helper, path)
    helper.__module__ = "mjlab.sim.randomization"
    with pytest.raises(ValueError): p.expansion_callable(helper, tmp_path)
    monkeypatch.setattr(inspect, "getsourcefile", lambda f: str(tmp_path))
    with pytest.raises(ValueError): p.expansion_callable(helper, path)


@pytest.mark.parametrize("damage", ["extra", "module", "hash", "block", "hook", "meta", "handle"])
def test_only_expected_passive_repeat_executable(damage):
    rows = [dict(deepcopy(p.EXPECTED_REPEAT), opaque_handle="wp_repeat_array_kernel_39317a34_1")]
    p.repeat_executables(rows)
    if damage == "extra": rows.append(deepcopy(rows[0]))
    else:
        field = dict(hash="module_source_options_hash", block="block_dim", hook="kernel_hook_count", handle="opaque_handle").get(damage, damage)
        rows[0][field] = "wrong"
    with pytest.raises(ValueError): p.repeat_executables(rows)


def probe_service():
    value = service(); value["unit"] = p.PREFIX + SOURCE[:8] + ".service"
    return value


@pytest.mark.parametrize("field", list(p.response.SERVICE_CAPS) + ["unit", "MainPID", "InvocationID", "Environment", "extra"])
def test_every_actual_probe_service_field_is_guarded(field):
    value = probe_service(); p.validate_service(value, SOURCE, pid=123)
    if field == "unit": value["unit"] = "microduck-ada-response-" + SOURCE[:8] + ".service"
    else: value["properties"][field] = "wrong"
    with pytest.raises(ValueError): p.validate_service(value, SOURCE, pid=123)


def receipt(tmp_path, monkeypatch):
    a, layout, report, banks, _ = packet()
    arrays = {k: a["before_collision" + k] for k in layout}
    # This fixture checks reception independent of frozen-schema checks above.
    # Its invented padding/model arrays are not claimed actual simulator fields.
    paths = list(report["child"]["input_manifest"])
    for i in range(347):
        key = "/model/array" + str(i); paths.append(key)
        report["child"]["input_manifest"][key] = dict(sha256="a" * 64)
    tree = {}
    for key in paths:
        node = tree; parts = key.split("/")[1:]
        for part in parts[:-1]: node = node.setdefault(part, {})
        node[parts[-1]] = None
    tree["data"].update(nworld=2, naconmax=256, naccdmax=256, njmax=512, njmax_pad=512, njmax_nnz=10240)
    rows, arraypaths = {}, []
    def walk(value, path):
        if path in paths: arraypaths.append(path); return
        if type(value) is dict:
            rows[path] = dict(kind="dict", keys=sorted(value))
            for k, v in value.items(): walk(v, path + "/" + k)
        else: rows[path] = dict(kind="int", value=value)
    walk(tree, "")
    manifest = dict(convention=p.packet.STATIC_CONVENTION, arrays=sorted(arraypaths), scalars=rows)
    output = tmp_path / "probe.json"
    expansion = dict(sha256=p.EXPANSION_SHA, bytes=100, record_sha256_matches=True)
    result = dict(plant=dict(selected_fields_sha256=report["child"]["plant"]["selected_fields_sha256"],
         options=report["child"]["plant"]["options"], collision_options=p.saved.p.COLLISION_OPTIONS,
         collision_flag_arm="unchanged-compiled-default"), options=dict(p.saved.p.WARP_COLLISION_OPTIONS, broadphase=0, broadphase_filter=11),
         model_array_sha256={k: "a" * 64 for k in paths if k.startswith("/model/")}, static_manifest=manifest,
         static_manifest_sha256=p.packet.digest(manifest), data_layout=layout,
         counters={k: [0] * (1 if k in {"nacon", "ncollision"} else 2) for k in p.saved.p.COUNTERS + ("nacon", "ncollision")},
         protocol=p.PROTOCOL, decision=p.DECISION, source=SOURCE, module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest(),
         source_audit_sha256=p.packet.digest({"frozen": True}), expansion_source=deepcopy(expansion),
         predecessor_report_sha256=p.saved.p.prior.REPORT_SHA256, input_file_sha256=report["files"],
         private_cache_was_absent=True, private_cache_files={"module.o": dict(bytes=4, sha256="c" * 64)},
         existing_cpu_executables=[dict(deepcopy(p.EXPECTED_REPEAT), opaque_handle="wp_repeat_array_kernel_39317a34_1")],
         running_service=probe_service(), **deepcopy(p.CONTRACT))
    monkeypatch.setattr(p.host, "read", lambda *args, **kwargs: Path(p.__file__).read_bytes())
    monkeypatch.setattr(p.saved.p.prior, "authenticated_banks", lambda root: (report, banks))
    monkeypatch.setattr(p.response.source, "audit", lambda: {"frozen": True})
    monkeypatch.setattr(p, "expansion_source", lambda: expansion)
    monkeypatch.setattr(p, "frozen_static_schema", p.packet.validate_static_structure)
    def write():
        path = output.with_suffix(".npz")
        if path.exists(): path.unlink()  # Only this test-owned temporary fixture.
        result["payload"] = p.saved.p.retain(path, arrays)
        output.write_text(json.dumps(result, allow_nan=False))
    write()
    return output, result, arrays, write


def test_complete_allocation_receipt_and_all114_leaves(tmp_path, monkeypatch):
    output, result, arrays, _ = receipt(tmp_path, monkeypatch)
    actual, raw = p.receive(output, SOURCE, tmp_path)
    assert actual == result and set(raw) == set(arrays) and len(raw) == 114
    assert actual["solver_calls"] == actual["fixture_warp_kinematics_calls"] == 0
    assert actual["source_expected_model_expansion_launches"] == 2 and not actual["training_authorized"]


@pytest.mark.parametrize("damage", ["extra", "source", "module", "audit", "expansion", "flag", "call", "modelhash", "manifest", "statichash",
    "capacity", "layout", "state", "counter", "missing", "dtype", "bytes", "nonfinite", "cache", "payloadhash", "leafhash"])
def test_closed_receipt_or_leaf_failures(tmp_path, monkeypatch, damage):
    output, r, a, write = receipt(tmp_path, monkeypatch)
    if damage == "extra": r["extra"] = True
    if damage == "source": r["source"] = "2" * 40
    if damage == "module": r["module_sha256"] = "0" * 64
    if damage == "audit": r["source_audit_sha256"] = "0" * 64
    if damage == "expansion": r["expansion_source"]["bytes"] += 1
    if damage == "flag": r["flags"]["training_authorized"] = True
    if damage == "call": r["solver_calls"] = 1
    if damage == "modelhash": r["model_array_sha256"]["/model/array0"] = "0" * 64
    if damage == "manifest": r["static_manifest"]["arrays"].pop()
    if damage == "statichash": r["static_manifest_sha256"] = "0" * 64
    if damage == "capacity":
        r["static_manifest"]["scalars"]["/data/njmax"]["value"] += 1
        r["static_manifest_sha256"] = p.packet.digest(r["static_manifest"])
    if damage == "layout": r["data_layout"]["/data/qacc"]["numpy_shape"] = [40]
    if damage == "state": a["/data/qpos"] = a["/data/qpos"].copy(); a["/data/qpos"][0] ^= 1
    if damage == "counter": a["/data/solver_niter"].view(np.int32)[0] = 1
    if damage == "missing": a.pop("/data/qacc")
    if damage == "dtype": a["/data/qacc"] = a["/data/qacc"].astype(np.int32)
    if damage == "bytes": a["/data/qacc"] = a["/data/qacc"][:-1]
    if damage == "nonfinite": a["/data/qacc"].view(np.float32)[0] = np.nan
    if damage == "cache": r["existing_cpu_executables"][0]["device"] = "cuda:0"
    if damage == "nonfinite":
        path = output.with_suffix(".npz"); path.unlink(); np.savez(path, **a)
        raw = path.read_bytes(); r["payload"].update(bytes=len(raw), sha256=sha256(raw).hexdigest())
        r["payload"]["arrays"]["/data/qacc"]["sha256"] = sha256(a["/data/qacc"].tobytes()).hexdigest()
        output.write_text(json.dumps(r))
    else: write()
    if damage == "payloadhash": r["payload"]["sha256"] = "0" * 64; output.write_text(json.dumps(r))
    if damage == "leafhash": r["payload"]["arrays"]["/data/qacc"]["sha256"] = "0" * 64; output.write_text(json.dumps(r))
    with pytest.raises(ValueError): p.receive(output, SOURCE, tmp_path)


def test_source_has_only_allocation_expansion_and_copies_no_fixture_physics():
    tree = ast.parse(Path(p.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attrs = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
    assert attrs.count("put_model") == attrs.count("make_data") == 1
    assert not set(attrs) & {"kinematics", "mj_kinematics", "forward", "mj_forward", "collision", "make_constraint", "solve", "step", "mj_step", "sensor_acc"}
    assert not any(isinstance(n.func, ast.Name) and n.func.id in {"staged_calls", "measured_staged_calls"} for n in calls)
    assert p.CONTRACT["source_expected_model_expansion_launches"] == 2
    expansion = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "expand_model_fields"]
    assert len(expansion) == 1 and not expansion[0].keywords
    assert [ast.unparse(a) for a in expansion[0].args] == ["model", "2", "['dof_frictionloss', 'dof_damping']"]
