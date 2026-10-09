"""Bounded CPU allocation/static probe; no fixture dynamics or GPU launcher."""
import argparse
import ast
import base64
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re

from mjlab_microduck import ada_measured_boundary_packet as packet

response = packet.response
saved = response.saved
host = saved.p.p.base.host
need = packet.need
PROTOCOL = "microduck-measured-model-cpu-allocation-probe-v1"
DECISION = "complete-cpu-model-statics-collected-not-runtime-admission"
PREFIX = "microduck-ada-model-probe-"
EXPANSION_SHA = "35d0bddcd6ef3b0317cb89987e0c42bbf1862fcd4beb795c1327ade814c2a64c"
EXPECTED_REPEAT = dict(module="repeat_array_kernel_39317a34", block_dim=1, device="cpu", kernel_hook_count=1,
    module_source_options_hash="64511f00215be403f7a05e8069fbed3ff8b6b02bb3b0a9560b56c2ee9f1f1847",
    meta={"repeat_array_kernel_d0305a41_cuda_kernel_backward_smem_bytes": 0,
          "repeat_array_kernel_d0305a41_cuda_kernel_forward_smem_bytes": 0}, loaded_binary_bytes_bound=False)
CONTRACT = dict(allocation_native_kinematics_calls=1, source_expected_model_expansion_launches=2,
    fixture_native_kinematics_calls=0, fixture_warp_kinematics_calls=0,
    collision_calls=0, constraint_calls=0, solver_calls=0, integration_steps=0,
    flags=host.FLAGS, simulator_qualified=False, training_authorized=False,
    physical_motion_authorized=False, actual_cpu_static_binding_checked=True,
    loaded_binary_bytes_bound=False)


def expansion_source():
    from importlib.metadata import distribution
    dist = distribution("mjlab")
    files = [f for f in dist.files if str(f) == "mjlab/sim/randomization.py"]
    need(len(files) == 1, "unique frozen model expansion source")
    item = files[0]; path = Path(dist.locate_file(item))
    need(path.is_file() and not path.is_symlink() and path.stat().st_size < 65536, "plain bounded expansion source")
    raw = path.read_bytes(); digest = sha256(raw).digest()
    need(digest.hex() == EXPANSION_SHA and item.hash and item.hash.mode == "sha256"
         and base64.urlsafe_b64encode(digest).decode().rstrip("=") == item.hash.value,
         "frozen wheel-bound model expansion bytes")
    return dict(sha256=digest.hex(), bytes=len(raw), record_sha256_matches=True)


def expansion_callable(function, module_file):
    import inspect
    from importlib.metadata import distribution
    path = Path(distribution("mjlab").locate_file("mjlab/sim/randomization.py"))
    need(inspect.isfunction(function) and function.__module__ == "mjlab.sim.randomization"
         and function.__name__ == "expand_model_fields" and Path(module_file).resolve(strict=True) == path.resolve(strict=True)
         and Path(inspect.getsourcefile(function)).resolve(strict=True) == path.resolve(strict=True),
         "executed expansion helper comes from the verified wheel source")
    return expansion_source()


def repeat_executables(rows):
    need(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict
         and set(rows[0]) == set(EXPECTED_REPEAT) | {"opaque_handle"}
         and packet.digest({k: v for k, v in rows[0].items() if k != "opaque_handle"}) == packet.digest(EXPECTED_REPEAT)
         and type(rows[0]["opaque_handle"]) is str
         and re.fullmatch(r"wp_repeat_array_kernel_39317a34_[0-9]+", rows[0]["opaque_handle"]),
         "only the retained source/options-bound CPU repeat executable, not a launch count")


def frozen_static_schema(manifest):
    """Pure source-backed completeness check, not a fresh runtime value walk."""
    from importlib.metadata import distribution
    import numpy as np
    packet.validate_static_structure(manifest)
    path = distribution("mujoco-warp").locate_file("mujoco_warp/_src/types.py")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == saved.SOURCE_FILES["mujoco_warp/_src/types.py"], "frozen static type source")
    classes = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef)}
    dataclasses = {name for name, n in classes.items() if any(ast.unparse(d).startswith("dataclasses.dataclass") for d in n.decorator_list)}
    rows, arrays = manifest["scalars"], set(manifest["arrays"])
    def typed(path, annotation):
        text = ast.unparse(annotation)
        if (isinstance(annotation, ast.Call) and text.startswith("array(")) or text.startswith("wp.array"):
            need(path in arrays, "every annotated Warp array is an array path"); return
        need(path in rows, "every annotated static child present"); row = rows[path]
        if isinstance(annotation, ast.Name) and annotation.id in dataclasses:
            name = annotation.id
            fields = [f for f in classes[name].body if isinstance(f, ast.AnnAssign) and isinstance(f.target, ast.Name)]
            need(row == dict(kind="dataclass", type="mujoco_warp._src.types." + name, fields=[f.target.id for f in fields]),
                 "every frozen actual dataclass field present in declaration order")
            for field in fields: typed(path + "/" + field.target.id, field.annotation)
        elif isinstance(annotation, ast.Subscript) and text.startswith("tuple["):
            need(row["kind"] == "tuple", "annotated tuple retains its exact kind")
            element = annotation.slice.elts[0]
            for i in range(row["size"]): typed(path + "/" + str(i), element)
        elif isinstance(annotation, ast.Name) and annotation.id in {"int", "bool", "float"}:
            kind = dict(int="int", bool="bool", float="float64")[annotation.id]
            if row["kind"] == "numpy-scalar":
                dtype = np.dtype(row["dtype"])
                need(np.issubdtype(dtype, {"int": np.integer, "bool": np.bool_, "float": np.floating}[annotation.id]),
                     "exact annotated numpy scalar family")
            else: need(row["kind"] == kind, "exact annotated primitive scalar kind")
        elif isinstance(annotation, ast.Name) and annotation.id in classes:
            need(row["kind"] == "enum" and row["type"] == "mujoco_warp._src.types." + annotation.id,
                 "exact annotated frozen enum type")
        elif text == "Callable | None":
            need(row == dict(kind="NoneType", value=None), "all frozen callback fields absent")
        else: raise ValueError("unsupported frozen static annotation: " + text)
    need(manifest["scalars"][""] == dict(kind="dict", keys=["data", "model"])
         and manifest["scalars"]["/model"]["type"] == "mujoco_warp._src.types.Model"
         and manifest["scalars"]["/data"]["type"] == "mujoco_warp._src.types.Data",
         "complete frozen root Model/Data dataclasses")
    typed("/model", ast.Name(id="Model")); typed("/data", ast.Name(id="Data"))


def validate_service(service, source, *, pid=None):
    need(type(service) is dict and set(service) == {"unit", "properties"}
         and service["unit"] == PREFIX + source[:8] + ".service", "closed source-bound model probe unit")
    response.validate_service(dict(service, unit="microduck-ada-response-" + source[:8] + ".service"), source, pid=pid)
    return service


def unit_identity(source):
    unit = PREFIX + source[:8] + ".service"
    props = tuple(response.SERVICE_CAPS) + ("MainPID", "InvocationID", "Environment")
    row = dict(line.split("=", 1) for line in host.read("systemctl", "--user", "show", unit,
                 *["--property=" + k for k in props]).splitlines())
    return validate_service(dict(unit=unit, properties=row), source, pid=os.getpid())


def run(report, banks):
    import numpy as np
    import warp as wp
    import mujoco_warp as mjwarp
    from mjlab.sim import randomization
    expand_model_fields = randomization.expand_model_fields
    expansion_callable(expand_model_fields, randomization.__file__)
    from mjlab_microduck.stance_warp_runtime import build_entity
    need(all(d.is_cpu for d in wp.get_devices()) and not saved.passive_executables(),
         "fresh CPU-only allocation before any kernel loads")
    native = build_entity().compile()
    plant = saved.p.bind_plant(native, report["child"]["plant"])
    for name in ("dof_frictionloss", "dof_damping"):
        getattr(native, name)[:] = banks["motor.npz"][name][0].astype(np.float64)
    with wp.ScopedDevice("cpu"):
        model = mjwarp.put_model(native)
        data = mjwarp.make_data(native, nworld=2, nconmax=128, naconmax=256,
                               naccdmax=256, njmax=512, njmax_nnz=10240)
        options = response.topology_guard(native, model, data)
        expand_model_fields(model, 2, ["dof_frictionloss", "dof_damping"])
        for name in ("dof_frictionloss", "dof_damping"):
            wp.copy(getattr(model, name), wp.array(banks["motor.npz"][name], dtype=wp.float32, device="cpu"))
        states = saved.p.state_inputs(report, banks)
        for name, value in states.items():
            array = getattr(data, name)
            wp.copy(array, wp.array(value.reshape(array.numpy().shape), dtype=array.dtype, device="cpu"))
        wp.synchronize_device("cpu")
        need(all(a.device.is_cpu and not a.device.is_cuda for a in saved.p.p.all_arrays(dict(model=model, data=data)).values()),
             "every actual Model/Data array is CPU")
        model_hashes = saved.p.model_binding(model, report, banks)
        manifest = packet.portable_static(dict(model=model, data=data), wp.array)
        static_sha = packet.bind_static_inventory(manifest, report, value=dict(model=model, data=data), array_type=wp.array)
        frozen_static_schema(manifest)
        response.exact_values({k: getattr(data, k).numpy().reshape(v.shape) for k, v in states.items()}, states,
                              "original seven CPU allocation state inputs")
        arrays, layout = response.snapshot_data(data, report)
        counters = {k: getattr(data, k).numpy().tolist() for k in saved.p.COUNTERS + ("nacon", "ncollision")}
        need(all(not any(v) for v in counters.values()), "no fixture collision/constraint/solver work")
        need(response.topology_guard(native, model, data) == options
             and saved.p.model_binding(model, report, banks) == model_hashes
             and packet.bind_static_inventory(manifest, report, value=dict(model=model, data=data), array_type=wp.array) == static_sha,
             "actual complete allocation bindings remain unchanged")
    return dict(plant=plant, options=options, model_array_sha256=model_hashes, static_manifest=manifest,
                static_manifest_sha256=static_sha, data_layout=layout, counters=counters, **CONTRACT), arrays


def receive(output, source, root, *, cache=None):
    """Portable closed packet reception; no allocation or runtime import."""
    import numpy as np
    output = Path(output)
    need(output.is_file() and not output.is_symlink() and output.stat().st_size <= 256 * 1024,
         "bounded plain model probe receipt")
    result = json.loads(output.read_bytes())
    keys = {"plant", "options", "model_array_sha256", "static_manifest", "static_manifest_sha256", "data_layout",
        "counters", "protocol", "decision", "source", "module_sha256", "source_audit_sha256", "expansion_source",
        "predecessor_report_sha256", "input_file_sha256", "private_cache_was_absent", "private_cache_files",
        "existing_cpu_executables", "payload", "running_service"} | set(CONTRACT)
    need(type(result) is dict and set(result) == keys and result["protocol"] == PROTOCOL and result["decision"] == DECISION
         and result["source"] == source and re.fullmatch(r"[0-9a-f]{40}", source)
         and all(type(result[k]) is type(v) and result[k] == v for k, v in CONTRACT.items())
         and all(type(v) is bool and v is False for v in result["flags"].values()), "closed unqualified allocation contract")
    need(result["module_sha256"] == sha256(host.read("git", "show", source + ":src/mjlab_microduck/ada_measured_model_probe.py", binary=True)).hexdigest()
         and result["source_audit_sha256"] == packet.digest(response.source.audit())
         and result["expansion_source"] == expansion_source(), "exact source-bound probe")
    report, banks = saved.p.prior.authenticated_banks(root)
    need(result["predecessor_report_sha256"] == saved.p.prior.REPORT_SHA256 and result["input_file_sha256"] == report["files"]
         and result["model_array_sha256"] == {k: v["sha256"] for k, v in report["child"]["input_manifest"].items() if k.startswith("/model/")}
         and result["options"] == dict(saved.p.WARP_COLLISION_OPTIONS, broadphase=0, broadphase_filter=11)
         and result["plant"] == dict(selected_fields_sha256=report["child"]["plant"]["selected_fields_sha256"],
              options=report["child"]["plant"]["options"], collision_options=saved.p.COLLISION_OPTIONS,
              collision_flag_arm="unchanged-compiled-default"), "same exact predecessor model and options")
    frozen_static_schema(result["static_manifest"])
    need(result["static_manifest"]["arrays"] == sorted(report["child"]["input_manifest"])
         and result["static_manifest_sha256"] == packet.digest(result["static_manifest"]), "complete declared static array inventory and digest")
    for k, v in dict(nworld=2, naconmax=256, naccdmax=256, njmax=512, njmax_pad=512, njmax_nnz=10240).items():
        need(result["static_manifest"]["scalars"]["/data/" + k] == dict(kind="int", value=v), "unchanged complete Data capacities")
    saved.provenance_layout(result); repeat_executables(result["existing_cpu_executables"])
    validate_service(result["running_service"], source)
    path, payload = output.with_suffix(".npz"), result["payload"]
    need(type(payload) is dict and set(payload) == {"file", "bytes", "sha256", "arrays"}
         and path.is_file() and not path.is_symlink() and path.name == payload["file"]
         and path.stat().st_size == payload["bytes"] < 16 * 1024**2
         and sha256(path.read_bytes()).hexdigest() == payload["sha256"], "exact complete allocation payload")
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as bank:
        need(len(bank.files) == len(set(bank.files)) == 114 and set(bank.files) == set(payload["arrays"]), "complete unique allocation bank")
        arrays = {k: bank[k].copy() for k in bank.files}
    expected = {k: row for k, row in report["child"]["input_manifest"].items() if k.startswith("/data/")}
    need(set(arrays) == set(result["data_layout"]) == set(expected), "all114 canonical allocation fields")
    decoded = {}
    for k, v in arrays.items():
        row = response.expanded_layout(expected[k])
        need(row == result["data_layout"][k] and v.dtype == np.uint8 and v.shape == (row["bytes"],)
             and payload["arrays"][k] == dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha256(v.tobytes()).hexdigest()),
             "complete exact allocation leaf")
        decoded[k] = np.frombuffer(v.tobytes(), dtype=row["numpy_dtype"]).reshape(row["numpy_shape"])
        need(np.isfinite(decoded[k]).all(), "finite full allocation field")
    counters = {k: decoded["/data/" + k].tolist() for k in saved.p.COUNTERS + ("nacon", "ncollision")}
    need(counters == result["counters"] and all(not any(v) for v in counters.values()), "actual zero fixture-work counters")
    for name, value in saved.p.state_inputs(report, banks).items():
        need(decoded["/data/" + name].tobytes() == value.tobytes(), "exact original state bytes")
    if cache is not None: need(saved.cache_files(Path(cache)) == result["private_cache_files"], "every native private cache byte")
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "cache"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and datetime.now(timezone.utc) < datetime(2026, 10, 9, 22, 50, tzinfo=timezone.utc),
         "CPU hidden and before campaign closeout reserve")
    identity = host.source_check(args.source)
    module = Path(__file__).read_bytes()
    need(module == host.read("git", "show", args.source + ":src/mjlab_microduck/ada_measured_model_probe.py", binary=True), "exact committed probe module")
    need(args.output.is_absolute() and args.cache.is_absolute() and not args.output.exists()
         and not args.output.with_suffix(".npz").exists() and not args.cache.exists()
         and args.output.parent.resolve(strict=True) == args.output.parent
         and args.cache.parent.resolve(strict=True) == args.cache.parent, "fresh canonical probe output/cache")
    service, services = unit_identity(args.source), host.service_snapshot()
    audit = response.source.audit(); expansion = expansion_source()
    report, banks = saved.p.prior.authenticated_banks(args.input)
    import warp as wp
    import torch
    args.cache.mkdir(); saved.configure_private_cpu_cache(wp, args.cache); wp.init()
    need(all(d.is_cpu for d in wp.get_devices()) and not torch.cuda.is_initialized(), "CPU-only probe runtime")
    result, arrays = run(report, banks)
    need(host.source_check(args.source) == identity and response.source.audit() == audit and expansion_source() == expansion
         and host.service_snapshot() == services and not torch.cuda.is_initialized(), "unchanged exact source/services and no CUDA")
    executables = saved.passive_executables(); repeat_executables(executables)
    result.update(protocol=PROTOCOL, decision=DECISION, source=args.source, module_sha256=sha256(module).hexdigest(),
        source_audit_sha256=packet.digest(audit), expansion_source=expansion, predecessor_report_sha256=saved.p.prior.REPORT_SHA256,
        input_file_sha256=report["files"], private_cache_was_absent=True, private_cache_files=saved.cache_files(args.cache),
        existing_cpu_executables=executables, payload=saved.p.retain(args.output.with_suffix(".npz"), arrays),
        running_service=service)
    host.write_json(args.output, result); receive(args.output, args.source, args.input, cache=args.cache)
    print(DECISION, result["static_manifest_sha256"], len(result["static_manifest"]["scalars"]))


if __name__ == "__main__": main()
