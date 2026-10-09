"""Saved-Ada-pose CPU collision arm; no fixture kinematics, solve or admission."""
import argparse
import base64
from hashlib import sha256
from importlib.metadata import distribution, version
import io
import json
import os
from pathlib import Path
import re

from mjlab_microduck import ada_contact_metadata_audit as metadata

p = metadata.replay
need = p.need
PROTOCOL = "microduck-ada-saved-pose-cpu-collision-v1"
CONTACT_TAILS = dict(dist=(), pos=(3,), frame=(3, 3), includemargin=(), friction=(5,),
                     solref=(2,), solreffriction=(2,), solimp=(5,), dim=(), geom=(2,),
                     flex=(2,), vert=(2,), efc_address=(4,), worldid=(), type=(), geomcollisionid=())
SOURCE_FILES = {
    "mujoco_warp/_src/collision_core.py": "3c12b154998af28e257ff80437ca4250546947a6b3bf2af7fa3f1210fdff7d74",
    "mujoco_warp/_src/collision_driver.py": "dcc32d3781e3201ff3c79e714f7b6bd854f36045c16068c8ac26a101eb652b8e",
    "mujoco_warp/_src/collision_primitive.py": "e515350dc6405494cdcf32b90ce32788fe7a5749c42439c877a426fce2bf9a72",
    "mujoco_warp/_src/io.py": "731a31f254274c6e04d9843b11bdcb0502cf9d49abf26aa1c82e52e1a13fa75a",
    "mujoco_warp/_src/smooth.py": "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f",
    "mujoco_warp/_src/types.py": "8f0b19d7b2bc039a419fa547ba6b732837327c0735f688ad34ace5afc4843e50",
    "warp/_src/context.py": "eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee",
    "warp/_src/build.py": "5993e58b44a916fc12f35afde1cededc59ab1db7df7a7eecd75551191d9197df",
    "warp/config.py": "e4b95ce936aa6d57aa11f5046879fe04ebdb8793107c88137c9ba0bd6b0b6cf5",
}


def library_sources():
    """Current installed Python bytes plus wheel RECORD binding, not binaries."""
    result = {}
    for package, prefix in (("mujoco-warp", "mujoco_warp/"), ("warp-lang", "warp/")):
        dist = distribution(package)
        files = [f for f in dist.files if str(f).startswith(prefix)
                 and (str(f).endswith(".py") if package == "mujoco-warp" else str(f) in SOURCE_FILES)]
        need(len(files) == (69 if package == "mujoco-warp" else 3), "complete predeclared Python source inventory")
        for item in files:
            name = str(item); path = Path(dist.locate_file(item))
            need(path.is_file() and not path.is_symlink() and path.stat().st_size < 2 * 1024**2,
                 "plain bounded installed source")
            raw = path.read_bytes(); digest = sha256(raw).digest()
            need(item.hash and item.hash.mode == "sha256"
                 and base64.urlsafe_b64encode(digest).decode().rstrip("=") == item.hash.value,
                 "unchanged installed wheel source bytes: " + name)
            result[name] = dict(bytes=len(raw), sha256=digest.hex(), record_sha256_matches=True)
    need(all(result[k]["sha256"] == v for k, v in SOURCE_FILES.items()), "frozen relevant collision/loader sources")
    return result


def supplied_poses(data, report, banks):
    """Decode exact logical pose bytes using the already-bound array layout."""
    import numpy as np
    values = {}
    for name in p.KINEMATIC:
        key = "/data/" + name; row = report["child"]["input_manifest"][key]
        array = getattr(data, name); raw = banks["prepared-inputs.npz"][key].tobytes()
        need(str(array.dtype) == row["dtype"] and list(array.shape) == row["shape"]
             and list(array.strides) == row["strides"] and len(raw) == row["bytes"]
             and sha256(raw).hexdigest() == row["sha256"], "complete exact supplied pose layout/bytes: " + name)
        value = np.frombuffer(raw, dtype=np.float32).reshape(array.numpy().shape).copy()
        need(value.dtype == np.float32 and np.isfinite(value).all(), "finite supplied pose")
        values[name] = value
    return values


def check_pose_bytes(actual, supplied):
    need(set(actual) == set(supplied) == set(p.KINEMATIC), "all nine supplied pose fields")
    for name in supplied:
        need(actual[name].shape == supplied[name].shape and actual[name].dtype == supplied[name].dtype
             and actual[name].tobytes() == supplied[name].tobytes(), "actual collision pose bytes unchanged: " + name)
    return {name: sha256(value.tobytes()).hexdigest() for name, value in actual.items()}


def contact_layout(contact):
    import numpy as np
    need(set(contact) == set(CONTACT_TAILS), "complete rigid candidate struct")
    count = len(contact["dist"])
    need(0 < count <= 256, "bounded active candidate prefix")
    ints = {"dim", "geom", "flex", "vert", "efc_address", "worldid", "type", "geomcollisionid"}
    for name, tail in CONTACT_TAILS.items():
        value = contact[name]
        need(value.shape == (count,) + tail and value.dtype == (np.int32 if name in ints else np.float32)
             and np.isfinite(value).all(), "complete finite candidate layout: " + name)
    need((contact["worldid"] == 1).all() and (contact["efc_address"] == -1).all(),
         "only shallow-world candidates, no constructed EFC")
    return count


def dispatch_guard(native, model, data):
    from mujoco_warp._src.types import BroadphaseType, BroadphaseFilter
    need(native.nflex == 0 and not model.has_sdf_geom and model.callback.contactfilter is None,
         "rigid no-SDF no-contactfilter arm")
    options = p.collision_options(model.opt, warp=True)
    need(model.opt.broadphase == BroadphaseType.NXN
         and model.opt.broadphase_filter == (BroadphaseFilter.PLANE | BroadphaseFilter.SPHERE | BroadphaseFilter.OBB)
         and data.naconmax == 256 and data.njmax == 512 and data.nworld == 2,
         "unchanged positive-capacity collision dispatch")
    return dict(options, broadphase=int(model.opt.broadphase), broadphase_filter=int(model.opt.broadphase_filter))


def passive_executables():
    """Only inspect already-held ModuleExec fields; never build, load or hash."""
    from warp._src.context import user_modules
    rows = []
    for name, module in sorted(user_modules.items()):
        for (context, block), executable in sorted(module.execs.items(), key=lambda x: str(x[0])):
            need(context is None and executable.device.is_cpu and not executable.device.is_cuda
                 and isinstance(executable.module_hash, bytes) and len(executable.module_hash) == 32,
                 "CPU-only existing executable identity")
            rows.append(dict(module=name, block_dim=block, opaque_handle=str(executable.handle),
                             module_source_options_hash=executable.module_hash.hex(),
                             device=str(executable.device), meta=executable.meta,
                             kernel_hook_count=len(executable.kernel_hooks), loaded_binary_bytes_bound=False))
    return rows


def cache_files(cache):
    result = {}
    for path in sorted(cache.rglob("*")):
        need(not path.is_symlink(), "no cache symlink")
        if path.is_dir(): continue
        need(path.is_file() and 0 < path.stat().st_size < 16 * 1024**2, "bounded plain CPU cache leaf")
        raw = path.read_bytes()
        result[str(path.relative_to(cache))] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
    need(0 < len(result) <= 128 and sum(x["bytes"] for x in result.values()) <= 96 * 1024**2,
         "bounded complete private CPU cache")
    return result


def provenance_layout(result):
    """Validate captured metadata without claiming cache files were loaded."""
    need(result["private_cache_was_absent"] is True and result["loaded_binary_bytes_bound"] is False,
         "fresh cache but no loaded-object byte claim")
    rows = result["existing_cpu_executables"]
    need(type(rows) is list and 0 < len(rows) <= 128, "bounded existing CPU executable inventory")
    seen = set()
    fields = {"module", "block_dim", "opaque_handle", "module_source_options_hash", "device", "meta", "kernel_hook_count", "loaded_binary_bytes_bound"}
    for row in rows:
        need(type(row) is dict and set(row) == fields and row["device"] == "cpu"
             and row["loaded_binary_bytes_bound"] is False
             and type(row["module"]) is str and type(row["opaque_handle"]) is str
             and type(row["block_dim"]) is int and row["block_dim"] > 0
             and re.fullmatch(r"[0-9a-f]{64}", row["module_source_options_hash"])
             and type(row["meta"]) is dict and type(row["kernel_hook_count"]) is int and row["kernel_hook_count"] >= 0,
             "passive CPU metadata, not loaded file identity")
        key = (row["module"], row["block_dim"])
        need(key not in seen, "unique existing CPU executable key"); seen.add(key)
    files = result["private_cache_files"]
    need(type(files) is dict and 0 < len(files) <= 128, "complete bounded CPU cache inventory")
    for name, row in files.items():
        need(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
             and type(row) is dict and set(row) == {"bytes", "sha256"}
             and type(row["bytes"]) is int and 0 < row["bytes"] < 16 * 1024**2
             and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "plain cache-relative file metadata")
    need(sum(x["bytes"] for x in files.values()) <= 96 * 1024**2, "bounded cache total")


def metadata_comparison(contact, active):
    import numpy as np
    cpu = {k: contact[k] for k in (set(metadata.INT_FIELDS) | set(metadata.FLOAT_FIELDS)) - {"slot"}}
    cpu["slot"] = np.arange(len(cpu["worldid"]), dtype=np.int32)
    ada = {k: active[("sidecar/" if k in ("type", "geomcollisionid") else "contacts/") + k]
           for k in set(metadata.INT_FIELDS) | set(metadata.FLOAT_FIELDS)}
    return metadata.compare(cpu, ada)


def comparison_stages(contact):
    import numpy as np
    return dict(cpu="collision-only before EFC construction", ada="retained forward after solve",
                ada_includemargin_retained=False,
                cpu_inside_includemargin_raw_slots=np.flatnonzero(contact["dist"] < contact["includemargin"]).tolist(),
                cpu_constraint_type_bit_raw_slots=np.flatnonzero((contact["type"] & 1) != 0).tolist())


def counter_binding(arrays, result):
    import numpy as np
    need(set(result["counters"]) == set(p.COUNTERS), "all EFC/solver counter names")
    for k in p.COUNTERS + ("nacon", "ncollision"):
        value = arrays["counter/" + k]
        need(value.dtype == np.int32 and value.shape == ((1,) if k in ("nacon", "ncollision") else (2,)),
             "complete actual collision/EFC counter layout")
        need(value.tolist() == ([result[k]] if k in ("nacon", "ncollision") else result["counters"][k]),
             "actual bank-to-JSON collision/EFC counter binding")
        if k in p.COUNTERS: need(not value.any(), "no constructed EFC or solver work")


def run(report, banks):
    import numpy as np
    import mujoco_warp as mjwarp
    import warp as wp
    from mjlab.sim.randomization import expand_model_fields
    from mjlab_microduck.stance_warp_runtime import build_entity
    need(not passive_executables(), "fresh process before any CPU kernel load")
    native = build_entity().compile()
    plant = p.bind_plant(native, report["child"]["plant"])
    for name in ("dof_frictionloss", "dof_damping"):
        getattr(native, name)[:] = banks["motor.npz"][name][0].astype(np.float64)
    with wp.ScopedDevice("cpu"):
        model = mjwarp.put_model(native)
        # make_data's ONE allocation-time native kinematics call is counted.
        # No fixture kinematics runs before or after loading saved poses.
        data = mjwarp.make_data(native, nworld=2, nconmax=128, njmax=512)
        options = dispatch_guard(native, model, data)
        expand_model_fields(model, 2, ["dof_frictionloss", "dof_damping"])
        for name in ("dof_frictionloss", "dof_damping"):
            wp.copy(getattr(model, name), wp.array(banks["motor.npz"][name], dtype=wp.float32, device="cpu"))
        states = p.state_inputs(report, banks)
        poses = supplied_poses(data, report, banks)
        for name, value in states.items():
            array = getattr(data, name)
            wp.copy(array, wp.array(value.reshape(array.numpy().shape), dtype=array.dtype, device="cpu"))
        for name, value in poses.items():
            wp.copy(getattr(data, name), wp.array(value, dtype=getattr(data, name).dtype, device="cpu"))
        wp.synchronize_device("cpu")
        model_hashes = p.model_binding(model, report, banks)
        before = {k: getattr(data, k).numpy().copy() for k in p.KINEMATIC}
        pose_hashes = check_pose_bytes(before, poses)
        need(int(data.nacon.numpy()[0]) == int(data.ncollision.numpy()[0]) == 0,
             "no stale collision counters before measured collision")
        need(all(not getattr(data, k).numpy().any() for k in p.COUNTERS), "no EFC/solver work before collision")
        mjwarp.collision(model, data)
        wp.synchronize_device("cpu")
        contact = p.warp_candidates(data)
        contact_layout(contact)
        ncollision = int(data.ncollision.numpy()[0])
        need(0 < ncollision <= data.naconmax and len(contact["dist"]) <= data.naconmax
             and (contact["worldid"] == 0).sum() == 0 and (contact["worldid"] == 1).sum() > 0,
             "no overflow and actual clear/shallow fixtures")
        need((contact["efc_address"] == -1).all(), "unconstructed active candidate addresses")
        counters = {k: getattr(data, k).numpy().tolist() for k in p.COUNTERS}
        need(all(not getattr(data, k).numpy().any() for k in p.COUNTERS), "zero EFC/solver counters after collision")
        after = {k: getattr(data, k).numpy().copy() for k in p.KINEMATIC}
        need(check_pose_bytes(after, poses) == pose_hashes, "saved actual poses survive collision unchanged")
        for name, value in states.items():
            need(getattr(data, name).numpy().tobytes() == value.reshape(getattr(data, name).numpy().shape).tobytes(),
                 "all seven state inputs unchanged")
        need(dispatch_guard(native, model, data) == options
             and p.model_binding(model, report, banks) == model_hashes, "unchanged collision model/options/callback")
    arrays = {"contact/" + k: v for k, v in contact.items()}
    arrays.update({"pose/" + k: v for k, v in after.items()})
    arrays.update({"state/" + k: v for k, v in states.items()})
    arrays.update({"counter/" + k: getattr(data, k).numpy().copy() for k in p.COUNTERS + ("nacon", "ncollision")})
    return dict(plant=plant, options=options, model_array_sha256=model_hashes, counters=counters,
                ncollision=ncollision, nacon=len(contact["dist"]), supplied_pose_sha256=pose_hashes,
                comparison=metadata_comparison(contact, banks["gpu-active.npz"]),
                actual_saved_pose_bytes_equal=True, allocation_native_kinematics_calls=1,
                fixture_native_kinematics_calls=0, fixture_warp_kinematics_calls=0,
                cpu_collision_calls=1, new_constraint_calls=0, new_solver_calls=0, new_integration_steps=0,
                contactfilter_present=False, sdf_present=False, flex_present=False,
                unused_rigid_contact_fields=["flex", "vert"],
                comparison_stages=comparison_stages(contact),
                flags=p.p.FLAGS, solver_qualified=False, training_authorized=False,
                physical_motion_authorized=False), arrays


def receive(output, source, root, *, cache=None):
    """Pure CPU reception; supplied bytes and label residuals are recomputed."""
    import numpy as np
    output = Path(output)
    need(output.is_file() and not output.is_symlink() and output.stat().st_size <= 256 * 1024, "bounded plain saved-pose JSON")
    result = json.loads(output.read_bytes())
    report, banks = p.prior.authenticated_banks(root)
    host = p.p.base.host
    need(result["protocol"] == PROTOCOL and result["decision"] == "saved-pose-collision-diagnostic-not-admission"
         and result["source"] == source and result["predecessor_report_sha256"] == p.prior.REPORT_SHA256
         and result["input_file_sha256"] == report["files"]
         and result["module_sha256"] == sha256(host.read("git", "show", source + ":src/mjlab_microduck/ada_saved_pose_collision.py", binary=True)).hexdigest(),
         "exact source/predecessor-bound saved-pose arm")
    need(result["flags"] == p.p.FLAGS and all(v is False for v in result["flags"].values())
         and result["solver_qualified"] is result["training_authorized"] is result["physical_motion_authorized"] is False
         and result["fixture_native_kinematics_calls"] == result["fixture_warp_kinematics_calls"] == 0
         and result["allocation_native_kinematics_calls"] == result["cpu_collision_calls"] == 1
         and result["new_constraint_calls"] == result["new_solver_calls"] == result["new_integration_steps"] == 0
         and result["contactfilter_present"] is result["sdf_present"] is result["flex_present"] is False
         and result["actual_saved_pose_bytes_equal"] is True
         and result["loaded_binary_bytes_bound"] is False, "unqualified exact call boundary")
    need(result["library_sources"] == library_sources()
         and result["plant"] == dict(selected_fields_sha256=report["child"]["plant"]["selected_fields_sha256"],
                                     options=report["child"]["plant"]["options"], collision_options=p.COLLISION_OPTIONS,
                                     collision_flag_arm="unchanged-compiled-default")
         and result["model_array_sha256"] == {k: v["sha256"] for k, v in report["child"]["input_manifest"].items() if k.startswith("/model/")}
         and result["options"] == dict(p.WARP_COLLISION_OPTIONS, broadphase=0, broadphase_filter=11)
         and set(result["counters"]) == set(p.COUNTERS)
         and all(type(row) is list and len(row) == 2 and all(type(v) is int and v == 0 for v in row)
                 for row in result["counters"].values())
         and all(result["versions"][k].split("+")[0] == v for k, v in p.VERSIONS.items()), "same sources/model/options/counters/packages")
    provenance_layout(result)
    path = output.with_suffix(".npz"); payload = result["payload"]
    need(path.is_file() and not path.is_symlink() and path.name == payload["file"]
         and path.stat().st_size == payload["bytes"] < 16 * 1024**2, "complete plain saved-pose bank")
    raw = path.read_bytes(); need(sha256(raw).hexdigest() == payload["sha256"], "exact saved-pose NPZ bytes")
    with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
        need(len(bank.files) == len(set(bank.files)) and set(bank.files) == set(payload["arrays"]), "complete unique saved-pose arrays")
        arrays = {k: bank[k].copy() for k in bank.files}
    expected = {"contact/" + k for k in p.CONTACT + ("worldid", "type", "geomcollisionid")}
    expected |= {"pose/" + k for k in p.KINEMATIC} | {"state/" + k for k in p.p.base.UNCHANGED}
    expected |= {"counter/" + k for k in p.COUNTERS + ("nacon", "ncollision")}
    need(set(arrays) == expected, "fixed complete saved-pose inventory")
    for k, value in arrays.items():
        row = payload["arrays"][k]
        need(str(value.dtype) == row["dtype"] and list(value.shape) == row["shape"]
             and sha256(value.tobytes()).hexdigest() == row["sha256"] and np.isfinite(value).all(), "finite byte-bound saved-pose array")
    saved = {k: np.frombuffer(banks["prepared-inputs.npz"]["/data/" + k].tobytes(), np.float32)
             .reshape(arrays["pose/" + k].shape) for k in p.KINEMATIC}
    for k in p.KINEMATIC:
        tail = (3, 3) if k.endswith("mat") else (4,) if k == "xquat" else (3,)
        shape = tuple(report["child"]["input_manifest"]["/data/" + k]["shape"]) + tail
        need(arrays["pose/" + k].shape == shape and arrays["pose/" + k].dtype == np.float32,
             "independently declared complete actual pose layout")
    need(check_pose_bytes({k: arrays["pose/" + k] for k in p.KINEMATIC}, saved) == result["supplied_pose_sha256"], "complete actual saved pose bytes")
    for k, value in p.state_inputs(report, banks).items():
        need(arrays["state/" + k].shape == value.shape and arrays["state/" + k].dtype == value.dtype
             and arrays["state/" + k].tobytes() == value.tobytes(), "exact seven-state byte reception")
    contact = {k.removeprefix("contact/"): v for k, v in arrays.items() if k.startswith("contact/")}
    counter_binding(arrays, result)
    need(result["nacon"] == contact_layout(contact) and 0 < result["ncollision"] <= 256
         and metadata_comparison(contact, banks["gpu-active.npz"]) == result["comparison"], "recomputed generator-label comparison")
    need(result["comparison_stages"] == comparison_stages(contact),
         "distinct candidate inclusion and post-solve comparison stages")
    if cache is not None: need(cache_files(Path(cache)) == result["private_cache_files"], "complete retained native CPU cache file bytes")
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "cache"): parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source", required=True)
    args = parser.parse_args(); host = p.p.base.host
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-hidden saved-pose collision")
    need(re.fullmatch(r"[0-9a-f]{40}", args.source) and Path.cwd().resolve() == Path(__file__).resolve().parents[2]
         and host.read("git", "rev-parse", "HEAD") == args.source and host.read("git", "branch", "--show-current") == host.BRANCH
         and not host.read("git", "status", "--porcelain"), "clean exact saved-pose source")
    raw = Path(__file__).read_bytes()
    need(raw == host.read("git", "show", args.source + ":src/mjlab_microduck/ada_saved_pose_collision.py", binary=True), "committed saved-pose module bytes")
    need(args.output.is_absolute() and args.cache.is_absolute() and not args.output.exists()
         and not args.output.with_suffix(".npz").exists() and not args.cache.exists()
         and args.output.parent.resolve(strict=True) == args.output.parent
         and args.cache.parent.resolve(strict=True) == args.cache.parent, "fresh canonical output and private cache")
    versions = {k: version(k) for k in p.VERSIONS}
    need(all(versions[k].split("+")[0] == v for k, v in p.VERSIONS.items()), "unchanged frozen packages")
    sources = library_sources(); report, banks = p.prior.authenticated_banks(args.input)
    import warp as wp
    import torch
    args.cache.mkdir()
    wp.config.kernel_cache_dir = str(args.cache)
    wp.config.enable_precompiled_headers = False
    wp.init()
    need(all(d.is_cpu for d in wp.get_devices()) and not torch.cuda.is_initialized(), "CPU-only devices")
    result, arrays = run(report, banks)
    need(not torch.cuda.is_initialized() and library_sources() == sources, "unchanged sources and hidden CUDA after collision")
    executables = passive_executables()
    need(executables and all(x["device"] == "cpu" for x in executables), "passively retained existing CPU executables")
    result.update(protocol=PROTOCOL, decision="saved-pose-collision-diagnostic-not-admission", source=args.source,
                  module_sha256=sha256(raw).hexdigest(), versions=versions,
                  predecessor_report_sha256=p.prior.REPORT_SHA256, input_file_sha256=report["files"],
                  library_sources=sources, private_cache_was_absent=True, private_cache_files=cache_files(args.cache),
                  existing_cpu_executables=executables, loaded_binary_bytes_bound=False,
                  cache_binding_convention="Complete fresh-cache file hashes, separately from opaque ModuleExec; no loaded-object byte claim",
                  payload=p.retain(args.output.with_suffix(".npz"), arrays))
    host.write_json(args.output, result)
    print(result["decision"], result["nacon"], result["comparison"]["maximum_common_field_residual"])


if __name__ == "__main__": main()
