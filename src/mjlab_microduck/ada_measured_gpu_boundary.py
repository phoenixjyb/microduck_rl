"""New complete Ada boundary measurement; not historical recovery or admission.

Stdlib-only owner uses the existing unchanged lease/watchdog/resource guards.
The child computes one new fixture kinematics and retains five complete banks.
No BAM proposal, integration, CPU solver control, ordinary forward or PPO.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import stat
import sys

from mjlab_microduck import ada_duck_forward as base
from mjlab_microduck import ada_measured_model_probe as allocation

packet = allocation.packet
response, saved, host, need = allocation.response, allocation.saved, base.host, base.need
MODULE = "mjlab_microduck.ada_measured_gpu_boundary"
PROTOCOL = "microduck-new-measured-ada-boundary-v1"
PREFIX = "microduck-ada-measured-boundary-"
BOUNDS, FLAGS = base.BOUNDS, base.FLAGS
PAYLOAD_MAX = 15 * 1024**2
CHILD_JSON_LIMIT = 128 * 1024
PAYLOADS = ("measured-stages.npz", "measured-statics.json", "measured-layout.json", "measured-analysis.json")
FILES = ("launch.json", "child.log", "child.json") + PAYLOADS
COLLECTION_DECISION = "new-measured-ada-boundary-collected-pending-reception-not-admission"
ALLOCATION_SOURCE = "03a8be9209597b158468ed2901089ba9358ced3b"
ALLOCATION_JSON_SHA = "469558cbbf19d11c4b7c2d14619dac6ceecba089d02dabbbbb3a6083fcb25eb1"
ALLOCATION_NPZ_SHA = "d997420e6449b4f57848bc7393d1a512d22d3ae262232c7879e1ab240989c80c"
STATIC_SHA = "8180631aae0b6d04130f140737c79cef65349b2b38ca9227134d637718fcf78e"
LEASE = dict(device=66306, inode=11419619, bytes=0)
FOREIGN = [dict(pid=1592, name="/home/converge/Tonghao/VLM/grounding_dino_cpp_dev/build_worker/grounding_dino_cpp_worker")]
CONTRACT = dict(allocation_native_kinematics_calls=1, source_expected_model_expansion_launches=2,
    fixture_warp_kinematics_calls=1, collision_calls=1, constraint_calls=1, solver_calls=1,
    fixture_native_forward_calls=0, fixture_native_kinematics_calls=0, ordinary_forward_calls=0,
    acceleration_sensor_calls=0, integration_steps=0, motor_preparations=0, optimizer_steps=0,
    same_bank_cpu_solver_control_executed=False, historical_live_gpu_boundary_established=False,
    loaded_binary_bytes_bound=False)


def specification():
    return dict(protocol=PROTOCOL, worlds=2, stages=list(response.STAGES), data_arrays_per_stage=114,
        model_arrays=347, measured_pose_fields=list(response.POSES), flags=FLAGS,
        numerical_acceptance_tolerance=None, **CONTRACT)


def directory(source):
    path = host.ROOT / "artifacts/evaluations" / ("ada-measured-boundary-" + source[:12])
    need(path.parent.resolve(strict=True) == path.parent, "canonical new measurement parent")
    return path


def identity(source):
    need(datetime.now(timezone.utc) < datetime(2026, 10, 9, 22, 50, tzinfo=timezone.utc), "before closeout reserve")
    result = base.identity(source)
    raw = Path(__file__).read_bytes()
    need(raw == host.read("git", "show", source + ":src/mjlab_microduck/ada_measured_gpu_boundary.py", binary=True), "exact committed measured module")
    return dict(result, measured_module_sha256=sha256(raw).hexdigest())


def owned_service(source):
    row = host.LOCK.stat(follow_symlinks=False)
    need(stat.S_ISREG(row.st_mode) and row.st_uid == os.getuid()
         and dict(device=row.st_dev, inode=row.st_ino, bytes=row.st_size) == LEASE
         and host.foreign_processes() == FOREIGN, "predeclared existing lease and preserved foreign owner before launch")
    return base.owned_service(source, prefix=PREFIX)


def allocation_reference(root):
    path = Path(__file__).resolve().parents[2] / "artifacts/tools/ada-measured-boundary/03a8be92-linux-model-probe.json"
    for file, digest in ((path, ALLOCATION_JSON_SHA), (path.with_suffix(".npz"), ALLOCATION_NPZ_SHA)):
        need(file.is_file() and not file.is_symlink() and file.stat().st_size < 1024**2
             and sha256(file.read_bytes()).hexdigest() == digest, "exact retained complete CPU allocation reference")
    value, _ = allocation.receive(path, ALLOCATION_SOURCE, root)
    need(value["static_manifest_sha256"] == STATIC_SHA, "predeclared complete static binding")
    return value


def require_device(model, data):
    arrays = saved.p.p.all_arrays(dict(model=model, data=data))
    need(len(arrays) == 461 and all(a.device.is_cuda and not a.device.is_cpu and a.device.arch == 89
         and a.device.ordinal == 0 and str(a.device) == "cuda:0" for a in arrays.values()), "all461 actual arrays on the selected Ada device")


def passive_executables():
    """Only already-held CUDA ModuleExec metadata, never build/load/hash calls."""
    from warp._src.context import user_modules
    rows = []
    for name, module in sorted(user_modules.items()):
        for (context, block), executable in sorted(module.execs.items(), key=lambda x: str(x[0])):
            need(context is not None and executable.device.is_cuda and not executable.device.is_cpu
                 and executable.device.arch == 89 and executable.device.ordinal == 0
                 and isinstance(executable.module_hash, bytes) and len(executable.module_hash) == 32,
                 "only already-held Ada executables")
            rows.append(dict(module=name, block_dim=block, opaque_handle=str(executable.handle),
                module_source_options_hash=executable.module_hash.hex(), device=str(executable.device),
                meta=executable.meta, kernel_hook_count=len(executable.kernel_hooks), loaded_binary_bytes_bound=False))
    return rows


def validate_provenance(value):
    need(value["private_cache_was_absent"] is True and type(value["private_cache_files"]) is dict
         and 0 < len(value["private_cache_files"]) <= 128, "complete fresh private cache metadata")
    for name, row in value["private_cache_files"].items():
        need(type(name) is str and Path(name).parts[0] in {k.lower() for k in host.CACHES}
             and not Path(name).is_absolute() and all(x not in ("", ".", "..") for x in name.split("/"))
             and type(row) is dict and set(row) == {"bytes", "sha256"}
             and type(row["bytes"]) is int and 0 < row["bytes"] < 16 * 1024**2
             and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "bounded private cache leaf identity")
    need(sum(x["bytes"] for x in value["private_cache_files"].values()) <= 96 * 1024**2, "unchanged private cache byte bound")
    rows = value["existing_gpu_executables"]
    need(type(rows) is list and 0 < len(rows) <= 128, "bounded passive executable metadata")
    seen = set()
    for row in rows:
        need(type(row) is dict and set(row) == {"module", "block_dim", "opaque_handle", "module_source_options_hash", "device", "meta", "kernel_hook_count", "loaded_binary_bytes_bound"}
             and type(row["module"]) is str and bool(row["module"]) and type(row["block_dim"]) is int and row["block_dim"] > 0
             and type(row["opaque_handle"]) is str and row["device"] == "cuda:0" and type(row["meta"]) is dict
             and type(row["kernel_hook_count"]) is int and row["kernel_hook_count"] >= 0
             and row["loaded_binary_bytes_bound"] is False and re.fullmatch(r"[0-9a-f]{64}", row["module_source_options_hash"]),
             "passive source/options metadata without loaded-object or executed-kernel identity")
        key = row["module"], row["block_dim"]; need(key not in seen, "unique passive executable"); seen.add(key)


def retain(root, name, value):
    import numpy as np
    need(name in PAYLOADS, "closed measurement payload name")
    if name.endswith(".npz"):
        need(sum(a.nbytes for a in value.values()) < 2 * 1024**2, "bounded complete stage arrays")
        with (root / name).open("xb") as stream:
            np.savez(stream, **value); stream.flush(); os.fsync(stream.fileno())
    else: host.write_json(root / name, value)
    raw = (root / name).read_bytes()
    need(len(raw) <= PAYLOAD_MAX, "bounded retained measurement payload")
    return dict(file=name, bytes=len(raw), sha256=sha256(raw).hexdigest())


def physics(root, device="cuda:0"):
    import numpy as np
    import warp as wp
    import mujoco_warp as mjwarp
    from mujoco_warp._src import smooth, forward, collision_driver, constraint, sensor, solver
    from mjlab.sim import randomization
    from mjlab_microduck.stance_warp_runtime import build_entity
    from mjlab_microduck.stance_forward_graph import binding
    need(device == "cuda:0" and not passive_executables(), "fresh Ada fixture before kernel loads")
    predecessor = host.ROOT / "artifacts/evaluations" / ("ada-duck-contact-" + saved.p.prior.SOURCE[:12])
    report, banks = saved.p.prior.authenticated_banks(predecessor)
    reference = allocation_reference(predecessor)
    audit, expansion = response.source.audit(), allocation.expansion_callable(randomization.expand_model_fields, randomization.__file__)
    native = build_entity().compile(); plant = saved.p.bind_plant(native, report["child"]["plant"])
    for name in ("dof_frictionloss", "dof_damping"):
        getattr(native, name)[:] = banks["motor.npz"][name][0].astype(np.float64)
    with wp.ScopedDevice(device):
        model = mjwarp.put_model(native)
        data = mjwarp.make_data(native, nworld=2, nconmax=128, naconmax=256, naccdmax=256, njmax=512, njmax_nnz=10240)
        options = response.topology_guard(native, model, data, device_kind="cuda")
        randomization.expand_model_fields(model, 2, ["dof_frictionloss", "dof_damping"])
        for name in ("dof_frictionloss", "dof_damping"):
            wp.copy(getattr(model, name), wp.array(banks["motor.npz"][name], dtype=wp.float32, device=device))
        states = saved.p.state_inputs(report, banks)
        states = {k: v.reshape(getattr(data, k).numpy().shape) for k, v in states.items()}
        for name, value in states.items():
            wp.copy(getattr(data, name), wp.array(value, dtype=getattr(data, name).dtype, device=device))
        wp.synchronize_device(device); require_device(model, data)
        model_hashes = saved.p.model_binding(model, report, banks)
        need(model_hashes == reference["model_array_sha256"], "all347 actual model bytes equal accepted CPU allocation before fixture physics")
        manifest = packet.portable_static(dict(model=model, data=data), wp.array)
        static_sha = packet.bind_static_inventory(manifest, report, value=dict(model=model, data=data), array_type=wp.array)
        allocation.frozen_static_schema(manifest)
        need(manifest == reference["static_manifest"] and static_sha == STATIC_SHA, "all actual GPU statics equal the retained complete CPU allocation")
        pointers = binding((model, data))
        need(all(not getattr(data, k).numpy().any() for k in saved.p.COUNTERS + ("nacon", "ncollision")), "no stale fixture work")
        boundaries, layout, poses, contacts = {}, None, None, {}
        def capture(stage):
            nonlocal layout, poses
            need(stage == response.STAGES[len(boundaries)], "one exact ordered measured boundary")
            wp.synchronize_device(device); require_device(model, data)
            actual_poses = {k: getattr(data, k).numpy().copy() for k in response.POSES}
            if poses is None: poses = actual_poses
            response.exact_values(actual_poses, poses, "eleven newly measured poses")
            response.exact_values({k: getattr(data, k).numpy().copy() for k in states}, states, "seven original state inputs")
            need(response.topology_guard(native, model, data, device_kind="cuda") == options
                 and saved.p.model_binding(model, report, banks) == model_hashes and binding((model, data)) == pointers
                 and packet.bind_static_inventory(manifest, report, value=dict(model=model, data=data), array_type=wp.array) == static_sha,
                 "unchanged actual complete model/statics/pointers at every boundary")
            raw, current = response.snapshot_data(data, report)
            need(layout is None or layout == current, "unchanged complete stage layout")
            layout = current; boundaries[stage] = raw
            if stage != "before_collision":
                candidates = saved.p.warp_candidates(data)
                need(0 < int(data.ncollision.numpy()[0]) <= 256, "bounded generated broadphase")
                if stage == "after_collision":
                    saved.contact_layout(candidates)
                    need(all(not getattr(data, k).numpy().any() for k in saved.p.COUNTERS), "no early construction or solve")
                else:
                    response.candidate_stability(contacts["after_collision"], candidates)
                    need((data.ne.numpy() == 0).all() and (data.nl.numpy() == 0).all() and (data.nf.numpy() == 14).all()
                         and (candidates["dim"] == 3).all()
                         and np.array_equal(data.nefc.numpy(), 14 + 4 * np.bincount(candidates["worldid"], minlength=2))
                         and (data.nefc.numpy() <= 512).all(), "bounded friction/contact-only constraints")
                    if stage != "after_solve": need(not data.solver_niter.numpy().any(), "no early solver")
                    else:
                        need(((data.solver_niter.numpy() >= 0) & (data.solver_niter.numpy() <= native.opt.iterations)).all(), "unchanged bounded solver iterations")
                        response.solver_write_fence(boundaries["before_solve"], raw)
                contacts[stage] = candidates
        packet.measured_staged_calls(model, data, dict(smooth=smooth, forward=forward, collision_driver=collision_driver,
            constraint=constraint, sensor=sensor, solver=solver), capture)
        need(tuple(boundaries) == response.STAGES and response.source.audit() == audit
             and allocation.expansion_source() == expansion, "complete frozen-source measured execution")
        arrays = {stage + key: raw for stage, bank in boundaries.items() for key, raw in bank.items()}
        analysis = packet.analyze_measured(arrays, layout, report, banks)
    payloads = [retain(root, name, value) for name, value in zip(PAYLOADS, (arrays, manifest, layout, analysis))]
    provenance = dict(private_cache_was_absent=True, private_cache_files=saved.cache_files(root / "private-cache"),
                      existing_gpu_executables=passive_executables())
    validate_provenance(provenance)
    return dict(plant=plant, options=options, model_array_sha256=model_hashes,
        static_manifest_sha256=static_sha, measured_pose_sha256={k: sha256(v.tobytes()).hexdigest() for k, v in poses.items()},
        in_process_binding_sha256=sha256(repr(pointers).encode()).hexdigest(), source_audit_sha256=packet.digest(audit),
        expansion_source=expansion, predecessor_report_sha256=saved.p.prior.REPORT_SHA256,
        input_file_sha256=report["files"], allocation_reference_sha256=ALLOCATION_JSON_SHA,
        allocation_payload_sha256=ALLOCATION_NPZ_SHA, payloads=payloads, **provenance)


def validate_child(value):
    keys = {"specification", "gpu_uuid", "name", "capability", "torch_cuda", "warp_arch", "warp_precompiled_headers",
        "plant", "options", "model_array_sha256", "static_manifest_sha256", "measured_pose_sha256", "in_process_binding_sha256",
        "source_audit_sha256", "expansion_source", "predecessor_report_sha256", "input_file_sha256", "allocation_reference_sha256",
        "allocation_payload_sha256", "payloads", "private_cache_was_absent", "private_cache_files", "existing_gpu_executables"}
    need(type(value) is dict and set(value) == keys and packet.digest(value["specification"]) == packet.digest(specification())
         and value["gpu_uuid"] == host.GPU and value["name"] == host.NAME and value["capability"] == [8, 9]
         and type(value["warp_arch"]) is int and value["warp_arch"] == 89
         and value["torch_cuda"] == "12.8" and value["warp_precompiled_headers"] is False,
         "closed actual unqualified Ada measurement")
    need(value["static_manifest_sha256"] == STATIC_SHA and value["allocation_reference_sha256"] == ALLOCATION_JSON_SHA
         and value["allocation_payload_sha256"] == ALLOCATION_NPZ_SHA
         and set(value["measured_pose_sha256"]) == set(response.POSES)
         and type(value["model_array_sha256"]) is dict and len(value["model_array_sha256"]) == 347,
         "complete measured pose/model/static bindings")
    for digest in list(value["measured_pose_sha256"].values()) + list(value["model_array_sha256"].values()) + [value["in_process_binding_sha256"], value["source_audit_sha256"]]:
        need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "exact complete binding digest")
    need(type(value["payloads"]) is list and [x["file"] for x in value["payloads"]] == list(PAYLOADS), "all four unique complete measurement payloads")
    for row in value["payloads"]:
        need(type(row) is dict and set(row) == {"file", "bytes", "sha256"} and type(row["bytes"]) is int
             and 0 < row["bytes"] <= (PAYLOAD_MAX if row["file"].endswith(".npz") else 256 * 1024)
             and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "bounded payload binding")
    validate_provenance(value)


def receive(root, source, predecessor, *, cache=False):
    """Complete portable receipt and arithmetic; imports no simulator runtime."""
    import numpy as np
    root = Path(root)
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source) and root.is_dir() and not root.is_symlink(), "plain exact measured packet")
    need({p.name for p in root.iterdir()} in (set(FILES) | {"report.json"}, set(FILES) | {"report.json", "private-cache"}), "complete unique retained measurement inventory")
    raw = {}
    for name in FILES + ("report.json",):
        path = root / name; limit = 256 * 1024 if name.endswith(".json") else PAYLOAD_MAX
        need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "plain bounded packet leaf")
        raw[name] = path.read_bytes()
    report = json.loads(raw["report.json"]); child = json.loads(raw["child.json"]); launch = json.loads(raw["launch.json"])
    validate_child(child)
    need(set(report) == {"specification", "source", "service", "bounds", "decision", "telemetry", "child_exit", "services_before", "lease", "foreign_before",
         "child", "services_unchanged", "foreign_unchanged", "lease_unchanged", "elapsed_seconds", "files"}
         and report["child"] == child and report["decision"] == COLLECTION_DECISION and report["specification"] == specification()
         and report["bounds"] == BOUNDS and type(report["child_exit"]) is int and report["child_exit"] == 0
         and all(report[k] is True for k in ("services_unchanged", "foreign_unchanged", "lease_unchanged"))
         and type(report["elapsed_seconds"]) is float and 0 < report["elapsed_seconds"] < BOUNDS["owner_seconds"]
         and set(report["files"]) == set(FILES)
         and all(sha256(raw[k]).hexdigest() == v for k, v in report["files"].items()), "exact successful supervisor and every retained byte")
    need(host.read("git", "hash-object", "uv.lock") == host.read("git", "rev-parse", source + ":uv.lock"), "current frozen lock bytes equal execution-source blob")
    expected_source = dict(commit=source, tree=host.read("git", "rev-parse", source + "^{tree}"), versions=host.VERSIONS,
        module_sha256=sha256(host.read("git", "show", source + ":src/mjlab_microduck/ada_runtime_smoke.py", binary=True)).hexdigest(),
        lock_sha256=sha256((Path(__file__).resolve().parents[2] / "uv.lock").read_bytes()).hexdigest(),
        interpreter=report["source"].get("interpreter"), prefix=str(host.ROOT / ".venv"),
        diagnostic_sha256=sha256(host.read("git", "show", source + ":src/mjlab_microduck/ada_duck_forward.py", binary=True)).hexdigest(),
        measured_module_sha256=sha256(host.read("git", "show", source + ":src/mjlab_microduck/ada_measured_gpu_boundary.py", binary=True)).hexdigest())
    need(report["source"] == expected_source and type(expected_source["interpreter"]) is str
         and expected_source["interpreter"].startswith("/home/converge/.local/share/uv/python/")
         and expected_source["interpreter"].endswith("/bin/python3.12"), "committed clean-source frozen native interpreter receipt")
    service = report["service"]; props = service["properties"]
    need(set(service) == {"unit", "invocation", "properties"} and service["unit"] == PREFIX + source[:12] + ".service"
         and re.fullmatch(r"[0-9a-f]{32}", service["invocation"]) and props.get("MainPID", "").isdigit() and int(props["MainPID"]) > 1
         and props == dict(MainPID=props["MainPID"], ActiveState="active", RuntimeMaxUSec="3min", MemoryMax=str(6 * 1024**3),
            CPUQuotaPerSecUSec="2s", TasksMax="64", Nice="10", KillMode="control-group", LimitFSIZE=str(16 * 1024**2), Restart="no", TimeoutStopUSec="10s"),
         "actual unchanged bounded measured owner service")
    services = {(scope + ":" + name): "inactive" for scope in ("system", "user") for name in host.SERVICES}
    need(report["services_before"] == services and report["lease"] == LEASE and report["foreign_before"] == FOREIGN,
         "predeclared preserved protected services, foreign owner and existing lease")
    telemetry = report["telemetry"]
    need(type(telemetry) is list and len(telemetry) >= 3, "baseline live-monitor and closeout telemetry")
    for sample in telemetry:
        need(type(sample) is dict and set(sample) == {"wall_time_unix", "gpu"} and type(sample["wall_time_unix"]) is float
             and sample["wall_time_unix"] < datetime(2026, 10, 9, 22, 50, tzinfo=timezone.utc).timestamp(), "pre-deadline telemetry")
        row = sample["gpu"]
        need(set(row) == {"uuid", "driver", "capability", "total_mib", "used_mib", "free_mib", "utilization_percent", "temperature_c"}
             and row["total_mib"] == 16376 and all(type(row[k]) is int and row[k] >= 0 for k in ("total_mib", "used_mib", "free_mib", "utilization_percent", "temperature_c"))
             and row["used_mib"] + row["free_mib"] <= row["total_mib"], "closed valid full GPU telemetry")
        host.capacity(sample, telemetry[0])
    need(all(a["wall_time_unix"] <= b["wall_time_unix"] for a, b in zip(telemetry, telemetry[1:])), "ordered telemetry")
    need(launch == dict(specification=specification(), source=expected_source, bounds=BOUNDS, baseline=telemetry[0], lease=LEASE, foreign=FOREIGN), "exact separate launch binding")
    prior, banks = saved.p.prior.authenticated_banks(predecessor); reference = allocation_reference(predecessor)
    need(child["source_audit_sha256"] == packet.digest(response.source.audit()) and child["expansion_source"] == allocation.expansion_source()
         and child["predecessor_report_sha256"] == saved.p.prior.REPORT_SHA256 and child["input_file_sha256"] == prior["files"]
         and child["model_array_sha256"] == reference["model_array_sha256"] and child["plant"] == reference["plant"] and child["options"] == reference["options"],
         "same exact model/plant/state inputs and frozen installed stage/helper sources")
    for item in child["payloads"]:
        need(len(raw[item["file"]]) == item["bytes"] and sha256(raw[item["file"]]).hexdigest() == item["sha256"], "exact separate payload digest")
    manifest, layout, analysis = [json.loads(raw[name]) for name in PAYLOADS[1:]]
    allocation.frozen_static_schema(manifest)
    need(manifest == reference["static_manifest"] and packet.digest(manifest) == STATIC_SHA, "complete exact GPU versus CPU retained statics")
    with np.load(io.BytesIO(raw[PAYLOADS[0]]), allow_pickle=False) as bank:
        need(len(bank.files) == len(set(bank.files)) == 570, "complete unique five114 raw leaves")
        arrays = {k: bank[k].copy() for k in bank.files}
    need(packet.analyze_measured(arrays, layout, prior, banks) == analysis, "complete independent exact-layout stage fences and ordered arithmetic")
    need(child["measured_pose_sha256"] == {k: sha256(arrays["before_collision/data/" + k].tobytes()).hexdigest() for k in response.POSES}, "all eleven actual measured poses")
    if cache: need(saved.cache_files(root / "private-cache") == child["private_cache_files"], "every native private cache byte, not loaded binary identity")
    return report, arrays, layout, manifest, analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--child", action="store_true")
    parser.add_argument("--lease-fd", type=int); args = parser.parse_args()
    profile = sys.modules[__name__]
    if args.child:
        need(args.lease_fd is not None and args.lease_fd >= 3, "inherited owned advisory lease")
        base.child(args.source, args.lease_fd, probe=profile)
    else:
        need(args.lease_fd is None, "owner acquires existing lease")
        base.supervise(args.source, probe=profile)


if __name__ == "__main__": main()
