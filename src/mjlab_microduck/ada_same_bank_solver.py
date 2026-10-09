"""Closed CPU solver-only control of the authentic NEW Ada before-solve bank.

No fixture kinematics/collision/construction/mass factorization/sensors/forces
are regenerated. The frozen public solve owns its fresh internal context and
Newton algebra. This is descriptive solver evidence, not simulator admission.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import re

from mjlab_microduck import ada_measured_gpu_boundary as measured

allocation, packet, response, saved, host, need = (measured.allocation, measured.packet,
    measured.response, measured.saved, measured.host, measured.need)
PROTOCOL = "microduck-new-ada-bank-cpu-solver-only-v1"
DECISION = "complete-new-ada-bank-cpu-solver-control-collected-not-admission"
PREFIX = "microduck-ada-same-bank-solver-"
GPU_SOURCE = "daebeb6ee97b1ce0c4a665331306bcb78a94204c"
GPU_REPORT_SHA = "5146e17ace38e5b0e2897cb0a3febf1202b66d9d69b757ac0398e2c581d267de"
CPU_RESPONSE_SOURCE = "05216efe9d33a80d7a839eaa20b7a57ea05dbe86"
CPU_RESPONSE_SHA = "af380438ef6da87e1e77f7eefcb5cdc1da8724fab757a8b9d2836c2f0a8386e9"
CPU_RESPONSE_NPZ_SHA = "b3c52c1463863389ba8161ac9fca0bd080b020e01b48920405904024c183dc34"
VERSIONS = dict(host.VERSIONS, **{"better-actuator-models": "1.0.1"})
BRANCHES = ("restored_before", "cpu_after_solve")
# Only frozen Newton solver/support generated modules, not stage generators.
ALLOWED_MODULES = {
    "linesearch_iterative__locals__kernel_c0167fd2", "mujoco_warp._src.solver", "mujoco_warp._src.support",
    "mul_m_dense__locals___mul_m_dense_b66df345", "repeat_array_kernel_39317a34",
    "solve_init_jaref__locals__kernel_5208712d", "update_constraint_efc__locals__kernel_32935e9d",
    "update_constraint_efc__locals__kernel_3c1a9816", "update_constraint_gauss_cost__locals__kernel_baf8e129",
    "update_gradient_JTDAJ_dense_tiled__locals__kernel_663e8482", "update_gradient_cholesky__locals__kernel_5ee47d83",
}
CONTRACT = dict(allocation_native_kinematics_calls=1, source_expected_model_expansion_launches=2,
    fixture_native_kinematics_calls=0, fixture_warp_kinematics_calls=0, fixture_native_forward_calls=0,
    collision_calls=0, constraint_construction_calls=0, pre_solve_mass_factorization_calls=0,
    sensor_stage_calls=0, force_stage_calls=0, ordinary_forward_calls=0, public_solver_calls=1,
    integration_steps=0, motor_preparations=0, optimizer_steps=0, restored_actual_data_arrays=114,
    restored_before_matches_new_gpu_bank=True, identical_kernel_execution_claimed=False,
    historical_live_gpu_boundary_established=False, loaded_binary_bytes_bound=False,
    solver_qualified=False, simulator_qualified=False, training_authorized=False, physical_motion_authorized=False,
    flags=host.FLAGS)


def exact_file(path, digest, limit):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit
         and sha256(path.read_bytes()).hexdigest() == digest, "exact retained plain reference bytes")


def receiver_source_check(source):
    """Portable receipt requires the clean execution revision, not latest code."""
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source), "exact portable receiver revision")
    root = Path(__file__).resolve().parents[2]
    need(host.read("git", "-C", str(root), "rev-parse", "HEAD") == source
         and host.read("git", "-C", str(root), "branch", "--show-current") == host.BRANCH
         and not host.read("git", "-C", str(root), "status", "--porcelain"),
         "portable receiver uses clean exact execution branch/revision")
    blob = host.read("git", "-C", str(root), "show", source + ":src/mjlab_microduck/ada_same_bank_solver.py", binary=True)
    need(Path(__file__).read_bytes() == blob, "current portable receiver module equals execution blob")


def gpu_packet(root, predecessor):
    root = Path(root)
    exact_file(root / "report.json", GPU_REPORT_SHA, 256 * 1024)
    return measured.receive(root, GPU_SOURCE, predecessor)


def provenance_reference(predecessor):
    path = Path(__file__).resolve().parents[2] / "artifacts/tools/ada-saved-pose-response/05216efe-linux-response.json"
    exact_file(path, CPU_RESPONSE_SHA, 256 * 1024); exact_file(path.with_suffix(".npz"), CPU_RESPONSE_NPZ_SHA, 2 * 1024**2)
    value, _ = response.receive(path, CPU_RESPONSE_SOURCE, predecessor)
    rows = {row["module"]: row for row in value["existing_cpu_executables"] if row["module"] in ALLOWED_MODULES}
    need(set(rows) == ALLOWED_MODULES, "complete retained CPU solver-only provenance whitelist")
    return rows


def validate_executables(rows, reference):
    need(type(rows) is list and 2 <= len(rows) <= len(ALLOWED_MODULES)
         and set(reference) == ALLOWED_MODULES, "bounded retained solver-only executable inventory")
    seen = set()
    for row in rows:
        need(type(row) is dict and row.get("module") in reference and row["module"] not in seen
             and set(row) == set(reference[row["module"]]), "unique closed solver-only executable row")
        baseline = reference[row["module"]]
        need(packet.digest({k: v for k, v in row.items() if k not in ("opaque_handle", "kernel_hook_count")}) ==
             packet.digest({k: v for k, v in baseline.items() if k not in ("opaque_handle", "kernel_hook_count")})
             and row["device"] == "cpu" and row["loaded_binary_bytes_bound"] is False
             and type(row["opaque_handle"]) is str and bool(row["opaque_handle"])
             and type(row["kernel_hook_count"]) is int and 0 < row["kernel_hook_count"] <= baseline["kernel_hook_count"],
             "same frozen CPU module/options/meta bytes; hooks are not launch counts")
        seen.add(row["module"])
    need({"repeat_array_kernel_39317a34", "mujoco_warp._src.solver"} <= seen,
         "model expansion and public solver existing metadata both present")


def validate_service(service, source, *, pid=None):
    need(type(service) is dict and set(service) == {"unit", "properties"}
         and service["unit"] == PREFIX + source[:8] + ".service", "closed source-bound solver control unit")
    response.validate_service(dict(service, unit="microduck-ada-response-" + source[:8] + ".service"), source, pid=pid)
    return service


def unit_identity(source):
    props = tuple(response.SERVICE_CAPS) + ("MainPID", "InvocationID", "Environment")
    unit = PREFIX + source[:8] + ".service"
    row = dict(line.split("=", 1) for line in host.read("systemctl", "--user", "show", unit,
        *["--property=" + k for k in props]).splitlines())
    return validate_service(dict(unit=unit, properties=row), source, pid=os.getpid())


def cpu_device(model, data):
    arrays = saved.p.p.all_arrays(dict(model=model, data=data))
    need(len(arrays) == 461 and all(a.device.is_cpu and not a.device.is_cuda for a in arrays.values()), "every actual model/data array on CPU")


def run(prior, banks, gpu_arrays, layout, manifest, gpu_child):
    """One public solve after actual complete restored-bank verification."""
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-hidden solver-only control")
    import numpy as np
    import warp as wp
    import mujoco_warp as mjwarp
    from mujoco_warp._src import solver
    from mjlab.sim import randomization
    from mjlab_microduck.stance_warp_runtime import build_entity
    from mjlab_microduck.stance_forward_graph import binding
    need(all(d.is_cpu for d in wp.get_devices()) and not saved.passive_executables(), "fresh CPU-only process before allocation kernels")
    expansion = allocation.expansion_callable(randomization.expand_model_fields, randomization.__file__)
    native = build_entity().compile(); plant = saved.p.bind_plant(native, prior["child"]["plant"])
    for name in ("dof_frictionloss", "dof_damping"):
        getattr(native, name)[:] = banks["motor.npz"][name][0].astype(np.float64)
    with wp.ScopedDevice("cpu"):
        model = mjwarp.put_model(native)
        data = mjwarp.make_data(native, nworld=2, nconmax=128, naconmax=256, naccdmax=256, njmax=512, njmax_nnz=10240)
        options = response.topology_guard(native, model, data)
        randomization.expand_model_fields(model, 2, ["dof_frictionloss", "dof_damping"])
        for name in ("dof_frictionloss", "dof_damping"):
            wp.copy(getattr(model, name), wp.array(banks["motor.npz"][name], dtype=wp.float32, device="cpu"))
        wp.synchronize_device("cpu"); cpu_device(model, data)
        model_hashes = saved.p.model_binding(model, prior, banks)
        need(model_hashes == gpu_child["model_array_sha256"] and plant == gpu_child["plant"] and options == gpu_child["options"],
             "actual complete CPU model/plant/options equal new measured GPU before restore")
        static_sha = packet.bind_static_inventory(manifest, prior, value=dict(model=model, data=data), array_type=wp.array)
        allocation.frozen_static_schema(manifest)
        need(static_sha == measured.STATIC_SHA, "all184 actual CPU statics equal the complete measured GPU manifest")
        allocation.repeat_executables(saved.passive_executables())
        pointers = binding((model, data))
        before = {k: gpu_arrays["before_solve" + k] for k in layout}
        gpu_after = {k: gpu_arrays["after_solve" + k] for k in layout}
        restore = packet.restore_complete_data(data, before, layout, prior, wp, model=model, static_manifest=manifest)
        restored, actual_layout = response.snapshot_data(data, prior)
        response.exact_values(restored, before, "all114 actual restored GPU pre-solve fields")
        need(actual_layout == layout and response.topology_guard(native, model, data) == options
             and saved.p.model_binding(model, prior, banks) == model_hashes and binding((model, data)) == pointers,
             "unchanged actual device/model/pointers at restored solver boundary")
        pre_solve_executables = saved.passive_executables(); allocation.repeat_executables(pre_solve_executables)
        # ONLY the frozen public solve after restoring ALL114 fields; it creates
        # its own fresh SolverContext and performs its own Newton algebra.
        solver.solve(model, data)
        wp.synchronize_device("cpu"); cpu_device(model, data)
        after, actual_layout = response.snapshot_data(data, prior)
        need(actual_layout == layout and response.topology_guard(native, model, data) == options
             and saved.p.model_binding(model, prior, banks) == model_hashes and binding((model, data)) == pointers
             and packet.bind_static_inventory(manifest, prior, value=dict(model=model, data=data), array_type=wp.array) == static_sha
             and allocation.expansion_source() == expansion, "complete unchanged actual bindings after solve")
        comparison = packet.compare_solver_banks(restored, gpu_after, after, layout, prior)
        need(((data.solver_niter.numpy() >= 0) & (data.solver_niter.numpy() <= native.opt.iterations)).all(), "unchanged iteration cap")
    return dict(plant=plant, options=options, model_array_sha256=model_hashes, static_manifest=manifest,
        static_manifest_sha256=static_sha, in_process_binding_sha256=sha256(repr(pointers).encode()).hexdigest(),
        in_process_binding_convention=response.STATIC_CONVENTION, data_layout=layout, restore_sha256=restore,
        pre_solve_executables=pre_solve_executables, comparison=comparison,
        cpu_counters={k: getattr(data, k).numpy().tolist() for k in saved.p.COUNTERS + ("nacon", "ncollision")},
        expansion_source=expansion, **CONTRACT), {branch + k: v for branch, bank in zip(BRANCHES, (restored, after)) for k, v in bank.items()}


def receive(output, source, gpu_root, predecessor, *, cache=None):
    """Portable complete reception; does not allocate or execute a runtime."""
    receiver_source_check(source)
    import numpy as np
    output = Path(output)
    need(output.is_file() and not output.is_symlink() and output.stat().st_size <= 256 * 1024, "plain bounded solver-control JSON")
    value = json.loads(output.read_bytes())
    keys = {"plant", "options", "model_array_sha256", "static_manifest", "static_manifest_sha256", "in_process_binding_sha256",
        "in_process_binding_convention", "data_layout", "restore_sha256", "pre_solve_executables", "comparison", "cpu_counters", "expansion_source",
        "protocol", "decision", "source", "module_sha256", "versions", "gpu_source", "gpu_report_sha256", "gpu_file_sha256",
        "source_audit_sha256", "provenance_reference_sha256", "private_cache_was_absent", "private_cache_files", "existing_cpu_executables",
        "payload", "running_service", "protected_services", "foreign_compute_processes"} | set(CONTRACT)
    need(type(value) is dict and set(value) == keys and re.fullmatch(r"[0-9a-f]{40}", source)
         and value["source"] == source and value["protocol"] == PROTOCOL and value["decision"] == DECISION
         and packet.digest({k: value[k] for k in CONTRACT}) == packet.digest(CONTRACT), "closed exact unqualified solver-control contract")
    need(value["module_sha256"] == sha256(host.read("git", "show", source + ":src/mjlab_microduck/ada_same_bank_solver.py", binary=True)).hexdigest()
         and value["source_audit_sha256"] == packet.digest(response.source.audit()) and value["expansion_source"] == allocation.expansion_source()
         and set(value["versions"]) == set(VERSIONS) and all(value["versions"][k].split("+")[0] == v for k, v in VERSIONS.items()), "committed module and frozen wheel sources/versions/helper")
    gpu, gpu_arrays, layout, manifest, _ = gpu_packet(gpu_root, predecessor)
    prior, _ = saved.p.prior.authenticated_banks(predecessor)
    need(value["gpu_source"] == GPU_SOURCE and value["gpu_report_sha256"] == GPU_REPORT_SHA and value["gpu_file_sha256"] == gpu["files"]
         and value["static_manifest"] == manifest and value["static_manifest_sha256"] == measured.STATIC_SHA
         and value["model_array_sha256"] == gpu["child"]["model_array_sha256"] and value["plant"] == gpu["child"]["plant"]
         and value["options"] == gpu["child"]["options"] and value["data_layout"] == layout
         and value["in_process_binding_convention"] == response.STATIC_CONVENTION
         and re.fullmatch(r"[0-9a-f]{64}", value["in_process_binding_sha256"]), "same exact authentic new GPU input/model/static bank")
    allocation.frozen_static_schema(value["static_manifest"])
    reference = provenance_reference(predecessor)
    need(value["provenance_reference_sha256"] == CPU_RESPONSE_SHA, "retained exact CPU provenance source")
    saved.provenance_layout(value); validate_executables(value["existing_cpu_executables"], reference)
    allocation.repeat_executables(value["pre_solve_executables"]); validate_service(value["running_service"], source)
    need(value["protected_services"] == {(scope + ":" + name): "inactive" for scope in ("system", "user") for name in host.SERVICES}
         and value["foreign_compute_processes"] == measured.FOREIGN, "preserved protected services and Dino owner")
    path, payload = output.with_suffix(".npz"), value["payload"]
    need(type(payload) is dict and set(payload) == {"file", "bytes", "sha256", "arrays"}
         and type(payload["bytes"]) is int and path.is_file() and not path.is_symlink() and path.name == payload["file"]
         and path.stat().st_size == payload["bytes"] < 2 * 1024**2 and sha256(path.read_bytes()).hexdigest() == payload["sha256"], "complete exact branch NPZ")
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as bank:
        need(len(bank.files) == len(set(bank.files)) == 228, "both complete unique114-field branches")
        arrays = {k: bank[k].copy() for k in bank.files}
    need(set(arrays) == {branch + k for branch in BRANCHES for k in layout} == set(payload["arrays"]), "closed complete branch inventory")
    for k, raw in arrays.items():
        need(payload["arrays"][k] == dict(shape=list(raw.shape), dtype=str(raw.dtype), sha256=sha256(raw.tobytes()).hexdigest()), "every complete retained branch leaf hash")
    before = {k: gpu_arrays["before_solve" + k] for k in layout}; gpu_after = {k: gpu_arrays["after_solve" + k] for k in layout}
    restored, after = [{k: arrays[branch + k] for k in layout} for branch in BRANCHES]
    response.exact_values(restored, before, "all114 restored GPU before-solve fields")
    need(value["restore_sha256"] == {k: sha256(v.tobytes()).hexdigest() for k, v in restored.items()}, "complete actual restore receipt")
    comparison = packet.compare_solver_banks(restored, gpu_after, after, layout, prior)
    need(comparison == value["comparison"], "every complete solver output/fence/difference independently recomputed")
    counters = {k: np.frombuffer(after["/data/" + k].tobytes(), np.int32).tolist() for k in saved.p.COUNTERS + ("nacon", "ncollision")}
    need(counters == value["cpu_counters"] and all(0 <= n <= prior["child"]["plant"]["options"]["iterations"] for n in counters["solver_niter"]), "actual complete bounded CPU solver counters")
    if cache is not None: need(saved.cache_files(Path(cache)) == value["private_cache_files"], "every native private cache byte, not loaded binary identity")
    return value, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "cache"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", required=True); args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and datetime.now(timezone.utc) < datetime(2026, 10, 9, 22, 50, tzinfo=timezone.utc), "CPU-hidden launch before closeout reserve")
    identity = host.source_check(args.source); module = Path(__file__).read_bytes()
    need(module == host.read("git", "show", args.source + ":src/mjlab_microduck/ada_same_bank_solver.py", binary=True), "executed committed solver-control module")
    expected_input = host.ROOT / "artifacts/evaluations" / ("ada-measured-boundary-" + GPU_SOURCE[:12])
    need(args.input == expected_input and args.input.resolve(strict=True) == expected_input
         and args.output.is_absolute() and args.cache.is_absolute() and args.output.name == args.source[:8] + "-linux-same-bank-solver.json"
         and args.cache.name == args.source[:8] + "-cpu-same-bank-cache"
         and args.output.parent == args.cache.parent == host.ROOT / "artifacts/tools/ada-measured-boundary"
         and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists()
         and not args.output.with_suffix(".npz").exists() and not args.cache.exists(), "fresh canonical closed source-bound control outputs/cache/input")
    predecessor = host.ROOT / "artifacts/evaluations/ada-duck-contact-0ce8c9d0a820"
    running, services, foreign = unit_identity(args.source), host.service_snapshot(), host.foreign_processes()
    need(foreign == measured.FOREIGN, "only unchanged predeclared Dino GPU owner")
    versions = {k: version(k) for k in VERSIONS}; need(versions == VERSIONS, "unchanged exact frozen packages including BAM")
    audit = response.source.audit(); gpu, gpu_arrays, layout, manifest, _ = gpu_packet(args.input, predecessor)
    reference = provenance_reference(predecessor); prior, banks = saved.p.prior.authenticated_banks(predecessor)
    import warp as wp
    import torch
    args.cache.mkdir(); saved.configure_private_cpu_cache(wp, args.cache); wp.init()
    need(all(d.is_cpu for d in wp.get_devices()) and not torch.cuda.is_initialized(), "CPU-only control runtime")
    value, arrays = run(prior, banks, gpu_arrays, layout, manifest, gpu["child"])
    need(host.source_check(args.source) == identity and response.source.audit() == audit
         and host.service_snapshot() == services and host.foreign_processes() == foreign and not torch.cuda.is_initialized(),
         "unchanged exact source/services/foreign owners and no initialized CUDA")
    executables = saved.passive_executables(); validate_executables(executables, reference)
    value.update(protocol=PROTOCOL, decision=DECISION, source=args.source, module_sha256=sha256(module).hexdigest(), versions=versions,
        gpu_source=GPU_SOURCE, gpu_report_sha256=GPU_REPORT_SHA, gpu_file_sha256=gpu["files"], source_audit_sha256=packet.digest(audit),
        provenance_reference_sha256=CPU_RESPONSE_SHA, private_cache_was_absent=True, private_cache_files=saved.cache_files(args.cache),
        existing_cpu_executables=executables, payload=saved.p.retain(args.output.with_suffix(".npz"), arrays),
        running_service=running, protected_services=services, foreign_compute_processes=foreign)
    host.write_json(args.output, value); receive(args.output, args.source, args.input, predecessor, cache=args.cache)
    print(DECISION, value["cpu_counters"], value["comparison"]["complete_solver_outputs"])


if __name__ == "__main__": main()
