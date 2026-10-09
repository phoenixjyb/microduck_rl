"""Two-world unconstrained Duck reset/forward diagnostic, never admission.

The owner is stdlib-only. Shared post-reboot health guards remain unchanged;
the child builds the actual stance plant but never initializes BAM or integrates.
"""

import argparse
from hashlib import sha256
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import fcntl
import math
import re

from mjlab_microduck import ada_runtime_smoke as host

PROTOCOL = "microduck-ada-unconstrained-reset-forward-oct9-v1"
MODULE = "mjlab_microduck.ada_duck_forward"
WORLDS = 2
FIELDS = ("qpos", "qvel", "time", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied",
          "xpos", "xquat", "xipos", "ximat", "subtree_com", "cinert", "crb", "cvel",
          "qfrc_bias", "qfrc_actuator", "qfrc_constraint", "qacc_smooth", "qacc")
UNCHANGED = FIELDS[:7]
COUNTERS = ("nf", "nefc", "ncon")
WIDTHS = dict(qpos=21, qvel=20, time=1, qacc_warmstart=20, ctrl=14,
              qfrc_applied=20, xfrc_applied=96, xpos=48, xquat=64, xipos=48,
              ximat=144, subtree_com=48, cinert=160, crb=160, cvel=96,
              qfrc_bias=20, qfrc_actuator=20, qfrc_constraint=20, qacc_smooth=20, qacc=20)
FLAGS = host.FLAGS
BOUNDS = dict(host.BOUNDS, owner_closeout_reserve_seconds=20)
need = host.need
PAYLOAD_MAX = 2 * 1024**2
CHILD_JSON_LIMIT = 65536
FILES = ("launch.json", "child.log", "child.json", "cpu-fields.npz", "gpu-fields.npz")
COLLECTION_DECISION = "ada-unconstrained-duck-forward-collected-pending-reception-not-training"


def selected_profile(probe):
    from types import SimpleNamespace
    profile = SimpleNamespace(**globals()) if probe is None else probe
    need(profile.MODULE in (MODULE, "mjlab_microduck.ada_duck_contact", "mjlab_microduck.ada_measured_gpu_boundary"), "closed bounded diagnostic profile")
    need(profile.BOUNDS == BOUNDS and profile.FLAGS == FLAGS, "unchanged diagnostic bounds and authority")
    return profile


def directory(source):
    path = host.ROOT / "artifacts/evaluations" / ("ada-duck-forward-" + source[:12])
    need(path.parent.resolve(strict=True) == path.parent, "canonical evidence parent")
    return path


def identity(source):
    value = host.source_check(source)
    raw = Path(__file__).read_bytes()
    need(raw == host.read("git", "show", source + ":src/mjlab_microduck/ada_duck_forward.py", binary=True),
         "committed Duck diagnostic module")
    return dict(**value, diagnostic_sha256=sha256(raw).hexdigest())


def specification():
    return dict(protocol=PROTOCOL, worlds=WORLDS, fields=list(FIELDS), unchanged=list(UNCHANGED),
                counters={key: 0 for key in COUNTERS}, cpu_dtype="float64", gpu_dtype="float32",
                input_cast="float32-then-float64-before-native-reference", integration_steps=0,
                motor_preparations=0, optimizer_steps=0, numerical_acceptance_tolerance=None,
                flags=FLAGS)


def comparison(cpu, gpu):
    """Complete flattened fields, with descriptive residuals, no dynamics gate."""
    import numpy as np
    need(set(cpu) == set(gpu) == set(FIELDS), "closed complete field inventory")
    result = {}
    for key in FIELDS:
        reference, actual = cpu[key], gpu[key]
        need(isinstance(reference, np.ndarray) and isinstance(actual, np.ndarray)
             and reference.dtype == np.float64 and actual.dtype == np.float32
             and reference.ndim == actual.ndim == 2 and reference.shape == actual.shape
             and reference.shape == (WORLDS, WIDTHS[key])
             and np.isfinite(reference).all() and np.isfinite(actual).all(), "complete finite field: " + key)
        cast = reference.astype(np.float32)
        delta = actual.astype(np.float64) - reference
        result[key] = dict(shape=list(actual.shape), values=int(actual.size),
                           cpu_sha256=sha256(reference.tobytes()).hexdigest(),
                           gpu_sha256=sha256(actual.tobytes()).hexdigest(),
                           float32_bit_mismatches=int(np.count_nonzero(cast.view(np.uint32) != actual.view(np.uint32))),
                           max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta**2))))
    return result


def retain_arrays(root, name, arrays):
    import numpy as np
    need(set(arrays) == set(FIELDS) and sum(value.nbytes for value in arrays.values()) <= 1024**2,
         "bounded complete uncompressed field payload")
    with (root / name).open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    raw = (root / name).read_bytes()
    need(len(raw) <= 2 * 1024**2, "bounded retained field file")
    return dict(file=name, bytes=len(raw), sha256=sha256(raw).hexdigest())


def physics(root, device="cuda:0"):
    import numpy as np
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp
    import mjlab  # Task discovery precedes BAM imports in the plant builder.
    from mjlab_microduck import stance_plant_evidence as plant
    from mjlab_microduck.football_flat_hold import place_on_floor

    native = plant.build_entity().compile()
    descriptor = plant.describe(native)
    need(descriptor == plant.reference() and (native.nq, native.nv, native.nu) == (21, 20, 14),
         "two fresh same-host compiled actual stance plants")
    cpu_data = mujoco.MjData(native)
    mujoco.mj_resetDataKeyframe(native, cpu_data, native.key("init_state").id)
    cpu_data.qpos[:7] = [0, 0, .2, 1, 0, 0, 0]
    cpu_data.qvel[:] = 0
    cpu_data.ctrl[:] = 0
    gaps = place_on_floor(native, cpu_data)
    # Both implementations receive exactly representable float32 state inputs.
    # Placement itself invokes native forward, so explicitly clear its scratch
    # warmstart/external forces before taking the paired reference.
    for key in UNCHANGED:
        if key == "qpos":
            cpu_data.qpos[:] = cpu_data.qpos.astype(np.float32).astype(np.float64)
        elif key == "time":
            cpu_data.time = 0.
        else:
            getattr(cpu_data, key)[:] = 0
    before_cpu = {key: np.asarray(getattr(cpu_data, key), dtype=np.float64).copy() for key in UNCHANGED}
    need(not native.dof_frictionloss.any(), "uninitialized BAM friction is zero; not a motor constraint test")
    with wp.ScopedDevice(device):
        model = mjwarp.put_model(native)
        data = mjwarp.put_data(native, cpu_data, nworld=WORLDS, nconmax=128, njmax=512)
    wp.synchronize_device(device)
    need(not model.is_sparse and data.naccdmax == data.naconmax, "dense fixed-capacity baseline")
    before_gpu = {key: getattr(data, key).numpy().copy() for key in UNCHANGED}
    for key in UNCHANGED:
        expect = np.tile(before_cpu[key].astype(np.float32).reshape(1, -1), (WORLDS, 1))
        actual = before_gpu[key].reshape(WORLDS, -1)
        need(actual.tobytes() == expect.tobytes(), "exact paired input bytes: " + key)
    mujoco.mj_forward(native, cpu_data)
    need(not cpu_data.warning.number.any() and all(int(getattr(cpu_data, key)) == 0 for key in COUNTERS),
         "native unconstrained no-contact baseline")
    with wp.ScopedDevice(device):
        mjwarp.forward(model, data)
    wp.synchronize_device(device)
    gpu_counters = {key: getattr(data, key).numpy().tolist() for key in ("nf", "nefc", "nacon")}
    need(gpu_counters == dict(nf=[0, 0], nefc=[0, 0], nacon=[0]), "GPU unconstrained no-contact baseline")
    counters = dict(cpu={key: int(getattr(cpu_data, key)) for key in COUNTERS}, gpu=gpu_counters)
    for key in UNCHANGED:
        need(np.asarray(getattr(cpu_data, key)).tobytes() == before_cpu[key].tobytes(), "native forward cannot integrate: " + key)
        need(getattr(data, key).numpy().tobytes() == before_gpu[key].tobytes(), "GPU forward cannot integrate: " + key)
    cpu = {key: np.tile(np.asarray(getattr(cpu_data, key), dtype=np.float64).reshape(1, -1), (WORLDS, 1)) for key in FIELDS}
    gpu = {key: getattr(data, key).numpy().reshape(WORLDS, -1).copy() for key in FIELDS}
    residuals = comparison(cpu, gpu)
    return dict(plant=descriptor, placement_gaps=gaps, counters=counters, residuals=residuals,
                payloads=[retain_arrays(root, "cpu-fields.npz", cpu), retain_arrays(root, "gpu-fields.npz", gpu)])


def validate_child(value):
    need(set(value) == {"specification", "gpu_uuid", "name", "capability", "torch_cuda", "warp_arch",
                        "warp_precompiled_headers", "plant", "placement_gaps", "counters", "residuals", "payloads"},
         "closed Duck child schema")
    need(value["specification"] == specification() and value["gpu_uuid"] == host.GPU
         and value["name"] == host.NAME and value["capability"] == [8, 9] and value["torch_cuda"] == "12.8"
         and value["warp_arch"] == 89 and value["warp_precompiled_headers"] is False, "actual Ada identity and fixed diagnostic")
    need(all(flag is False for flag in value["specification"]["flags"].values()), "strict false qualification flags")
    plant = value["plant"]
    need(type(plant) is dict and set(plant) == {"protocol", "assets", "topology", "joints", "qids", "dofs",
         "floor", "feet", "ranges", "initial_qpos", "selected_fields", "selected_fields_sha256", "options"}
         and plant["protocol"] == "football-b1n-compiled-plant-v1" and plant["topology"][:3] == [21, 20, 14]
         and type(plant["assets"]) is dict and bool(plant["assets"])
         and all(type(h) is str and re.fullmatch(r"[0-9a-f]{64}", h) for h in plant["assets"].values())
         and type(plant["selected_fields_sha256"]) is str
         and re.fullmatch(r"[0-9a-f]{64}", plant["selected_fields_sha256"]), "compiled plant envelope; fresh binding deferred to receiver")
    need(type(value["placement_gaps"]) is list and len(value["placement_gaps"]) == 2
         and all(type(gap) is float and math.isfinite(gap) and 0 <= gap <= .0003 for gap in value["placement_gaps"]),
         "two finite placement-band gaps")
    need(value["counters"] == dict(cpu={key: 0 for key in COUNTERS}, gpu=dict(nf=[0, 0], nefc=[0, 0], nacon=[0]))
         and set(value["residuals"]) == set(FIELDS),
         "complete unconstrained forward receipt")
    need(all(type(x) is int for x in value["counters"]["cpu"].values())
         and all(type(x) is int for row in value["counters"]["gpu"].values() for x in row), "strict zero counter types")
    for key, row in value["residuals"].items():
        need(set(row) == {"shape", "values", "cpu_sha256", "gpu_sha256", "float32_bit_mismatches", "max_abs", "rms"}
             and row["shape"] == [WORLDS, WIDTHS[key]] and all(type(x) is int for x in row["shape"])
             and type(row["values"]) is int and row["values"] == WORLDS * WIDTHS[key]
             and type(row["float32_bit_mismatches"]) is int and 0 <= row["float32_bit_mismatches"] <= row["values"]
             and all(type(row[h]) is str and re.fullmatch(r"[0-9a-f]{64}", row[h]) for h in ("cpu_sha256", "gpu_sha256"))
             and all(type(row[x]) is float and math.isfinite(row[x]) and row[x] >= 0 for x in ("max_abs", "rms")),
             "complete finite residual field: " + key)
        if key in UNCHANGED:
            need(row["float32_bit_mismatches"] == 0 and row["max_abs"] == row["rms"] == 0., "no integration fields")
    need([p["file"] for p in value["payloads"]] == ["cpu-fields.npz", "gpu-fields.npz"], "paired payloads")
    for item in value["payloads"]:
        need(set(item) == {"file", "bytes", "sha256"} and type(item["bytes"]) is int
             and 0 < item["bytes"] <= 2 * 1024**2 and type(item["sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "bounded payload identity")


def receive_payloads(root, value):
    """CPU-only independent recomputation, before accepting retained evidence."""
    import numpy as np
    validate_child(value)
    arrays = []
    for item in value["payloads"]:
        path = root / item["file"]
        need(not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"] <= 2 * 1024**2,
             "plain bounded payload")
        raw = path.read_bytes()
        need(sha256(raw).hexdigest() == item["sha256"], "full payload hash")
        with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
            need(set(bank.files) == set(FIELDS) and len(bank.files) == len(FIELDS), "exact NPZ field inventory")
            arrays.append({key: bank[key].copy() for key in FIELDS})
    need(comparison(*arrays) == value["residuals"], "independently recomputed every residual row")
    return dict(payloads_verified=True, residuals_recomputed=True, flags=FLAGS)


def child(source, fd, *, probe=None):
    profile = selected_profile(probe)
    profile.identity(source)
    host.lease_identity(fd)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.environ.get("CUDA_CACHE_DISABLE") == "1", "fresh capped Ada child")
    root = profile.directory(source)
    for key in host.CACHES:
        need(Path(os.environ.get(key, "")).resolve() == root / "private-cache" / key.lower(), "private cache binding")
    import torch
    import warp as wp
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized() and torch.cuda.is_available() and torch.cuda.device_count() == 1, "fresh single CUDA")
    props = torch.cuda.get_device_properties(0)
    actual_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(props.uuid.bytes)))
    need(actual_uuid == host.GPU and props.name == host.NAME and torch.cuda.get_device_capability(0) == (8, 9)
         and torch.version.cuda == "12.8", "actual Torch Ada binding")
    torch.cuda.set_per_process_memory_fraction(BOUNDS["torch_allocator_bytes"] / props.total_memory, 0)
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    wp.config.use_precompiled_headers = False
    wp.init()
    device = wp.get_device("cuda:0")
    need(device.arch == 89 and device.ordinal == 0 and device.is_cuda, "actual Warp Ada binding")
    result = dict(specification=profile.specification(), gpu_uuid=actual_uuid, name=props.name,
                  capability=[8, 9], torch_cuda=torch.version.cuda, warp_arch=device.arch,
                  warp_precompiled_headers=wp.config.use_precompiled_headers, **profile.physics(root))
    profile.validate_child(result)
    host.write_json(root / "child.json", result)


def owned_service(source, *, prefix="microduck-ada-duck-forward-"):
    need(prefix in ("microduck-ada-duck-forward-", "microduck-ada-duck-contact-", "microduck-ada-measured-boundary-"), "closed diagnostic unit prefix")
    unit = prefix + source[:12] + ".service"
    keys = ("MainPID", "ActiveState", "RuntimeMaxUSec", "MemoryMax", "CPUQuotaPerSecUSec",
            "TasksMax", "Nice", "KillMode", "LimitFSIZE", "Restart", "TimeoutStopUSec")
    values = dict(line.split("=", 1) for line in host.read("systemctl", "--user", "show", unit,
                  *sum((["-p", key] for key in keys), [])).splitlines())
    need(values == dict(MainPID=str(os.getpid()), ActiveState="active", RuntimeMaxUSec="3min",
         MemoryMax=str(6 * 1024**3), CPUQuotaPerSecUSec="2s", TasksMax="64", Nice="10",
         KillMode="control-group", LimitFSIZE=str(16 * 1024**2), Restart="no", TimeoutStopUSec="10s"), "actual bounded Duck owner unit")
    return dict(unit=unit, invocation=os.environ["INVOCATION_ID"], properties=values)


def supervise(source, *, probe=None):
    profile = selected_profile(probe)
    started = time.monotonic()
    host.READ_DEADLINE = started + BOUNDS["owner_seconds"]
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not {"torch", "warp", "mujoco", "mujoco_warp"}.intersection(sys.modules),
         "fresh CPU-only owner")
    inputs, service, services = profile.identity(source), profile.owned_service(source), host.service_snapshot()
    root = profile.directory(source)
    root.mkdir(mode=0o700, exist_ok=False)
    report = dict(specification=profile.specification(), source=inputs, service=service, bounds=BOUNDS,
                  decision="failed", telemetry=[], child_exit=None, services_before=services)
    proc = watchdog = fd = None
    expired = threading.Event()
    try:
        fd = os.open(host.LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        lease = host.lease_identity(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline = host.telemetry()
        report["telemetry"].append(baseline)
        host.capacity(baseline)
        foreign = host.foreign_processes()
        report.update(lease=lease, foreign_before=foreign)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", CUDA_CACHE_DISABLE="1")
        for key in host.CACHES:
            cache = root / "private-cache" / key.lower()
            cache.mkdir(parents=True, exist_ok=False)
            env[key] = str(cache)
        host.write_json(root / "launch.json", dict(specification=profile.specification(), source=inputs,
                        bounds=BOUNDS, baseline=baseline, lease=lease, foreign=foreign))
        with (root / "child.log").open("xb") as log:
            proc = subprocess.Popen([sys.executable, "-m", profile.MODULE, "--source", source,
                "--child", "--lease-fd", str(fd)], env=env, pass_fds=(fd,), stdout=log, stderr=subprocess.STDOUT)
            host.READ_DEADLINE = min(host.READ_DEADLINE - BOUNDS["owner_closeout_reserve_seconds"],
                                     time.monotonic() + BOUNDS["child_seconds"])
            def stop_owned():
                if proc.poll() is None:
                    expired.set()
                    proc.kill()
            watchdog = threading.Timer(max(0, host.READ_DEADLINE - time.monotonic()), stop_owned)
            watchdog.daemon = True
            watchdog.start()
            while True:
                sample = host.telemetry()
                report["telemetry"].append(sample)
                host.capacity(sample, baseline)
                need(host.service_snapshot() == services and host.lease_identity(fd) == lease
                     and host.foreign_processes(proc.pid) == foreign, "preserved foreign owners/services/lease")
                need(not expired.is_set() and time.monotonic() < host.READ_DEADLINE, "owned child deadline")
                if proc.poll() is not None:
                    break
                time.sleep(.5)
            need(proc.returncode == 0, "owned Duck child failed; inspect retained child.log")
            watchdog.cancel()
            watchdog.join(timeout=1)
        host.READ_DEADLINE = started + BOUNDS["owner_seconds"]
        post = host.telemetry()
        report["telemetry"].append(post)
        host.capacity(post, baseline)
        need(profile.identity(source) == inputs and host.service_snapshot() == services
             and host.foreign_processes() == foreign and host.lease_identity(fd) == lease, "post-exit bindings")
        need((root / "child.json").stat().st_size <= profile.CHILD_JSON_LIMIT, "bounded child receipt")
        result = json.loads((root / "child.json").read_bytes())
        profile.validate_child(result)
        for item in result["payloads"]:
            raw = (root / item["file"]).read_bytes()
            need(len(raw) == item["bytes"] <= profile.PAYLOAD_MAX and sha256(raw).hexdigest() == item["sha256"], "paired field byte binding")
        need(time.monotonic() < host.READ_DEADLINE, "owner closeout deadline")
        report.update(decision=profile.COLLECTION_DECISION, child=result,
                      services_unchanged=True, foreign_unchanged=True, lease_unchanged=True)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join(timeout=1)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if fd is not None:
            os.close(fd)
        report.update(elapsed_seconds=time.monotonic() - started, child_exit=None if proc is None else proc.returncode)
        report["files"] = {name: sha256((root / name).read_bytes()).hexdigest() for name in profile.FILES if (root / name).is_file()}
        host.write_json(root / "report.json", report)
        host.READ_DEADLINE = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args()
    if args.child:
        need(args.lease_fd is not None and args.lease_fd >= 3, "inherited advisory lease")
        child(args.source, args.lease_fd)
    else:
        need(args.lease_fd is None, "owner acquires lease")
        supervise(args.source)


if __name__ == "__main__":
    main()
