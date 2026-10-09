"""Bounded Ada tensor/kernel health check, never simulator or training admission.

The stdlib-only owner takes the existing advisory Duck lock and supervises one
fresh child. No predecessor collector, runtime gate, solver or policy is used.
"""

import argparse
import fcntl
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
import time
import uuid

PROTOCOL = "microduck-ada-runtime-smoke-oct9-v1"
ROOT = Path("/home/converge/work/microduck_rl-athletics-obstacle-curriculum")
BRANCH = "feat/athletics-obstacle-curriculum"
MACHINE = "0c79e415429b4933a400159bfa79a34d"
GPU = "GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd"
DRIVER = "595.91.07"
NAME = "NVIDIA GeForce RTX 4090 Laptop GPU"
LOCK = Path("/home/converge/.local/state/microduck-gpu0.lock")
VERSIONS = {"torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
            "mujoco-warp": "3.8.1", "mjlab": "1.3.0"}
SERVICES = ("recomo-ai-mission-vllm.service", "recomo-ai-mission-subject-model-worker.service")
BOUNDS = dict(service_seconds=180, owner_seconds=150, child_seconds=120,
              free_reserve_mib=10240, total_used_mib=12288, aggregate_growth_mib=2048,
              temperature_c=65, utilization_percent=85, elements=32,
              torch_allocator_bytes=64 * 1024**2)
FLAGS = dict(exclusive_gpu_claimed=False, runtime_origin_authenticated=False,
             solver_qualified=False, simulator_qualified=False, training_authorized=False,
             learned_skill_accepted=False, physical_motion_authorized=False)
CACHES = ("XDG_CACHE_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR",
          "CUDA_CACHE_PATH", "WARP_CACHE_PATH")
READ_DEADLINE = None


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(*command, binary=False):
    remaining = 5 if READ_DEADLINE is None else min(5, READ_DEADLINE - time.monotonic())
    need(remaining > 0, "read deadline")
    result = subprocess.run(command, capture_output=True, text=not binary, check=True, timeout=remaining)
    need(len(result.stdout if binary else result.stdout.encode()) <= 65536, "bounded command output")
    return result.stdout if binary else result.stdout.strip()


def parse_gpu(raw):
    rows = raw.splitlines()
    need(len(rows) == 1, "one NVIDIA GPU")
    parts = [part.strip() for part in rows[0].split(",")]
    need(len(parts) == 8 and parts[:3] == [GPU, DRIVER, "8.9"], "exact Ada GPU/driver/capability")
    need(all(re.fullmatch(r"[0-9]+", value) for value in parts[3:]), "plain GPU counters")
    total, used, free, utilization, temperature = map(int, parts[3:])
    need(total == 16376 and 0 <= used <= total and 0 <= free <= total and used + free <= total
         and 0 <= utilization <= 100 and 0 <= temperature <= 120, "valid GPU counters")
    return dict(uuid=GPU, driver=DRIVER, capability=[8, 9], total_mib=total,
                used_mib=used, free_mib=free, utilization_percent=utilization, temperature_c=temperature)


def telemetry():
    query = "uuid,driver_version,compute_cap,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
    return dict(wall_time_unix=time.time(), gpu=parse_gpu(read("nvidia-smi", "--query-gpu=" + query,
                                                            "--format=csv,noheader,nounits")))


def capacity(sample, baseline=None):
    row = sample["gpu"]
    need(row["uuid"] == GPU and row["driver"] == DRIVER and row["capability"] == [8, 9], "GPU identity changed")
    need(row["free_mib"] >= BOUNDS["free_reserve_mib"] and row["used_mib"] <= BOUNDS["total_used_mib"],
         "shared VRAM reserve")
    need(row["temperature_c"] < BOUNDS["temperature_c"] and row["utilization_percent"] <= BOUNDS["utilization_percent"],
         "shared demand/thermal guard")
    if baseline is not None:
        need(row["used_mib"] - baseline["gpu"]["used_mib"] <= BOUNDS["aggregate_growth_mib"], "aggregate VRAM growth")
    return sample


def source_check(source):
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source), "exact source SHA")
    need(sys.platform == "linux" and Path.cwd().resolve() == ROOT
         and Path(__file__).resolve().parents[2] == ROOT, "exact Linux checkout")
    need(read("git", "rev-parse", "HEAD") == source and read("git", "branch", "--show-current") == BRANCH
         and not read("git", "status", "--porcelain"), "clean exact branch/source")
    raw = Path(__file__).read_bytes()
    need(raw == read("git", "show", source + ":src/mjlab_microduck/ada_runtime_smoke.py", binary=True),
         "executed committed module")
    need(Path("/etc/machine-id").read_text().strip() == MACHINE, "exact authorized machine")
    venv = ROOT / ".venv"
    need(not venv.is_symlink() and Path(sys.prefix).resolve() == venv
         and Path(sys.executable).parent.parent.resolve() == venv
         and Path(sys.executable).resolve() == (venv / "bin/python").resolve(), "actual frozen venv interpreter")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    need(versions == VERSIONS, "frozen dependency versions")
    need(all(os.environ.get(key) == "1" for key in
             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
         "pre-import single-thread limits")
    return dict(commit=source, tree=read("git", "rev-parse", "HEAD^{tree}"), versions=versions,
                module_sha256=sha256(raw).hexdigest(), lock_sha256=sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
                interpreter=str(Path(sys.executable).resolve()), prefix=str(venv))


def service_snapshot():
    result = {}
    for name in SERVICES:
        for scope in ([], ["--user"]):
            state = read("systemctl", *scope, "show", name, "-p", "ActiveState", "--value")
            need(state == "inactive", "AI mission GPU services must stay inactive")
            result[("user:" if scope else "system:") + name] = state
    return result


def foreign_processes(owned_pid=None):
    raw = read("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits")
    result = []
    for line in raw.splitlines():
        pid, name = [part.strip() for part in line.split(",", 1)]
        need(pid.isdigit(), "plain compute PID")
        if int(pid) != owned_pid:
            result.append(dict(pid=int(pid), name=name))
    return sorted(result, key=lambda item: item["pid"])


def lease_identity(fd):
    actual, linked = os.fstat(fd), LOCK.stat(follow_symlinks=False)
    need(stat.S_ISREG(actual.st_mode) and actual.st_size == 0 and actual.st_uid == os.getuid()
         and (actual.st_dev, actual.st_ino) == (linked.st_dev, linked.st_ino), "same existing owned advisory lease")
    return dict(device=actual.st_dev, inode=actual.st_ino, bytes=actual.st_size)


def directory(source):
    path = ROOT / "artifacts/evaluations" / ("ada-runtime-smoke-" + source[:12])
    need(path.parent.resolve(strict=True) == path.parent, "canonical evidence parent")
    return path


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    need(len(raw) <= 256 * 1024, "bounded evidence JSON")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def expected_values():
    return [float(index * 2 + 1) for index in range(BOUNDS["elements"])]


def warp_affine(wp, device):
    """Fixed toy kernel only; also exercisable on Warp CPU in a separate test."""
    @wp.kernel(enable_backward=False)
    def affine(inputs: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)):
        index = wp.tid()
        output[index] = inputs[index] * 2.0 + 1.0

    inputs = wp.array([float(i) for i in range(BOUNDS["elements"])], dtype=wp.float32, device=device)
    output = wp.empty(BOUNDS["elements"], dtype=wp.float32, device=device)
    wp.launch(affine, dim=BOUNDS["elements"], inputs=[inputs], outputs=[output], device=device)
    wp.synchronize_device(device)
    return output.numpy().tolist()


def validate_child(result):
    need(set(result) == {"protocol", "gpu_uuid", "name", "capability", "torch_cuda", "torch", "warp",
                         "elements", "torch_values", "warp_values", "torch_peak_bytes", "flags"}, "closed child schema")
    need(result["protocol"] == PROTOCOL and result["gpu_uuid"] == GPU and result["name"] == NAME
         and result["capability"] == [8, 9] and result["torch_cuda"] == "12.8"
         and result["torch"] == "2.9.1" and result["warp"] == "1.12.0"
         and type(result["elements"]) is int and result["elements"] == BOUNDS["elements"], "exact child identity")
    for name in ("torch_values", "warp_values"):
        need(type(result[name]) is list and len(result[name]) == BOUNDS["elements"]
             and all(type(value) is float for value in result[name])
             and result[name] == expected_values(), "every element matches independent CPU arithmetic")
    need(type(result["torch_peak_bytes"]) is int and 0 < result["torch_peak_bytes"] <= BOUNDS["torch_allocator_bytes"],
         "bounded Torch tensor allocation, not total VRAM")
    need(type(result["flags"]) is dict and set(result["flags"]) == set(FLAGS)
         and all(value is False for value in result["flags"].values()), "all qualification flags remain false")


def child(source, fd):
    source_check(source)
    lease_identity(fd)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "one exposed child GPU")
    root = directory(source)
    for key in CACHES:
        need(Path(os.environ.get(key, "")).resolve() == root / "private-cache" / key.lower(), "private cache binding")
    import torch
    import warp as wp

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized() and torch.cuda.is_available() and torch.cuda.device_count() == 1, "fresh single CUDA device")
    properties = torch.cuda.get_device_properties(0)
    device_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(properties.uuid.bytes)))
    need(device_uuid == GPU and properties.name == NAME and torch.cuda.get_device_capability(0) == (8, 9)
         and torch.version.cuda == "12.8", "actual Ada CUDA device")
    torch.cuda.set_per_process_memory_fraction(BOUNDS["torch_allocator_bytes"] / properties.total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        values = torch.arange(BOUNDS["elements"], dtype=torch.float32, device="cuda:0")
        torch_output = values * 2.0 + 1.0
        torch.cuda.synchronize(0)
        torch_values = torch_output.cpu().tolist()
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    wp.init()
    device = wp.get_device("cuda:0")
    need(device.is_cuda and device.arch == 89 and device.ordinal == 0, "actual Warp Ada device")

    warp_values = warp_affine(wp, device)
    result = dict(protocol=PROTOCOL, gpu_uuid=device_uuid, name=properties.name, capability=[8, 9],
                  torch_cuda=torch.version.cuda, torch=importlib.metadata.version("torch"),
                  warp=wp.__version__, elements=BOUNDS["elements"], torch_values=torch_values,
                  warp_values=warp_values, torch_peak_bytes=torch.cuda.max_memory_allocated(0), flags=FLAGS)
    validate_child(result)
    write_json(root / "child.json", result)


def owned_service(source):
    unit = "microduck-ada-runtime-smoke-" + source[:12] + ".service"
    values = dict(line.split("=", 1) for line in read("systemctl", "--user", "show", unit,
        "-p", "MainPID", "-p", "ActiveState", "-p", "RuntimeMaxUSec", "-p", "MemoryMax",
        "-p", "CPUQuotaPerSecUSec", "-p", "TasksMax", "-p", "Nice", "-p", "KillMode",
        "-p", "LimitFSIZE", "-p", "Restart").splitlines())
    need(values == dict(MainPID=str(os.getpid()), ActiveState="active", RuntimeMaxUSec="3min",
         MemoryMax=str(6 * 1024**3), CPUQuotaPerSecUSec="2s", TasksMax="64", Nice="10",
         KillMode="control-group", LimitFSIZE=str(16 * 1024**2), Restart="no"), "actual bounded owner unit")
    return dict(unit=unit, invocation=os.environ["INVOCATION_ID"], properties=values)


def supervise(source):
    global READ_DEADLINE
    started = time.monotonic()
    READ_DEADLINE = started + BOUNDS["owner_seconds"]
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not {"torch", "warp"}.intersection(sys.modules), "CPU-only fresh owner")
    identity, service, services = source_check(source), owned_service(source), service_snapshot()
    root = directory(source)
    root.mkdir(mode=0o700, exist_ok=False)
    report = dict(protocol=PROTOCOL, source=identity, service=service, bounds=BOUNDS, flags=FLAGS,
                  decision="failed", telemetry=[], child_exit=None, services_before=services)
    proc = None
    watchdog = None
    fd = None
    expired = threading.Event()
    try:
        fd = os.open(LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        lease = lease_identity(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline = telemetry()
        report["telemetry"].append(baseline)
        capacity(baseline)
        foreign = foreign_processes()
        report.update(lease=lease, foreign_before=foreign)
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
        for key in CACHES:
            cache = root / "private-cache" / key.lower()
            cache.mkdir(parents=True, exist_ok=False)
            child_env[key] = str(cache)
        write_json(root / "launch.json", dict(protocol=PROTOCOL, source=identity, bounds=BOUNDS,
                   baseline=baseline, lease=lease, foreign=foreign, flags=FLAGS))
        with (root / "child.log").open("xb") as log:
            proc = subprocess.Popen([sys.executable, "-m", "mjlab_microduck.ada_runtime_smoke", "--source", source,
                "--child", "--lease-fd", str(fd)], env=child_env, pass_fds=(fd,), stdout=log, stderr=subprocess.STDOUT)
            READ_DEADLINE = min(READ_DEADLINE, time.monotonic() + BOUNDS["child_seconds"])
            def stop_owned_child():
                if proc.poll() is None:
                    expired.set()
                    proc.kill()
            watchdog = threading.Timer(max(0, READ_DEADLINE - time.monotonic()), stop_owned_child)
            watchdog.daemon = True
            watchdog.start()
            while True:
                sample = telemetry()
                report["telemetry"].append(sample)
                capacity(sample, baseline)
                need(service_snapshot() == services and lease_identity(fd) == lease
                     and foreign_processes(proc.pid) == foreign, "preserved services/lease/foreign compute owners")
                need(not expired.is_set() and time.monotonic() < READ_DEADLINE, "owned child deadline")
                if proc.poll() is not None:
                    break
                time.sleep(.5)
            need(proc.returncode == 0, "owned child failed; inspect retained child.log")
            watchdog.cancel()
            watchdog.join(timeout=1)
        READ_DEADLINE = started + BOUNDS["owner_seconds"]
        post_exit = telemetry()
        report["telemetry"].append(post_exit)
        capacity(post_exit, baseline)
        need(source_check(source) == identity and service_snapshot() == services
             and foreign_processes() == foreign and lease_identity(fd) == lease, "stable post-exit bindings")
        path = root / "child.json"
        need(path.stat().st_size <= 65536, "bounded child receipt")
        result = json.loads(path.read_bytes())
        validate_child(result)
        need(time.monotonic() < READ_DEADLINE, "owner closeout deadline")
        report.update(decision="ada-torch-warp-smoke-complete-not-training", child=result,
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
        report["elapsed_seconds"] = time.monotonic() - started
        report["child_exit"] = None if proc is None else proc.returncode
        report["files"] = {name: sha256((root / name).read_bytes()).hexdigest()
                           for name in ("launch.json", "child.log", "child.json") if (root / name).is_file()}
        write_json(root / "report.json", report)
        READ_DEADLINE = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args()
    if args.child:
        need(args.lease_fd is not None and args.lease_fd >= 3, "inherited existing lease FD")
        child(args.source, args.lease_fd)
    else:
        need(args.lease_fd is None, "owner acquires lease")
        supervise(args.source)


if __name__ == "__main__":
    main()
