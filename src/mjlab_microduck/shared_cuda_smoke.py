"""A separately capped CUDA coexistence diagnostic, never Duck admission.

The CPU owner preserves the existing FilmBrain lease and samples Windows and
WSL. Only a fresh child imports Torch; it runs fixed elementwise operations,
not the unqualified robot simulator, a policy, or an optimizer. Old idle gates
are deliberately neither imported nor changed.
"""

import argparse
import base64
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

PROTOCOL = "microduck-shared-cuda-smoke-oct8-v1"
ROOT = Path("/home/yanbo/work/microduck_rl-com-entry-20261006")
BRANCH = "feat/athletics-obstacle-curriculum"
MACHINE = "7d6778c98cb345788b8c1a410f19ad35"
GPU = "GPU-7d72b360-33bc-2cee-3ff4-a954474011b5"
DRIVER = "595.95"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
LOCK = Path("/home/yanbo/data/recomo/film-brain/runtimes/wan22-fun-camera-3e8d686-py312-cu130/wan-gpu.lock")
VENV = Path("/home/yanbo/work/microduck_rl-stance-replication-20260930/.venv")
PROTECTED = ("recomo-ai-mission-vllm.service", "recomo-ai-mission-subject-model-worker.service")
FILMBRAIN = {"recomo-filmbrain-observatory.service": 521,
             "recomo-filmbrain-video-playground.service": 298048}
VERSIONS = {"torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
            "mujoco-warp": "3.8.1", "mjlab": "1.3.0"}
BOUNDS = dict(service_seconds=120, owner_seconds=90, child_seconds=60,
              system_ram_bytes=4 * 1024**3, total_used_mib=12288,
              free_reserve_mib=10240, aggregate_growth_mib=2048,
              temperature_c=65, utilization_percent=85,
              torch_allocator_bytes=512 * 1024**2,
              tensor_peak_bytes=128 * 1024**2, rounds=8, elements=4 * 1024**2)
FLAGS = dict(exclusive_gpu_claimed=False, simulator_qualified=False,
             training_authorized=False, learned_skill_accepted=False,
             physical_motion_authorized=False)
QUERY = "uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
READ_DEADLINE = None
WINDOWS_SCRIPT = r"""
$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$smi=Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
$rows=@(& $smi --query-gpu=uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
if($LASTEXITCODE -ne 0){throw 'Windows GPU telemetry failed'}
$engines=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object UtilizationPercentage -GT 0 | Sort-Object UtilizationPercentage -Descending | Select-Object -First 8 Name,UtilizationPercentage)
[ordered]@{machine=$env:COMPUTERNAME;sampled_at=(Get-Date -Format o);gpu_csv=$rows;active_engines=$engines}|ConvertTo-Json -Depth 5 -Compress
"""


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_result(*args, timeout=8):
    if READ_DEADLINE is not None:
        remaining = READ_DEADLINE - time.monotonic()
        need(remaining > 0, "owned diagnostic read deadline")
        timeout = min(timeout, remaining)
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=True)
    need(len(result.stdout.encode()) <= 65536 and len(result.stderr.encode()) <= 65536,
         "bounded read-only telemetry output")
    return result


def read(*args, timeout=8):
    return read_result(*args, timeout=timeout).stdout.strip()


def parse_gpu(raw):
    rows = raw.splitlines()
    need(len(rows) == 1, "one physical GPU telemetry row")
    parts = [value.strip() for value in rows[0].split(",")]
    need(len(parts) == 7 and parts[:2] == [GPU, DRIVER], "unchanged GPU UUID and driver")
    need(all(re.fullmatch(r"[0-9]+", value) for value in parts[2:]), "plain GPU counters")
    total, used, free, util, temp = map(int, parts[2:])
    need(total == 24467 and 0 <= used <= total and 0 <= free <= total
         and 0 <= util <= 100 and 0 <= temp <= 120, "valid physical GPU counters")
    return dict(uuid=GPU, driver=DRIVER, total_mib=total, used_mib=used,
                free_mib=free, utilization_percent=util, temperature_c=temp)


def telemetry():
    encoded = base64.b64encode(WINDOWS_SCRIPT.encode("utf-16le")).decode()
    windows = json.loads(read(POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive",
                              "-EncodedCommand", encoded))
    need(windows["machine"] == "DESKTOP-HNKBDR1"
         and type(windows["gpu_csv"]) is list and len(windows["gpu_csv"]) == 1,
         "Windows telemetry identity")
    return dict(wall_time_unix=time.time(),
                wsl=parse_gpu(read(SMI, "--query-gpu=" + QUERY, "--format=csv,noheader,nounits")),
                windows=parse_gpu(windows["gpu_csv"][0]),
                windows_sampled_at=windows["sampled_at"],
                windows_active_engines=windows["active_engines"],
                counters_simultaneous=False)


def capacity(sample, baseline=None):
    for view in ("wsl", "windows"):
        current = sample[view]
        need(current["uuid"] == GPU and current["driver"] == DRIVER, "device identity changed")
        need(current["free_mib"] >= BOUNDS["free_reserve_mib"]
             and current["used_mib"] <= BOUNDS["total_used_mib"], "shared memory reserve breached")
        need(current["temperature_c"] < BOUNDS["temperature_c"], "shared thermal guard")
        need(current["utilization_percent"] <= BOUNDS["utilization_percent"], "shared demand guard")
        if baseline is not None:
            need(current["used_mib"] - baseline[view]["used_mib"] <= BOUNDS["aggregate_growth_mib"],
                 "aggregate memory growth guard; not per-process attribution")
    return sample


def source_check(source):
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source), "exact Git source")
    need(Path.cwd().resolve() == ROOT and Path(__file__).resolve().parents[2] == ROOT
         and read("git", "rev-parse", "HEAD") == source
         and read("git", "branch", "--show-current") == BRANCH
         and not read("git", "status", "--porcelain"), "clean exact source worktree")
    raw = Path(__file__).read_bytes()
    committed = read_result("git", "show", source + ":src/mjlab_microduck/shared_cuda_smoke.py").stdout.encode()
    need(raw == committed, "executed module matches committed bytes")
    need(Path("/etc/machine-id").read_text().strip() == MACHINE, "exact WSL machine")
    need((ROOT / ".venv").is_symlink() and (ROOT / ".venv").resolve() == VENV,
         "existing frozen environment alias")
    need(Path(sys.prefix).resolve() == VENV and Path(sys.executable).parent.parent.resolve() == VENV
         and Path(sys.executable).resolve() == (VENV / "bin/python").resolve(),
         "actual existing frozen venv interpreter and prefix")
    need({name: importlib.metadata.version(name) for name in VERSIONS} == VERSIONS,
         "unchanged frozen package versions")
    return dict(commit=source, module_sha256=sha256(raw).hexdigest(), versions=VERSIONS,
                interpreter=str(Path(sys.executable).resolve()), prefix=str(VENV))


def service_snapshot():
    result = {}
    for name in PROTECTED:
        for scope in ([], ["--user"]):
            value = read("systemctl", *scope, "show", name, "-p", "ActiveState", "--value")
            need(value == "inactive", "protected GPU service must remain inactive")
            result[("user:" if scope else "system:") + name] = value
    for name, pid in FILMBRAIN.items():
        values = dict(line.split("=", 1) for line in read(
            "systemctl", "--user", "show", name, "-p", "ActiveState", "-p", "MainPID", "-p", "NRestarts").splitlines())
        need(values == dict(ActiveState="active", MainPID=str(pid), NRestarts="0"),
             "preserved FilmBrain PID/state/restarts")
        result[name] = values
    return result


def lease_identity(fd):
    actual, linked = os.fstat(fd), LOCK.stat(follow_symlinks=False)
    need(stat.S_ISREG(actual.st_mode) and actual.st_size == 0
         and (actual.st_dev, actual.st_ino) == (linked.st_dev, linked.st_ino)
         == (2096, 35886), "same pre-existing empty FilmBrain lease")
    return dict(device=actual.st_dev, inode=actual.st_ino, bytes=actual.st_size)


def directory(source):
    path = ROOT / "artifacts/evaluations" / ("shared-cuda-smoke-" + source[:12])
    need(path.parent.resolve(strict=True) == path.parent, "canonical retained evidence parent")
    return path


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    need(len(raw) <= 256 * 1024, "bounded evidence JSON")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_child(result):
    need(set(result) == {"protocol", "device", "gpu_uuid", "gpu_name", "capability", "torch", "torch_cuda",
         "rounds", "elements", "all_values_exact", "autograd_enabled", "allocator_cap_bytes",
         "peak_allocated_bytes", "peak_reserved_bytes", "flags"}, "closed CUDA-only receipt schema")
    need(result["protocol"] == PROTOCOL and result["device"] == "cuda:0" and result["gpu_uuid"] == GPU
         and result["gpu_name"] == "NVIDIA RTX PRO 4000 Blackwell" and result["torch_cuda"] == "12.8"
         and result["capability"] == [12, 0] and result["torch"] == "2.9.1"
         and result["rounds"] == BOUNDS["rounds"] and result["elements"] == BOUNDS["elements"]
         and result["all_values_exact"] is True and result["autograd_enabled"] is False
         and result["allocator_cap_bytes"] == BOUNDS["torch_allocator_bytes"]
         and 0 < result["peak_allocated_bytes"] <= result["peak_reserved_bytes"]
         <= BOUNDS["tensor_peak_bytes"] and result["flags"] == FLAGS,
         "exact bounded CUDA-only child receipt")


def physical_uuid(raw):
    need(type(raw) is list and len(raw) == 16
         and all(type(value) is int and 0 <= value <= 255 for value in raw), "literal 16-byte CUDA UUID")
    return "GPU-" + str(uuid.UUID(bytes=bytes(raw)))


def child(source, fd):
    source_check(source)
    lease_identity(fd)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "explicit single CUDA child")
    root = directory(source)
    cache = root / "private-cache"
    for key in ("XDG_CACHE_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
        need(Path(os.environ.get(key, "")).resolve() == cache / key.lower(), "private cache binding")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized() and torch.cuda.is_available(), "fresh CUDA context")
    need(torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (12, 0),
         "actual sm120 child device")
    properties = torch.cuda.get_device_properties(0)
    mapped_uuid = physical_uuid(properties.uuid.bytes)
    need(mapped_uuid == GPU and properties.name == "NVIDIA RTX PRO 4000 Blackwell"
         and str(torch.version.cuda) == "12.8", "actual mapped child GPU and frozen Torch runtime")
    torch.cuda.set_per_process_memory_fraction(BOUNDS["torch_allocator_bytes"] / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        left = torch.full((BOUNDS["elements"],), 1.25, dtype=torch.float32, device="cuda:0")
        right = torch.full_like(left, 2.5)
        for _ in range(BOUNDS["rounds"]):
            product = left * right
            torch.cuda.synchronize(0)
            need(bool(torch.all(product == 3.125).item()), "exact deterministic tensor result")
            del product
            time.sleep(1)
        result = dict(protocol=PROTOCOL, device="cuda:0", capability=list(torch.cuda.get_device_capability(0)),
                      gpu_uuid=mapped_uuid, gpu_name=properties.name,
                      torch=importlib.metadata.version("torch"), torch_cuda=str(torch.version.cuda),
                      rounds=BOUNDS["rounds"], elements=BOUNDS["elements"], all_values_exact=True,
                      autograd_enabled=torch.is_grad_enabled(), allocator_cap_bytes=BOUNDS["torch_allocator_bytes"],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(0), flags=FLAGS)
    validate_child(result)
    write_json(root / "child.json", result)


def owned_service(source):
    unit = "microduck-shared-cuda-smoke-" + source[:12] + ".service"
    values = dict(line.split("=", 1) for line in read("systemctl", "--user", "show", unit,
        "-p", "MainPID", "-p", "ActiveState", "-p", "RuntimeMaxUSec", "-p", "MemoryMax",
        "-p", "CPUQuotaPerSecUSec", "-p", "TasksMax", "-p", "Nice", "-p", "KillMode", "-p", "LimitFSIZE").splitlines())
    need(values == dict(MainPID=str(os.getpid()), ActiveState="active", RuntimeMaxUSec="2min",
         MemoryMax=str(BOUNDS["system_ram_bytes"]), CPUQuotaPerSecUSec="1s", TasksMax="64",
         Nice="10", KillMode="control-group", LimitFSIZE=str(4 * 1024**2)), "actual bounded owner service")
    return dict(unit=unit, properties=values)


def supervise(source):
    global READ_DEADLINE
    started = time.monotonic()
    READ_DEADLINE = started + BOUNDS["owner_seconds"]
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules,
         "CPU-only owner before admission")
    identity = source_check(source)
    service = owned_service(source)
    services = service_snapshot()
    env_info = (ROOT / ".venv").lstat()
    frozen_env = (env_info.st_dev, env_info.st_ino, env_info.st_mtime_ns, env_info.st_ctime_ns)
    root = directory(source)
    root.mkdir(mode=0o700, exist_ok=False)
    report = dict(protocol=PROTOCOL, source=identity, service=service, bounds=BOUNDS,
                  flags=FLAGS, decision="failed", telemetry=[], child_exit=None,
                  services_before=services, failure_stage="shared-capacity-admission",
                  claim="Tensor CUDA coexistence only; no simulator or learner admission.")
    proc = None
    watchdog = None
    expired = threading.Event()
    try:
        fd = os.open(LOCK, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            lease = lease_identity(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report["lease"] = lease
            baseline = telemetry()
            report["telemetry"].append(baseline)
            capacity(baseline)
            child_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
            for key in ("XDG_CACHE_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
                cache = root / "private-cache" / key.lower()
                cache.mkdir(parents=True, exist_ok=False)
                child_env[key] = str(cache)
            write_json(root / "launch.json", dict(protocol=PROTOCOL, source=identity, bounds=BOUNDS,
                       baseline=baseline, lease=lease, flags=FLAGS, windows_script_sha256=sha256(WINDOWS_SCRIPT.encode()).hexdigest()))
            with (root / "child.log").open("xb") as log:
                proc = subprocess.Popen([sys.executable, "-m", "mjlab_microduck.shared_cuda_smoke",
                    "--source", source, "--child", "--lease-fd", str(fd)], env=child_env,
                    pass_fds=(fd,), stdout=log, stderr=subprocess.STDOUT)
                child_started = time.monotonic()
                report["failure_stage"] = "owned-child-monitor"
                READ_DEADLINE = min(READ_DEADLINE, child_started + BOUNDS["child_seconds"])
                def hard_stop_owned_child():
                    if proc.poll() is None:
                        expired.set()
                        proc.kill()  # Independent watchdog, this Popen child only.
                watchdog = threading.Timer(max(0, READ_DEADLINE - time.monotonic()), hard_stop_owned_child)
                watchdog.daemon = True
                watchdog.start()
                while True:
                    sample = telemetry()
                    report["telemetry"].append(sample)
                    capacity(sample, baseline)
                    need(service_snapshot() == services and lease_identity(fd) == lease,
                         "preserved services and lease during diagnostic")
                    need(not expired.is_set() and time.monotonic() - child_started < BOUNDS["child_seconds"]
                         and time.monotonic() - started < BOUNDS["owner_seconds"], "owned diagnostic deadline")
                    if proc.poll() is not None:
                        break
                    time.sleep(.5)
                report["child_exit"] = proc.returncode
                need(proc.returncode == 0, "owned CUDA child failed; inspect retained child.log")
                watchdog.cancel()
                watchdog.join(timeout=1)
                READ_DEADLINE = started + BOUNDS["owner_seconds"]
            report["failure_stage"] = "post-exit-closeout"
            post_exit = telemetry()
            report["telemetry"].append(post_exit)
            capacity(post_exit, baseline)
            report["post_exit_telemetry"] = post_exit
            need(time.monotonic() < READ_DEADLINE and lease_identity(fd) == lease, "post-exit deadline and preserved lease")
            need((root / "child.json").stat().st_size <= 65536, "bounded child receipt")
            result = json.loads((root / "child.json").read_bytes())
            validate_child(result)
            need(source_check(source) == identity and service_snapshot() == services, "stable source and services after child")
            after = (ROOT / ".venv").lstat()
            need(frozen_env == (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns), "unchanged frozen environment alias")
            need(time.monotonic() < READ_DEADLINE, "owner closeout deadline before success")
            report.update(decision="shared-cuda-tensor-smoke-complete-not-training", child=result,
                          services_unchanged=True, lease_unchanged=True, environment_alias_unchanged=True,
                          failure_stage=None)
        finally:
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join(timeout=1)
            if proc is not None and proc.poll() is None:
                proc.terminate()  # This Popen child only, never a foreign process.
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            os.close(fd)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if proc is not None:
            report["child_exit"] = proc.returncode
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
        need(args.lease_fd is None, "owner acquires lease itself")
        supervise(args.source)


if __name__ == "__main__":
    main()
