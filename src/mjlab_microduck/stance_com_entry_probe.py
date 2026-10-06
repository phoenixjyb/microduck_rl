"""One leased, source-bound actual-entry CoM diagnostic; never a learner."""

import argparse
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
from importlib.metadata import distribution, version
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time

from mjlab_microduck.stance_com_entry_receiver import FLAGS, SERVICE_CAPS, canonical

PROTOCOL = "microduck-com-actual-entry-probe-v1"
MODULE = "mjlab_microduck.stance_com_entry_probe"
BASE = "538a29fe360ded475ac58e4fa6ed482a5ddf9f50"
OLD_ROOT = Path("/home/yanbo/work/microduck_rl-stance-replication-20260930")
ROOT = Path("/home/yanbo/work/microduck_rl-com-entry-20261006")
BRANCH = "feat/athletics-obstacle-curriculum"
ORIGIN = "https://github.com/phoenixjyb/microduck_rl.git"
MACHINE = "7d6778c98cb345788b8c1a410f19ad35"
GPU = "GPU-7d72b360-33bc-2cee-3ff4-a954474011b5"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
LOCK = Path(
    "/home/yanbo/data/recomo/film-brain/runtimes/wan22-fun-camera-3e8d686-py312-cu130/wan-gpu.lock"
)
VERSIONS = {
    "torch": "2.9.1",
    "warp-lang": "1.12.0",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.8.1",
    "mjlab": "1.3.0",
    "better-actuator-models": "1.0.1",
}
CUTOFF = datetime(2026, 10, 6, 5, 0, tzinfo=timezone.utc).timestamp()
SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS, MARGIN = 300, 240, 300, 60
MEMORY = 6 * 1024**3
EXPECTED_TESTS = 1855
OWN = {
    f"src/mjlab_microduck/stance_com_entry_{name}.py"
    for name in ("capture", "probe", "receiver")
}
OWN |= {
    f"tests/test_stance_com_entry_{name}.py"
    for name in ("capture", "probe", "receiver")
}
OWN.add("docs/experiments/2026-10-06-com-actual-entry-diagnostic.md")
PREDECESSORS = {
    "artifacts/tools/stance-crb-runtime-partial-owner-terminal-2ecee471f7b9.json": "dc2aa2d344fcad58d2054a2238c15d0d3678724852160834dc5a2a29bef8151a",
    "artifacts/tools/stance-crb-runtime-partial-mac-2ecee471f7b9/partial-mac-verification-2ecee471f7b9.json": "5d77483122dc390a53c5d277828f9d8eb7196856b0d950e2445bf9694ab4b167",
    "artifacts/tools/stance-cpu-plant-fields-2ecee471f7b9-20261006-0720/fields.json": "e8160827a52db740a6edc2e89973e5313f0020ed7be79ef609c903a1cf8f6ace",
}
BAD_LOG = re.compile(
    r"\b(?:nan|nonfinite|overflow|warning|traceback)\b|CUDA error", re.I
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(*command):
    return subprocess.check_output(command, text=True, timeout=15).strip()


def bounded(path, limit, *, allow_empty=False):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(
            stat.S_ISREG(before.st_mode)
            and (allow_empty or before.st_size > 0)
            and before.st_size <= limit,
            "bounded regular file",
        )
        raw = b""
        while len(raw) <= limit:
            chunk = os.read(fd, min(65536, limit + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        after = os.fstat(fd)
        need(
            len(raw) == before.st_size
            and (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "stable full file read",
        )
        return raw
    finally:
        os.close(fd)


def write(path, raw, limit):
    need(type(raw) is bytes and 0 < len(raw) <= limit, "bounded output bytes")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check_window(reserve, now=None):
    now = time.time() if now is None else now
    need(
        type(reserve) is int
        and reserve > 0
        and type(now) in (int, float)
        and math.isfinite(now)
        and now >= 0
        and now + reserve < CUTOFF,
        "complete job and closeout before 13:00",
    )


def unit(source, mode):
    need(
        type(source) is str
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and mode in ("tests", "run"),
        "declared source/unit",
    )
    return f"microduck-com-entry-{mode}-{source[:12]}.service"


def output(source, mode):
    unit(source, mode)
    return (
        ROOT
        / "artifacts"
        / ("tools" if mode == "tests" else "evaluations")
        / f"com-entry-{mode}-{source[:12]}"
    )


def source_binding(source):
    unit(source, "run")
    need(
        Path.cwd().resolve() == ROOT and Path(__file__).resolve().parents[2] == ROOT,
        "exact new diagnostic checkout and imported probe",
    )
    need(
        read("git", "rev-parse", "HEAD") == source
        and read("git", "branch", "--show-current") == BRANCH
        and read("git", "remote", "get-url", "origin") == ORIGIN
        and not read("git", "status", "--porcelain"),
        "clean exact branch/origin/source",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, source], check=True, timeout=15
    )
    need(
        set(read("git", "diff", "--name-only", BASE, source).splitlines()) <= OWN,
        "new seven-path source fence",
    )
    leaves = {}
    for path in read(
        "git", "ls-files", "src", "tests", "pyproject.toml", "uv.lock", *sorted(OWN)
    ).splitlines():
        raw = bounded(ROOT / path, 4 * 1024**2, allow_empty=True)
        expected = subprocess.check_output(
            ["git", "show", f"{source}:{path}"], timeout=15
        )
        need(raw == expected, "whole committed source leaf " + path)
        leaves[path] = sha256(raw).hexdigest()
    return {
        "source": source,
        "branch": BRANCH,
        "tree": read("git", "rev-parse", "HEAD^{tree}"),
        "leaf_count": len(leaves),
        "leaves_sha256": sha256(canonical(leaves)).hexdigest(),
    }


def service(source, mode, *, live_pid):
    text = read("systemctl", "--user", "show", unit(source, mode))
    values = dict(row.split("=", 1) for row in text.splitlines() if "=" in row)
    need(
        all(values.get(key) == value for key, value in SERVICE_CAPS.items()),
        "actual retained service caps",
    )
    need(
        values["MainPID"] == str(live_pid)
        and values["ActiveState"] == "active"
        and values["SubState"] == "running"
        and re.fullmatch(r"[0-9a-f]{32}", values["InvocationID"]),
        "actual running retained user service",
    )
    return {
        **{key: values[key] for key in SERVICE_CAPS},
        **{
            key: values[key]
            for key in ("MainPID", "InvocationID", "ActiveState", "SubState")
        },
    }


def packages():
    actual = {name: version(name) for name in VERSIONS}
    need(
        actual == VERSIONS
        and platform.python_version() == "3.12.13"
        and platform.machine() == "x86_64"
        and Path(sys.prefix).resolve() == (OLD_ROOT / ".venv").resolve(),
        "frozen native interpreter/packages",
    )
    trees = {}
    for name, package in (
        ("mjlab", "mjlab"),
        ("mujoco-warp", "mujoco_warp"),
        ("better-actuator-models", "bam"),
    ):
        root = Path(distribution(name).locate_file(package))
        files = {
            str(path.relative_to(root)): sha256(bounded(path, 1024**2)).hexdigest()
            for path in sorted(root.rglob("*.py"))
        }
        need(5 < len(files) < 10000, "bounded installed Python tree")
        trees[name] = {
            "files": len(files),
            "sha256": sha256(canonical(files)).hexdigest(),
        }
    return {
        "versions": actual,
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "python_trees": trees,
    }


def host():
    need(
        sys.platform == "linux"
        and read("hostname") == "DESKTOP-HNKBDR1"
        and Path("/etc/machine-id").read_text().strip() == MACHINE,
        "exact authorized host",
    )
    gpu = [
        value.strip()
        for value in read(
            SMI,
            "--query-gpu=uuid,driver_version,temperature.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ).split(",")
    ]
    need(
        len(gpu) == 5 and gpu[:2] == [GPU, "595.95"] and int(gpu[2]) < 75,
        "GPU identity/driver/temperature",
    )
    for namespace in ((), ("--user",)):
        for name in (
            "recomo-ai-mission-vllm.service",
            "recomo-ai-mission-subject-model-worker.service",
        ):
            need(
                read(
                    "systemctl",
                    *namespace,
                    "show",
                    name,
                    "-p",
                    "ActiveState",
                    "--value",
                )
                == "inactive"
                and read(
                    "systemctl", *namespace, "show", name, "-p", "MainPID", "--value"
                )
                == "0",
                "protected services inactive",
            )
    for name, pid in (
        ("recomo-filmbrain-observatory.service", "521"),
        ("recomo-filmbrain-video-playground.service", "298048"),
    ):
        need(
            read("systemctl", "--user", "show", name, "-p", "ActiveState", "--value")
            == "active"
            and read("systemctl", "--user", "show", name, "-p", "MainPID", "--value")
            == pid
            and read("systemctl", "--user", "show", name, "-p", "NRestarts", "--value")
            == "0",
            "FilmBrain preserved",
        )
    pids = []
    text = read(
        SMI, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
    )
    for row in text.splitlines():
        pid, memory = row.split(",")
        pids.append({"pid": int(pid), "memory_mib": int(memory)})
    return {
        "machine": MACHINE,
        "gpu": GPU,
        "driver": "595.95",
        "temperature_c": int(gpu[2]),
        "used_mib": int(gpu[3]),
        "free_mib": int(gpu[4]),
        "processes": pids,
    }


def predecessors():
    need(
        read("git", "-C", str(OLD_ROOT), "rev-parse", "HEAD")
        == "2ecee471f7b999822ed19defd7e2d1c7ad09df07"
        and not read("git", "-C", str(OLD_ROOT), "status", "--porcelain"),
        "old evidence checkout preserved",
    )
    raw = {name: bounded(OLD_ROOT / name, 256 * 1024) for name in PREDECESSORS}
    need(
        all(
            sha256(raw[name]).hexdigest() == digest
            for name, digest in PREDECESSORS.items()
        ),
        "immutable predecessor whole hashes",
    )
    report = json.loads(
        raw[next(name for name in raw if "partial-mac-verification" in name)]
    )
    need(
        report["all_exact_raw_bits"] is False
        and report["closeout_failed_retained"] is True,
        "original partial rejection remains closed",
    )
    return dict(PREDECESSORS)


def inherited_lease(fd):
    need(type(fd) is int and fd >= 3, "inherited lease FD")
    expected, actual = os.lstat(LOCK), os.fstat(fd)
    need(
        stat.S_ISREG(expected.st_mode)
        and expected.st_size == 0
        and (expected.st_dev, expected.st_ino) == (actual.st_dev, actual.st_ino)
        and os.get_inheritable(fd),
        "existing exact inherited FilmBrain lock",
    )
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return {"device": actual.st_dev, "inode": actual.st_ino, "bytes": actual.st_size}


def test_files():
    from mjlab_microduck.stance_crb_runtime_probe import TEST_FILES

    return ["tests/" + name for name in TEST_FILES] + [
        "tests/test_stance_forward_reduction_plan.py",
        "tests/test_stance_forward_entry_reference.py",
        "tests/test_stance_forward_entry_repeat_analysis.py",
        "tests/test_stance_com_entry_capture.py",
        "tests/test_stance_com_entry_probe.py",
        "tests/test_stance_com_entry_receiver.py",
    ]


def child(source, lease_fd, owner_pid, declaration_sha):
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.getppid() == owner_pid,
        "one actual CUDA child/parent",
    )
    inherited_lease(lease_fd)
    check_window(CHILD_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    binding, runtime, live = (
        source_binding(source),
        packages(),
        service(source, "run", live_pid=owner_pid),
    )
    directory = output(source, "run")
    declaration_raw = bounded(directory / "declaration.json", 128 * 1024)
    need(
        sha256(declaration_raw).hexdigest() == declaration_sha,
        "whole declaration before child decode",
    )
    declaration = json.loads(declaration_raw)
    need(
        declaration["source_binding"] == binding
        and declaration["packages"] == runtime
        and declaration["service_properties"] == live
        and declaration["flags"] == FLAGS,
        "exact native child declaration",
    )
    sample = host()
    need(
        not sample["processes"] and sample["used_mib"] <= 1024,
        "idle GPU before child CUDA init",
    )
    import numpy as np
    import torch
    import warp as wp
    from mjlab_microduck import (
        stance_com_entry_capture as capture,
        stance_plant_evidence as plant,
    )
    from mjlab_microduck import stance_warp_integrator as integrator
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

    need(not torch.cuda.is_initialized(), "no early CUDA initialization")
    for name, module in tuple(sys.modules.items()):
        if name.startswith("mjlab_microduck") and getattr(module, "__file__", None):
            need(
                Path(module.__file__)
                .resolve()
                .is_relative_to(ROOT / "src/mjlab_microduck"),
                "new-checkout imports only",
            )
    torch.manual_seed(673)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(677)
    integration_calls = 0
    original_integrate = integrator.EulerCandidateCommit.integrate

    def refuse_integration(*args, **kwargs):
        nonlocal integration_calls
        integration_calls += 1
        raise ValueError("integration forbidden in actual-entry diagnostic")

    integrator.EulerCandidateCommit.integrate = refuse_integration
    try:
        scope = capture.ComEntryCapture()
        with scope:
            env = WarpStanceRuntime(
                nworld=64, device="cuda:0", solved_field_check="packed"
            )
            fixed_names = (
                "qpos",
                "qvel",
                "time",
                "ctrl",
                "qfrc_applied",
                "xfrc_applied",
            )
            fixed = {
                name: getattr(env.data, name)
                .numpy()
                .astype("<f4", copy=False)
                .tobytes()
                for name in fixed_names
            }
            env._forward()
        descriptor = plant.describe(env.native)
        need(
            descriptor["selected_fields_sha256"]
            == "6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f",
            "fresh CPU compiled descriptor matches retained native",
        )
        need(
            env.forward_graph is None
            and not torch.count_nonzero(env.steps)
            and integration_calls == 0,
            "no graph or physics integration",
        )
        for index, raw in enumerate(scope.entries):
            write(directory / f"entry{index}.bin", raw, 12288)
        receipt = {
            "protocol": PROTOCOL + ":child",
            "source": source,
            "source_binding": binding,
            "declaration_sha256": declaration_sha,
            "owner_pid": owner_pid,
            "child_pid": os.getpid(),
            "child_ppid": os.getppid(),
            "capture_receipt": scope.receipt,
            "compiled_descriptor": descriptor,
            "runtime_ntendon": env.model.ntendon,
            "integration_calls": integration_calls,
            "physics_steps": int(env.steps.sum()),
            "graph_created": False,
            "actor_model_created": False,
            "optimizer_created": False,
            "storage_created": False,
            "flags": dict(FLAGS),
            "repeat_receipts": {},
        }
        if scope.entries[0] != scope.entries[1]:
            receipt.update(
                decision="fresh-entry-inputs-differ", repeat_rng_unchanged=False
            )
        else:
            cpu_rng, cuda_rng = (
                torch.random.get_rng_state().clone(),
                torch.cuda.get_rng_state(0).clone(),
            )
            for mode in ("concurrent", "serial"):
                with wp.ScopedDevice(env.wp_device):
                    bank, row = capture.repeat_entry(
                        env.model,
                        env.data,
                        scope.entries[0],
                        sha256(scope.entries[0]).hexdigest(),
                        mode,
                    )
                write(directory / (mode + ".bin"), bank, 393216)
                receipt["repeat_receipts"][mode] = row
            need(
                torch.equal(cpu_rng, torch.random.get_rng_state())
                and torch.equal(cuda_rng, torch.cuda.get_rng_state(0)),
                "repeat preserves caller CPU/CUDA RNG",
            )
            receipt.update(
                decision="fixed-input-banks-retained",
                repeat_rng_unchanged=True,
                repeat_cpu_rng_sha256=sha256(cpu_rng.numpy().tobytes()).hexdigest(),
                repeat_cuda_rng_sha256=sha256(
                    cuda_rng.cpu().numpy().tobytes()
                ).hexdigest(),
            )
        unchanged = all(
            getattr(env.data, name).numpy().astype("<f4", copy=False).tobytes() == raw
            for name, raw in fixed.items()
        )
        need(
            unchanged and np.isfinite(env.data.subtree_com.numpy()).all(),
            "fixed integration/control state unchanged",
        )
        receipt["fixed_state_unchanged"] = unchanged
        for name, module in tuple(sys.modules.items()):
            if name.startswith("mjlab_microduck") and getattr(module, "__file__", None):
                need(
                    Path(module.__file__)
                    .resolve()
                    .is_relative_to(ROOT / "src/mjlab_microduck"),
                    "new-checkout imports remain bound at child closure",
                )
        need(
            source_binding(source) == binding and packages() == runtime,
            "unchanged source/packages at child closeout",
        )
        write(directory / "child.json", canonical(receipt), 32 * 1024)
    finally:
        if integrator.EulerCandidateCommit.integrate is refuse_integration:
            integrator.EulerCandidateCommit.integrate = original_integrate


def run_tests(source, directory, binding, live, runtime):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only test service")
    test_paths = test_files()
    log = directory / "pytest.log"
    with log.open("xb") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--junitxml=" + str(directory / "junit.xml"),
                *test_paths,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=CHILD_SECONDS,
        )
        stream.flush()
        os.fsync(stream.fileno())
    need(result.returncode == 0, "new exact native CPU suite passed")
    import xml.etree.ElementTree as ET

    fd = os.open(directory / "junit.xml", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        need(stat.S_ISREG(os.fstat(fd).st_mode), "regular generated JUnit file")
        os.fsync(fd)
    finally:
        os.close(fd)
    xml = bounded(directory / "junit.xml", 1024**2)
    suites = list(ET.fromstring(xml).iter("testsuite"))
    need(
        suites
        and sum(int(s.get("tests", 0)) for s in suites) == EXPECTED_TESTS
        and all(
            int(s.get(k, 0)) == 0
            for s in suites
            for k in ("errors", "failures", "skipped")
        ),
        "zero skips/failures/errors",
    )
    receipt = {
        "protocol": PROTOCOL + ":tests",
        "source": source,
        "source_binding": binding,
        "tests": sum(int(s.get("tests", 0)) for s in suites),
        "test_files": test_paths,
        "service_properties": live,
        "packages": runtime,
        "junit_sha256": sha256(xml).hexdigest(),
        "log_sha256": sha256(bounded(log, 1024**2)).hexdigest(),
        "flags": dict(FLAGS),
    }
    need(source_binding(source) == binding, "clean test source at closure")
    write(directory / "receipt.json", canonical(receipt), 128 * 1024)


def run(source, directory, binding, live, runtime):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-hidden supervisor")
    test_raw = bounded(output(source, "tests") / "receipt.json", 128 * 1024)
    test = json.loads(test_raw)
    need(
        test["source_binding"] == binding
        and test["packages"] == runtime
        and type(test["tests"]) is int
        and test["tests"] == EXPECTED_TESTS
        and test["test_files"] == test_files()
        and test["source"] == source
        and test["protocol"] == PROTOCOL + ":tests",
        "new source-addressed CPU tests",
    )
    before = os.lstat(LOCK)
    need(
        stat.S_ISREG(before.st_mode) and before.st_size == 0,
        "existing regular lease file",
    )
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    process = None
    try:
        need(
            (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
            == (before.st_dev, before.st_ino),
            "stable lease inode",
        )
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sample = host()
        need(
            not sample["processes"] and sample["used_mib"] <= 1024,
            "fresh leased idle GPU",
        )
        check_window(SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN)
        declaration = {
            "protocol": PROTOCOL + ":declaration",
            "source": source,
            "source_binding": binding,
            "packages": runtime,
            "host": sample,
            "service_properties": live,
            "predecessors": predecessors(),
            "tests_receipt_sha256": sha256(test_raw).hexdigest(),
            "flags": dict(FLAGS),
        }
        declaration_raw = canonical(declaration)
        write(directory / "declaration.json", declaration_raw, 128 * 1024)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        with (directory / "child.log").open("xb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    MODULE,
                    "--mode",
                    "child",
                    "--source",
                    source,
                    "--lease-fd",
                    str(fd),
                    "--owner-pid",
                    str(os.getpid()),
                    "--declaration-sha",
                    sha256(declaration_raw).hexdigest(),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                pass_fds=(fd,),
            )
            start, observations, seen, observed_ppid = time.monotonic(), [], False, None
            while process.poll() is None:
                need(
                    time.monotonic() - start < CHILD_SECONDS, "bounded CUDA child time"
                )
                stat_path = Path(f"/proc/{process.pid}/stat")
                if stat_path.exists():
                    observed_ppid = int(
                        stat_path.read_text().rsplit(") ", 1)[1].split()[1]
                    )
                    need(
                        observed_ppid == os.getpid(), "actual child parent observation"
                    )
                current = host()
                need(
                    all(row["pid"] == process.pid for row in current["processes"]),
                    "no overlapping GPU process",
                )
                seen |= any(row["pid"] == process.pid for row in current["processes"])
                observations.append(
                    {
                        "elapsed": time.monotonic() - start,
                        "child_pid": process.pid,
                        "host": current,
                    }
                )
                need(
                    len(observations) <= 480 and log.tell() <= 1024**2,
                    "bounded monitoring/log",
                )
                time.sleep(0.5)
            log.flush()
            os.fsync(log.fileno())
        need(
            process.returncode == 0 and seen and observed_ppid == os.getpid(),
            "observed successful GPU child",
        )
        child_log = bounded(directory / "child.log", 1024**2)
        filtered = "\n".join(
            line
            for line in child_log.decode().splitlines()
            if line != "[mdp] Patches 1-2 active: NaN-safe reward/advantage"
        )
        need(not BAD_LOG.search(filtered), "finite warning-free owned GPU log")
        child_raw = bounded(directory / "child.json", 32 * 1024)
        report = {
            "protocol": PROTOCOL + ":report",
            "source": source,
            "source_binding": binding,
            "child_sha256": sha256(child_raw).hexdigest(),
            "returncode": process.returncode,
            "observed_child_pid": process.pid,
            "observed_child_ppid": observed_ppid,
            "gpu_child_observed": seen,
            "monitor": observations,
            "flags": dict(FLAGS),
        }
        need(
            sum(path.stat().st_size for path in directory.iterdir()) < 16 * 1024**2,
            "total output cap",
        )
        need(
            source_binding(source) == binding and packages() == runtime,
            "unchanged owner source/packages at closure",
        )
        write(directory / "report.json", canonical(report), 256 * 1024)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=("tests", "run", "child"), required=True)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--declaration-sha")
    args = parser.parse_args()
    if args.mode == "child":
        child(args.source, args.lease_fd, args.owner_pid, args.declaration_sha)
        return
    check_window(
        (2 if args.mode == "tests" else 1) * SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN
    )
    binding = source_binding(args.source)
    live, runtime = service(args.source, args.mode, live_pid=os.getpid()), packages()
    host()
    directory = output(args.source, args.mode)
    directory.mkdir(parents=True, exist_ok=False)
    function = run_tests if args.mode == "tests" else run
    function(args.source, directory, binding, live, runtime)


if __name__ == "__main__":
    main()
