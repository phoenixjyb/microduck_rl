"""Fresh, capped synthetic CUDA prefix probe. Never a training admission.

The parent has no GPU imports. Only its inherited-lease child creates CUDA
arrays, compiles two isolated retained modules and performs thirty launches.
"""

import argparse
from datetime import datetime, timezone
import fcntl
from hashlib import sha1, sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time

PROTOCOL = "microduck-friction-prefix-cuda-probe-oct8-v1"
ROOT = Path("/home/yanbo/work/microduck_rl-com-entry-20261006")
BRANCH = "feat/athletics-obstacle-curriculum"
BASE = "4452976d981c90d61568fe57d0b43510a610c32a"
CUTOFF = datetime(2026, 10, 7, 23, 30, tzinfo=timezone.utc).timestamp()
SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS, MARGIN = 600, 540, 240, 60
CPU_SERVICE_SECONDS, CPU_TEST_SECONDS = 660, 600
LOCK = Path(
    "/home/yanbo/data/recomo/film-brain/runtimes/wan22-fun-camera-3e8d686-py312-cu130/wan-gpu.lock"
)
SMI = "/usr/lib/wsl/lib/nvidia-smi"
MACHINE = "7d6778c98cb345788b8c1a410f19ad35"
GPU = "GPU-7d72b360-33bc-2cee-3ff4-a954474011b5"
CONTEXT_SHA = "eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee"
WARP_SOURCES_SHA = "4aa3c865b7e523e1c0bef175f80ed51b00543569246d524914e78c333cdf9e6c"
LIBRARIES = {
    "warp.so": (
        283675616,
        "4afdc3ddd8d4c7e4f68837e9b1f4268e767ef8527769321b152dde8cc3b11acd",
    ),
    "warp-clang.so": (
        67215456,
        "f8e0f74720067ee5606f189463d55142928087bc45b59d0e3e5c2b2385425a3a",
    ),
}
# Filled with the exact collected source-test total before source declaration.
EXPECTED_TESTS = 2671
COMPILER_CONFIG = {
    "mode": "release",
    "optimization_level": None,
    "verify_fp": False,
    "llvm_cuda": False,
    "cache_kernels": True,
    "verify_autograd_array_access": False,
    "use_precompiled_headers": False,
}
BASE_FIXTURE_SHA = "8e96c5077f248aad06cacab4c3a45f17cd6b3c35c09c7a46457281916e8046d9"
PREFIX_SHA = "0ff83581832f3264d9f6b447b089fee285f845f12a5a4eb3614a057b82ba0bad"
VERSIONS = {
    "torch": "2.9.1",
    "warp-lang": "1.12.0",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.8.1",
    "mjlab": "1.3.0",
    "better-actuator-models": "1.0.1",
}
FLAGS = {
    name: False
    for name in (
        "runtime_cause_proven",
        "native_qualified",
        "full_window_qualified",
        "training_authorized",
        "physical_acceptance",
    )
}
INPUT_NAMES = ("frictionloss", "qvel", "invweight", "solref", "solimp", "timestep")
OUTPUT_NAMES = (
    "nf",
    "nefc",
    "type",
    "id",
    "row_nnz",
    "row_adr",
    "col_ind",
    "J",
    "pos",
    "margin",
    "D",
    "vel",
    "aref",
    "frictionloss",
    "efc_nnz",
)
CASE_NAMES = (
    "all-dofs",
    "append-exact-fill",
    "broadcast",
    "direct-solref",
    "empty",
    "maximum",
    "mixed-broadcast",
    "overflow",
    "per-world",
    "signed-zero",
)
OWN = {
    "src/mjlab_microduck/stance_friction_prefix_cuda_probe.py",
    "src/mjlab_microduck/stance_friction_prefix_cuda_receiver.py",
    "src/mjlab_microduck/stance_friction_prefix_cuda_numerical.py",
    "tests/test_stance_friction_prefix_cuda_probe.py",
    "tests/test_stance_friction_prefix_cuda_receiver.py",
    "tests/test_stance_friction_prefix_cuda_numerical.py",
    "docs/experiments/2026-10-08-dense-friction-cuda-component.md",
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def configure_compiler(wp):
    """Process-local PCH disablement bounds files; installed config is untouched."""
    current = {key: getattr(wp.config, key) for key in COMPILER_CONFIG}
    need(
        canonical(current)
        == canonical(COMPILER_CONFIG | {"use_precompiled_headers": True}),
        "fresh frozen compiler defaults before local PCH disablement",
    )
    wp.config.use_precompiled_headers = False
    return {key: getattr(wp.config, key) for key in COMPILER_CONFIG}


def read(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def stat_identity(item):
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)


def check_window(reserve, now=None):
    need(type(reserve) is int and reserve >= 0, "plain nonnegative cutoff reserve")
    need(
        (time.time() if now is None else now) + reserve < CUTOFF,
        "Oct8 07:30 Shanghai cutoff and full reserve",
    )


def unit(source, mode):
    need(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source),
        "exact source SHA",
    )
    need(mode in ("tests", "run"), "bounded unit mode")
    return f"microduck-friction-prefix-cuda-{mode}-{source[:12]}.service"


def output(source, mode):
    unit(source, mode)
    return (
        ROOT
        / "artifacts"
        / ("tools" if mode == "tests" else "evaluations")
        / f"friction-prefix-cuda-{mode}-{source[:12]}"
    )


def digest_file(path, cap=16 * 1024**2):
    path = Path(path)
    need(path.resolve(strict=True) == path.absolute(), "no symlink file or parent")
    before = path.stat(follow_symlinks=False)
    need(
        stat.S_ISREG(before.st_mode) and before.st_size <= cap,
        "bounded regular whole file",
    )
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
        after = os.fstat(handle.fileno())
    current = path.stat(follow_symlinks=False)
    need(
        stat_identity(before) == stat_identity(after) == stat_identity(current),
        "stable whole file",
    )
    need(len(raw) == before.st_size, "whole file length")
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def library_binding():
    """Authenticate frozen compiler/runtime libraries without importing Warp."""
    dist = importlib.metadata.distribution("warp-lang")
    result = {}
    for name, (size, expected) in LIBRARIES.items():
        path = Path(dist.locate_file("warp/bin/" + name)).absolute()
        need(path.resolve(strict=True) == path, "plain frozen runtime library path")
        before = path.stat(follow_symlinks=False)
        need(
            stat.S_ISREG(before.st_mode) and before.st_size == size,
            "frozen library length",
        )
        digest = sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024**2):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        current = path.stat(follow_symlinks=False)
        need(
            stat_identity(before) == stat_identity(after) == stat_identity(current)
            and digest.hexdigest() == expected,
            "unchanged complete frozen native library bytes",
        )
        result[name] = {"path": str(path), "bytes": size, "sha256": expected}
    return result


def toolchain_source():
    root = Path(
        importlib.metadata.distribution("warp-lang").locate_file("warp")
    ).absolute()
    paths = sorted(
        set(root.rglob("*.py"))
        | {path for path in (root / "native").rglob("*") if path.is_file()}
    )
    leaves = {str(path.relative_to(root)): digest_file(path) for path in paths}
    need(
        len(leaves) == 460
        and sum(value["bytes"] for value in leaves.values()) == 9292904
        and sha256(canonical(leaves)).hexdigest() == WARP_SOURCES_SHA,
        "complete frozen Warp Python/compiler-header source tree",
    )
    return {"root": str(root), "leaves": leaves, "sha256": WARP_SOURCES_SHA}


def write(path, value, cap=8 * 1024**2):
    raw = value if type(value) is bytes else canonical(value)
    need(len(raw) <= cap, "bounded exclusive packet")
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def source_binding(source):
    unit(source, "run")
    need(
        Path.cwd().resolve() == ROOT and Path(__file__).resolve().parents[2] == ROOT,
        "authorized lean worktree",
    )
    need(
        read("git", "rev-parse", "HEAD") == source
        and read("git", "branch", "--show-current") == BRANCH
        and not read("git", "status", "--porcelain"),
        "exact clean source and branch",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, source], check=True, timeout=15
    )
    changed = set(read("git", "diff", "--name-only", BASE, source).splitlines())
    need(changed <= OWN and OWN <= changed, "exact seven-path reviewed source fence")
    leaves = {}
    tree = subprocess.check_output(
        ["git", "ls-tree", "-rz", "--full-tree", source], timeout=15
    )
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, path_raw = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = path_raw.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed file")
        path = ROOT / name
        need(
            path.resolve(strict=True) == path
            and stat.S_ISREG(path.stat(follow_symlinks=False).st_mode),
            "unchanged regular source file, not symlink",
        )
        raw = path.read_bytes()
        need(
            sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid,
            "complete committed blob " + name,
        )
        leaves[name] = {
            "bytes": len(raw),
            "git_blob": oid,
            "sha256": sha256(raw).hexdigest(),
        }
    return {
        "source": source,
        "tree": read("git", "rev-parse", source + "^{tree}"),
        "branch": BRANCH,
        "leaves": leaves,
    }


def packages():
    actual = {name: importlib.metadata.version(name) for name in VERSIONS}
    need(
        actual == VERSIONS
        and platform.python_version() == "3.12.13"
        and platform.machine() == "x86_64",
        "frozen WSL Python and packages",
    )
    dist = importlib.metadata.distribution("warp-lang")
    context = Path(dist.locate_file("warp/_src/context.py")).resolve()
    need(
        digest_file(context) == {"bytes": 408818, "sha256": CONTEXT_SHA},
        "frozen Warp loader source",
    )
    return actual


def host(allowed_pids=()):
    need(
        sys.platform == "linux"
        and read("hostname") == "DESKTOP-HNKBDR1"
        and Path("/etc/machine-id").read_text().strip() == MACHINE,
        "authorized 100.98 machine",
    )
    fields = [
        part.strip()
        for part in read(
            SMI,
            "--query-gpu=uuid,driver_version,temperature.gpu,memory.used,driver_model.current",
            "--format=csv,noheader,nounits",
        ).split(",")
    ]
    need(
        len(fields) == 5
        and fields[:2] == [GPU, "595.95"]
        and fields[4] == "WDDM"
        and int(fields[2]) < 75,
        "fixed GPU/driver and temperature gate",
    )
    pids = []
    for line in read(
        SMI, "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ).splitlines():
        need(
            line.isdecimal() and int(line) in allowed_pids,
            "no foreign GPU compute workload",
        )
        pids.append(int(line))
    protected, filmbrain = {}, {}
    for namespace in ((), ("--user",)):
        for name in (
            "recomo-ai-mission-vllm.service",
            "recomo-ai-mission-subject-model-worker.service",
        ):
            props = dict(
                line.split("=", 1)
                for line in read(
                    "systemctl",
                    *namespace,
                    "show",
                    name,
                    "-p",
                    "ActiveState",
                    "-p",
                    "MainPID",
                ).splitlines()
            )
            need(
                props == {"ActiveState": "inactive", "MainPID": "0"},
                "protected services remain inactive",
            )
            protected[("user:" if namespace else "system:") + name] = props
    for name, pid in (
        ("recomo-filmbrain-observatory.service", "521"),
        ("recomo-filmbrain-video-playground.service", "298048"),
    ):
        props = dict(
            line.split("=", 1)
            for line in read(
                "systemctl",
                "--user",
                "show",
                name,
                "-p",
                "ActiveState",
                "-p",
                "MainPID",
                "-p",
                "NRestarts",
            ).splitlines()
        )
        need(
            props == {"ActiveState": "active", "MainPID": pid, "NRestarts": "0"},
            "FilmBrain unchanged",
        )
        filmbrain[name] = props
    return {
        "machine": MACHINE,
        "gpu": GPU,
        "driver": fields[1],
        "temperature_c": int(fields[2]),
        "used_mib": int(fields[3]),
        "compute_pids": pids,
        "protected": protected,
        "filmbrain": filmbrain,
    }


def lease_identity(fd):
    path, actual = LOCK.stat(follow_symlinks=False), os.fstat(fd)
    need(
        stat.S_ISREG(path.st_mode)
        and path.st_size == 0
        and (path.st_dev, path.st_ino) == (actual.st_dev, actual.st_ino),
        "same existing empty GPU lease",
    )
    return {
        "path": str(LOCK),
        "device": actual.st_dev,
        "inode": actual.st_ino,
        "bytes": actual.st_size,
    }


def service_properties(source, mode):
    keys = (
        "Id",
        "MainPID",
        "RuntimeMaxUSec",
        "MemoryMax",
        "CPUQuotaPerSecUSec",
        "TasksMax",
        "Nice",
        "LimitFSIZE",
        "Restart",
        "KillMode",
        "InvocationID",
    )
    props = dict(
        line.split("=", 1)
        for line in read(
            "systemctl",
            "--user",
            "show",
            unit(source, mode),
            *[part for key in keys for part in ("-p", key)],
        ).splitlines()
    )
    expected = {
        "Id": unit(source, mode),
        "MainPID": str(os.getpid()),
        "RuntimeMaxUSec": "10min" if mode == "run" else "11min",
        "MemoryMax": str(6 * 1024**3),
        "CPUQuotaPerSecUSec": "2s",
        "TasksMax": "64",
        "Nice": "10",
        "LimitFSIZE": str(16 * 1024**2 if mode == "run" else 64 * 1024**2),
        "Restart": "no",
        "KillMode": "control-group",
    }
    need(
        all(props.get(key) == value for key, value in expected.items())
        and re.fullmatch(r"[0-9a-f]{32}", props.get("InvocationID", "")),
        "actual bounded owner unit and invocation",
    )
    return props


def prior_test_files():
    path = (
        ROOT
        / "artifacts/tools/cuda-artifact-guard-release-mac-oct7/source-bindings.json"
    )
    need(
        digest_file(path)
        == {
            "bytes": 4190,
            "sha256": "015b5290c3b998300e5dee30c1dc8237a34fbdfaced6e584e6337b056c1434da",
        },
        "authenticated retained 82-file CPU scope before decode",
    )
    files = json.loads(path.read_bytes())["files"]
    need(
        type(files) is list
        and len(files) == len(set(files)) == 82
        and all(
            type(name) is str
            and name.startswith("tests/test_")
            and name.endswith(".py")
            for name in files
        ),
        "retained exact test scope",
    )
    return files + sorted(path for path in OWN if path.startswith("tests/"))


def tests(source):
    import xml.etree.ElementTree as ET

    check_window(CPU_SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "CPU prerequisite with CUDA hidden",
    )
    need(
        not os.environ.get("PYTEST_ADDOPTS") and not os.environ.get("PYTEST_PLUGINS"),
        "no external test selection or plugin injection",
    )
    before, versions = source_binding(source), packages()
    props = service_properties(source, "tests")
    host_before = host()
    directory = output(source, "tests")
    directory.mkdir()
    files = prior_test_files()
    test_env = {
        **os.environ,
        "WARP_CACHE_PATH": str(directory / "private-cpu-warp-cache"),
    }
    with (directory / "pytest.log").open("xb") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *files,
                "--junitxml=" + str(directory / "junit.xml"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=CPU_TEST_SECONDS,
            env=test_env,
        )
        log.flush()
        os.fsync(log.fileno())
    need(result.returncode == 0, "exact-source CPU regression success")
    suites = list(
        ET.fromstring((directory / "junit.xml").read_bytes()).iter("testsuite")
    )
    need(
        len(suites) == 1
        and int(suites[0].attrib["tests"]) == EXPECTED_TESTS > 0
        and all(
            suites[0].attrib[key] == "0" for key in ("errors", "failures", "skipped")
        ),
        "exact test count and zero CPU prerequisite omissions",
    )
    after = source_binding(source)
    need(before == after, "all committed source bytes unchanged through tests")
    record = {
        "protocol": PROTOCOL + ":tests",
        "source_binding": before,
        "packages": versions,
        "unit": props,
        "files": files,
        "tests": int(suites[0].attrib["tests"]),
        "host_before": host_before,
        "host_after": host(),
        "flags": FLAGS,
    }
    write(directory / "receipt.json", record)
    write(
        directory / "inventory.json",
        {
            name: digest_file(directory / name, 64 * 1024**2)
            for name in ("receipt.json", "pytest.log", "junit.xml")
        },
    )
    return record


def launch_arguments(role, inputs, bank, worlds, capacity):
    """Literal frozen dense signatures; testable without importing Warp."""
    fl, vel, inv, ref, imp, dt = (inputs[name] for name in INPUT_NAMES)
    if role == "original":
        return (
            (worlds, 20),
            [20, dt, 0, ref, imp, fl, inv, False, vel, capacity, capacity * 20],
            [bank[name] for name in OUTPUT_NAMES],
        )
    need(role in ("candidate0", "candidate1"), "literal launch role")
    names = (
        "nf",
        "nefc",
        "type",
        "id",
        "J",
        "pos",
        "margin",
        "D",
        "vel",
        "aref",
        "frictionloss",
    )
    return (
        worlds,
        [20, dt, 0, ref, imp, fl, inv, vel, capacity],
        [bank[name] for name in names],
    )


def child(source, lease_fd, owner_pid, declaration_sha):
    """GPU imports occur only after source/declaration/inherited-lease checks."""
    check_window(CHILD_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    need(
        os.getppid() == owner_pid and os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "one owner and literal CUDA visibility",
    )
    before = source_binding(source)
    versions = packages()
    libraries = library_binding()
    warp_sources = toolchain_source()
    lease_before = lease_identity(lease_fd)
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = output(source, "run")
    declaration_path = directory / "declaration.json"
    need(
        digest_file(declaration_path)["sha256"] == declaration_sha,
        "whole declaration authenticated before decode",
    )
    declaration = json.loads(declaration_path.read_bytes())
    need(
        declaration["source_binding"] == before
        and declaration["owner_pid"] == owner_pid
        and declaration["lease"] == lease_before
        and declaration["libraries"] == libraries
        and declaration["warp_sources"] == warp_sources,
        "same declared source, owner, compiler sources, libraries and lease",
    )
    private_cache = directory / "private-warp-cache"
    need(
        os.environ.get("WARP_CACHE_PATH") == str(private_cache)
        and not private_cache.exists(),
        "fresh process-private Warp cache, never shared cache",
    )

    import numpy as np
    import warp as wp
    from warp._src import context as ctx
    from mujoco_warp._src.types import vec5
    from mjlab_microduck import stance_friction_prefix_cpu_fixture as prefix
    from mjlab_microduck import stance_friction_row_cpu_fixture as base
    from mjlab_microduck import stance_cuda_artifact_binding as artifacts

    need(
        digest_file(Path(prefix.__file__))["sha256"] == PREFIX_SHA
        and digest_file(Path(base.__file__))["sha256"] == BASE_FIXTURE_SHA,
        "unchanged borrowed literal inputs and banks",
    )
    need(
        base._tree_sha256() == base.TREE_SHA256
        and digest_file(base._SOURCE_PATH)["sha256"] == base.CONSTRAINT_SHA256,
        "frozen MuJoCo-Warp tree and constraint source",
    )
    compiler_config = configure_compiler(wp)
    wp.init()
    need(
        str(ctx.runtime.core._name) == libraries["warp.so"]["path"]
        and str(ctx.runtime.llvm._name) == libraries["warp-clang.so"]["path"],
        "actual loaded frozen runtime and compiler library paths",
    )
    device = wp.get_device("cuda:0")
    need(
        len(wp.get_cuda_devices()) == 1
        and device.is_cuda
        and str(device) == "cuda:0"
        and not device.is_capturing
        and Path(wp.config.kernel_cache_dir).resolve() == private_cache.resolve(),
        "only eager CUDA and own fresh cache",
    )
    stream = wp.Stream(device)
    need(
        type(stream.cuda_stream) is int
        and stream.cuda_stream != 0
        and stream.device is device,
        "dedicated actual nonzero CUDA stream",
    )
    kernels = {"original": base._ORIGINAL_KERNEL, "candidate": base._CANDIDATE_KERNEL}
    need(
        kernels["original"].module is not kernels["candidate"].module,
        "two distinct original and candidate modules",
    )
    held = {
        "launch": (wp.launch, wp.launch.__code__),
        "load": (ctx.Module.load, ctx.Module.load.__code__),
        "compile": (ctx.Module._compile, ctx.Module._compile.__code__),
        "hooks": (
            ctx.ModuleExec.get_kernel_hooks,
            ctx.ModuleExec.get_kernel_hooks.__code__,
        ),
        "hash": (ctx.Module.get_module_hash, ctx.Module.get_module_hash.__code__),
        "load_cuda": (wp._src.build.load_cuda, wp._src.build.load_cuda.__code__),
        "build_cuda": (wp._src.build.build_cuda, wp._src.build.build_cuda.__code__),
        "synchronize": (wp.synchronize_stream, wp.synchronize_stream.__code__),
    }
    kernel_ids = {
        name: (kernel, kernel.func, kernel.func.__code__)
        for name, kernel in kernels.items()
    }
    borrowed = {
        name: (getattr(prefix, name), getattr(prefix, name).__code__)
        for name in (
            "predeclared_prefix_fixture_values",
            "_prefix_values",
            "_bank_record",
        )
    }
    row_function = base.constraint._efc_row
    row_function_code = (row_function, row_function.func, row_function.func.__code__)

    def guards():
        need(
            canonical({key: getattr(wp.config, key) for key in COMPILER_CONFIG})
            == canonical(compiler_config)
            == canonical(COMPILER_CONFIG),
            "same explicit compiler settings with PCH disabled",
        )
        current = {
            "launch": wp.launch,
            "load": ctx.Module.load,
            "compile": ctx.Module._compile,
            "hooks": ctx.ModuleExec.get_kernel_hooks,
            "hash": ctx.Module.get_module_hash,
            "load_cuda": wp._src.build.load_cuda,
            "build_cuda": wp._src.build.build_cuda,
            "synchronize": wp.synchronize_stream,
        }
        need(
            all(
                current[name] is function and function.__code__ is code
                for name, (function, code) in held.items()
            ),
            "held frozen runtime entry points",
        )
        need(
            all(
                kernels[name] is kernel
                and kernel.func is function
                and function.__code__ is code
                for name, (kernel, function, code) in kernel_ids.items()
            ),
            "held original and candidate code",
        )
        need(
            all(
                getattr(prefix, name) is function and function.__code__ is code
                for name, (function, code) in borrowed.items()
            ),
            "held borrowed literal helper code",
        )
        need(
            base._ORIGINAL_KERNEL is kernels["original"]
            and base._CANDIDATE_KERNEL is kernels["candidate"]
            and base.constraint._friction_dof is kernels["original"],
            "same frozen kernel aliases",
        )
        need(
            base.constraint._efc_row is row_function_code[0]
            and row_function.func is row_function_code[1]
            and row_function.func.__code__ is row_function_code[2],
            "held shared row construction function",
        )
        need(
            device.is_cuda
            and not device.is_capturing
            and stream.device is device
            and stream.cuda_stream == stream_handle,
            "same eager device and explicit stream",
        )

    stream_handle = stream.cuda_stream
    compiled = {}
    bindings = {}
    for role, kernel in kernels.items():
        guards()
        module = kernel.module
        artifacts.assert_fresh_module(module, device, 256)
        module_hash = held["hash"][0](module, 256)
        options = canonical(module.options | kernel.options)
        target_arch = module._get_compile_arch(device)
        need(
            type(target_arch) is int and target_arch == device.arch,
            "native device architecture, no implicit target substitution",
        )
        artifact_dir = directory / ("compiled-" + role)
        need(not artifact_dir.exists(), "unique unused retained compile directory")
        compiled_now = held["compile"][0](
            module, device, artifact_dir, role + ".cubin", target_arch, False
        )
        need(
            compiled_now is True and not module.execs,
            "fresh compile performed without loading CUDA executable",
        )
        binary = artifact_dir / (role + ".cubin")
        meta = artifact_dir / module._get_meta_name()
        cu = artifact_dir / (module.get_module_identifier() + ".cu")
        need(
            set(artifact_dir.iterdir()) == {binary, meta, cu},
            "complete retained generated source, CUBIN and metadata",
        )
        binary_info, meta_info = (
            digest_file(binary, artifacts.MAX_BINARY_BYTES),
            digest_file(meta, artifacts.MAX_META_BYTES),
        )
        need(
            binary.read_bytes()[:4] == b"\x7fELF",
            "retained CUBIN ELF input, not mislabeled PTX",
        )
        artifact = artifacts.retain_artifact(
            binary,
            meta,
            binary_size=binary_info["bytes"],
            binary_sha256=binary_info["sha256"],
            metadata_size=meta_info["bytes"],
            metadata_sha256=meta_info["sha256"],
        )
        artifacts.assert_fresh_module(module, device, 256)
        executable = held["load"][0](
            module,
            device,
            block_dim=256,
            binary_path=str(binary),
            output_arch=target_arch,
            meta_path=str(meta),
        )
        need(
            executable is not None
            and executable is module.execs.get((device.context, 256)),
            "exact explicit-load returned executable",
        )
        hooks = held["hooks"][0](executable, kernel)
        binding = artifacts.bind_loaded_module(
            artifact,
            kernel,
            device,
            executable,
            hooks,
            block_dim=256,
            expected_module_hash=module_hash,
        )
        need(
            canonical(module.options | kernel.options) == options,
            "source/options unchanged through compile and explicit load",
        )
        bindings[role] = binding
        compiled[role] = {
            "compiler_config": compiler_config,
            "binding": binding.record(),
            "module_options": json.loads(options),
            "output_arch": target_arch,
            "generated_source": {"path": str(cu), **digest_file(cu)},
            "binary": {"path": str(binary), **binary_info},
            "metadata": {"path": str(meta), **meta_info},
            "explicit_load": {
                "module_object_id": id(module),
                "returned_executable_id": id(executable),
                "device_object_id": id(device),
                "block_dim": 256,
                "binary_path": str(binary),
                "meta_path": str(meta),
                "output_arch": target_arch,
                "fresh_cache_before": True,
            },
            "runtime_entries": {
                name: {
                    "object_id": id(function),
                    "code_id": id(code),
                    "qualname": code.co_qualname,
                }
                for name, (function, code) in held.items()
            },
        }

    def snapshot(values):
        return {
            name: {
                "shape": list(value.shape),
                "dtype": "<f4",
                "u32": value.view("<u4").reshape(-1).tolist(),
            }
            for name, value in values.items()
        }

    def allocation(array):
        value = array.numpy()
        need(
            array.device is device and array.is_contiguous and int(array.ptr) > 0,
            "actual owned contiguous CUDA allocation",
        )
        return {
            "object_id": id(array),
            "pointer": int(array.ptr),
            "device": str(array.device),
            "context": device.context,
            "warp_dtype": str(array.dtype),
            "shape": list(array.shape),
            "strides": list(array.strides),
            "bytes": value.nbytes,
            "host_shape": list(value.shape),
            "host_dtype": str(value.dtype),
        }

    cases, evidence, launches = {}, {}, []
    values_by_case = borrowed["predeclared_prefix_fixture_values"][0]()
    need(set(values_by_case) == set(CASE_NAMES), "exact ten literal cases")
    with wp.ScopedStream(stream):
        for case_name in CASE_NAMES:
            values = values_by_case[case_name]
            worlds, cap = values["qvel"].shape[0], values["njmax"]
            initial = borrowed["_prefix_values"][0](worlds, cap, values["initial_nefc"])
            host_inputs = {
                name: np.array(values[name], copy=True, order="C")
                for name in INPUT_NAMES
            }
            input_snapshot = snapshot(host_inputs)
            runs, run_evidence, keep_alive = {}, {}, []
            for side in ("original", "candidate0", "candidate1"):
                guards()
                role = "original" if side == "original" else "candidate"
                bound = bindings[role]
                bound.assert_unchanged()
                dtypes = {"solref": wp.vec2, "solimp": vec5}
                inputs = {
                    name: wp.array(
                        value.copy(), dtype=dtypes.get(name, wp.float32), device=device
                    )
                    for name, value in host_inputs.items()
                }
                bank = {
                    name: wp.array(
                        value.copy(),
                        dtype=wp.int32 if value.dtype.kind == "i" else wp.float32,
                        device=device,
                    )
                    for name, value in initial.items()
                }
                keep_alive.extend([*inputs.values(), *bank.values()])
                before_bank = borrowed["_bank_record"][0](
                    {name: value.numpy().copy() for name, value in bank.items()}
                )
                input_before = snapshot(
                    {name: value.numpy().copy() for name, value in inputs.items()}
                )
                allocs = {
                    "inputs": {
                        name: allocation(value) for name, value in inputs.items()
                    },
                    "outputs": {
                        name: allocation(value) for name, value in bank.items()
                    },
                }
                dim, args, outs = launch_arguments(side, inputs, bank, worlds, cap)
                check_window(CLOSEOUT_SECONDS + MARGIN)
                binding_before = bound.record()
                guards()
                held["launch"][0](
                    bound.kernel,
                    dim=dim,
                    inputs=args,
                    outputs=outs,
                    device=device,
                    stream=stream,
                    block_dim=256,
                    record_tape=False,
                )
                held["synchronize"][0](stream)
                guards()
                binding_after = bound.record()
                after_bank = borrowed["_bank_record"][0](
                    {name: value.numpy().copy() for name, value in bank.items()}
                )
                input_after = snapshot(
                    {name: value.numpy().copy() for name, value in inputs.items()}
                )
                need(
                    input_before == input_after == input_snapshot
                    and binding_before == binding_after,
                    "actual inputs and module hooks unchanged through launch",
                )
                runs[side] = {
                    "before": before_bank,
                    "after": after_bank,
                    "inputs_before": input_before,
                    "inputs_after": input_after,
                }
                post_allocs = {
                    "inputs": {
                        name: allocation(value) for name, value in inputs.items()
                    },
                    "outputs": {
                        name: allocation(value) for name, value in bank.items()
                    },
                }
                need(allocs == post_allocs, "same live buffers through actual launch")
                run_evidence[side] = allocs
                launches.append(
                    {
                        "index": len(launches),
                        "case": case_name,
                        "role": side,
                        "kernel_role": role,
                        "dim": list(dim) if type(dim) is tuple else dim,
                        "block_dim": 256,
                        "stream_handle": stream_handle,
                        "stream_object_id": id(stream),
                        "context": device.context,
                        "binding_before": binding_before,
                        "binding_after": binding_after,
                        "argument_array_pointers": [
                            int(value.ptr) for value in args if hasattr(value, "ptr")
                        ],
                        "argument_scalars": [
                            value for value in args if type(value) in (int, bool)
                        ],
                        "output_array_pointers": [int(value.ptr) for value in outs],
                    }
                )
            all_allocs = [
                entry
                for side in run_evidence.values()
                for group in side.values()
                for entry in group.values()
            ]
            intervals = sorted(
                (entry["pointer"], entry["pointer"] + entry["bytes"])
                for entry in all_allocs
            )
            need(
                len(all_allocs) == 63
                and len({entry["object_id"] for entry in all_allocs}) == 63
                and all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
                "63 independent simultaneously live nonoverlapping buffers",
            )
            cases[case_name] = {
                "device": "cuda:0",
                "input_snapshot": input_snapshot,
                "initial_nefc": {
                    "shape": [worlds],
                    "dtype": "<i4",
                    "i32": values["initial_nefc"].tolist(),
                },
                "initial_nf": {"shape": [worlds], "dtype": "<i4", "i32": [0] * worlds},
                "runs": runs,
            }
            evidence[case_name] = run_evidence
    guards()
    need(len(launches) == 30, "exact thirty synthetic launches")
    after = source_binding(source)
    need(
        before == after
        and lease_before == lease_identity(lease_fd)
        and libraries == library_binding()
        and warp_sources == toolchain_source(),
        "source, compiler headers, native libraries and inherited lease unchanged through child",
    )
    return {
        "protocol": PROTOCOL + ":child",
        "source_binding_before": before,
        "source_binding_after": after,
        "packages": versions,
        "libraries": libraries,
        "warp_sources": warp_sources,
        "owner_pid": owner_pid,
        "child_pid": os.getpid(),
        "lease": lease_before,
        "declaration_sha256": declaration_sha,
        "device": {
            "alias": str(device),
            "object_id": id(device),
            "context": device.context,
            "arch": device.arch,
            "name": device.name,
            "stream": stream_handle,
            "stream_object_id": id(stream),
            "toolkit_version": list(ctx.runtime.toolkit_version),
            "driver_version": list(ctx.runtime.driver_version),
        },
        "compiled": compiled,
        "cases": cases,
        "allocations": evidence,
        "launches": launches,
        "flags": FLAGS,
    }


def run(source, tests_inventory_sha):
    from mjlab_microduck import stance_friction_prefix_cuda_receiver as receiver
    import xml.etree.ElementTree as ET

    check_window(SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    before, versions, props = (
        source_binding(source),
        packages(),
        service_properties(source, "run"),
    )
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "owner has CUDA hidden")
    prerequisite = output(source, "tests")
    need(
        type(tests_inventory_sha) is str
        and re.fullmatch(r"[0-9a-f]{64}", tests_inventory_sha),
        "externally anchored exact CPU prerequisite inventory SHA",
    )
    need(
        digest_file(prerequisite / "inventory.json")["sha256"] == tests_inventory_sha,
        "CPU inventory whole bytes authenticated before decode",
    )
    test_inventory = receiver.decode((prerequisite / "inventory.json").read_bytes())
    test_raw = receiver.authenticate(prerequisite, test_inventory, cap=64 * 1024**2)
    test_record = receiver.decode(test_raw["receipt.json"])
    need(
        test_record["source_binding"] == before
        and test_record["packages"] == versions
        and test_record["files"] == prior_test_files()
        and test_record["flags"] == FLAGS,
        "retained same-source CPU prerequisite",
    )
    need(
        test_record["tests"] == EXPECTED_TESTS > 0
        and set(test_raw) == {"receipt.json", "pytest.log", "junit.xml"},
        "exact completed CPU test scope",
    )
    suites = list(ET.fromstring(test_raw["junit.xml"]).iter("testsuite"))
    need(
        len(suites) == 1
        and int(suites[0].attrib["tests"]) == EXPECTED_TESTS
        and all(
            suites[0].attrib[key] == "0" for key in ("errors", "failures", "skipped")
        ),
        "independently authenticated CPU JUnit exact count and zero omissions",
    )
    terminal = dict(
        line.split("=", 1)
        for line in read(
            "systemctl",
            "--user",
            "show",
            unit(source, "tests"),
            "-p",
            "MainPID",
            "-p",
            "SubState",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
            "-p",
            "InvocationID",
        ).splitlines()
    )
    need(
        terminal
        == {
            "MainPID": "0",
            "SubState": "exited",
            "Result": "success",
            "ExecMainStatus": "0",
            "InvocationID": test_record["unit"]["InvocationID"],
        },
        "successful retained CPU test unit is no longer running",
    )
    libraries = library_binding()
    warp_sources = toolchain_source()
    directory = output(source, "run")
    directory.mkdir()
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    process = None
    samples = []
    try:
        lease = lease_identity(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_host = host()
        need(initial_host["used_mib"] <= 1024, "idle GPU before fresh child")
        declaration = {
            "protocol": PROTOCOL + ":declaration",
            "source_binding": before,
            "packages": versions,
            "libraries": libraries,
            "warp_sources": warp_sources,
            "unit": props,
            "owner_pid": os.getpid(),
            "lease": lease,
            "host_before": initial_host,
            "tests": test_record,
            "tests_inventory_sha256": tests_inventory_sha,
            "tests_inventory": test_inventory,
            "tests_terminal": terminal,
            "case_order": list(CASE_NAMES),
            "launches": 30,
            "started_utc_ns": time.time_ns(),
            "cutoff_utc": "2026-10-07T23:30:00Z",
            "child_timeout_seconds": CHILD_SECONDS,
            "flags": FLAGS,
        }
        write(directory / "declaration.json", declaration)
        declaration_sha = sha256(canonical(declaration)).hexdigest()
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "WARP_CACHE_PATH": str(directory / "private-warp-cache"),
        }
        with (directory / "child.log").open("xb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mjlab_microduck.stance_friction_prefix_cuda_probe",
                    "--mode",
                    "child",
                    "--source",
                    source,
                    "--lease-fd",
                    str(fd),
                    "--owner-pid",
                    str(os.getpid()),
                    "--declaration-sha",
                    declaration_sha,
                ],
                env=env,
                pass_fds=(fd,),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            started = time.monotonic()
            while process.poll() is None:
                need(
                    time.monotonic() - started < CHILD_SECONDS,
                    "bounded fresh child timeout",
                )
                check_window(CLOSEOUT_SECONDS + MARGIN)
                samples.append(
                    {
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        **host((process.pid,)),
                    }
                )
                time.sleep(2)
            log.flush()
            os.fsync(log.fileno())
        need(
            process.returncode == 0,
            "fresh CUDA child success; diagnose retained log before any retry",
        )
        final_host = host()
        need(
            before == source_binding(source)
            and lease == lease_identity(fd)
            and libraries == library_binding()
            and warp_sources == toolchain_source(),
            "owner source, compiler headers, native libraries and lease closure",
        )
        write(
            directory / "owner.json",
            {
                "protocol": PROTOCOL + ":owner",
                "source_binding": before,
                "owner_pid": os.getpid(),
                "child_pid": process.pid,
                "child_exit": process.returncode,
                "finished_utc_ns": time.time_ns(),
                "host_after": final_host,
                "samples": samples,
                "flags": FLAGS,
            },
        )
        inventory = {
            str(path.relative_to(directory)): digest_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        report = receiver.verify_run(
            directory,
            inventory,
            expected_source=before,
            expected_tests_sha=tests_inventory_sha,
        )
        closeout = (
            ROOT / "artifacts/tools" / f"friction-prefix-cuda-closeout-{source[:12]}"
        )
        closeout.mkdir()
        write(closeout / "inventory.json", inventory)
        write(closeout / "receiver.json", report)
        return report
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if not (directory / "failure.json").exists():
            write(
                directory / "failure.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "owner_pid": os.getpid(),
                    "child_pid": process.pid if process else None,
                    "child_exit": process.returncode if process else None,
                    "samples": samples,
                    "flags": FLAGS,
                },
            )
        raise
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tests", "run", "child"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--declaration-sha")
    parser.add_argument("--tests-inventory-sha")
    args = parser.parse_args(argv)
    if args.mode == "tests":
        result = tests(args.source)
    elif args.mode == "run":
        result = run(args.source, args.tests_inventory_sha)
    else:
        need(
            type(args.lease_fd) is int
            and args.lease_fd >= 3
            and type(args.owner_pid) is int
            and re.fullmatch(r"[0-9a-f]{64}", args.declaration_sha or ""),
            "literal inherited child arguments",
        )
        result = child(args.source, args.lease_fd, args.owner_pid, args.declaration_sha)
        write(output(args.source, "run") / "child.packet.json", result)
    print(
        json.dumps(
            {"protocol": PROTOCOL, "mode": args.mode, "flags": FLAGS, "complete": True},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
