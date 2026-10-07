"""Separately owned passive contact-boundary tick; never training admission.

The owner keeps CUDA hidden.  A separately leased child authenticates its
declaration before importing CUDA packages and retains every raw packet.
"""

import argparse
import fcntl
from hashlib import sha1, sha256
import inspect
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from mjlab_microduck import stance_friction_prefix_cuda_probe as prefix

PROTOCOL = "microduck-contact-boundary-tick-oct8-v1"
ROOT = Path("/home/yanbo/work/microduck_rl-com-entry-20261006")
BASE = "0ad275806cb64383535fa2338f307fc91dccea99"
BRANCH = "feat/athletics-obstacle-curriculum"
CUTOFF = 1791415800  # Newly declared Oct8 07:30 Asia/Shanghai owner cutoff.
SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS, MARGIN = 600, 540, 240, 60
CPU_SERVICE_SECONDS, CPU_TEST_SECONDS = 660, 600
EXPECTED_TESTS = 3069  # Exact reviewed 94-file collection; receipts still required.
CPU_TEST_THREADS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
OWNER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
CUDA_CHILD_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
THREAD_BUDGET = {
    "owner": OWNER_THREAD_ENV,
    "cuda_child": CUDA_CHILD_THREAD_ENV,
}
TEST_FILES_SHA256 = "253d755fc09f0b5286695a85303353ff6349e7f4bb88cbd3b97dea5b1872fb6a"
LOCK = prefix.LOCK
FLAGS = prefix.FLAGS
ARMS = ("original", "candidate0", "candidate1")
CALLER_RNG_SEEDS = {"cpu": 673, "cuda": 677}
OWN = {
    "src/mjlab_microduck/stance_contact_boundary_control.py",
    "src/mjlab_microduck/stance_contact_boundary_probe.py",
    "src/mjlab_microduck/stance_contact_boundary_receiver.py",
    "tests/test_stance_contact_boundary_control.py",
    "tests/test_stance_contact_boundary_probe.py",
    "tests/test_stance_contact_boundary_receiver.py",
    "docs/experiments/2026-10-08-passive-contact-boundary.md",
}

HISTORICAL_SOURCE = "5973184031ba9537e88dd066bd14faf8e5a9b4a4"
HISTORICAL_RAW_INVENTORY_SHA256 = (
    "2db2959673fc3719f7e3b0a050f6003e96d973efc60cb3a2def281768ce23fff"
)
HISTORICAL_RECEIVER_SHA256 = (
    "2601315fa33c5366fd6df716ee657bdd51f201fab1b81e56a3e3795ca9bf2c17"
)
HISTORICAL_REPLAY_SHA256 = (
    "acb87ac697792a832de4ff214c5570ffffe278b7ebba1ec27d3573f04c9bf578"
)
HISTORICAL_TESTS_INVENTORY_SHA256 = (
    "4f1f4e1a7adc38a792a9bd914f5b7e32fb768db92b1419ecdee5b3ecfea3496c"
)
HISTORICAL_EXPECTED_TESTS_SHA256 = (
    "4f1f4e1a7adc38a792a9bd914f5b7e32fb768db92b1419ecdee5b3ecfea3496c"
)
HISTORICAL_TERMINAL_SHA256 = (
    "9ba0f4372ed687b84009b5e3ee36ebbc0858c1b95d1ad5136cc81d18563cf124"
)
VERSIONS = prefix.VERSIONS
COMPILER_CONFIG = prefix.COMPILER_CONFIG
BASE_FIXTURE_SHA = "8e96c5077f248aad06cacab4c3a45f17cd6b3c35c09c7a46457281916e8046d9"
PREFIX_FIXTURE_SHA = "0ff83581832f3264d9f6b447b089fee285f845f12a5a4eb3614a057b82ba0bad"
NEGATIVE_SOURCE = "321a72a6b7d3d64808dafdbc36309052495cc85b"
NEGATIVE_ANCHORS = {
    "inventory.json": (
        60551,
        "233e52baa6d73fc9b23b4149f013d783638bdaa4ff814ab6eacbea765ae4bb58",
    ),
    "receiver.json": (
        1024216,
        "69a1616f48189de605f9efd0e6d4eb58372031d14ff9c6bc6e36d779383884f2",
    ),
    "mac-receiver.json": (
        1024216,
        "69a1616f48189de605f9efd0e6d4eb58372031d14ff9c6bc6e36d779383884f2",
    ),
    "mac-replay.json": (
        842,
        "6e75c19aa3b5464a8b60b28a69669818d55b1f5f4fb9b6ce030974229a6093a6",
    ),
    "mac-first-delta.json": (
        32437,
        "58d9fe994557b1cafc13e488577dcde1d9d7a4407ff5609723356e47be213dff",
    ),
    "terminal.txt": (
        104,
        "4edd349d407c77b66d4bf2890710eadeb6b193c43c322f86fe903213b4f87bc3",
    ),
}


need = prefix.need
canonical = prefix.canonical
read = prefix.read
digest_file = prefix.digest_file
write = prefix.write
environment_binding = prefix.environment_binding
library_binding = prefix.library_binding
toolchain_source = prefix.toolchain_source
packages = prefix.packages
host = prefix.host
lease_identity = prefix.lease_identity
configure_compiler = prefix.configure_compiler


def check_window(reserve, now=None):
    need(type(reserve) is int and reserve >= 0, "plain nonnegative cutoff reserve")
    need(
        (time.time() if now is None else now) + reserve < CUTOFF,
        "Oct8 07:30 cutoff and full reserve",
    )


def unit(source, mode):
    need(
        type(source) is str
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and mode in ("tests", "run"),
        "exact source and bounded unit mode",
    )
    return f"microduck-contact-boundary-tick-{mode}-{source[:12]}.service"


def output(source, mode):
    unit(source, mode)
    return (
        ROOT
        / "artifacts"
        / ("tools" if mode == "tests" else "evaluations")
        / f"contact-boundary-tick-{mode}-{source[:12]}"
    )


def source_binding(source):
    """Bind every committed blob and the literal seven-path source delta."""
    unit(source, "run")
    need(
        Path.cwd().resolve() == ROOT
        and Path(__file__).resolve().parents[2] == ROOT
        and read("git", "rev-parse", "HEAD") == source
        and read("git", "branch", "--show-current") == BRANCH
        and not read("git", "status", "--porcelain"),
        "exact clean owner worktree, branch and source SHA",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, source], check=True, timeout=15
    )
    changed = set(read("git", "diff", "--name-only", BASE, source).splitlines())
    need(changed == OWN, "exact seven-path experiment source fence")
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
        need(
            mode in ("100644", "100755") and kind == "blob",
            "plain committed source leaf",
        )
        path = ROOT / name
        need(
            path.resolve(strict=True) == path
            and stat.S_ISREG(path.stat(follow_symlinks=False).st_mode),
            "plain source path " + name,
        )
        raw = path.read_bytes()
        need(
            sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid,
            "whole committed Git blob " + name,
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
        "actual bounded source-derived owner service",
    )
    return props


def test_files():
    from mjlab_microduck import stance_friction_runtime_probe as prior

    files = prior.test_files() + ["tests/test_stance_raw_carrier_delta.py"]
    result = files + sorted(name for name in OWN if name.startswith("tests/"))
    need(
        len(files) == 91
        and len(result) == len(set(result)) == 94
        and sha256(canonical(result)).hexdigest() == TEST_FILES_SHA256,
        "separately frozen 94-file contact-boundary scope",
    )
    return result


def _raw_inventory(
    directory,
    anchors,
    *,
    total_cap=768 * 1024**2,
    external_inventory=False,
    cpu_cache=False,
):
    """Authenticate a closed plain-file inventory before decoding any member."""
    directory = Path(directory).absolute()
    need(
        directory.resolve(strict=True) == directory and directory.is_dir(),
        "plain inventory root",
    )
    for path in directory.rglob("*"):
        info = path.stat(follow_symlinks=False)
        need(
            not path.is_symlink()
            and (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)),
            "no symlink or special inventory nodes",
        )
    need(
        type(anchors) is dict and 0 < len(anchors) <= 1024,
        "bounded external raw inventory",
    )
    result, total = {}, 0
    for name, anchor in anchors.items():
        rel = Path(name)
        need(
            type(name) is str
            and str(rel) == name
            and not rel.is_absolute()
            and ".." not in rel.parts,
            "safe relative inventory path",
        )
        need(
            type(anchor) is dict and set(anchor) == {"bytes", "sha256"},
            "whole-file anchor shape",
        )
        size = anchor["bytes"]
        need(
            type(size) is int
            and 0 <= size <= 16 * 1024**2
            and re.fullmatch(r"[0-9a-f]{64}", anchor["sha256"]),
            "per-leaf inventory caps",
        )
        path = directory / rel
        need(path.resolve(strict=True) == path, "no symlink inventory member")
        raw_info = digest_file(path, 16 * 1024**2)
        need(raw_info == anchor, "whole-file inventory digest")
        raw = path.read_bytes()
        need(
            len(raw) == size and sha256(raw).hexdigest() == anchor["sha256"],
            "stable whole-file inventory bytes",
        )
        total += size
        need(total <= total_cap, "bounded inventory total")
        result[name] = raw
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
        and not (
            cpu_cache
            and path.relative_to(directory).parts[0] == "private-cpu-warp-cache"
        )
    }
    need(
        cpu_cache is False or external_inventory is True,
        "CPU compile cache is auxiliary to prerequisite-only three-leaf proof",
    )
    expected = set(anchors) | ({"inventory.json"} if external_inventory else set())
    need(actual == expected, "exact closed raw inventory leaf set")
    return result


def historical_synthetic():
    """Authenticate original synthetic proof before allowing its interpretation."""
    from mjlab_microduck import stance_friction_prefix_cuda_receiver as old_receiver

    closeout = ROOT / "artifacts/tools/friction-prefix-cuda-closeout-5973184031ba"
    inventory_path = closeout / "inventory.json"
    need(
        digest_file(inventory_path, 128 * 1024)
        == {"bytes": 1353, "sha256": HISTORICAL_RAW_INVENTORY_SHA256},
        "historical inventory whole-byte anchor",
    )
    inventory_raw = inventory_path.read_bytes()
    inventory = json.loads(inventory_raw)
    raw = _raw_inventory(
        ROOT / "artifacts/evaluations/friction-prefix-cuda-run-5973184031ba", inventory
    )
    need(len(raw) == 10, "exact historical ten-leaf raw CUDA inventory")
    proof_anchors = {
        "receiver.json": (89489, HISTORICAL_RECEIVER_SHA256),
        "mac-receiver.json": (89489, HISTORICAL_RECEIVER_SHA256),
        "mac-replay.json": (586, HISTORICAL_REPLAY_SHA256),
        "terminal.txt": (178, HISTORICAL_TERMINAL_SHA256),
    }
    proof_raw = {}
    for name, (size, digest) in proof_anchors.items():
        path = closeout / name
        need(
            digest_file(path, 128 * 1024) == {"bytes": size, "sha256": digest},
            "whole historical closeout proof " + name,
        )
        proof_raw[name] = path.read_bytes()
    receiver_raw = proof_raw["receiver.json"]
    replay_raw = proof_raw["mac-replay.json"]
    mac_receiver_raw = proof_raw["mac-receiver.json"]
    terminal_raw = proof_raw["terminal.txt"]
    need(
        sha256(receiver_raw).hexdigest() == HISTORICAL_RECEIVER_SHA256
        and sha256(replay_raw).hexdigest() == HISTORICAL_REPLAY_SHA256
        and sha256(mac_receiver_raw).hexdigest() == HISTORICAL_RECEIVER_SHA256
        and receiver_raw == mac_receiver_raw
        and sha256(terminal_raw).hexdigest() == HISTORICAL_TERMINAL_SHA256,
        "whole native/Mac receiver, replay and terminal proofs",
    )
    tests_root = ROOT / "artifacts/tools/friction-prefix-cuda-tests-5973184031ba"
    tests_inventory_path = tests_root / "inventory.json"
    need(
        digest_file(tests_inventory_path, 128 * 1024)["sha256"]
        == HISTORICAL_TESTS_INVENTORY_SHA256,
        "historical CPU inventory anchor",
    )
    tests_inventory_raw = tests_inventory_path.read_bytes()
    tests_inventory = json.loads(tests_inventory_raw)
    tests_raw = _raw_inventory(
        tests_root,
        tests_inventory,
        total_cap=64 * 1024**2,
        external_inventory=True,
        cpu_cache=True,
    )
    receipt = json.loads(tests_raw["receipt.json"])
    need(
        receipt["protocol"] == prefix.PROTOCOL + ":tests"
        and receipt["source_binding"]["source"] == HISTORICAL_SOURCE
        and receipt["tests"] == old_receiver.EXPECTED_TESTS == 2687
        and len(receipt["files"]) == 85
        and sha256(tests_inventory_raw).hexdigest() == HISTORICAL_EXPECTED_TESTS_SHA256,
        "historical CPU receipt source and exact 85-file run",
    )
    replay_proof = json.loads(replay_raw)
    need(
        replay_proof["complete_raw_inventory_matches"] is True
        and replay_proof["deterministic_receiver_bytes_match"] is True
        and replay_proof["raw_inventory"]["sha256"] == HISTORICAL_RAW_INVENTORY_SHA256
        and replay_proof["source"] == HISTORICAL_SOURCE
        and replay_proof["native_tests_inventory_sha256"]
        == HISTORICAL_TESTS_INVENTORY_SHA256,
        "independent Mac replay proof bound to historical raw inventory/source",
    )
    report = old_receiver.verify_run(
        ROOT / "artifacts/evaluations/friction-prefix-cuda-run-5973184031ba",
        inventory,
        expected_source=receipt["source_binding"],
        expected_tests_sha=sha256(tests_inventory_raw).hexdigest(),
    )
    need(
        canonical(report) == receiver_raw,
        "independently recomputed original receiver bytes",
    )
    return {
        "source": HISTORICAL_SOURCE,
        "inventory_sha256": HISTORICAL_RAW_INVENTORY_SHA256,
        "receiver_sha256": HISTORICAL_RECEIVER_SHA256,
        "mac_replay_sha256": HISTORICAL_REPLAY_SHA256,
        "tests_inventory_sha256": HISTORICAL_TESTS_INVENTORY_SHA256,
        "tests": 2687,
        "decision": report["decision"],
    }


def historical_negative():
    """Reauthenticate and replay the earlier negative, never replace its decision."""
    from mjlab_microduck import stance_friction_runtime_receiver as old_receiver

    closeout = ROOT / "artifacts/tools/friction-runtime-tick-closeout-321a72a6b7d3"
    proof = {}
    for name, (size, digest) in NEGATIVE_ANCHORS.items():
        path = closeout / name
        need(
            digest_file(path, 2 * 1024**2) == {"bytes": size, "sha256": digest},
            "whole retained negative anchor " + name,
        )
        value = path.read_bytes()
        need(
            len(value) == size and sha256(value).hexdigest() == digest,
            "stable retained negative anchor " + name,
        )
        proof[name] = value
    inventory = json.loads(proof["inventory.json"])
    directory = ROOT / "artifacts/evaluations/friction-runtime-tick-run-321a72a6b7d3"
    raw = old_receiver.authenticate(directory, inventory)
    need(
        len(raw) == 414 and sum(map(len, raw.values())) == 526737887,
        "complete earlier negative raw inventory",
    )
    declaration = old_receiver.json_packet(raw["declaration.json"])
    binding = declaration["source_binding"]
    need(binding["source"] == NEGATIVE_SOURCE, "earlier negative execution source")
    report = old_receiver.verify_run(
        directory,
        inventory,
        expected_source=binding,
        expected_tests_sha=declaration["tests_inventory_sha256"],
    )
    need(
        canonical(report) == proof["receiver.json"] == proof["mac-receiver.json"],
        "independent earlier negative receiver bytes",
    )
    replay, delta = (
        json.loads(proof[name]) for name in ("mac-replay.json", "mac-first-delta.json")
    )
    need(
        report["decision"]
        == replay["decision"]
        == "real-model-one-tick-candidate-repeat-negative"
        and report["flags"] == replay["flags"] == delta["flags"] == FLAGS
        and replay["source"] == delta["execution_source"] == NEGATIVE_SOURCE
        and replay["all_whole_leaves_authenticated"] is True
        and replay["native_receiver_bytes_equal"] is True
        and replay["raw_inventory"]
        == {"bytes": 60551, "sha256": NEGATIVE_ANCHORS["inventory.json"][1]}
        and delta["raw_inventory_sha256"] == NEGATIVE_ANCHORS["inventory.json"][1]
        and delta["independent_receiver_sha256"]
        == NEGATIVE_ANCHORS["receiver.json"][1],
        "unchanged non-admitting independently replayed negative",
    )
    return {
        "source": NEGATIVE_SOURCE,
        "decision": report["decision"],
        "anchors": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in NEGATIVE_ANCHORS.items()
        },
        "raw_files": len(raw),
        "raw_bytes": sum(map(len, raw.values())),
        "flags": dict(FLAGS),
    }


def check_contact_specialization(kernel):
    need(
        kernel.func.__code__.co_freevars == ("IS_ELLIPTIC", "IS_SPARSE")
        and type(kernel.func.__closure__) is tuple
        and len(kernel.func.__closure__) == 2
        and all(cell.cell_contents is False for cell in kernel.func.__closure__),
        "literal frozen pyramidal/dense closure specialization",
    )


def contact_factory_binding(constraint):
    """Hold cached factory/code and its literal dense-pyramidal target."""
    factory = constraint._efc_contact_init
    function = inspect.unwrap(factory)
    code = getattr(function, "__code__", None)
    need(callable(factory) and code is not None, "real cached contact factory code")
    kernel = factory(0, False)
    need(
        callable(kernel.func) and getattr(kernel.func, "__code__", None) is not None,
        "real contact kernel function code",
    )
    check_contact_specialization(kernel)
    module_name = kernel.func.__module__
    python_module = sys.modules.get(module_name)
    need(
        python_module is not None and kernel.func.__globals__ is python_module.__dict__,
        "actual frozen contact function Python namespace",
    )
    return {
        "factory": factory,
        "function": function,
        "code": code,
        "kernel": kernel,
        "kernel_function": kernel.func,
        "kernel_code": kernel.func.__code__,
        "module": kernel.module,
        "python_module_name": module_name,
        "python_module": python_module,
    }


def check_contact_factory(constraint, held):
    need(
        constraint._efc_contact_init is held["factory"]
        and inspect.unwrap(held["factory"]) is held["function"]
        and held["function"].__code__ is held["code"]
        and held["factory"](0, False) is held["kernel"]
        and held["kernel"].func is held["kernel_function"]
        and held["kernel_function"].__code__ is held["kernel_code"]
        and held["kernel"].module is held["module"]
        and held["kernel_function"].__module__ == held["python_module_name"]
        and sys.modules.get(held["python_module_name"]) is held["python_module"]
        and held["kernel_function"].__globals__ is held["python_module"].__dict__,
        "unchanged held cached contact factory, target and module",
    )
    check_contact_specialization(held["kernel"])


def capture_plan():
    """Literal passive-only boundary and capacity declaration, no CUDA import."""
    from mjlab_microduck import stance_contact_boundary_control as control

    return {
        "protocol": control.PROTOCOL,
        "forwards": list(range(control.SAMPLED_FORWARDS)),
        "arms": list(ARMS),
        "packet_count_per_arm": 21,
        "max_bytes_per_arm": control.MAX_INSTANCE_BYTES,
        "max_bytes_all_arms": control.MAX_ALL_ARMS_BYTES,
        "contact_input_order": list(control.CONTACT_INPUT_ORDER),
        "contact_output_order": list(control.CONTACT_OUTPUT_ORDER),
        "context_order": list(control.CONTEXT_FIELDS),
        "complete_order": list(control.COMPLETE_ORDER),
        "stale_prior_fields": list(control.STALE_PRIOR_FIELDS),
        "phase": "construction-complete-BEFORE-solver",
        "flags": dict(FLAGS),
    }


def _checked_junit(raw):
    import xml.etree.ElementTree as ET

    need(type(raw) is bytes and 0 < len(raw) <= 1024**2, "bounded exact JUnit")
    suites = list(ET.fromstring(raw).iter("testsuite"))
    need(
        EXPECTED_TESTS > 0
        and len(suites) == 1
        and int(suites[0].attrib["tests"]) == EXPECTED_TESTS
        and all(
            suites[0].attrib[key] == "0" for key in ("errors", "failures", "skipped")
        ),
        "literal exact source-test count with zero omissions",
    )
    return int(suites[0].attrib["tests"])


def cpu_test_environment(cache):
    """Cap only the CPU pytest child pools; never mutate owner/CUDA settings."""
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU tests keep CUDA hidden")
    return {
        **os.environ,
        **CPU_TEST_THREADS,
        "WARP_CACHE_PATH": str(cache),
    }


def tests(source):
    check_window(CPU_SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "CPU prerequisite keeps CUDA hidden",
    )
    need(
        not os.environ.get("PYTEST_ADDOPTS") and not os.environ.get("PYTEST_PLUGINS"),
        "no injected test selection/plugins",
    )
    binding, version_pin = source_binding(source), packages()
    alias = environment_binding()
    service = service_properties(source, "tests")
    host_before = host()
    directory = output(source, "tests")
    directory.mkdir(parents=True, exist_ok=False)
    files = test_files()
    env = cpu_test_environment(directory / "private-cpu-warp-cache")
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
            env=env,
        )
        log.flush()
        os.fsync(log.fileno())
    need(result.returncode == 0, "frozen exact-source CPU suite passed")
    junit = (directory / "junit.xml").read_bytes()
    count = _checked_junit(junit)
    need(
        binding == source_binding(source),
        "full committed source closure unchanged after CPU tests",
    )
    need(version_pin == packages(), "packages unchanged across CPU suite")
    need(
        environment_binding() == alias,
        "frozen environment alias unchanged after CPU tests",
    )
    record = {
        "protocol": PROTOCOL + ":tests",
        "source_binding": binding,
        "packages": version_pin,
        "environment_alias": alias,
        "unit": service,
        "files": files,
        "tests": count,
        "cpu_test_threads": dict(CPU_TEST_THREADS),
        "host_before": host_before,
        "host_after": host(),
        "flags": FLAGS,
    }
    write(directory / "receipt.json", record)
    inventory = {
        name: digest_file(directory / name, 64 * 1024**2)
        for name in ("receipt.json", "pytest.log", "junit.xml")
    }
    write(directory / "inventory.json", inventory)
    return record


def runtime_thread_snapshot(role, *, require_fresh=False):
    """Observe selected process-local pools before native imports."""
    need(role in THREAD_BUDGET, "literal owner or CUDA-child thread role")
    expected = THREAD_BUDGET[role]
    settings = {name: os.environ.get(name) for name in expected}
    need(settings == expected, "exact source-bound " + role + " thread environment")
    if require_fresh:
        need(
            not any(name in sys.modules for name in ("numpy", "torch", "warp")),
            "fresh native process before NumPy/Torch/Warp imports",
        )
    status = Path("/proc/self/status")
    need(status.is_file(), "Linux native thread-count status")
    rows = [
        line for line in status.read_text().splitlines() if line.startswith("Threads:")
    ]
    need(len(rows) == 1, "one observed native thread-count row")
    try:
        threads = int(rows[0].split()[1])
    except (IndexError, ValueError) as error:
        raise ValueError("plain observed native thread count") from error
    need(type(threads) is int and 1 <= threads <= 64, "bounded native thread count")
    if require_fresh:
        need(threads == 1, "fresh single-thread process before native imports")
    return {"role": role, "pid": os.getpid(), "settings": settings, "threads": threads}


def phase_thread_snapshot(role, phase_record, expected_phase):
    """Normalize one flushed native phase into the exact snapshot schema."""
    need(role in THREAD_BUDGET, "literal owner or CUDA-child thread role")
    expected = THREAD_BUDGET[role]
    need(
        type(phase_record) is dict
        and set(phase_record)
        == {"protocol", "phase", "role", "pid", "threads", "observed_cpu_thread_env"}
        and phase_record["protocol"] == PROTOCOL + ":phase"
        and phase_record["phase"] == expected_phase
        and phase_record["pid"] == os.getpid()
        and phase_record["observed_cpu_thread_env"] == expected,
        "exact flushed thread phase " + expected_phase,
    )
    threads = phase_record["threads"]
    need(type(threads) is int and 1 <= threads <= 64, "bounded phase thread count")
    if role == "cuda_child" and expected_phase == "torch-import-start":
        need(threads == 1, "fresh single-thread child at torch import boundary")
    return {
        "role": role,
        "pid": phase_record["pid"],
        "settings": dict(expected),
        "threads": threads,
    }


def runtime_child_environment(cache):
    """Build the leased child's environment without mutating its owner."""
    return {
        **os.environ,
        **CUDA_CHILD_THREAD_ENV,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONFAULTHANDLER": "1",
        "WARP_CACHE_PATH": str(cache),
    }


def diagnostic_phase(phase, role=None):
    """Flushed non-admitting breadcrumbs; observe, never alter native thread pools."""
    need(
        type(phase) is str and 0 < len(phase) <= 64,
        "bounded diagnostic phase label",
    )
    need(role is None or role in {*ARMS, "candidate", "contact"}, "diagnostic role")
    threads = None
    status = Path("/proc/self/status")
    if status.exists():
        rows = [
            line
            for line in status.read_text().splitlines()
            if line.startswith("Threads:")
        ]
        need(len(rows) == 1, "one observed native thread-count row")
        threads = int(rows[0].split()[1])
    settings = {name: os.environ.get(name) for name in CPU_TEST_THREADS}
    need(
        all(value is None or len(value) <= 64 for value in settings.values()),
        "bounded observed thread settings",
    )
    record = {
        "protocol": PROTOCOL + ":phase",
        "phase": phase,
        "role": role,
        "pid": os.getpid(),
        "threads": threads,
        "observed_cpu_thread_env": settings,
    }
    raw = canonical(record)
    need(
        len(raw) <= 1024 and os.write(2, raw) == len(raw),
        "complete bounded phase write",
    )
    return record


def _compile_modules(wp, ctx, artifacts, device, directory):
    from mujoco_warp._src import constraint
    from mjlab_microduck.stance_friction_runtime_kernel import ascending_friction_dof

    roles = {
        "original": constraint._friction_dof,
        "candidate": ascending_friction_dof,
        "contact": constraint._efc_contact_init(0, False),
    }
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
    records, bindings = {}, {}
    for role, kernel in roles.items():
        module = kernel.module
        artifacts.assert_fresh_module(module, device, 256)
        module_hash = held["hash"][0](module, 256)
        options = canonical(module.options | kernel.options)
        arch = module._get_compile_arch(device)
        need(
            type(arch) is int and arch == device.arch == 120,
            "native sm120 compile target",
        )
        artifact_dir = directory / ("compiled-" + role)
        need(not artifact_dir.exists(), "unique private compiled module directory")
        diagnostic_phase("compile-start", role)
        compiled = held["compile"][0](
            module, device, artifact_dir, role + ".cubin", arch, False
        )
        diagnostic_phase("compile-done", role)
        need(
            compiled is True and not module.execs,
            "fresh compilation before explicit load",
        )
        binary = artifact_dir / (role + ".cubin")
        meta = artifact_dir / module._get_meta_name()
        cu = artifact_dir / (module.get_module_identifier() + ".cu")
        need(
            set(artifact_dir.iterdir()) == {binary, meta, cu},
            "complete generated source/CUBIN/metadata set",
        )
        binary_info, meta_info = digest_file(binary), digest_file(meta)
        need(binary.read_bytes()[:4] == b"\x7fELF", "retained ELF CUBIN")
        artifact = artifacts.retain_artifact(
            binary,
            meta,
            binary_size=binary_info["bytes"],
            binary_sha256=binary_info["sha256"],
            metadata_size=meta_info["bytes"],
            metadata_sha256=meta_info["sha256"],
        )
        artifacts.assert_fresh_module(module, device, 256)
        diagnostic_phase("load-start", role)
        executable = held["load"][0](
            module,
            device,
            block_dim=256,
            binary_path=str(binary),
            output_arch=arch,
            meta_path=str(meta),
        )
        diagnostic_phase("load-done", role)
        need(
            executable is not None
            and executable is module.execs.get((device.context, 256)),
            "actual explicit module load",
        )
        diagnostic_phase("hooks-start", role)
        hooks = held["hooks"][0](executable, kernel)
        diagnostic_phase("hooks-done", role)
        diagnostic_phase("bind-start", role)
        bound = artifacts.bind_loaded_module(
            artifact,
            kernel,
            device,
            executable,
            hooks,
            block_dim=256,
            expected_module_hash=module_hash,
        )
        diagnostic_phase("bind-done", role)
        need(
            canonical(module.options | kernel.options) == options,
            "compiled source and options unchanged",
        )
        bindings[role] = bound
        records[role] = {
            "compiler_config": {
                key: getattr(wp.config, key) for key in COMPILER_CONFIG
            },
            "binding": bound.record(),
            "module_options": json.loads(options),
            "output_arch": arch,
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
                "output_arch": arch,
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
    need(
        all(bindings[role] is not None for role in roles),
        "both original and candidate bound",
    )
    return records, bindings, roles


def frozen_dispatch_entries(constraint, fixture):
    """Pin the executable kernel and shared row helper to captured frozen code."""
    need(
        constraint._friction_dof is fixture._ORIGINAL_KERNEL
        and constraint._friction_dof.func is fixture._ORIGINAL_KERNEL_FUNC
        and fixture._ORIGINAL_KERNEL_FUNC.__code__ is fixture._ORIGINAL_KERNEL_CODE
        and constraint._efc_row is fixture._ORIGINAL_EFC_ROW
        and constraint._efc_row.func is fixture._ORIGINAL_EFC_FUNC
        and fixture._ORIGINAL_EFC_FUNC.__code__ is fixture._ORIGINAL_EFC_CODE,
        "frozen friction kernel and shared row helper identity/code",
    )


def initialize_caller_rng(torch):
    """Set the declared fresh caller streams once; recipe forks preserve them."""
    need(not torch.cuda.is_initialized(), "fresh Torch CUDA before caller RNG setup")
    torch.manual_seed(CALLER_RNG_SEEDS["cpu"])
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(CALLER_RNG_SEEDS["cuda"])
    need(
        torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
        "single declared Torch CUDA device for caller streams",
    )
    raw = {
        "cpu": torch.get_rng_state().cpu().numpy().tobytes(),
        "cuda": torch.cuda.get_rng_state(0).cpu().numpy().tobytes(),
    }
    need(
        len(raw["cpu"]) == 5056
        and sha256(raw["cpu"]).hexdigest()
        == "ba8adae6f1ee70135e097a78de4f08bb885703e3eca406e93e9acf7aafaba8fa"
        and 0 < len(raw["cuda"]) <= 16384,
        "actual bounded seeded caller streams",
    )
    return {
        "seeds": dict(CALLER_RNG_SEEDS),
        "states": {
            name: {"bytes": len(value), "sha256": sha256(value).hexdigest()}
            for name, value in raw.items()
        },
    }


def child(source, lease_fd, owner_pid, declaration_sha):
    check_window(CHILD_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    need(
        type(EXPECTED_TESTS) is int and EXPECTED_TESTS > 0,
        "fully frozen CPU collection before child CUDA imports",
    )
    need(
        os.getppid() == owner_pid and os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "one declared owner and literal child CUDA visibility",
    )
    binding = source_binding(source)
    package_pin, alias = packages(), environment_binding()
    libraries, warp_sources = library_binding(), toolchain_source()
    lease = lease_identity(lease_fd)
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = output(source, "run")
    declaration_path = directory / "declaration.json"
    need(
        digest_file(declaration_path)["sha256"] == declaration_sha,
        "whole declaration authenticated before decoding",
    )
    declaration = json.loads(declaration_path.read_bytes())
    need(
        declaration["protocol"] == PROTOCOL + ":declaration"
        and declaration["flags"] == FLAGS
        and declaration["capture_plan"] == capture_plan()
        and declaration["source_binding"] == binding
        and declaration["packages"] == package_pin
        and declaration["environment_alias"] == alias
        and declaration["libraries"] == libraries
        and declaration["warp_sources"] == warp_sources
        and declaration["thread_budget"] == THREAD_BUDGET
        and declaration["owner_pid"] == owner_pid
        and declaration["lease"] == lease
        and declaration["caller_rng_seeds"] == CALLER_RNG_SEEDS,
        "same exact source, package, alias, library, compiler-source and lease declaration",
    )
    cache = directory / "private-warp-cache"
    need(
        os.environ.get("WARP_CACHE_PATH") == str(cache) and not cache.exists(),
        "fresh process-private Warp cache",
    )

    need(
        os.environ.get("PYTHONFAULTHANDLER") == "1",
        "child fatal-signal logging enabled at startup",
    )
    runtime_thread_snapshot("cuda_child", require_fresh=True)
    torch_import_record = diagnostic_phase("torch-import-start")
    child_before_import = phase_thread_snapshot(
        "cuda_child", torch_import_record, "torch-import-start"
    )
    import torch

    diagnostic_phase("torch-import-done")
    caller_rng = initialize_caller_rng(torch)
    diagnostic_phase("caller-rng-initialized")
    import warp as wp
    from warp._src import context as ctx
    from mujoco_warp._src import constraint
    from mjlab_microduck import stance_bam_load_probe as bam
    from mjlab_microduck import stance_contact_boundary_control as control
    from mjlab_microduck import stance_friction_runtime_kernel as runtime_kernel
    from mjlab_microduck import stance_cuda_artifact_binding as artifacts

    base = __import__("mjlab_microduck.stance_friction_row_cpu_fixture", fromlist=["_"])
    fixture = __import__(
        "mjlab_microduck.stance_friction_prefix_cpu_fixture", fromlist=["_"]
    )
    need(
        digest_file(Path(base.__file__))["sha256"] == BASE_FIXTURE_SHA
        and digest_file(Path(fixture.__file__))["sha256"] == PREFIX_FIXTURE_SHA,
        "unchanged original frozen plant/kernel fixture sources",
    )
    need(
        base._tree_sha256() == base.TREE_SHA256
        and digest_file(base._SOURCE_PATH)["sha256"] == base.CONSTRAINT_SHA256,
        "unchanged pinned MuJoCo-Warp tree and constraint source",
    )
    compiler_config = configure_compiler(wp)
    diagnostic_phase("warp-init-start")
    wp.init()
    warp_init_record = diagnostic_phase("warp-init-done")
    after_warp_init = phase_thread_snapshot(
        "cuda_child", warp_init_record, "warp-init-done"
    )
    need(
        str(ctx.runtime.core._name) == libraries["warp.so"]["path"]
        and str(ctx.runtime.llvm._name) == libraries["warp-clang.so"]["path"],
        "loaded frozen Warp runtime/compiler library paths",
    )
    device = wp.get_device("cuda:0")
    need(
        len(wp.get_cuda_devices()) == 1
        and device.is_cuda
        and str(device) == "cuda:0"
        and not device.is_capturing
        and Path(wp.config.kernel_cache_dir).resolve() == cache.resolve(),
        "single eager CUDA device and private cache",
    )
    stream = wp.Stream(device)
    need(
        type(stream.cuda_stream) is int
        and stream.cuda_stream > 0
        and stream.device is device,
        "one dedicated nonzero CUDA stream",
    )
    held_runtime = {
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
    recipe_case, recipe_case_code = bam._case, bam._case.__code__
    frozen_dispatch_entries(constraint, base)
    held_contact = contact_factory_binding(constraint)
    compiled, bound, _roles = _compile_modules(wp, ctx, artifacts, device, directory)
    diagnostic_phase("three-explicit-loads-complete")
    check_contact_factory(constraint, held_contact)
    diagnostic_phase("contact-factory-checked")
    need(_roles["contact"] is held_contact["kernel"], "compiled literal contact target")
    active_observer = None

    def guard():
        frozen_dispatch_entries(constraint, base)
        check_contact_factory(constraint, held_contact)
        need(
            environment_binding() == alias, "frozen environment alias through dispatch"
        )
        need(
            canonical({key: getattr(wp.config, key) for key in COMPILER_CONFIG})
            == canonical(compiler_config),
            "frozen compiler configuration through dispatch",
        )
        need(
            hasattr(ctx.runtime, "tape") and ctx.runtime.tape is None,
            "no active Warp Tape at dispatch",
        )
        need(
            wp.get_stream(device) is stream,
            "same dedicated nonzero stream is active for recipe dispatch",
        )
        current_launch = (
            active_observer.launch_wrapper
            if active_observer is not None and active_observer.inside
            else held_runtime["launch"][0]
        )
        need(
            wp.launch is current_launch,
            "original launch or currently owned observer wrapper",
        )
        current_runtime = {
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
                current_runtime[name] is function and function.__code__ is code
                for name, (function, code) in held_runtime.items()
                if name != "launch"
            ),
            "frozen Warp compile/load/hook/hash/build/synchronize entries",
        )
        need(
            bam._case is recipe_case and bam._case.__code__ is recipe_case_code,
            "unchanged frozen BAM recipe callable",
        )
        for value in bound.values():
            value.assert_unchanged()

    recipe_arms, arm_observers = {}, {}
    after_recipe = None

    def sink(relative, raw):
        rel = Path(relative)
        need(
            not rel.is_absolute()
            and ".." not in rel.parts
            and type(raw) is bytes
            and len(raw) <= 16 * 1024**2,
            "plain bounded runtime packet",
        )
        target = directory / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        need(
            target.resolve().parent == target.parent.resolve(),
            "packet parent remains within owned run root",
        )
        write(target, raw, cap=16 * 1024**2)

    candidate = runtime_kernel.ascending_friction_dof
    for arm in ARMS:
        diagnostic_phase("arm-start", arm)
        selected_bound = bound["original" if arm == "original" else "candidate"]
        with wp.ScopedDevice(device), wp.ScopedStream(stream):
            observer = control.RuntimeContactBoundaryObserver(
                arm,
                wp=wp,
                constraint=constraint,
                candidate=candidate,
                contact_kernel=_roles["contact"],
                device=device,
                stream=stream,
                sink=sink,
                guard=guard,
            )
            active_observer = observer
            guard()
            with observer:
                selected_bound.assert_unchanged()
                diagnostic_phase("recipe-case-start", arm)
                recipe, packets = recipe_case(torch, wp)
                recipe_record = diagnostic_phase("recipe-case-done", arm)
                after_recipe = phase_thread_snapshot(
                    "cuda_child", recipe_record, "recipe-case-done"
                )
                guard()
            arm_observers[arm] = observer.receipt()
            recipe_arms.setdefault(arm, {})["boundary"] = observer.boundary_receipt()
            active_observer = None
        recipe_arms[arm] = {
            "boundary": recipe_arms[arm]["boundary"],
            "record": recipe,
            "packets": {
                name: {
                    "path": arm + "/" + name,
                    "bytes": len(raw),
                    "sha256": sha256(raw).hexdigest(),
                }
                for name, raw in packets.items()
            },
        }
        for name, raw in packets.items():
            sink(arm + "/" + name, raw)
    need(after_recipe is not None, "completed source-bound recipe thread snapshot")
    need(
        sum(len(row["entries"]) for row in arm_observers.values()) == 63,
        "exact 21 target entries per three fresh sequential arms",
    )
    for row in bound.values():
        row.assert_unchanged()
    need(
        binding == source_binding(source)
        and lease == lease_identity(lease_fd)
        and libraries == library_binding()
        and warp_sources == toolchain_source(),
        "closed exact source, lease, libraries and Warp toolchain",
    )
    device_record = {
        "alias": str(device),
        "object_id": id(device),
        "context": device.context,
        "arch": device.arch,
        "name": str(device),
        "hardware_gpu_name": device.name,
        "stream": stream.cuda_stream,
        "stream_object_id": id(stream),
        "toolkit_version": list(ctx.runtime.toolkit_version),
        "driver_version": list(ctx.runtime.driver_version),
    }
    return {
        "thread_budget": {
            role: dict(settings) for role, settings in THREAD_BUDGET.items()
        },
        "thread_observations": {
            "pre_import": child_before_import,
            "after_warp_init": after_warp_init,
            "after_recipe": after_recipe,
        },
        "protocol": PROTOCOL + ":child",
        "source_binding": binding,
        "packages": package_pin,
        "environment_alias": alias,
        "libraries": libraries,
        "warp_sources": warp_sources,
        "owner_pid": owner_pid,
        "child_pid": os.getpid(),
        "lease": lease,
        "declaration_sha256": declaration_sha,
        "device": device_record,
        "compiler_config": compiler_config,
        "compiled": compiled,
        "caller_rng": caller_rng,
        "arms": {
            name: {
                "recipe": recipe_arms[name]["record"],
                "boundary": recipe_arms[name]["boundary"],
                "observer": arm_observers[name],
                "packets": recipe_arms[name]["packets"],
            }
            for name in ARMS
        },
        "flags": FLAGS,
    }


def _mac_test_prerequisite(source, binding, package_pin, inventory_sha):
    directory = (
        ROOT / "artifacts/tools" / f"contact-boundary-tick-mac-tests-{source[:12]}"
    )
    inventory_path = directory / "inventory.json"
    need(
        digest_file(inventory_path, 128 * 1024)["sha256"] == inventory_sha,
        "externally anchored Mac CPU inventory",
    )
    inventory_raw = inventory_path.read_bytes()
    inventory = json.loads(inventory_raw)
    need(
        sha256(canonical(inventory)).hexdigest() == inventory_sha,
        "canonical Mac inventory matches external anchor",
    )
    need(
        type(inventory) is dict
        and set(inventory) == {"receipt.json", "pytest.log", "junit.xml"},
        "exact three-leaf Mac CPU inventory",
    )
    raw = _raw_inventory(
        directory, inventory, total_cap=64 * 1024**2, external_inventory=True
    )
    receipt = json.loads(raw["receipt.json"])
    need(
        receipt
        == {
            "protocol": PROTOCOL + ":mac-tests",
            "source_binding": binding,
            "packages": package_pin,
            "files": test_files(),
            "tests": EXPECTED_TESTS,
            "cpu_test_threads": dict(CPU_TEST_THREADS),
            "python": "3.12.12",
            "machine": "arm64",
            "flags": FLAGS,
        },
        "exact independently retained Mac CPU receipt and 94-file scope",
    )
    count = _checked_junit(raw["junit.xml"])
    return {
        "inventory_sha256": inventory_sha,
        "inventory": inventory,
        "receipt": receipt,
        "junit": {"tests": count, "errors": 0, "failures": 0, "skipped": 0},
        "raw": raw,
    }


def run(source, tests_inventory_sha, mac_tests_inventory_sha):
    owner_thread_start = runtime_thread_snapshot("owner", require_fresh=True)
    thread_budget = {role: dict(settings) for role, settings in THREAD_BUDGET.items()}
    from mjlab_microduck import stance_contact_boundary_receiver as receiver

    check_window(SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    binding, package_pin = source_binding(source), packages()
    alias, service = environment_binding(), service_properties(source, "run")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden in owner")
    need(
        type(tests_inventory_sha) is str
        and re.fullmatch(r"[0-9a-f]{64}", tests_inventory_sha),
        "external CPU inventory SHA anchor",
    )
    prerequisite = output(source, "tests")
    need(
        digest_file(prerequisite / "inventory.json")["sha256"] == tests_inventory_sha,
        "CPU prerequisite inventory whole bytes before decode",
    )
    test_inventory = json.loads((prerequisite / "inventory.json").read_bytes())
    need(
        sha256(canonical(test_inventory)).hexdigest() == tests_inventory_sha,
        "canonical native test inventory matches external anchor",
    )
    test_raw = _raw_inventory(
        prerequisite,
        test_inventory,
        total_cap=64 * 1024**2,
        external_inventory=True,
        cpu_cache=True,
    )
    test_receipt = json.loads(test_raw["receipt.json"])
    need(
        test_receipt["source_binding"] == binding
        and test_receipt["packages"] == package_pin
        and test_receipt["environment_alias"] == alias
        and test_receipt["files"] == test_files()
        and test_receipt["tests"] == EXPECTED_TESTS > 0,
        "same-source complete CPU prerequisite",
    )
    receiver.cpu_thread_settings(test_receipt)
    test_terminal = dict(
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
        test_terminal
        == {
            "MainPID": "0",
            "SubState": "exited",
            "Result": "success",
            "ExecMainStatus": "0",
            "InvocationID": test_receipt["unit"]["InvocationID"],
        },
        "successful completed CPU service terminal",
    )
    historical = historical_synthetic()
    negative = historical_negative()
    need(
        type(mac_tests_inventory_sha) is str
        and re.fullmatch(r"[0-9a-f]{64}", mac_tests_inventory_sha),
        "external Mac inventory SHA anchor",
    )
    mac_tests = _mac_test_prerequisite(
        source, binding, package_pin, mac_tests_inventory_sha
    )
    need(
        _checked_junit(test_raw["junit.xml"]) == EXPECTED_TESTS,
        "whole-authenticated CPU JUnit exact count and zero omissions",
    )

    libraries, warp_sources = library_binding(), toolchain_source()
    directory = output(source, "run")
    directory.mkdir(parents=True, exist_ok=False)
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    process, samples = None, []
    try:
        lease = lease_identity(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_host = host()
        need(
            initial_host["compute_pids"] == [] and initial_host["used_mib"] <= 1024,
            "idle leased GPU before child",
        )
        declaration = {
            "protocol": PROTOCOL + ":declaration",
            "source_binding": binding,
            "packages": package_pin,
            "environment_alias": alias,
            "libraries": libraries,
            "warp_sources": warp_sources,
            "unit": service,
            "owner_pid": os.getpid(),
            "thread_budget": thread_budget,
            "owner_thread_start": owner_thread_start,
            "lease": lease,
            "host_before": initial_host,
            "tests": test_receipt,
            "tests_inventory_sha256": tests_inventory_sha,
            "tests_inventory": test_inventory,
            "tests_terminal": test_terminal,
            "mac_tests_inventory": mac_tests["inventory"],
            "mac_tests_inventory_sha256": mac_tests_inventory_sha,
            "dependency": historical,
            "negative_dependency": negative,
            "capture_plan": capture_plan(),
            "case_order": list(ARMS),
            "caller_rng_seeds": dict(CALLER_RNG_SEEDS),
            "started_utc_ns": time.time_ns(),
            "cutoff_utc": "2026-10-07T23:30:00Z",
            "child_timeout_seconds": CHILD_SECONDS,
            "flags": FLAGS,
        }
        write(directory / "declaration.json", declaration)
        tests_copy = directory / "tests"
        tests_copy.mkdir()
        for name, raw in test_raw.items():
            write(tests_copy / name, raw, cap=16 * 1024**2)
        mac_copy = directory / "mac-tests"
        mac_copy.mkdir()
        for name, raw in mac_tests["raw"].items():
            write(mac_copy / name, raw, cap=16 * 1024**2)
        declaration_sha = sha256(canonical(declaration)).hexdigest()
        env = runtime_child_environment(directory / "private-warp-cache")
        check_window(CHILD_SECONDS + CLOSEOUT_SECONDS + MARGIN)
        with (directory / "child.log").open("xb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mjlab_microduck.stance_contact_boundary_probe",
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
            seen = False
            observed_ppid = None
            while process.poll() is None:
                need(
                    time.monotonic() - started < CHILD_SECONDS,
                    "fresh CUDA child timeout",
                )
                check_window(CLOSEOUT_SECONDS + MARGIN)
                proc_stat = Path(f"/proc/{process.pid}/stat")
                if proc_stat.exists():
                    observed_ppid = int(
                        proc_stat.read_text().rsplit(") ", 1)[1].split()[1]
                    )
                    need(
                        observed_ppid == os.getpid(),
                        "observed child parent is this owner",
                    )
                sample = host((process.pid,))
                need(
                    all(pid == process.pid for pid in sample["compute_pids"]),
                    "only owned CUDA child visible",
                )
                need(
                    sample["used_mib"] <= 6 * 1024 and sample["temperature_c"] < 75,
                    "bounded GPU memory/temperature",
                )
                seen |= process.pid in sample["compute_pids"]
                samples.append(
                    {
                        **sample,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "child_pid": process.pid,
                    }
                )
                need(
                    len(samples) <= 270 and log.tell() <= 16 * 1024**2,
                    "bounded owner samples and child log",
                )
                time.sleep(2)
            log.flush()
            os.fsync(log.fileno())
        need(
            process.returncode == 0 and seen and observed_ppid == os.getpid(),
            "observed successful owned GPU child",
        )
        owner_after_child = runtime_thread_snapshot("owner")
        final_host = host()
        need(
            final_host["compute_pids"] == [] and final_host["used_mib"] <= 1024,
            "GPU idle after child",
        )
        need(
            binding == source_binding(source)
            and package_pin == packages()
            and alias == environment_binding()
            and lease == lease_identity(fd)
            and libraries == library_binding()
            and warp_sources == toolchain_source(),
            "owner source, packages, alias, lease, libraries and compiler-source closure",
        )
        write(
            directory / "owner.json",
            {
                "protocol": PROTOCOL + ":owner",
                "source_binding": binding,
                "thread_budget": thread_budget,
                "thread_observations": {
                    "pre_import": owner_thread_start,
                    "after_child": owner_after_child,
                },
                "packages": package_pin,
                "environment_alias": alias,
                "lease": lease,
                "owner_pid": os.getpid(),
                "child_pid": process.pid,
                "child_exit": process.returncode,
                "child_observed": seen,
                "child_ppid": observed_ppid,
                "finished_utc_ns": time.time_ns(),
                "host_after": final_host,
                "samples": samples,
                "flags": FLAGS,
            },
        )
        inventory = {
            str(path.relative_to(directory)): digest_file(path, 16 * 1024**2)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        need(
            len(inventory) <= 1024
            and sum(item["bytes"] for item in inventory.values()) <= 768 * 1024**2,
            "complete bounded runtime inventory",
        )
        report = receiver.verify_run(
            directory,
            inventory,
            expected_source=binding,
            expected_tests_sha=tests_inventory_sha,
        )
        closeout = (
            ROOT / "artifacts/tools" / f"contact-boundary-tick-closeout-{source[:12]}"
        )
        closeout.mkdir(parents=True, exist_ok=False)
        write(closeout / "inventory.json", inventory)
        write(closeout / "receiver.json", report, cap=8 * 1024**2)
        return report
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        failure = directory / "failure.json"
        if not failure.exists():
            write(
                failure,
                {
                    "protocol": PROTOCOL + ":failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
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
    parser.add_argument("--mac-tests-inventory-sha")
    args = parser.parse_args(argv)
    if args.mode == "tests":
        result = tests(args.source)
    elif args.mode == "run":
        need(
            args.mac_tests_inventory_sha is not None,
            "external Mac CPU inventory SHA required",
        )
        result = run(
            args.source, args.tests_inventory_sha, args.mac_tests_inventory_sha
        )
    else:
        need(
            type(args.lease_fd) is int
            and args.lease_fd >= 3
            and type(args.owner_pid) is int
            and re.fullmatch(r"[0-9a-f]{64}", args.declaration_sha or ""),
            "literal inherited child arguments",
        )
        result = child(args.source, args.lease_fd, args.owner_pid, args.declaration_sha)
        write(output(args.source, "run") / "child.json", result)
    print(
        json.dumps(
            {"protocol": PROTOCOL, "mode": args.mode, "flags": FLAGS, "complete": True},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
