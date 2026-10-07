"""Whole-byte, numerical replay receiver for the bounded Oct8 CUDA component.

This is an observation checker, not a sandbox or a simulator admission. Its
inventory and expected source must come from the independent controlling caller.
No CUDA runtime, compiler, robot model, learner or optimizer is imported.
"""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat

from mjlab_microduck import stance_friction_prefix_cuda_numerical as numerical

PROTOCOL = "microduck-friction-prefix-cuda-probe-oct8-v1"
FLAGS = {
    key: False
    for key in (
        "runtime_cause_proven",
        "native_qualified",
        "full_window_qualified",
        "training_authorized",
        "physical_acceptance",
    )
}
CASES = (
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
INPUTS = ("frictionloss", "qvel", "invweight", "solref", "solimp", "timestep")
OUTPUTS = (
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
ROOT = "/home/yanbo/work/microduck_rl-com-entry-20261006"
BRANCH = "feat/athletics-obstacle-curriculum"
GPU = "GPU-7d72b360-33bc-2cee-3ff4-a954474011b5"
MACHINE = "7d6778c98cb345788b8c1a410f19ad35"
VERSIONS = {
    "torch": "2.9.1",
    "warp-lang": "1.12.0",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.8.1",
    "mjlab": "1.3.0",
    "better-actuator-models": "1.0.1",
}
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
EXPECTED_TESTS = 2671
STRUCTURAL_KEYS = (
    "all_prefix_rows_preserved",
    "all_inactive_suffix_preserved",
    "all_sparse_scratch_unchanged",
    "counts_and_addresses_complete",
    "candidate_rows_ascending",
    "candidate_replay_addressed_exact",
    "candidate_replay_full_bank_bit_identical",
    "original_candidate_addressed_exact",
)
COMPILER_CONFIG = {
    "mode": "release",
    "optimization_level": None,
    "verify_fp": False,
    "llvm_cuda": False,
    "cache_kernels": True,
    "verify_autograd_array_access": False,
    "use_precompiled_headers": False,
}


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def matrix_matches(cases):
    """Overflow stays negative AND must remain a well-formed controlled sample."""
    need(
        type(cases) is dict and set(cases) == set(CASES),
        "exact ten recomputed matrix cases",
    )
    for name, row in cases.items():
        need(
            type(row) is dict
            and set(STRUCTURAL_KEYS)
            | {
                "component_exact_without_overflow",
                "overflow_negative",
                "overflow_cases_match_pinned_matrix",
            }
            <= set(row),
            "complete recomputed structural matrix fields",
        )
        overflow = name in {"overflow", "maximum"}
        if not all(row[key] is True for key in STRUCTURAL_KEYS):
            return False
        if (
            row["component_exact_without_overflow"] is not (not overflow)
            or row["overflow_negative"] is not overflow
            or row["overflow_cases_match_pinned_matrix"] is not True
        ):
            return False
    return True


def need(ok, message):
    if not ok:
        raise ValueError(message)


def plain(value, minimum=0, maximum=2**64 - 1):
    need(type(value) is int and minimum <= value <= maximum, "plain bounded integer")
    return value


def sha(value):
    need(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
        "plain exact SHA256",
    )
    return value


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "unique packet JSON keys")
        result[key] = value
    return result


def decode(raw):
    def reject(value):
        raise ValueError("finite packet JSON: " + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def stat_identity(item):
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)


def authenticate(directory, inventory, cap=16 * 1024**2):
    """Read every externally anchored whole file before interpreting any packet."""
    directory = Path(directory).absolute()
    need(directory.resolve(strict=True) == directory, "plain retained root")
    need(
        type(inventory) is dict and 0 < len(inventory) <= 512,
        "bounded external inventory",
    )
    result, total = {}, 0
    for name, info in inventory.items():
        need(
            type(name) is str
            and name == str(Path(name))
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts,
            "safe relative anchored path",
        )
        need(
            type(info) is dict and set(info) == {"bytes", "sha256"},
            "external whole-file anchor fields",
        )
        size = plain(info["bytes"], 0, cap)
        sha(info["sha256"])
        path = directory / name
        need(path.resolve(strict=True) == path, "no symlink retained file or parent")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            need(
                stat.S_ISREG(before.st_mode) and before.st_size == size,
                "regular exact retained length",
            )
            chunks, remaining = [], size + 1
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after, current = os.fstat(fd), path.stat(follow_symlinks=False)
            need(
                stat_identity(before) == stat_identity(after) == stat_identity(current)
                and path.resolve(strict=True) == path,
                "stable retained bytes and path",
            )
            need(
                len(raw) == size and sha256(raw).hexdigest() == info["sha256"],
                "whole retained authentication before JSON decode",
            )
        finally:
            os.close(fd)
        total += size
        need(total <= 128 * 1024**2, "bounded complete raw inventory bytes")
        result[name] = raw
    return result


def flags(value):
    need(
        type(value) is dict
        and set(value) == set(FLAGS)
        and all(value[key] is False for key in FLAGS),
        "all admission flags remain literally false",
    )


def source(value, expected):
    need(
        type(expected) is dict and type(value) is dict and value == expected,
        "externally pinned whole source binding",
    )
    need(
        set(value) == {"source", "tree", "branch", "leaves"}
        and value["branch"] == BRANCH,
        "exact source schema and feature branch",
    )
    need(
        all(
            type(value[key]) is str and re.fullmatch(r"[0-9a-f]{40}", value[key])
            for key in ("source", "tree")
        ),
        "exact committed source and tree",
    )
    leaves = value["leaves"]
    need(
        type(leaves) is dict and 1 <= len(leaves) <= 2000,
        "complete externally matched source leaves",
    )
    for name, info in leaves.items():
        need(
            type(name) is str
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and type(info) is dict
            and set(info) == {"bytes", "sha256", "git_blob"},
            "plain tracked source leaves",
        )
        plain(info["bytes"], 0, 16 * 1024**2)
        sha(info["sha256"])
        need(
            type(info["git_blob"]) is str
            and re.fullmatch(r"[0-9a-f]{40}", info["git_blob"]),
            "committed blob anchor",
        )


def host(value, child=None, idle=False):
    need(
        type(value) is dict
        and {
            "machine",
            "gpu",
            "driver",
            "temperature_c",
            "used_mib",
            "compute_pids",
            "protected",
            "filmbrain",
        }
        <= set(value),
        "complete observed host gate",
    )
    need(
        value["machine"] == MACHINE
        and value["gpu"] == GPU
        and value["driver"] == "595.95",
        "fixed observed 100.98 GPU",
    )
    plain(value["temperature_c"], 0, 74)
    plain(value["used_mib"], 0, 1024 if idle else 24576)
    need(
        type(value["compute_pids"]) is list
        and all(type(pid) is int and pid == child for pid in value["compute_pids"])
        and (not idle or not value["compute_pids"]),
        "no foreign observed GPU processes",
    )
    protected = {
        namespace + name: {"ActiveState": "inactive", "MainPID": "0"}
        for namespace in ("system:", "user:")
        for name in (
            "recomo-ai-mission-vllm.service",
            "recomo-ai-mission-subject-model-worker.service",
        )
    }
    need(
        value["protected"] == protected
        and value["filmbrain"]
        == {
            "recomo-filmbrain-observatory.service": {
                "ActiveState": "active",
                "MainPID": "521",
                "NRestarts": "0",
            },
            "recomo-filmbrain-video-playground.service": {
                "ActiveState": "active",
                "MainPID": "298048",
                "NRestarts": "0",
            },
        },
        "protected services inactive and FilmBrain unchanged",
    )


def unit(value, source_sha, mode, owner):
    need(
        type(value) is dict
        and set(value)
        == {
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
        },
        "actual bounded unit fields",
    )
    expected = {
        "Id": f"microduck-friction-prefix-cuda-{mode}-{source_sha[:12]}.service",
        "MainPID": str(owner),
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
        all(value[key] == item for key, item in expected.items())
        and re.fullmatch(r"[0-9a-f]{32}", value["InvocationID"]),
        "actual unit limits and invocation",
    )


def contiguous(shape):
    stride, result = 4, []
    for dim in reversed(shape):
        result.append(stride)
        stride *= plain(dim, 1, 640)
    return list(reversed(result)), stride


def allocation(row, shape, dtype, device):
    need(
        type(row) is dict
        and set(row)
        == {
            "object_id",
            "pointer",
            "device",
            "context",
            "warp_dtype",
            "shape",
            "strides",
            "bytes",
            "host_shape",
            "host_dtype",
        },
        "complete actual CUDA allocation fields",
    )
    plain(row["object_id"], 1)
    plain(row["pointer"], 1)
    plain(row["context"], 1)
    need(
        row["device"] == "cuda:0" and row["context"] == device["context"],
        "same actual device and context per buffer",
    )
    strides, size = contiguous(shape)
    need(
        type(row["host_shape"]) is list
        and all(type(dim) is int for dim in row["host_shape"])
        and row["host_shape"] == shape
        and row["bytes"] == size
        and type(row["bytes"]) is int,
        "whole host allocation shape and bytes",
    )
    vector = dtype in ("solref", "solimp")
    warp_shape, warp_strides = (
        (shape[:-1], strides[:-1]) if vector else (shape, strides)
    )
    expected_dtype = (
        "<class 'warp._src.types.vec2f'>"
        if dtype == "solref"
        else "<class 'mujoco_warp._src.types.vec5f'>"
        if dtype == "solimp"
        else "<class 'warp._src.types.int32'>"
        if dtype == "int32"
        else "<class 'warp._src.types.float32'>"
    )
    need(
        row["host_dtype"] == ("int32" if dtype == "int32" else "float32")
        and row["warp_dtype"] == expected_dtype,
        "actual frozen Warp scalar/vector dtype",
    )
    need(
        type(row["shape"]) is list
        and type(row["strides"]) is list
        and all(type(item) is int for item in row["shape"] + row["strides"])
        and row["shape"] == warp_shape
        and row["strides"] == warp_strides,
        "actual contiguous Warp vector shape/strides",
    )
    return row["pointer"], row["pointer"] + size


def compiled(value, raw, inventory, directory, device):
    need(
        type(value) is dict and set(value) == {"original", "candidate"},
        "two retained compiled modules",
    )
    identities, entries = [], None
    for role in ("original", "candidate"):
        row = value[role]
        need(
            type(row) is dict
            and set(row)
            == {
                "compiler_config",
                "binding",
                "module_options",
                "output_arch",
                "generated_source",
                "binary",
                "metadata",
                "explicit_load",
                "runtime_entries",
            },
            "complete compile and explicit-load witness",
        )
        binding = row["binding"]
        need(
            canonical(row["compiler_config"]) == canonical(COMPILER_CONFIG),
            "actual explicit bounded compiler config and PCH disabled",
        )
        need(
            binding["protocol"] == "microduck-retained-cuda-artifact-binding-oct7-v1"
            and binding["artifact_format"] == "cubin"
            and binding["device"] == "cuda:0"
            and binding["device_arch"] == device["arch"]
            and binding["context"] == device["context"]
            and binding["block_dim"] == 256,
            "actual retained CUBIN device/cache key",
        )
        sha(binding["module_hash"])
        for key in ("module_handle", "forward_handle"):
            plain(binding[key], 1)
        plain(binding["forward_smem_bytes"], 0, 1024**2)
        need(
            all(
                binding[key] is False
                for key in (
                    "driver_jit_machine_code_observed",
                    "loaded_binary_bytes_observed",
                    "native_execution_qualified",
                    "training_authorized",
                    "physical_acceptance",
                )
            ),
            "opaque loaded/JIT bytes and non-admission",
        )
        need(
            type(binding["symbol"]) is str
            and binding["symbol"].endswith("_cuda_kernel_forward"),
            "actual forward symbol",
        )
        ids = binding["observed_object_ids"]
        need(
            type(ids) is dict
            and set(ids) == {"kernel", "module", "device", "executable", "hooks"}
            and all(type(item) is int and item > 0 for item in ids.values())
            and ids["device"] == device["object_id"],
            "actual retained object identities",
        )
        identities.append(ids)
        options = row["module_options"]
        need(
            type(options) is dict
            and options.get("enable_backward") is False
            and options.get("strip_hash") is False
            and type(options.get("block_dim")) is int
            and options["block_dim"] == 256
            and row["output_arch"] == device["arch"],
            "actual merged forward-only compile options",
        )
        paths = {}
        for kind in ("generated_source", "binary", "metadata"):
            info = row[kind]
            need(
                type(info) is dict and set(info) == {"path", "bytes", "sha256"},
                "retained compilation file anchor",
            )
            path = Path(info["path"])
            need(
                path.parent == Path(directory) / ("compiled-" + role),
                "own isolated retained compile output",
            )
            name = str(path.relative_to(directory))
            need(
                name in raw
                and inventory[name] == {key: info[key] for key in ("bytes", "sha256")},
                "compiled file belongs to whole authenticated raw inventory",
            )
            paths[kind] = name
        need(
            raw[paths["binary"]][:4] == b"\x7fELF"
            and paths["binary"].endswith("/" + role + ".cubin")
            and paths["generated_source"].endswith(".cu")
            and paths["metadata"].endswith(".meta"),
            "actual generated CUDA source and ELF CUBIN input",
        )
        meta = decode(raw[paths["metadata"]])
        need(
            type(meta) is dict
            and all(
                type(key) is str
                and key.endswith(
                    (
                        "_cuda_kernel_forward_smem_bytes",
                        "_cuda_kernel_backward_smem_bytes",
                    )
                )
                and type(item) is int
                and 0 <= item <= 1024**2
                for key, item in meta.items()
            )
            and meta.get(binding["symbol"] + "_smem_bytes")
            == binding["forward_smem_bytes"],
            "whole metadata agrees with forward hook",
        )
        for kind in ("binary", "metadata"):
            need(
                binding[kind + "_path"] == row[kind]["path"]
                and binding[kind + "_bytes"] == row[kind]["bytes"]
                and binding[kind + "_sha256"] == row[kind]["sha256"],
                "bound loaded input file identity",
            )
        load = row["explicit_load"]
        need(
            load
            == {
                "module_object_id": ids["module"],
                "returned_executable_id": ids["executable"],
                "device_object_id": ids["device"],
                "block_dim": 256,
                "binary_path": row["binary"]["path"],
                "meta_path": row["metadata"]["path"],
                "output_arch": device["arch"],
                "fresh_cache_before": True,
            }
            and load["fresh_cache_before"] is True,
            "exact explicit-load arguments/result with empty cache witness",
        )
        runtime = row["runtime_entries"]
        need(
            type(runtime) is dict
            and set(runtime)
            == {
                "launch",
                "load",
                "compile",
                "hooks",
                "hash",
                "load_cuda",
                "build_cuda",
                "synchronize",
            },
            "held frozen runtime entry point set",
        )
        for record in runtime.values():
            need(
                type(record) is dict
                and set(record) == {"object_id", "code_id", "qualname"}
                and type(record["qualname"]) is str,
                "held runtime code identity fields",
            )
            plain(record["object_id"], 1)
            plain(record["code_id"], 1)
        need(
            entries is None or entries == runtime,
            "same held runtime entries for both modules",
        )
        entries = runtime
    need(
        all(
            identities[0][key] != identities[1][key]
            for key in ("kernel", "module", "executable", "hooks")
        ),
        "distinct original and candidate loaded objects",
    )


def verify_run(directory, inventory, *, expected_source, expected_tests_sha):
    """Independent replay with externally supplied source and raw-file anchors."""
    directory = Path(directory)
    raw = authenticate(directory, inventory)
    need(
        {"declaration.json", "owner.json", "child.packet.json", "child.log"} <= set(raw)
        and "failure.json" not in raw,
        "complete successful raw owner/child evidence",
    )
    declaration, owner, child = (
        decode(raw[name])
        for name in ("declaration.json", "owner.json", "child.packet.json")
    )
    for record, suffix in (
        (declaration, "declaration"),
        (owner, "owner"),
        (child, "child"),
    ):
        need(
            type(record) is dict and record["protocol"] == PROTOCOL + ":" + suffix,
            "exact new CUDA-only packet protocol",
        )
        flags(record["flags"])
    for binding in (
        declaration["source_binding"],
        owner["source_binding"],
        child["source_binding_before"],
        child["source_binding_after"],
    ):
        source(binding, expected_source)
    source_sha = expected_source["source"]
    native_directory = (
        ROOT + "/artifacts/evaluations/friction-prefix-cuda-run-" + source_sha[:12]
    )
    owner_pid, child_pid = plain(owner["owner_pid"], 1), plain(owner["child_pid"], 1)
    need(
        owner_pid != child_pid
        and child["owner_pid"] == declaration["owner_pid"] == owner_pid
        and child["child_pid"] == child_pid
        and type(owner["child_exit"]) is int
        and owner["child_exit"] == 0,
        "fresh distinct successfully completed child under owner",
    )
    unit(declaration["unit"], source_sha, "run", owner_pid)
    need(
        declaration["packages"] == child["packages"] == VERSIONS,
        "frozen exact package imports",
    )
    need(
        declaration["libraries"] == child["libraries"]
        and set(child["libraries"]) == set(LIBRARIES),
        "same actual frozen compiler/runtime library binding",
    )
    for name, (size, digest) in LIBRARIES.items():
        need(
            child["libraries"][name]
            == {
                "path": ROOT + "/.venv/lib/python3.12/site-packages/warp/bin/" + name,
                "bytes": size,
                "sha256": digest,
            },
            "whole frozen native library anchor",
        )
    warp_sources = child["warp_sources"]
    need(
        declaration["warp_sources"] == warp_sources
        and warp_sources["root"] == ROOT + "/.venv/lib/python3.12/site-packages/warp"
        and warp_sources["sha256"]
        == "4aa3c865b7e523e1c0bef175f80ed51b00543569246d524914e78c333cdf9e6c",
        "same complete frozen compiler source/header binding",
    )
    leaves = warp_sources["leaves"]
    need(
        type(leaves) is dict
        and len(leaves) == 460
        and sum(plain(info["bytes"]) for info in leaves.values()) == 9292904
        and sha256(
            (
                json.dumps(
                    leaves, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            ).encode()
        ).hexdigest()
        == warp_sources["sha256"],
        "whole pinned Warp Python/compiler-header source inventory",
    )
    lease = declaration["lease"]
    need(
        type(lease) is dict
        and set(lease) == {"path", "device", "inode", "bytes"}
        and lease["path"]
        == "/home/yanbo/data/recomo/film-brain/runtimes/wan22-fun-camera-3e8d686-py312-cu130/wan-gpu.lock"
        and lease["bytes"] == 0
        and type(lease["bytes"]) is int
        and child["lease"] == lease,
        "same inherited existing empty GPU lease",
    )
    plain(lease["device"], 1)
    plain(lease["inode"], 1)
    need(
        child["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest()
        and declaration["case_order"] == list(CASES)
        and declaration["launches"] == 30
        and type(declaration["launches"]) is int
        and declaration["cutoff_utc"] == "2026-10-07T23:30:00Z"
        and declaration["child_timeout_seconds"] == 540,
        "whole anchored declaration and bounded literal launch plan",
    )
    started, finished = (
        plain(declaration["started_utc_ns"], 1),
        plain(owner["finished_utc_ns"], 1),
    )
    need(
        started < finished < 1791415800 * 10**9 and finished - started < 600 * 10**9,
        "observed bounded job closure before authorization cutoff",
    )
    need(
        declaration["tests_inventory_sha256"] == sha(expected_tests_sha),
        "externally anchored same-source CPU prerequisite",
    )
    tests = declaration["tests"]
    source(tests["source_binding"], expected_source)
    flags(tests["flags"])
    need(
        tests["protocol"] == PROTOCOL + ":tests"
        and tests["packages"] == VERSIONS
        and type(tests["files"]) is list
        and len(tests["files"]) == len(set(tests["files"])) == 85
        and all(
            type(name) is str
            and name.startswith("tests/test_")
            and name.endswith(".py")
            for name in tests["files"]
        ),
        "exact source-test scope",
    )
    need(
        type(tests["tests"]) is int and tests["tests"] == EXPECTED_TESTS > 0,
        "exact predeclared collected source-test count",
    )
    unit(tests["unit"], source_sha, "tests", plain(int(tests["unit"]["MainPID"]), 1))
    need(
        declaration["tests_terminal"]
        == {
            "MainPID": "0",
            "SubState": "exited",
            "Result": "success",
            "ExecMainStatus": "0",
            "InvocationID": tests["unit"]["InvocationID"],
        },
        "retained CPU prerequisite is complete before CUDA",
    )
    # Its separate inventory is externally anchored and fully replayed by caller.
    need(
        type(declaration["tests_inventory"]) is dict
        and set(declaration["tests_inventory"])
        == {"receipt.json", "pytest.log", "junit.xml"},
        "complete retained CPU-test inventory",
    )
    test_inventory = declaration["tests_inventory"]
    need(
        sha256(canonical(test_inventory)).hexdigest() == expected_tests_sha
        and test_inventory["receipt.json"]
        == {
            "bytes": len(canonical(tests)),
            "sha256": sha256(canonical(tests)).hexdigest(),
        },
        "CPU receipt is whole-bound to independently anchored test inventory",
    )
    host(declaration["host_before"], idle=True)
    host(owner["host_after"], idle=True)
    need(
        type(owner["samples"]) is list and len(owner["samples"]) <= 300,
        "bounded owner monitoring samples",
    )
    for sample in owner["samples"]:
        need(
            type(sample["elapsed_seconds"]) in (int, float)
            and math.isfinite(sample["elapsed_seconds"])
            and 0 <= sample["elapsed_seconds"] < 540,
            "bounded finite monitoring time",
        )
        host(sample, child_pid)
    device = child["device"]
    need(
        type(device) is dict
        and set(device)
        == {
            "alias",
            "object_id",
            "context",
            "arch",
            "name",
            "stream",
            "stream_object_id",
            "toolkit_version",
            "driver_version",
        }
        and device["alias"] == "cuda:0"
        and type(device["name"]) is str
        and "Blackwell" in device["name"],
        "actual declared CUDA device",
    )
    for key in ("object_id", "context", "stream", "stream_object_id"):
        plain(device[key], 1)
    plain(device["arch"], 100, 120)
    for key in ("toolkit_version", "driver_version"):
        need(
            type(device[key]) is list
            and len(device[key]) == 2
            and all(type(item) is int and item >= 0 for item in device[key]),
            "actual plain toolkit/driver version",
        )
    compiled(child["compiled"], raw, inventory, native_directory, device)
    need(
        type(child["allocations"]) is dict
        and set(child["allocations"]) == set(CASES)
        and type(child["launches"]) is list
        and len(child["launches"]) == 30,
        "full ten-case allocation and thirty-launch packets",
    )
    report = numerical.audit_cases(child["cases"])
    for index, case_name in enumerate(CASES):
        case = child["cases"][case_name]
        worlds, cap = numerical.pinned._CASES[case_name][:2]
        intervals, ids = [], []
        for ordinal, side in enumerate(("original", "candidate0", "candidate1")):
            groups = child["allocations"][case_name][side]
            need(
                type(groups) is dict
                and set(groups) == {"inputs", "outputs"}
                and set(groups["inputs"]) == set(INPUTS)
                and set(groups["outputs"]) == set(OUTPUTS),
                "complete fresh input/output allocations",
            )
            for name in INPUTS:
                intervals.append(
                    allocation(
                        groups["inputs"][name],
                        case["input_snapshot"][name]["shape"],
                        name,
                        device,
                    )
                )
                ids.append(groups["inputs"][name]["object_id"])
            for name in OUTPUTS:
                dtype = (
                    "int32"
                    if numerical.pinned._BANK_DTYPES[name] == "i32"
                    else "float32"
                )
                intervals.append(
                    allocation(
                        groups["outputs"][name],
                        case["runs"][side]["before"][name]["shape"],
                        dtype,
                        device,
                    )
                )
                ids.append(groups["outputs"][name]["object_id"])
            launch = child["launches"][index * 3 + ordinal]
            kernel_role = "original" if side == "original" else "candidate"
            binding = child["compiled"][kernel_role]["binding"]
            need(
                launch["index"] == index * 3 + ordinal
                and type(launch["index"]) is int
                and launch["case"] == case_name
                and launch["role"] == side
                and launch["kernel_role"] == kernel_role,
                "exact retained launch order and kernel",
            )
            need(
                launch["binding_before"] == launch["binding_after"] == binding
                and launch["context"] == device["context"]
                and launch["stream_handle"] == device["stream"]
                and launch["stream_object_id"] == device["stream_object_id"]
                and launch["block_dim"] == 256,
                "same explicit stream, binding and cache through launch",
            )
            need(
                (
                    type(launch["dim"]) is list
                    and all(type(item) is int for item in launch["dim"])
                    if side == "original"
                    else type(launch["dim"]) is int
                )
                and launch["dim"] == ([worlds, 20] if side == "original" else worlds),
                "plain literal launch dimensions",
            )
            need(
                launch["argument_array_pointers"]
                == [
                    groups["inputs"][name]["pointer"]
                    for name in (
                        "timestep",
                        "solref",
                        "solimp",
                        "frictionloss",
                        "invweight",
                        "qvel",
                    )
                ],
                "actual frozen argument-array order",
            )
            scalars = (
                [20, 0, False, cap, cap * 20] if side == "original" else [20, 0, cap]
            )
            need(
                launch["argument_scalars"] == scalars
                and [type(item) for item in launch["argument_scalars"]]
                == [type(item) for item in scalars],
                "plain literal frozen scalar signature",
            )
            names = (
                OUTPUTS
                if side == "original"
                else (
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
            )
            need(
                launch["output_array_pointers"]
                == [groups["outputs"][name]["pointer"] for name in names],
                "actual frozen output-array order",
            )
        intervals.sort()
        need(
            len(intervals) == len(ids) == len(set(ids)) == 63
            and all(
                left[1] <= right[0] for left, right in zip(intervals, intervals[1:])
            ),
            "63 simultaneously live independent buffers per case",
        )
    positive = [
        name
        for name, result in report["cases"].items()
        if result["component_exact_without_overflow"]
    ]
    negative = [name for name in CASES if name not in positive]
    expected = matrix_matches(report["cases"])
    return {
        "protocol": PROTOCOL + ":receiver",
        "source": source_sha,
        "raw_file_count": len(raw),
        "raw_total_bytes": sum(len(value) for value in raw.values()),
        "inventory_sha256": sha256(
            (
                json.dumps(
                    inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            ).encode()
        ).hexdigest(),
        "component_matrix_matches_predeclaration": expected,
        "positive_cases": positive,
        "negative_cases": negative,
        "decision": "synthetic-cuda-component-eight-exact-two-overflow-negative"
        if expected
        else "synthetic-cuda-component-numerical-negative",
        "numerical": report,
        "flags": FLAGS,
    }
