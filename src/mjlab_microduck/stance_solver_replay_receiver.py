"""Independent CPU receiver for one guarded dense-solver scratch replay.

This module deliberately imports only the Python standard library. It checks
the whole retained artifact inventory before decoding records or packet bytes;
it does not execute CUDA, load machine code, qualify training, or infer a
runtime cause from numerical agreement.
"""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import struct

from mjlab_microduck import stance_solver_binary_scope as binary_scope
from mjlab_microduck import stance_solver_disassembly_scope as disassembly
from mjlab_microduck import stance_solver_executable as executable_contract
from mjlab_microduck import stance_solver_packets as packet_contract
from mjlab_microduck import stance_solver_scratch as scratch_contract
from mjlab_microduck import stance_solver_target_binding as target_contract

PROTOCOL = "microduck-dense-solver-replay-receiver-oct8-v1"
PROBE_PROTOCOL = "microduck-dense-solver-replay-probe-oct8-v1"
PINNED_SOURCE_BRANCH = "feat/athletics-obstacle-curriculum"
PINNED_GPU_UUID = "GPU-7d72b360-33bc-2cee-3ff4-a954474011b5"
NATIVE_ROOT = "/home/yanbo/work/microduck_rl-com-entry-20261006"
FROZEN_ENV = "/home/yanbo/work/microduck_rl-stance-replication-20260930/.venv"
FROZEN_WARP_ROOT = FROZEN_ENV + "/lib/python3.12/site-packages/warp"
FROZEN_MW_ROOT = FROZEN_ENV + "/lib/python3.12/site-packages/mujoco_warp"
GPU_DRIVER = "595.95"
REPLAY_BOUNDS = {
    "service_seconds": 480, "child_seconds": 240, "probe_seconds": 25,
    "cleanup_seconds": 5, "closeout_seconds": 120, "margin_seconds": 60,
    "system_ram_bytes": 6 * 1024**3, "tasks": 64,
    "file_bytes": 16 * 1024**2, "total_used_mib": 12288,
    "free_reserve_mib": 10240, "aggregate_growth_mib": 2048,
    "temperature_c": 65, "utilization_percent": 85,
}
UNIT_CAPS = {
    "Type": "exec", "RuntimeMaxUSec": "8min", "TimeoutStopUSec": "10s",
    "MemoryMax": str(6 * 1024**3), "CPUQuotaPerSecUSec": "2s", "Nice": "10",
    "TasksMax": "64", "LimitFSIZE": str(16 * 1024**2),
    "KillMode": "control-group", "Restart": "no", "NRestarts": "0",
}
PACKAGE_PINS = {
    "torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
    "mujoco-warp": "3.8.1", "mjlab": "1.3.0", "better-actuator-models": "1.0.1",
}
RUNTIME_LIBRARY_PINS = {
    "warp.so": (283675616, "4afdc3ddd8d4c7e4f68837e9b1f4268e767ef8527769321b152dde8cc3b11acd"),
    "warp-clang.so": (67215456, "f8e0f74720067ee5606f189463d55142928087bc45b59d0e3e5c2b2385425a3a"),
}
WARP_SOURCE_TREE_SHA256 = "4aa3c865b7e523e1c0bef175f80ed51b00543569246d524914e78c333cdf9e6c"
MUJOCO_WARP_SOURCE_TREE_SHA256 = "188d58bfac6ab54e51adfc80a6c504d782aec43856cbb15e54b4109f0ce30a1d"
TOOL_SHA256 = "ad1d0f0699f46603416eb58e7b9fdf9a293e95f9312c9df7fd6fda56a6c30d41"
TOOL_BYTES = 569008
SOURCE_SCOPE = {
    "src/mjlab_microduck/stance_solver_replay_probe.py",
    "src/mjlab_microduck/stance_solver_replay_receiver.py",
    "tests/test_stance_solver_replay_probe.py",
    "tests/test_stance_solver_replay_receiver.py",
    "docs/experiments/2026-10-08-dense-solver-replay-diagnostic.md",
}
REPLAY_TESTS = (
    "tests/test_stance_solver_scratch.py", "tests/test_stance_solver_supervisor.py",
    "tests/test_stance_solver_packets.py", "tests/test_stance_solver_executable.py",
    "tests/test_stance_solver_dispatch_guard.py", "tests/test_stance_solver_disassembly_scope.py",
    "tests/test_stance_solver_target_binding.py", "tests/test_stance_cuda_artifact_binding.py",
    "tests/test_stance_solver_binary_scope.py", "tests/test_stance_solver_replay_probe.py",
    "tests/test_stance_solver_replay_receiver.py",
)
MAX_FILES = 256
MAX_FILE_BYTES = 16 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2
MAX_JSON_BYTES = 8 * 1024**2
ABS_TOLERANCE = 2.0e-5
REL_TOLERANCE = 2.0e-5
HISTORICAL_REPLAY_BYTES = 5_375_152
HISTORICAL_REPLAY_SHA256 = "b47122d67e3568bf4e413c3905b0ca329e727ee900b168f93dd4c5f27bce4441"

FLAGS = {
    "numerical_acceptance": False,
    "native_qualified": False,
    "full_window_qualified": False,
    "runtime_cause_proven": False,
    "training_authorized": False,
    "physical_acceptance": False,
}

SPECS = packet_contract.SPECS
PACKET_BYTES = packet_contract.PACKET_BYTES
_FIELD_LAYOUT = {}
_offset = 0
for _name, _shape, _dtype, _wire, _width in SPECS:
    _size = math.prod(_shape) * _width
    _FIELD_LAYOUT[_name] = (_shape, _wire, _offset, _size)
    _offset += _size
del _offset, _name, _shape, _dtype, _wire, _width, _size


def need(value, message):
    if not value:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "unique JSON object keys")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("finite JSON number " + value)


def _json(raw, label):
    need(type(raw) is bytes and 0 < len(raw) <= MAX_JSON_BYTES,
         "bounded JSON bytes " + label)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=_invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("valid JSON " + label) from error
    need(type(value) is dict, "JSON object " + label)
    return value


def _safe_name(name):
    need(type(name) is str and bool(name) and name == str(Path(name)),
         "canonical relative artifact name")
    path = Path(name)
    need(not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)
         and "\\" not in name,
         "contained plain artifact name")
    return path


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _canonical(value):
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as error:
        raise ValueError("plain finite canonical JSON") from error


def _hash(value, label):
    need(type(value) is str and len(value) == 64
         and all(c in "0123456789abcdef" for c in value),
         "lowercase SHA256 " + label)


def _check_runtime(runtime):
    need(type(runtime) is dict and set(runtime) == {
        "environment_alias", "packages", "libraries", "warp_sources",
        "mujoco_warp_sources", "tool",
    }, "complete frozen runtime declaration")
    alias = runtime["environment_alias"]
    need(type(alias) is dict and set(alias) == {
        "path", "target", "device", "inode", "bytes", "mtime_ns", "ctime_ns",
    } and alias["path"] == NATIVE_ROOT + "/.venv"
         and alias["target"] == FROZEN_ENV
         and type(alias["device"]) is int and alias["device"] > 0
         and alias["inode"] == 1_794_097
         and alias["bytes"] == len(FROZEN_ENV.encode())
         and type(alias["mtime_ns"]) is int and alias["mtime_ns"] > 0
         and type(alias["ctime_ns"]) is int and alias["ctime_ns"] > 0,
         "existing frozen venv symlink identity")
    need(runtime["packages"] == PACKAGE_PINS, "literal frozen package version set")

    libraries = runtime["libraries"]
    need(type(libraries) is dict and set(libraries) == set(RUNTIME_LIBRARY_PINS),
         "complete frozen Warp runtime libraries")
    for name, (size, digest) in RUNTIME_LIBRARY_PINS.items():
        row = libraries[name]
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
             and row["path"] == FROZEN_WARP_ROOT + "/bin/" + name
             and row["bytes"] == size and row["sha256"] == digest,
             "whole pinned runtime library " + name)

    warp = runtime["warp_sources"]
    need(type(warp) is dict and set(warp) == {"root", "leaves", "sha256"}
         and warp["root"] == FROZEN_WARP_ROOT
         and warp["sha256"] == WARP_SOURCE_TREE_SHA256,
         "pinned Warp Python and compiler source tree")
    warp_leaves = warp["leaves"]
    need(type(warp_leaves) is dict and len(warp_leaves) == 460,
         "bounded complete Warp source leaf set")
    warp_total = 0
    for name, row in warp_leaves.items():
        _safe_name(name)
        need(type(row) is dict and set(row) == {"bytes", "sha256"}
             and type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_FILE_BYTES,
             "Warp source leaf size")
        _hash(row["sha256"], "Warp source leaf")
        warp_total += row["bytes"]
    need(warp_total == 9_292_904
         and sha256(_canonical(warp_leaves)).hexdigest() == WARP_SOURCE_TREE_SHA256,
         "whole Warp source leaf inventory hash")

    mujoco_warp = runtime["mujoco_warp_sources"]
    need(type(mujoco_warp) is dict and set(mujoco_warp) == {"root", "leaves", "sha256"}
         and mujoco_warp["root"] == FROZEN_MW_ROOT
         and mujoco_warp["sha256"] == MUJOCO_WARP_SOURCE_TREE_SHA256
         and type(mujoco_warp["leaves"]) is dict and len(mujoco_warp["leaves"]) == 69,
         "pinned MuJoCo Warp source tree")
    for name, digest in mujoco_warp["leaves"].items():
        _safe_name(name)
        _hash(digest, "MuJoCo Warp source leaf")
    need(sha256(_canonical(mujoco_warp["leaves"])).hexdigest()
         == MUJOCO_WARP_SOURCE_TREE_SHA256,
         "whole MuJoCo Warp source leaf inventory hash")

    tool = runtime["tool"]
    need(type(tool) is dict and set(tool) == {"path", "bytes", "sha256", "version"}
         and tool["path"] == FROZEN_ENV + "/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump"
         and tool["bytes"] == TOOL_BYTES and tool["sha256"] == TOOL_SHA256
         and type(tool["version"]) is str and len(tool["version"]) < 4096
         and "V12.8.55" in tool["version"],
         "whole pinned cuobjdump tool identity")


def _check_declaration_context(declaration, source, owner_pid, deadline, device_record):
    need(declaration.get("bounds") == REPLAY_BOUNDS,
         "literal bounded GPU replay admission settings")
    need(type(deadline) in (int, float) and math.isfinite(deadline)
         and declaration.get("deadline_unix") == deadline and deadline > 0,
         "same finite owner and child deadline")
    need(declaration.get("exclusive_gpu_claimed") is False,
         "shared GPU lane remains nonexclusive")
    _check_runtime(declaration.get("runtime"))

    service = declaration.get("service")
    expected_name = "microduck-dense-solver-replay-" + source[:12] + ".service"
    need(type(service) is dict and set(service) == {
        "name", "Id", "Type", "RuntimeMaxUSec", "TimeoutStopUSec", "MemoryMax",
        "CPUQuotaPerSecUSec", "Nice", "TasksMax", "LimitFSIZE", "KillMode",
        "Restart", "NRestarts", "MainPID", "ActiveState", "ControlGroup", "InvocationID",
    } and service.get("name") == expected_name
         and all(service.get(key) == value for key, value in UNIT_CAPS.items())
         and service.get("Id") == expected_name
         and service.get("MainPID") == str(owner_pid)
         and service.get("ActiveState") == "active"
         and type(service.get("InvocationID")) is str
         and len(service["InvocationID"]) == 32
         and all(c in "0123456789abcdef" for c in service["InvocationID"])
         and type(service.get("ControlGroup")) is str
         and service["ControlGroup"] == "/user.slice/user-1000.slice/user@1000.service/app.slice/" + expected_name
         and ".." not in Path(service["ControlGroup"]).parts,
         "same active owner unit with exact bounded service caps")

    lease = declaration.get("lease")
    need(type(lease) is dict and set(lease) == {"device", "inode", "bytes"}
         and lease == {"device": 2096, "inode": 35886, "bytes": 0},
         "same pre-existing empty shared GPU lease")
    need(declaration["runtime"]["environment_alias"]["device"] == lease["device"],
         "runtime environment and shared GPU lease share the same filesystem device")
    expected_services = {
        "system:recomo-ai-mission-vllm.service": "inactive",
        "user:recomo-ai-mission-vllm.service": "inactive",
        "system:recomo-ai-mission-subject-model-worker.service": "inactive",
        "user:recomo-ai-mission-subject-model-worker.service": "inactive",
        "recomo-filmbrain-observatory.service": {
            "ActiveState": "active", "MainPID": "521", "NRestarts": "0",
        },
        "recomo-filmbrain-video-playground.service": {
            "ActiveState": "active", "MainPID": "298048", "NRestarts": "0",
        },
    }
    need(declaration.get("services") == expected_services,
         "protected service and FilmBrain baseline identities")

    baseline = declaration.get("baseline")
    need(type(baseline) is dict and set(baseline) == {
        "wall_time_unix", "wsl", "windows", "windows_sampled_at",
        "windows_active_engines", "counters_simultaneous",
    } and type(baseline["wall_time_unix"]) in (int, float)
         and math.isfinite(baseline["wall_time_unix"])
         and type(baseline["windows_sampled_at"]) is str
         and type(baseline["windows_active_engines"]) is list
         and baseline["counters_simultaneous"] is False,
         "complete shared host telemetry baseline")
    for name in ("wsl", "windows"):
        sample = baseline[name]
        need(type(sample) is dict and set(sample) == {
            "uuid", "driver", "total_mib", "used_mib", "free_mib",
            "utilization_percent", "temperature_c",
        } and sample["uuid"] == PINNED_GPU_UUID and sample["driver"] == GPU_DRIVER
             and sample["total_mib"] == 24467
             and type(sample["used_mib"]) is int and 0 <= sample["used_mib"] <= REPLAY_BOUNDS["total_used_mib"]
             and type(sample["free_mib"]) is int and sample["free_mib"] >= REPLAY_BOUNDS["free_reserve_mib"]
             and type(sample["utilization_percent"]) is int
             and sample["utilization_percent"] <= REPLAY_BOUNDS["utilization_percent"]
             and type(sample["temperature_c"]) is int
             and sample["temperature_c"] < REPLAY_BOUNDS["temperature_c"]
             and sample["used_mib"] + sample["free_mib"] <= sample["total_mib"],
             "shared capacity telemetry and safe admission bounds " + name)
        need(sample["uuid"] == device_record.get("gpu_uuid"),
             "admitted shared GPU telemetry matches actual replay device UUID")

    cpu_tests = declaration.get("cpu_tests")
    need(type(cpu_tests) is dict and set(cpu_tests) == {"path", "bytes", "sha256", "evidence"}
         and type(cpu_tests["path"]) is str and Path(cpu_tests["path"]).is_absolute()
         and type(cpu_tests["bytes"]) is int and cpu_tests["bytes"] > 0,
         "paired CPU prerequisite record")
    _hash(cpu_tests["sha256"], "CPU prerequisite record")
    evidence = cpu_tests["evidence"]
    need(type(evidence) is dict and set(evidence) == {
        "source", "test_files", "mac", "native", "flags",
    } and evidence["source"] == source and evidence["test_files"] == list(REPLAY_TESTS)
         and evidence["flags"] == target_contract.FLAGS,
         "same-source CPU test prerequisite declaration")
    for platform in ("mac", "native"):
        anchor = evidence[platform]
        need(type(anchor) is dict and set(anchor) == {"path", "bytes", "sha256", "tests"}
             and type(anchor["path"]) is str and Path(anchor["path"]).is_absolute()
             and str(Path(anchor["path"])).startswith(NATIVE_ROOT + "/artifacts/tools/")
             and type(anchor["bytes"]) is int and anchor["bytes"] > 0
             and type(anchor["tests"]) is int and anchor["tests"] > 335,
             "bounded retained " + platform + " CPU test XML record")
        _hash(anchor["sha256"], platform + " test XML")
    need(evidence["mac"]["tests"] == evidence["native"]["tests"],
         "matched paired CPU test totals")


def _check_supervision(artifact_bytes, receipt):
    names = [name for name in artifact_bytes if Path(name).name == "supervision.json"]
    need(len(names) == 1, "one authenticated joined-child supervision record")
    record = _json(artifact_bytes[names[0]], "supervision record")
    need(type(record) is dict and set(record) == {
        "result", "kernel_cgroup_only_owner", "unit_retirement_independently_required", "flags",
    } and record["kernel_cgroup_only_owner"] is True
         and record["unit_retirement_independently_required"] is True
         and record["flags"] == target_contract.FLAGS,
         "owner supervision keeps cgroup retirement as an independent delivery gate")
    result = record["result"]
    need(type(result) is dict and set(result) == {
        "decision", "returncode", "root", "elapsed", "observed_session_members",
        "process_tree_retirement_proven", "native_qualified", "training_authorized",
    } and result["decision"] == "reviewed-owned-root-exited-no-native-qualification"
         and result["returncode"] == 0
         and type(result["root"]) is list and len(result["root"]) == 2
         and result["root"][0] == receipt["child_pid"]
         and type(result["root"][1]) is int and result["root"][1] > 0
         and type(result["elapsed"]) in (int, float)
         and math.isfinite(result["elapsed"]) and 0 <= result["elapsed"] <= REPLAY_BOUNDS["child_seconds"]
         and result["observed_session_members"] == []
         and result["process_tree_retirement_proven"] is False
         and result["native_qualified"] is False
         and result["training_authorized"] is False,
         "successful supervised child matches receipt PID and has no native qualification")
    return {"owner_observed_root_pid": result["root"][0],
            "owner_observed_root_birth": result["root"][1],
            "unit_retirement_independently_required": True}


def authenticate(root, inventory):
    """Authenticate the complete exact regular-file inventory before parsing."""
    need(type(root) is type(Path()) and root.is_absolute()
         and root.resolve(strict=True) == root and root.is_dir(),
         "absolute canonical artifact root")
    need(type(inventory) is dict and 0 < len(inventory) <= MAX_FILES,
         "bounded complete artifact inventory")
    expected = set()
    total = 0
    for name, anchor in inventory.items():
        path = _safe_name(name)
        need(type(anchor) is dict and set(anchor) == {"bytes", "sha256"},
             "whole artifact inventory anchor")
        size, digest = anchor["bytes"], anchor["sha256"]
        need(type(size) is int and 0 <= size <= MAX_FILE_BYTES,
             "bounded artifact size")
        need(type(digest) is str and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest),
             "lowercase artifact SHA256")
        total += size
        expected.add(str(path))
    need(total <= MAX_TOTAL_BYTES, "bounded complete artifact bytes")

    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "no symlink artifact nodes")
        need(path.resolve(strict=True) == path, "canonical artifact node")
        mode = path.stat(follow_symlinks=False).st_mode
        need(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "regular artifact tree nodes")
        if stat.S_ISREG(mode):
            actual.add(str(path.relative_to(root)))
    need(actual == expected, "exact complete artifact leaf inventory")

    retained = {}
    for name in sorted(expected):
        path = root / name
        anchor = inventory[name]
        need(path.resolve(strict=True) == path, "canonical artifact file path")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_size == anchor["bytes"],
                 "bounded regular artifact file")
            chunks, remaining = [], anchor["bytes"] + 1
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(fd)
            current = path.stat(follow_symlinks=False)
            need(_identity(before) == _identity(after) == _identity(current)
                 and stat.S_ISREG(current.st_mode)
                 and path.resolve(strict=True) == path,
                 "stable artifact identity during full read")
        finally:
            os.close(fd)
        need(len(raw) == anchor["bytes"] and sha256(raw).hexdigest() == anchor["sha256"],
             "whole artifact bytes match external inventory")
        retained[name] = raw
    return retained


def _field_bytes(raw, phase_record, label):
    need(type(phase_record) is dict
         and set(phase_record) == {"bytes", "sha256", "fields"}
         and type(phase_record["bytes"]) is int
         and phase_record["bytes"] == len(raw)
         and phase_record["sha256"] == sha256(raw).hexdigest(),
         "whole " + label + " packet record binding")
    rows = phase_record["fields"]
    need(type(rows) is list and len(rows) == len(SPECS),
         "complete fixed packet fields " + label)
    decoded, offset = {}, 0
    for row, (name, shape, _dtype, wire, width) in zip(rows, SPECS):
        size = math.prod(shape) * width
        need(type(row) is dict and set(row) == {
            "name", "shape", "dtype", "offset", "bytes", "sha256",
        } and row["name"] == name and row["shape"] == list(shape)
             and row["dtype"] == wire and type(row["offset"]) is int
             and row["offset"] == offset and type(row["bytes"]) is int
             and row["bytes"] == size,
             "literal ordered packet field schema " + name)
        chunk = raw[offset:offset + size]
        need(len(chunk) == size and row["sha256"] == sha256(chunk).hexdigest(),
             "whole packet field bytes " + name)
        decoded[name] = chunk
        offset += size
    need(offset == len(raw) == PACKET_BYTES, "no trailing packet bytes")
    return decoded


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_at(raw, index):
    return struct.unpack_from("<f", raw, index * 4)[0]


def compare_packets(before_raw, after_raw, packet_record):
    """Check full input-bank immutability and CPU float32 solver replay."""
    need(type(packet_record) is dict and set(packet_record) == {
        "protocol", "decision", "packets", "complete_pair", "input_bytes_unchanged",
        "packet_bytes", "pair_bytes", "copy_stream_handle", "timing_changed_by_readback",
        "numerical_acceptance", "driver_loaded_code_observed", "flags",
    }, "exact guarded packet pair receipt schema")
    need(packet_record["protocol"] == packet_contract.PROTOCOL
         and packet_record["decision"] == "guarded-packets-only-not-qualification"
         and packet_record["complete_pair"] is True
         and packet_record["packet_bytes"] == PACKET_BYTES
         and packet_record["pair_bytes"] == 2 * PACKET_BYTES
         and type(packet_record["copy_stream_handle"]) is int
         and packet_record["copy_stream_handle"] > 0
         and packet_record["timing_changed_by_readback"] is True
         and packet_record["numerical_acceptance"] is False
         and packet_record["driver_loaded_code_observed"] is False
         and packet_record["flags"] == target_contract.FLAGS,
         "literal complete guarded packet declaration")
    packets = packet_record["packets"]
    need(type(packets) is dict and set(packets) == {"before", "after"},
         "exact before and after packet records")
    before = _field_bytes(before_raw, packets["before"], "before")
    after = _field_bytes(after_raw, packets["after"], "after")
    inputs = tuple(name for name, *_ in SPECS if name != "qfrc_constraint")
    unchanged = all(before[name] == after[name] for name in inputs)
    need(packet_record["input_bytes_unchanged"] is unchanged and unchanged,
         "entire nefc/J/force/done input banks unchanged")

    counts = tuple(struct.unpack_from("<i", before["nefc"], i * 4)[0]
                   for i in range(64))
    done_raw = before["done"]
    need(all(0 <= count <= 512 for count in counts), "bounded per-world active row counts")
    need(len(done_raw) == 64 and all(value in (0, 1) for value in done_raw),
         "canonical per-world done flags")

    world_rows = []
    for world, count in enumerate(counts):
        before_output, after_output = before["qfrc_constraint"], after["qfrc_constraint"]
        max_abs = 0.0
        max_rel = 0.0
        mismatches = 0
        if done_raw[world]:
            # The kernel returns before any store; compare all 20 output bits.
            start = world * 20 * 4
            same = before_output[start:start + 80] == after_output[start:start + 80]
            need(same, "done world retains complete qfrc_constraint row")
            world_rows.append({"world": world, "done": True, "nefc": count,
                               "compared_dofs": 20, "mismatches": 0,
                               "max_abs_error": 0.0, "max_rel_error": 0.0,
                               "result": "done-row-preserved"})
            continue

        for dof in range(20):
            expected = 0.0
            for row in range(count):
                index = (world * 512 + row) * 20 + dof
                jacobian = _f32_at(before["J"], index)
                force = _f32_at(before["force"], world * 512 + row)
                need(math.isfinite(jacobian) and math.isfinite(force),
                     "finite active Jacobian and force values")
                product = _f32(jacobian * force)
                expected = _f32(expected + product)
            actual = _f32_at(after_output, world * 20 + dof)
            need(math.isfinite(expected) and math.isfinite(actual),
                 "finite CPU reference and returned qfrc values")
            error = abs(actual - expected)
            scale = max(abs(actual), abs(expected))
            relative = error / scale if scale else 0.0
            max_abs, max_rel = max(max_abs, error), max(max_rel, relative)
            if error > ABS_TOLERANCE + REL_TOLERANCE * scale:
                mismatches += 1
        world_rows.append({"world": world, "done": False, "nefc": count,
                           "compared_dofs": 20, "mismatches": mismatches,
                           "max_abs_error": max_abs, "max_rel_error": max_rel,
                           "result": "match" if mismatches == 0 else "numerical-mismatch"})
    return {
        "input_banks_unchanged": True,
        "reference": "ascending-efc-row-float32-product-and-accumulation",
        "absolute_tolerance": ABS_TOLERANCE,
        "relative_tolerance": REL_TOLERANCE,
        "worlds": world_rows,
        "numerical_match": all(row["mismatches"] == 0 for row in world_rows),
        "numerical_acceptance": False,
    }


def _binding_check(executable, guard, files, artifact_bytes, native_root, device_record):
    """Bind offline CUBIN/SASS and observed launch records to one target identity."""
    need(type(executable) is dict and executable.get("protocol") == executable_contract.PROTOCOL
         and executable.get("decision") == "dense-solver-executable-prepared-not-dispatch-or-qualification"
         and executable.get("target") == binary_scope.TARGET,
         "prepared exact dense executable record")
    need(type(executable.get("source")) is dict
         and executable["source"].get("path")
         == FROZEN_ENV + "/lib/python3.12/site-packages/mujoco_warp/_src/solver.py"
         and executable["source"].get("sha256") == target_contract.SOLVER_SHA256
         and type(executable["source"].get("identity")) is list
         and len(executable["source"]["identity"]) == 5
         and all(type(value) is int and value >= 0
                 for value in executable["source"]["identity"]),
         "whole pinned installed solver source identity")
    need(type(guard) is dict and guard.get("protocol") == "microduck-dense-solver-dispatch-guard-oct8-v1"
         and guard.get("decision") == "dense-solver-dispatch-guard-complete-not-qualification"
         and guard.get("observed_target_calls") == 1
         and guard.get("caller") == "_update_constraint"
         and guard.get("dimensions") == [64, 20]
         and guard.get("dispatch_line") == 2197
         and guard.get("input_order") == ["d.nefc", "d.efc.J", "d.efc.force", "d.njmax", "ctx.done"]
         and guard.get("output_order") == ["d.qfrc_constraint"]
         and guard.get("flags") == target_contract.FLAGS
         and guard.get("numerical_acceptance") is False
         and guard.get("driver_loaded_code_observed") is False,
         "one complete exact dense target dispatch record")
    need(type(files) is dict and set(files) == {
        "cubin", "metadata", "generated_source", "sass", "packet_before", "packet_after",
    }, "closed replay artifact roles")
    need(all(type(name) is str and name in artifact_bytes for name in files.values())
         and len(set(files.values())) == len(files),
         "all receipt roles name authenticated inventory leaves")
    cubin = artifact_bytes[files["cubin"]]
    metadata = artifact_bytes[files["metadata"]]
    generated_source = artifact_bytes[files["generated_source"]]
    sass = artifact_bytes[files["sass"]]
    selected = disassembly.verify_target_disassembly(cubin, sass)
    compile_record = executable.get("compile")
    need(type(compile_record) is dict and compile_record.get("performed") is True
         and compile_record.get("output_arch") == 120 and compile_record.get("use_ptx") is False,
         "fresh sm120 CUBIN compile record")
    generated = compile_record.get("generated")
    need(type(generated) is dict and set(generated) == {"source", "metadata", "binary"},
         "complete compile output inventory")
    binary_info, meta_info = generated["binary"], generated["metadata"]
    source_info = generated["source"]
    role_records = ((binary_info, cubin, "CUBIN", "cubin"),
                    (meta_info, metadata, "metadata", "metadata"),
                    (source_info, generated_source, "generated source", "generated_source"))
    for info, raw, role, file_role in role_records:
        need(type(info) is dict and set(info) == {"path", "bytes", "sha256", "identity"}
             and type(info["path"]) is str and Path(info["path"]).is_absolute()
             and type(info["identity"]) is list and len(info["identity"]) == 5
             and all(type(v) is int for v in info["identity"])
             and info["bytes"] == len(raw) and info["sha256"] == sha256(raw).hexdigest(),
             "authenticated generated " + role + " record")
        path = Path(info["path"])
        need(path.parent == Path(native_root) / "compiled-dense-solver"
             and path.name and path.name not in (".", ".."),
             "generated " + role + " path is inside declared native compiled directory")
        need(files[file_role] == str(path.relative_to(Path(native_root))),
             "inventory role path matches native-relative generated " + role + " path")
    need(executable.get("offline_disassembly") == selected
         and compile_record.get("target_compile_input") == disassembly.select_target_cubin(cubin),
         "whole CUBIN and SASS bind exact target compile input")
    need(selected["target"]["symbol"] == executable.get("loaded_binding", {}).get("symbol"),
         "offline target symbol matches explicitly loaded executable")
    target_declarations = re.findall(
        rb'extern "C" __global__ void ([A-Za-z_][A-Za-z_0-9]*_cuda_kernel_forward)\s*\(',
        generated_source,
    )
    need(target_declarations.count(selected["target"]["symbol"].encode("ascii")) == 1,
         "generated CUDA source declares the exact CUBIN target symbol once")
    binding = executable.get("loaded_binding")
    guard_binding = guard.get("binding")
    need(type(binding) is dict and type(guard_binding) is dict
         and binding == guard_binding and binding.get("artifact_format") == "cubin"
         and binding.get("binary_sha256") == sha256(cubin).hexdigest()
         and binding.get("metadata_sha256") == sha256(metadata).hexdigest()
         and binding.get("binary_path") == binary_info["path"]
         and binding.get("binary_bytes") == binary_info["bytes"]
         and binding.get("metadata_path") == meta_info["path"]
         and binding.get("metadata_bytes") == meta_info["bytes"]
         and binding.get("driver_jit_machine_code_observed") is False
         and binding.get("loaded_binary_bytes_observed") is False
         and binding.get("native_execution_qualified") is False,
         "same loaded CUBIN binding in executable and dispatch records")
    metadata_map = executable_contract.artifacts._metadata(metadata)
    need(metadata_map.get(selected["target"]["symbol"] + "_smem_bytes")
         == binding.get("forward_smem_bytes"),
         "authenticated CUBIN metadata target shared-memory binding")
    need(binding.get("training_authorized") is False
         and binding.get("physical_acceptance") is False
         and executable.get("flags") == target_contract.FLAGS
         and guard.get("flags") == target_contract.FLAGS,
         "executable, binding, and guard preserve all qualification gates")
    need(compile_record.get("module_hash") == binding.get("module_hash")
         and type(compile_record.get("module_hash")) is str
         and len(compile_record["module_hash"]) == 64,
         "compile and explicit-load records share the exact module hash")
    loaded = executable.get("explicit_load")
    ids = binding.get("observed_object_ids")
    device = executable.get("device")
    need(type(loaded) is dict and type(ids) is dict and type(device) is dict
         and loaded.get("returned_executable_is_cache_entry") is True
         and loaded.get("output_arch") == 120 and loaded.get("block_dim") == 256
         and loaded.get("device_object_id") == device.get("object_id")
         and loaded.get("executable_object_id") == ids.get("executable")
         and loaded.get("module_object_id") == ids.get("module")
         and guard.get("stream_handle") == guard.get("packets", {}).get("copy_stream_handle"),
         "same explicitly loaded device/module/executable and captured stream")
    need(type(device_record) is dict and set(device_record) == {
        "alias", "arch", "context", "object_id", "gpu_uuid",
    } and device_record == {
        "alias": device.get("alias"), "arch": device.get("arch"),
        "context": device.get("context"), "object_id": device.get("object_id"),
        "gpu_uuid": device_record.get("gpu_uuid"),
    } and type(device_record["gpu_uuid"]) is str
         and device_record["gpu_uuid"] == PINNED_GPU_UUID
         and device_record["alias"] == "cuda:0",
         "receipt device identity matches prepared executable")
    need(device_record["arch"] == 120 and device_record["context"] == binding.get("context")
         and device_record["object_id"] == ids.get("device"),
         "same sm120 CUDA context and device object across records")
    need(loaded.get("binary_path") == binary_info["path"]
         and loaded.get("metadata_path") == meta_info["path"],
         "explicit-load paths match retained generated CUBIN and metadata")
    layout_rows = guard.get("layouts")
    expected_shapes = ([64], [64, 512, 20], [64, 512], [64], [64, 20])
    need(type(layout_rows) is list and len(layout_rows) == len(expected_shapes),
         "complete five-array dispatch layout evidence")
    for layout, shape in zip(layout_rows, expected_shapes):
        need(type(layout) is dict and set(layout) == {
            "object_id", "pointer", "shape", "dtype_object_id",
        } and type(layout["object_id"]) is int and layout["object_id"] > 0
             and type(layout["pointer"]) is int and layout["pointer"] > 0
             and layout["shape"] == shape
             and type(layout["dtype_object_id"]) is int and layout["dtype_object_id"] > 0,
             "literal dense launch array identity and layout")
    need(len({row["object_id"] for row in layout_rows}) == 5
         and len({row["pointer"] for row in layout_rows}) == 5,
         "five distinct target array allocations")
    dtype_ids = [row["dtype_object_id"] for row in layout_rows]
    need(dtype_ids[1] == dtype_ids[2] == dtype_ids[4]
         and len({dtype_ids[0], dtype_ids[1], dtype_ids[3]}) == 3,
         "literal int32/float32/bool dtype identity relationships")
    need(type(guard.get("stream_object_id")) is int and guard["stream_object_id"] > 0,
         "positive retained dispatch stream object identity")
    byte_widths = (4, 4, 4, 1, 4)
    spans = sorted((row["pointer"], row["pointer"] + math.prod(shape) * width)
                   for row, shape, width in zip(layout_rows, expected_shapes, byte_widths))
    need(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])),
         "five dispatch array address ranges do not overlap")
    need(executable.get("actual_dispatch_observed") is False
         and executable.get("loaded_binary_bytes_observed") is False
         and executable.get("driver_jit_machine_code_observed") is False
         and executable.get("numerical_cause_proven") is False,
         "executable record does not overclaim runtime observation")
    return {
        "compile_input_cubin_bound": True,
        "offline_sass_matches_target_cubin": True,
        "explicit_load_record_bound_to_target": True,
        "dispatch_record_same_loaded_binding": True,
        "driver_loaded_code_observed": False,
        "native_executable_qualification": False,
    }


def receive(artifact_root, inventory, receipt_name="receipt.json"):
    """Authenticate one bounded artifact directory and independently replay it.

    The envelope binds source, declaration, owner/child identities, device,
    native output root, executable, guard, scratch and file roles. Files name
    authenticated leaves for the CUBIN, metadata, generated source, SASS and
    two raw complete packet banks. The external inventory includes every
    output leaf, including the receipt itself.
    """
    # Authentication intentionally precedes receipt JSON or packet decoding.
    artifact_bytes = authenticate(artifact_root, inventory)
    receipt_path = str(_safe_name(receipt_name))
    need(receipt_path in artifact_bytes, "receipt is an authenticated inventory leaf")
    receipt_raw = artifact_bytes[receipt_path]
    receipt = _json(receipt_raw, "receipt")
    need(_canonical(receipt) == receipt_raw, "receipt uses canonical JSON bytes")
    need(set(receipt) == {
        "protocol", "source", "declaration_sha256", "owner_pid", "child_pid",
        "device", "native_root", "executable", "guard", "scratch", "files", "flags",
    } and receipt["protocol"] == PROBE_PROTOCOL,
         "exact dense solver replay receiver envelope")
    source = receipt["source"]
    need(type(source) is str and len(source) == 40
         and all(c in "0123456789abcdef" for c in source),
         "literal lowercase source commit identity")
    need(type(receipt["owner_pid"]) is int and receipt["owner_pid"] > 0
         and type(receipt["child_pid"]) is int and receipt["child_pid"] > 0
         and receipt["owner_pid"] != receipt["child_pid"],
         "distinct positive owner and child process identities")
    device_record = receipt["device"]
    need(type(device_record) is dict and set(device_record) == {
        "alias", "arch", "context", "object_id", "gpu_uuid",
    } and device_record["alias"] == "cuda:0" and device_record["arch"] == 120
         and type(device_record["context"]) is int and device_record["context"] > 0
         and type(device_record["object_id"]) is int and device_record["object_id"] > 0
         and device_record["gpu_uuid"] == PINNED_GPU_UUID,
         "literal receipt device alias, architecture, context, object, and UUID")
    need(receipt["flags"] == target_contract.FLAGS,
         "probe receipt keeps every qualification false")
    native_root = receipt["native_root"]
    need(type(native_root) is str and Path(native_root).is_absolute()
         and str(Path(native_root)) == native_root and native_root.startswith(
        NATIVE_ROOT + "/artifacts/evaluations/dense-solver-replay-"
    ) and native_root.endswith(source[:12]) and ".." not in Path(native_root).parts,
         "native compiled root is source-anchored and canonical")
    declarations = [name for name in artifact_bytes if Path(name).name == "declaration.json"]
    need(len(declarations) == 1, "one authenticated owner declaration")
    declaration_raw = artifact_bytes[declarations[0]]
    need(sha256(declaration_raw).hexdigest() == receipt["declaration_sha256"],
         "whole declaration hash matches owner receipt")
    declaration = _json(declaration_raw, "owner declaration")
    need(_canonical(declaration) == declaration_raw,
         "owner declaration uses the exact canonical JSON byte representation")
    need(set(declaration) == {
        "protocol", "source", "source_binding", "native_root", "owner_pid", "runtime",
        "service", "cpu_tests", "lease", "services", "baseline", "bounds",
        "deadline_unix", "replay_input", "exclusive_gpu_claimed", "flags",
    } and declaration.get("protocol") == PROBE_PROTOCOL
         and declaration.get("source") == source and declaration.get("native_root") == native_root
         and declaration.get("owner_pid") == receipt["owner_pid"]
         and declaration.get("flags") == target_contract.FLAGS,
         "receipt source and native root match authenticated declaration")
    source_binding = declaration.get("source_binding")
    need(type(source_binding) is dict and set(source_binding) == {
        "source", "tree", "branch", "leaves",
    } and source_binding.get("source") == source
         and type(source_binding.get("tree")) is str
         and len(source_binding["tree"]) == 40
         and all(c in "0123456789abcdef" for c in source_binding["tree"])
         and source_binding.get("branch") == PINNED_SOURCE_BRANCH
         and type(source_binding.get("leaves")) is list
         and bool(source_binding["leaves"]),
         "pinned source commit, tree, branch, and source leaf declaration")
    seen_source_leaves = set()
    for leaf in source_binding["leaves"]:
        need(type(leaf) is dict and set(leaf) == {"path", "bytes", "git_blob", "sha256"}
             and type(leaf["path"]) is str and leaf["path"] == str(_safe_name(leaf["path"]))
             and type(leaf["bytes"]) is int and leaf["bytes"] >= 0
             and type(leaf["git_blob"]) is str and len(leaf["git_blob"]) == 40
             and all(c in "0123456789abcdef" for c in leaf["git_blob"])
             and type(leaf["sha256"]) is str and len(leaf["sha256"]) == 64
             and all(c in "0123456789abcdef" for c in leaf["sha256"])
             and leaf["path"] not in seen_source_leaves,
             "authenticated declaration source leaf anchor")
        seen_source_leaves.add(leaf["path"])
    need(SOURCE_SCOPE <= seen_source_leaves,
         "source tree contains the exact five-path reviewed replay scope")
    _check_declaration_context(declaration, source, receipt["owner_pid"],
                               declaration.get("deadline_unix"), device_record)
    replay_input = declaration.get("replay_input")
    need(type(replay_input) is dict and set(replay_input) == {
        "inventory_sha256", "metadata_sha256", "raw_sha256", "raw_bytes",
    }
         and replay_input.get("inventory_sha256") == "8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625"
         and replay_input.get("metadata_sha256") == "d19e16bcbf8152629b18c514ff627ff8417de1c4b4ed760d9ab0add30f867d91"
         and replay_input.get("raw_sha256") == HISTORICAL_REPLAY_SHA256
         and replay_input.get("raw_bytes") == HISTORICAL_REPLAY_BYTES,
         "declaration pins the documented historical forward-four replay bank")
    scratch = receipt["scratch"]
    need(type(scratch) is dict and scratch.get("protocol") == scratch_contract.PROTOCOL
         and scratch.get("decision") == "dense-solver-scratch-restored-no-dispatch-or-qualification"
         and scratch.get("source_sha256") == HISTORICAL_REPLAY_SHA256
         and scratch.get("source_bytes") == HISTORICAL_REPLAY_BYTES
         and scratch.get("retained_source_bytes") is True
         and scratch.get("solver_called") is False
         and scratch.get("forward_called") is False
         and scratch.get("restored_fields") == list(scratch_contract.RESTORE_ORDER),
         "scratch receipt binds known historical forward-four replay bank")
    need(scratch.get("flags") == scratch_contract.FLAGS,
         "scratch receipt keeps every qualification false")
    supervision = _check_supervision(artifact_bytes, receipt)
    files = receipt["files"]
    binding = _binding_check(receipt["executable"], receipt["guard"], files,
                             artifact_bytes, native_root, receipt["device"])
    pair = receipt["guard"].get("packets")
    comparison = compare_packets(artifact_bytes[files["packet_before"]],
                                 artifact_bytes[files["packet_after"]], pair)
    return {
        "protocol": PROTOCOL,
        "decision": "authenticated-one-launch-numerical-replay-only",
        "artifact_count": len(artifact_bytes),
        "artifact_bytes": sum(map(len, artifact_bytes.values())),
        "numerical": comparison,
        "executable_binding": binding,
        "supervision": supervision,
        "qualification": dict(FLAGS),
    }
