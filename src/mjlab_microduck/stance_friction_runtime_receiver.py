"""Independent whole-byte receiver for the newly declared one-tick diagnostic.

No native reexecution or admission occurs. Historical receivers are borrowed
only for unchanged pure shape/numerical/component checks, never their owner
protocols, source fences, cutoffs or qualification.
"""

from hashlib import sha256
from math import isfinite, prod
import os
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET

from mjlab_microduck import stance_friction_prefix_cuda_receiver as component
from mjlab_microduck import stance_friction_prefix_cuda_probe as pins
from mjlab_microduck import stance_friction_runtime_numerical as numerical
from mjlab_microduck import stance_bam_load_receiver as recipe_checks
from mjlab_microduck import stance_serial_step_receiver as serial_checks
from mjlab_microduck import stance_com_coupled_receiver as rng_checks

PROTOCOL = "microduck-friction-runtime-receiver-oct8-v1"
DATA_PROTOCOL = "microduck-friction-runtime-tick-oct8-v1"
CONTROL_PROTOCOL = "microduck-friction-runtime-control-oct8-v1"
ARMS = numerical.ARMS
FLAGS = dict(pins.FLAGS)
MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES = 1024, 16 * 1024**2, 768 * 1024**2
MAX_JSON_BYTES = 8 * 1024**2
canonical, need, decode = component.canonical, component.need, component.decode
plain, sha = component.plain, component.sha


def safe_name(name):
    need(
        type(name) is str
        and bool(name)
        and name == str(Path(name))
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts,
        "plain relative artifact name",
    )
    return name


def authenticate(directory, inventory):
    """Authenticate the exact whole leaf set before interpreting any packet."""
    directory = Path(directory).absolute()
    need(
        directory.resolve(strict=True) == directory and directory.is_dir(),
        "plain artifact root",
    )
    need(
        type(inventory) is dict and 0 < len(inventory) <= MAX_FILES,
        "bounded complete inventory",
    )
    leaves = set()
    for path in directory.rglob("*"):
        need(
            not path.is_symlink() and (path.is_file() or path.is_dir()),
            "no special artifact nodes",
        )
        if path.is_file():
            leaves.add(str(path.relative_to(directory)))
    need(leaves == set(inventory), "exact externally anchored leaf set")
    total = 0
    for name, info in inventory.items():
        safe_name(name)
        need(
            type(info) is dict and set(info) == {"bytes", "sha256"},
            "whole-file anchor schema",
        )
        total += plain(info["bytes"], 0, MAX_FILE_BYTES)
        sha(info["sha256"])
    need(total <= MAX_TOTAL_BYTES, "bounded total raw bytes")
    result = {}
    for name, info in sorted(inventory.items()):
        path = directory / name
        need(path.resolve(strict=True) == path, "plain stable retained path")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            need(
                stat.S_ISREG(before.st_mode) and before.st_size == info["bytes"],
                "regular exact file length",
            )
            chunks, remaining = [], info["bytes"] + 1
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after, current = os.fstat(fd), path.stat(follow_symlinks=False)
            need(
                component.stat_identity(before)
                == component.stat_identity(after)
                == component.stat_identity(current)
                and path.resolve(strict=True) == path,
                "stable whole-byte artifact identity",
            )
            need(
                len(raw) == info["bytes"] and sha256(raw).hexdigest() == info["sha256"],
                "whole bytes authenticated before interpretation",
            )
            result[name] = raw
        finally:
            os.close(fd)
    return result


def json_packet(raw):
    need(type(raw) is bytes and len(raw) <= MAX_JSON_BYTES, "bounded JSON packet")
    return decode(raw)


def unpack_packet(record, raw, *, order):
    need(
        type(record) is dict and set(record) == {"path", "bytes", "sha256", "fields"},
        "binary section schema",
    )
    name = safe_name(record["path"])
    need(
        name in raw
        and record["bytes"] == len(raw[name])
        and record["sha256"] == sha256(raw[name]).hexdigest(),
        "section whole-byte binding",
    )
    fields = record["fields"]
    need(
        type(fields) is dict and set(fields) == set(order),
        "complete binary section fields",
    )
    result, offset = {}, 0
    for key in order:
        value = fields[key]
        need(
            type(value) is dict and set(value) == {"offset", "bytes", "shape", "dtype"},
            "raw field metadata",
        )
        shape = value["shape"]
        need(type(shape) is list and len(shape) <= 4, "bounded raw field rank")
        for dimension in shape:
            plain(dimension, 0, 10240)
        size = prod(shape) * 4
        need(
            type(value["offset"]) is int
            and value["offset"] == offset
            and type(value["bytes"]) is int
            and value["bytes"] == size
            and type(value["dtype"]) is str
            and value["dtype"] in ("<f4", "<i4"),
            "exact nonoverlapping four-byte field slice",
        )
        result[key] = {
            "shape": shape,
            "dtype": value["dtype"],
            "raw": raw[name][offset : offset + size],
        }
        need(len(result[key]["raw"]) == size, "complete raw field slice")
        offset += size
    need(offset == len(raw[name]), "no trailing unclaimed section bytes")
    return result


def layout(value, carrier, device):
    need(
        type(value) is dict
        and set(value)
        == {
            "object_id",
            "pointer",
            "span",
            "device",
            "context",
            "warp_dtype",
            "shape",
            "strides",
            "host_shape",
            "host_dtype",
            "bytes",
        },
        "complete allocation layout",
    )
    for key in ("object_id", "context"):
        plain(value[key], 1)
    span = plain(value["span"], 0, MAX_FILE_BYTES)
    if value["pointer"] is None:
        need(span == 0, "null pointer only for empty allocation")
        pointer = 0  # Address-range calculation only; retained metadata stays null.
    else:
        pointer = plain(value["pointer"])
    need(
        value["device"] == "cuda:0" and value["context"] == device["context"],
        "same actual CUDA context",
    )
    need(
        value["host_shape"] == carrier["shape"]
        and value["bytes"] == len(carrier["raw"])
        and value["host_dtype"]
        == ("float32" if carrier["dtype"] == "<f4" else "int32"),
        "actual host carrier layout",
    )
    shape, strides = value["shape"], value["strides"]
    need(
        type(shape) is list
        and type(strides) is list
        and len(shape) == len(strides)
        and shape == carrier["shape"][: len(shape)],
        "logical/vector shape relationship",
    )
    for x in shape + strides:
        plain(x, 0, MAX_FILE_BYTES)
    item = 4 * prod(carrier["shape"][len(shape) :])
    expected_span = (
        0
        if not prod(shape)
        else item + sum((s - 1) * t for s, t in zip(shape, strides))
    )
    expected_strides, width = [], item
    for dimension in reversed(shape):
        expected_strides.insert(0, width)
        width *= dimension
    if prod(shape):
        need(
            strides == expected_strides
            or (strides[0] == 0 and strides[1:] == expected_strides[1:]),
            "contiguous or leading broadcast layout",
        )
    need(
        span == expected_span and (pointer > 0 or span == 0),
        "actual allocation span and pointer",
    )
    expected_dtype = {
        "<i4": "<class 'warp._src.types.int32'>",
        "<f4": "<class 'warp._src.types.float32'>",
    }[carrier["dtype"]]
    if len(carrier["shape"]) == len(shape) + 1:
        need(
            carrier["dtype"] == "<f4" and carrier["shape"][-1] in (2, 5),
            "declared vector input only",
        )
        expected_dtype = (
            "<class 'warp._src.types.vec2f'>"
            if carrier["shape"][-1] == 2
            else "<class 'mujoco_warp._src.types.vec5f'>"
        )
    need(value["warp_dtype"] == expected_dtype, "actual pinned Warp dtype")
    return pointer, pointer + span


def observer(record, raw, arm, device, compiled, recipe):
    need(
        type(record) is dict
        and set(record)
        == {"protocol", "arm", "entries", "other_launches", "owned_hooks_restored"}
        and record["protocol"] == CONTROL_PROTOCOL
        and record["arm"] == arm
        and record["owned_hooks_restored"] is True,
        "closed new observer schema",
    )
    entries = record["entries"]
    need(type(entries) is list and len(entries) == 21, "exact21 per-arm targets")
    result, stable, identities = [], None, None
    selected = compiled["original" if arm == "original" else "candidate"]
    for index, row in enumerate(entries):
        need(
            type(row) is dict
            and set(row)
            == {
                "index",
                "arm",
                "packets",
                "layouts",
                "dim",
                "scalars",
                "kernel",
                "kernel_object_id",
                "kernel_function_id",
                "kernel_code_id",
                "launch_function_id",
                "launch_code_id",
                "make_function_id",
                "make_code_id",
                "stream",
                "device_context",
                "model_object_id",
                "data_object_id",
                "record_tape",
            },
            "exact target witness schema",
        )
        need(
            type(row["index"]) is int
            and row["index"] == index
            and row["arm"] == arm
            and row["record_tape"] is False
            and row["stream"] == device["stream"] > 0
            and row["device_context"] == device["context"],
            "actual ordered eager target context",
        )
        ids = {
            key: plain(row[key], 1)
            for key in (
                "kernel_object_id",
                "kernel_function_id",
                "kernel_code_id",
                "launch_function_id",
                "launch_code_id",
                "make_function_id",
                "make_code_id",
                "model_object_id",
                "data_object_id",
            )
        }
        need(
            ids["kernel_object_id"]
            == selected["binding"]["observed_object_ids"]["kernel"]
            and ids["launch_function_id"]
            == selected["runtime_entries"]["launch"]["object_id"]
            and ids["launch_code_id"]
            == selected["runtime_entries"]["launch"]["code_id"],
            "selected bound kernel and held launch entry",
        )
        need(
            row["kernel"] in selected["binding"]["symbol"]
            and row["kernel"]
            == ("_friction_dof" if arm == "original" else "ascending_friction_dof"),
            "literal selected kernel key",
        )
        need(
            identities is None or identities == ids,
            "stable per-arm callable/model/data identities",
        )
        identities = ids
        scope_call = recipe["control_scope"]["calls"][index]
        need(
            ids["model_object_id"] == scope_call["model_id"]
            and ids["data_object_id"] == scope_call["data_id"]
            and row["stream"] == scope_call["stream"],
            "friction entry tied to same actual serial forward",
        )
        packets = row["packets"]
        need(
            type(packets) is dict
            and set(packets)
            == {"inputs.before", "inputs.after", "bank.before", "bank.after"},
            "four complete target sections",
        )
        values = {}
        for label, order in (
            ("inputs", numerical.INPUT_NAMES),
            ("bank", numerical.BANK_NAMES),
        ):
            for phase in ("before", "after"):
                info = packets[label + "." + phase]
                need(
                    info["path"] == f"{arm}/entry-{index:02d}.{label}.{phase}.bin",
                    "literal target packet path",
                )
                values[label + "." + phase] = unpack_packet(info, raw, order=order)
        layouts = row["layouts"]
        all_carriers = {"input." + k: v for k, v in values["inputs.before"].items()} | {
            "bank." + k: v for k, v in values["bank.before"].items()
        }
        need(
            type(layouts) is dict and set(layouts) == set(all_carriers),
            "complete target allocation set",
        )
        ranges = []
        for key, carrier in all_carriers.items():
            bounds = layout(layouts[key], carrier, device)
            if bounds[1] > bounds[0]:
                ranges.append(bounds)
        ranges.sort()
        need(
            all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])),
            "target allocation ranges do not alias",
        )
        current = {k: v for k, v in layouts.items() if k != "bank.efc_nnz"}
        need(
            stable is None or stable == current,
            "same complete per-arm layouts except ephemeral counter",
        )
        stable = current
        result.append(
            dict(
                inputs=values["inputs.before"],
                inputs_after=values["inputs.after"],
                before=values["bank.before"],
                after=values["bank.after"],
                scalars=row["scalars"],
                dim=row["dim"],
            )
        )
    others = record["other_launches"]
    need(
        type(others) is list and 21 <= len(others) <= 4096,
        "bounded non-target launch trace",
    )
    for row in others:
        need(
            type(row) is dict and set(row) == {"forward", "kernel"},
            "non-target trace schema",
        )
        plain(row["forward"], 0, 20)
        need(
            type(row["kernel"]) is str
            and row["kernel"] not in ("_friction_dof", "ascending_friction_dof"),
            "non-target unchanged passthrough trace",
        )
    return result


def recipe(record, packet_info, raw, arm):
    need(
        type(record) is dict
        and set(record)
        == {
            "load_observer",
            "compiled_descriptor",
            "ntendon",
            "nominal_parameters",
            "physics_steps",
            "graph_created",
            "actor_model_created",
            "optimizer_created",
            "storage_created",
            "explicit_reset_calls",
            "control_scope",
            "rng_metadata",
            "ledger",
            "rng_end_boundary",
            "control_capture",
            "control_ids",
        },
        "unchanged one-tick recipe record schema",
    )
    for key in (
        "graph_created",
        "actor_model_created",
        "optimizer_created",
        "storage_created",
    ):
        need(record[key] is False, "no graph/actor/optimizer/storage")
    need(
        type(record["ntendon"]) is int
        and record["ntendon"] == 0
        and type(record["explicit_reset_calls"]) is int
        and record["explicit_reset_calls"] == 0
        and type(record["physics_steps"]) is int
        and record["physics_steps"] == 10
        and record["control_capture"] is True
        and record["rng_end_boundary"] == "after-one-nominal-step",
        "one nominal tick only",
    )
    need(
        sha256(canonical(record["compiled_descriptor"])).hexdigest()
        == recipe_checks.DESCRIPTOR_SHA256
        and record["nominal_parameters"]
        == dict(
            voltage=7.5, drop_gain=0.1, kp_scale=1.0, kd_scale=1.0, friction_scale=1.0
        ),
        "unchanged native plant and motor",
    )
    need(
        type(packet_info) is dict and set(packet_info) == set(recipe_checks.CASE_CAPS),
        "complete recipe packets",
    )
    packets = {}
    for name, cap in recipe_checks.CASE_CAPS.items():
        info = packet_info[name]
        need(
            type(info) is dict
            and set(info) == {"path", "bytes", "sha256"}
            and info["path"] == arm + "/" + name
            and info["path"] in raw
            and 0 < len(raw[info["path"]]) <= cap
            and info["bytes"] == len(raw[info["path"]])
            and info["sha256"] == sha256(raw[info["path"]]).hexdigest(),
            "whole recipe packet binding",
        )
        packets[name] = raw[info["path"]]
    initial, final = serial_checks.checked_frames(
        packets["frames.bin"], record["compiled_descriptor"]
    )
    serial_checks.checked_motor(
        packets["motor.bin"], packets["masks.bin"], record["compiled_descriptor"], final
    )
    serial_checks.checked_ledger(record["ledger"])
    rng_checks.checked_rng(packets["rng.bin"], record["rng_metadata"])
    chunks = {
        name: recipe_checks._chunks(packets[name], size, 21, name)
        for name, size in (
            ("rne.entries.bin", 24576),
            ("rne.outputs.bin", 24576),
            ("com.entries.bin", 12288),
            ("com.weighted.bin", 12288),
        )
    }
    serial_checks.checked_scope(record["control_scope"], chunks)
    # Explicit name adapter for the unchanged pure ten-proposal validator.
    # No historical two-arm owner envelope or admission is constructed.
    recipe_checks.checked_load_receipt(
        record["load_observer"],
        record,
        {arm + "." + k: v for k, v in packets.items()},
        arm,
    )
    need(
        all(type(x) is int for x in record["control_ids"])
        and record["control_ids"] == list(range(14)),
        "literal controlled actuator IDs",
    )
    return packets


def bounded_unit(value, source, mode, owner):
    plain(owner, 1)
    need(
        type(source) is str
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and mode in ("tests", "run"),
        "literal source and unit mode",
    )
    expected = dict(
        Id=f"microduck-friction-runtime-tick-{mode}-{source[:12]}.service",
        MainPID=str(owner),
        RuntimeMaxUSec="10min" if mode == "run" else "11min",
        MemoryMax=str(6 * 1024**3),
        CPUQuotaPerSecUSec="2s",
        TasksMax="64",
        Nice="10",
        LimitFSIZE=str(16 * 1024**2 if mode == "run" else 64 * 1024**2),
        Restart="no",
        KillMode="control-group",
    )
    need(
        type(value) is dict
        and set(value) == set(expected) | {"InvocationID"}
        and all(value[k] == v for k, v in expected.items())
        and type(value["InvocationID"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", value["InvocationID"]),
        "actual new bounded unit and invocation",
    )


def frozen_owner(declaration, child, owner):
    """Independent literal package, library/header and inherited-lease pins."""
    expected_libraries = {
        name: {
            "path": str(
                pins.VENV_TARGET / "lib/python3.12/site-packages/warp/bin" / name
            ),
            "bytes": size,
            "sha256": digest,
        }
        for name, (size, digest) in pins.LIBRARIES.items()
    }
    need(
        declaration["libraries"] == child["libraries"] == expected_libraries,
        "exact canonical loaded library whole-byte pins",
    )
    headers = declaration["warp_sources"]
    need(
        type(headers) is dict
        and set(headers) == {"root", "leaves", "sha256"}
        and headers == child["warp_sources"]
        and headers["root"]
        == str(pins.VENV_TARGET / "lib/python3.12/site-packages/warp")
        and type(headers["leaves"]) is dict
        and len(headers["leaves"]) == 460
        and headers["sha256"] == pins.WARP_SOURCES_SHA
        and sha256(canonical(headers["leaves"])).hexdigest() == pins.WARP_SOURCES_SHA,
        "exact complete frozen Warp source/header inventory",
    )
    need(
        sum(
            plain(info["bytes"], 0, MAX_FILE_BYTES)
            for info in headers["leaves"].values()
        )
        == 9292904,
        "complete frozen header bytes",
    )
    for name, info in headers["leaves"].items():
        safe_name(name)
        need(
            type(info) is dict and set(info) == {"bytes", "sha256"},
            "header anchor schema",
        )
        sha(info["sha256"])
    lease = declaration["lease"]
    need(
        type(lease) is dict
        and set(lease) == {"path", "device", "inode", "bytes"}
        and lease["path"] == str(pins.LOCK)
        and type(lease["bytes"]) is int
        and lease["bytes"] == 0
        and lease == child["lease"] == owner["lease"],
        "same existing empty inherited GPU lease",
    )
    plain(lease["device"], 1)
    plain(lease["inode"], 1)


def cpu_proof(declaration, raw, expected_source, expected_tests_sha):
    from mjlab_microduck import stance_friction_runtime_probe as producer

    need(
        type(declaration) is dict
        and {
            "tests_inventory_sha256",
            "tests_inventory",
            "mac_tests_inventory",
            "mac_tests_inventory_sha256",
            "tests",
            "tests_terminal",
        }
        <= set(declaration),
        "complete CPU proof declaration schema",
    )
    sha(expected_tests_sha)
    need(
        declaration["tests_inventory_sha256"] == expected_tests_sha
        and sha256(canonical(declaration["tests_inventory"])).hexdigest()
        == expected_tests_sha,
        "externally anchored current-source CPU inventory",
    )
    for mode, field in (
        ("tests", "tests_inventory"),
        ("mac-tests", "mac_tests_inventory"),
    ):
        inventory = declaration[field]
        need(
            type(inventory) is dict
            and set(inventory) == {"receipt.json", "pytest.log", "junit.xml"},
            "complete CPU prerequisite raw set",
        )
        for name, info in inventory.items():
            packet = raw[mode + "/" + name]
            need(
                info == {"bytes": len(packet), "sha256": sha256(packet).hexdigest()},
                "authenticated CPU prerequisite leaf",
            )
        proof = json_packet(raw[mode + "/receipt.json"])
        component.source(proof["source_binding"], expected_source)
        component.flags(proof["flags"])
        need(
            proof["packages"] == pins.VERSIONS
            and type(proof["files"]) is list
            and len(proof["files"]) == len(set(proof["files"])) == 90
            and sha256(canonical(proof["files"])).hexdigest()
            == producer.TEST_FILES_SHA256
            and type(proof["tests"]) is int
            and proof["tests"] == producer.EXPECTED_TESTS > 0,
            "exact90-file current-source CPU proof",
        )
        suites = list(ET.fromstring(raw[mode + "/junit.xml"]).iter("testsuite"))
        need(
            len(suites) == 1
            and int(suites[0].attrib["tests"]) == producer.EXPECTED_TESTS
            and all(
                suites[0].attrib[k] == "0" for k in ("skipped", "errors", "failures")
            ),
            "exact CPU collection and zero omissions",
        )
        if mode == "tests":
            need(proof == declaration["tests"], "whole native test receipt")
            component.environment(proof["environment_alias"])
            bounded_unit(
                proof["unit"],
                expected_source["source"],
                "tests",
                int(proof["unit"]["MainPID"]),
            )
            terminal = declaration["tests_terminal"]
            need(
                terminal["MainPID"] == "0"
                and terminal["SubState"] == "exited"
                and terminal["Result"] == "success"
                and terminal["ExecMainStatus"] == "0"
                and terminal["InvocationID"] == proof["unit"]["InvocationID"],
                "completed native CPU prerequisite",
            )
        else:
            need(
                proof["protocol"] == DATA_PROTOCOL + ":mac-tests"
                and proof["python"] == "3.12.12"
                and proof["machine"] == "arm64",
                "separate frozen Mac CPU prerequisite",
            )
            sha(declaration["mac_tests_inventory_sha256"])
            need(
                sha256(canonical(inventory)).hexdigest()
                == declaration["mac_tests_inventory_sha256"],
                "whole Mac inventory anchor",
            )


def owner_window(declaration, owner, child_pid):
    """Check observed ownership and the predeclared bounded time window."""
    from mjlab_microduck import stance_friction_runtime_probe as producer

    pid = plain(declaration["owner_pid"], 1)
    need(
        type(owner["child_observed"]) is bool
        and owner["child_observed"] is True
        and type(owner["child_ppid"]) is int
        and owner["child_ppid"] == pid
        and child_pid != pid,
        "observed separate child with actual owner parent",
    )
    start = plain(declaration["started_utc_ns"], 1)
    finish = plain(owner["finished_utc_ns"], 1)
    need(
        type(declaration["child_timeout_seconds"]) is int
        and declaration["child_timeout_seconds"] == producer.CHILD_SECONDS
        and start < finish <= start + producer.SERVICE_SECONDS * 10**9
        and start + 900 * 10**9 < int(producer.CUTOFF * 10**9)
        and finish < int(producer.CUTOFF * 10**9),
        "bounded owner duration and absolute launch reserve",
    )
    need(
        type(owner["samples"]) is list and 1 <= len(owner["samples"]) <= 270,
        "bounded monitored owner samples",
    )
    previous = -1.0
    for sample in owner["samples"]:
        elapsed = sample["elapsed_seconds"]
        need(
            type(elapsed) in (int, float)
            and isfinite(elapsed)
            and previous <= elapsed <= producer.CHILD_SECONDS
            and type(sample["child_pid"]) is int
            and sample["child_pid"] == child_pid,
            "ordered finite samples for the owned bounded child",
        )
        previous = elapsed


def verify_run(directory, inventory, *, expected_source, expected_tests_sha):
    raw = authenticate(directory, inventory)
    need(
        {"declaration.json", "child.json", "owner.json", "child.log"} <= set(raw),
        "complete fresh runtime owner raw set",
    )
    declaration, child, owner = (
        json_packet(raw[k + ".json"]) for k in ("declaration", "child", "owner")
    )
    common = {
        "protocol",
        "source_binding",
        "packages",
        "environment_alias",
        "owner_pid",
        "flags",
    }
    schemas = (
        (
            declaration,
            common
            | {
                "libraries",
                "warp_sources",
                "unit",
                "lease",
                "host_before",
                "tests",
                "tests_inventory_sha256",
                "tests_inventory",
                "tests_terminal",
                "mac_tests_inventory",
                "mac_tests_inventory_sha256",
                "dependency",
                "case_order",
                "started_utc_ns",
                "cutoff_utc",
                "child_timeout_seconds",
            },
        ),
        (
            child,
            common
            | {
                "libraries",
                "warp_sources",
                "child_pid",
                "lease",
                "declaration_sha256",
                "device",
                "compiler_config",
                "compiled",
                "arms",
            },
        ),
        (
            owner,
            common
            | {
                "lease",
                "child_pid",
                "child_exit",
                "child_observed",
                "child_ppid",
                "finished_utc_ns",
                "host_after",
                "samples",
            },
        ),
    )
    need(
        all(
            type(record) is dict and set(record) == schema for record, schema in schemas
        ),
        "exact fresh runtime packet schemas",
    )
    for record, suffix in (
        (declaration, "declaration"),
        (child, "child"),
        (owner, "owner"),
    ):
        need(
            record["protocol"] == DATA_PROTOCOL + ":" + suffix,
            "new scoped owner packet protocol",
        )
        component.source(record["source_binding"], expected_source)
        component.flags(record["flags"])
        component.environment(record["environment_alias"])
    need(
        child["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest(),
        "whole child declaration binding",
    )
    cpu_proof(declaration, raw, expected_source, expected_tests_sha)
    pid, child_pid = plain(declaration["owner_pid"], 1), plain(child["child_pid"], 1)
    need(
        owner["owner_pid"] == child["owner_pid"] == pid
        and owner["child_pid"] == child_pid
        and type(owner["child_exit"]) is int
        and owner["child_exit"] == 0,
        "same successfully completed leased child",
    )
    bounded_unit(declaration["unit"], expected_source["source"], "run", pid)
    need(
        declaration["unit"]["InvocationID"]
        != declaration["tests"]["unit"]["InvocationID"],
        "fresh separate run invocation",
    )
    need(
        declaration["packages"]
        == child["packages"]
        == owner["packages"]
        == pins.VERSIONS
        and declaration["libraries"] == child["libraries"]
        and declaration["warp_sources"] == child["warp_sources"]
        and declaration["environment_alias"]
        == child["environment_alias"]
        == owner["environment_alias"]
        and declaration["lease"] == child["lease"] == owner["lease"],
        "same frozen packages/libraries/headers/environment/lease",
    )
    frozen_owner(declaration, child, owner)
    component.host(declaration["host_before"], idle=True)
    component.host(owner["host_after"], idle=True)
    owner_window(declaration, owner, child_pid)
    for sample in owner["samples"]:
        component.host(sample, child=child_pid)
    need(
        declaration["case_order"] == list(ARMS)
        and declaration["cutoff_utc"] == "2026-10-07T23:30:00Z",
        "three sequential arms and absolute cutoff",
    )
    dependency = declaration["dependency"]
    need(
        dependency["decision"]
        == "synthetic-cuda-component-eight-exact-two-overflow-negative"
        and dependency["source"] == "5973184031ba9537e88dd066bd14faf8e5a9b4a4"
        and dependency["inventory_sha256"]
        == "2db2959673fc3719f7e3b0a050f6003e96d973efc60cb3a2def281768ce23fff"
        and dependency["receiver_sha256"]
        == "2601315fa33c5366fd6df716ee657bdd51f201fab1b81e56a3e3795ca9bf2c17"
        and dependency["mac_replay_sha256"]
        == "acb87ac697792a832de4ff214c5570ffffe278b7ebba1ec27d3573f04c9bf578",
        "separately accepted historical synthetic dependency",
    )
    device = child["device"]
    need(
        device["name"] == "cuda:0"
        and type(device["arch"]) is int
        and device["arch"] == 120,
        "actual sm120 eager device",
    )
    for key in ("object_id", "context", "stream"):
        plain(device[key], 1)
    native_directory = (
        pins.ROOT
        / "artifacts/evaluations"
        / ("friction-runtime-tick-run-" + expected_source["source"][:12])
    )
    component.compiled(child["compiled"], raw, inventory, native_directory, device)
    arms = child["arms"]
    need(type(arms) is dict and set(arms) == set(ARMS), "exact fresh three-arm records")
    entries, packets = {}, {}
    for arm in ARMS:
        need(
            type(arms[arm]) is dict
            and set(arms[arm]) == {"recipe", "observer", "packets"},
            "complete arm schema",
        )
        packets[arm] = recipe(arms[arm]["recipe"], arms[arm]["packets"], raw, arm)
        entries[arm] = observer(
            arms[arm]["observer"],
            raw,
            arm,
            device,
            child["compiled"],
            arms[arm]["recipe"],
        )
    report = numerical.audit_entries(entries)
    repeat = {
        name: packets["candidate0"][name] == packets["candidate1"][name]
        for name in recipe_checks.CASE_CAPS
    }
    recipe_repeat = all(
        arms["candidate0"]["recipe"][key] == arms["candidate1"]["recipe"][key]
        for key in (
            "ledger",
            "compiled_descriptor",
            "nominal_parameters",
            "control_ids",
        )
    )
    # The numerical component supplies its own strict decision, never provenance.
    numerical_exact = report["candidate_replay"]["exact"]
    paired = report["original_candidate_comparison"]
    matched_entries_exact = (
        paired["matched_entries"] > 0 and paired["first_matched_delta"] is None
    )
    structural_exact = all(
        report["arms"][arm]["structurally_exact"]
        and not report["arms"][arm]["overflow"]
        for arm in ARMS
    )
    passed = (
        structural_exact
        and numerical_exact
        and matched_entries_exact
        and all(repeat.values())
        and recipe_repeat
    )
    return {
        "protocol": PROTOCOL,
        "source_binding": expected_source,
        "decision": "real-model-one-tick-candidate-repeat-exact"
        if passed
        else "real-model-one-tick-candidate-repeat-negative",
        "synthetic_dependency": dependency,
        "entries": 63,
        "proposals": 30,
        "numerical": report,
        "recipe_packets_repeat": repeat,
        "recipe_state_repeat": recipe_repeat,
        "matched_original_entries_exact": matched_entries_exact,
        "flags": FLAGS,
    }
