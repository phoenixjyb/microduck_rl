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
from mjlab_microduck import stance_friction_runtime_receiver as historical_runtime
from mjlab_microduck import stance_bam_load_receiver as recipe_checks
from mjlab_microduck import stance_serial_step_receiver as serial_checks
from mjlab_microduck import stance_com_coupled_receiver as rng_checks

PROTOCOL = "microduck-contact-boundary-receiver-oct8-v1"
DATA_PROTOCOL = "microduck-contact-boundary-tick-oct8-v1"
CONTROL_PROTOCOL = "microduck-friction-runtime-control-oct8-v1"
BOUNDARY_CONTROL_PROTOCOL = "microduck-contact-boundary-control-oct8-v1"
ARMS = numerical.ARMS
FLAGS = dict(pins.FLAGS)
THREAD_ENV_REQUIRED = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES = 1024, 16 * 1024**2, 768 * 1024**2
MAX_JSON_BYTES = 8 * 1024**2
MAX_BOUNDARY_ARM_BYTES = 80 * 1024**2
MAX_BOUNDARY_BYTES = 240 * 1024**2
BOUNDARY_FORWARDS = 7
CONTACT_CAPACITY = 8192
CONTACT_NJMAX = 512
CONTACT_NJMAX_NNZ = 10240
CONTACT_FACTORY_QUALNAME = "_efc_contact_init"
HISTORICAL_NEGATIVE = {
    "source": "321a72a6b7d3d64808dafdbc36309052495cc85b",
    "raw_inventory_sha256": "233e52baa6d73fc9b23b4149f013d783638bdaa4ff814ab6eacbea765ae4bb58",
    "receiver_sha256": "69a1616f48189de605f9efd0e6d4eb58372031d14ff9c6bc6e36d779383884f2",
    "mac_replay_sha256": "6e75c19aa3b5464a8b60b28a69669818d55b1f5f4fb9b6ce030974229a6093a6",
    "mac_first_delta_sha256": "58d9fe994557b1cafc13e488577dcde1d9d7a4407ff5609723356e47be213dff",
}
HISTORICAL_NEGATIVE_FILES = {
    "inventory.json": {
        "bytes": 60551,
        "sha256": HISTORICAL_NEGATIVE["raw_inventory_sha256"],
    },
    "receiver.json": {
        "bytes": 1024216,
        "sha256": HISTORICAL_NEGATIVE["receiver_sha256"],
    },
    "mac-receiver.json": {
        "bytes": 1024216,
        "sha256": HISTORICAL_NEGATIVE["receiver_sha256"],
    },
    "mac-replay.json": {
        "bytes": 842,
        "sha256": HISTORICAL_NEGATIVE["mac_replay_sha256"],
    },
    "mac-first-delta.json": {
        "bytes": 32437,
        "sha256": HISTORICAL_NEGATIVE["mac_first_delta_sha256"],
    },
    "terminal.txt": {
        "bytes": 104,
        "sha256": "4edd349d407c77b66d4bf2890710eadeb6b193c43c322f86fe903213b4f87bc3",
    },
}
CONTACT_INPUT_ORDER = (
    "model.body_weldid",
    "model.body_dofnum",
    "model.body_dofadr",
    "model.dof_parentid",
    "model.geom_bodyid",
    "model.flex_vertadr",
    "model.flex_vertbodyid",
    "contact.nacon",
    "contact.dist",
    "contact.dim",
    "contact.includemargin",
    "contact.worldid",
    "contact.geom",
    "contact.flex",
    "contact.vert",
    "contact.type",
)
CONTACT_OUTPUT_ORDER = (
    "data.nefc",
    "contact.efc_address",
    "efc.id",
    "efc.J_rownnz",
    "efc.J_rowadr",
    "local.efc_nnz",
)
CONTACT_CONTEXT_ORDER = (
    "context.geomcollisionid",
    "context.pos",
    "context.frame",
)
CONTACT_PACKET_ORDER = (
    CONTACT_INPUT_ORDER + CONTACT_OUTPUT_ORDER + CONTACT_CONTEXT_ORDER
)
COMPLETE_ORDER = (
    "data.ne",
    "data.nf",
    "data.nl",
    "data.nefc",
    "data.qpos",
    "data.qvel",
    "data.qacc",
    "data.qacc_warmstart",
    "data.ctrl",
    "efc.type",
    "efc.id",
    "efc.J_rownnz",
    "efc.J_rowadr",
    "efc.J_colind",
    "efc.J",
    "efc.pos",
    "efc.margin",
    "efc.D",
    "efc.vel",
    "efc.aref",
    "efc.frictionloss",
    "efc.force",
    "efc.state",
    "efc.Ma",
    "efc.Jqvel",
    "local.efc_nnz",
)
STALE_PRIOR_FIELDS = ("efc.force", "efc.state", "efc.Ma", "efc.Jqvel")
# Keep the historical capture declaration above immutable. Dense construction
# clears/recomputes Jqvel before the current solver; prospective side views must
# not repeat the old phase-label mistake.
SIDE_VIEW_PRIOR_SOLVER_FIELDS = ("efc.force", "efc.state", "efc.Ma")
DENSE_CONSTRUCTION_SOURCE_SHA256 = (
    "b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53"
)
CONSTRUCTION_ROW_FIELDS = (
    "efc.J",
    "efc.pos",
    "efc.margin",
    "efc.D",
    "efc.vel",
    "efc.aref",
    "efc.frictionloss",
    "efc.Jqvel",
)
POST_FORWARD_LOAD_PAIRS = ((0, 0), (2, 1), (4, 2), (6, 3))
MAX_ROW_VIEW_SAMPLES = 128
SOLVED_LOAD_CARRIERS = {
    "data.nefc": ((64,), "<i4"),
    "efc.id": ((64, 512), "<i4"),
    "efc.type": ((64, 512), "<i4"),
    "efc.force": ((64, 512), "<f4"),
    "data.qfrc_constraint": ((64, 20), "<f4"),
}
CONTACT_CARRIERS = {
    "model.body_weldid": ((16,), "<i4"),
    "model.body_dofnum": ((16,), "<i4"),
    "model.body_dofadr": ((16,), "<i4"),
    "model.dof_parentid": ((20,), "<i4"),
    "model.geom_bodyid": ((82,), "<i4"),
    "model.flex_vertadr": ((0,), "<i4"),
    "model.flex_vertbodyid": ((0,), "<i4"),
    "contact.nacon": ((1,), "<i4"),
    "contact.dist": ((CONTACT_CAPACITY,), "<f4"),
    "contact.dim": ((CONTACT_CAPACITY,), "<i4"),
    "contact.includemargin": ((CONTACT_CAPACITY,), "<f4"),
    "contact.worldid": ((CONTACT_CAPACITY,), "<i4"),
    "contact.geom": ((CONTACT_CAPACITY, 2), "<i4"),
    "contact.flex": ((CONTACT_CAPACITY, 2), "<i4"),
    "contact.vert": ((CONTACT_CAPACITY, 2), "<i4"),
    "contact.type": ((CONTACT_CAPACITY,), "<i4"),
    "data.nefc": ((64,), "<i4"),
    "contact.efc_address": ((CONTACT_CAPACITY, 4), "<i4"),
    "efc.id": ((64, 512), "<i4"),
    "efc.J_rownnz": ((64, 0), "<i4"),
    "efc.J_rowadr": ((64, 0), "<i4"),
    "local.efc_nnz": ((64,), "<i4"),
    "context.geomcollisionid": ((CONTACT_CAPACITY,), "<i4"),
    "context.pos": ((CONTACT_CAPACITY, 3), "<f4"),
    "context.frame": ((CONTACT_CAPACITY, 3, 3), "<f4"),
}
COMPLETE_CARRIERS = {
    "data.ne": ((64,), "<i4"),
    "data.nf": ((64,), "<i4"),
    "data.nl": ((64,), "<i4"),
    "data.nefc": ((64,), "<i4"),
    "data.qpos": ((64, 21), "<f4"),
    "data.qvel": ((64, 20), "<f4"),
    "data.qacc": ((64, 20), "<f4"),
    "data.qacc_warmstart": ((64, 20), "<f4"),
    "data.ctrl": ((64, 14), "<f4"),
    "efc.type": ((64, 512), "<i4"),
    "efc.id": ((64, 512), "<i4"),
    "efc.J_rownnz": ((64, 0), "<i4"),
    "efc.J_rowadr": ((64, 0), "<i4"),
    "efc.J_colind": ((64, 0, 0), "<i4"),
    "efc.J": ((64, 512, 20), "<f4"),
    "efc.pos": ((64, 512), "<f4"),
    "efc.margin": ((64, 512), "<f4"),
    "efc.D": ((64, 512), "<f4"),
    "efc.vel": ((64, 512), "<f4"),
    "efc.aref": ((64, 512), "<f4"),
    "efc.frictionloss": ((64, 512), "<f4"),
    "efc.force": ((64, 512), "<f4"),
    "efc.state": ((64, 512), "<i4"),
    "efc.Ma": ((64, 20), "<f4"),
    "efc.Jqvel": ((64, 512), "<f4"),
    "local.efc_nnz": ((64,), "<i4"),
}
EFC_FIELDS = tuple(name for name in COMPLETE_ORDER if name.startswith("efc."))
EFC_BYTES = 4_068_352
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
    if carrier["dtype"] == "<f4" and carrier["shape"][-2:] == [3, 3]:
        need(
            shape == carrier["shape"][:-2],
            "literal original mat33 logical array shape, not a reinterpreted vector view",
        )
        expected_dtype = "<class 'warp._src.types.mat33f'>"
    elif len(carrier["shape"]) > len(shape):
        vector_types = {
            ("<f4", 2): "<class 'warp._src.types.vec2f'>",
            ("<f4", 3): "<class 'warp._src.types.vec3f'>",
            ("<f4", 5): "<class 'mujoco_warp._src.types.vec5f'>",
            ("<i4", 2): "<class 'warp._src.types.vec2i'>",
        }
        key = (carrier["dtype"], carrier["shape"][-1])
        need(key in vector_types, "declared vector/matrix carrier only")
        expected_dtype = vector_types[key]
    need(value["warp_dtype"] == expected_dtype, "actual pinned Warp dtype")
    return pointer, pointer + span


def numerical_entry_projection(entries):
    """Project already ownership-checked rows without changing numeric carriers.

    Contact metadata stays in the original rows. The historical numerical
    component still receives and validates its exact unchanged six-field ABI.
    This adapter grants no ownership or qualification on its own.
    """
    need(type(entries) is dict and set(entries) == set(ARMS), "exact observer arms")
    enriched_fields = numerical.ENTRY_FIELDS | {"layouts", "identities", "stream"}
    projected = {}
    for arm in ARMS:
        rows = entries[arm]
        need(type(rows) is list and len(rows) == 21, "exact observer forward count")
        projected[arm] = []
        for index, row in enumerate(rows):
            need(
                type(row) is dict and set(row) == enriched_fields,
                f"exact enriched observer fields {arm}.forward{index}",
            )
            projected[arm].append({key: row[key] for key in numerical.ENTRY_FIELDS})
    return projected


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
                layouts=layouts,
                identities=ids,
                stream=row["stream"],
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


def compiled_contact(value, raw, inventory, directory, device):
    """Validate the explicit contact role without aliasing it as friction."""
    need(
        type(value) is dict
        and set(value)
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
        "explicit contact compiled-module role",
    )
    binding = value["binding"]
    need(
        canonical(value["compiler_config"]) == canonical(component.COMPILER_CONFIG),
        "actual explicit bounded contact compiler config and PCH disabled",
    )
    need(
        type(binding) is dict
        and set(binding)
        == {
            "protocol",
            "artifact_format",
            "binary_sha256",
            "binary_path",
            "binary_bytes",
            "metadata_sha256",
            "metadata_path",
            "metadata_bytes",
            "module_hash",
            "block_dim",
            "context",
            "module_handle",
            "forward_handle",
            "forward_smem_bytes",
            "symbol",
            "device",
            "device_arch",
            "observed_object_ids",
            "driver_jit_machine_code_observed",
            "loaded_binary_bytes_observed",
            "native_execution_qualified",
            "training_authorized",
            "physical_acceptance",
        }
        and binding["protocol"] == "microduck-retained-cuda-artifact-binding-oct7-v1"
        and binding["artifact_format"] == "cubin"
        and binding["device"] == "cuda:0"
        and binding["device_arch"] == device["arch"]
        and binding["context"] == device["context"]
        and binding["block_dim"] == 256,
        "contact CUBIN role bound to actual sm120 context",
    )
    plain(binding["context"], 1)
    plain(binding["device_arch"], 1)
    plain(binding["block_dim"], 1, 1024)
    plain(binding["binary_bytes"], 1, MAX_FILE_BYTES)
    plain(binding["metadata_bytes"], 1, MAX_FILE_BYTES)
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
        "opaque contact loaded/JIT bytes and non-admission",
    )
    need(
        type(binding["symbol"]) is str
        and "efc_contact_init" in binding["symbol"]
        and binding["symbol"].endswith("_cuda_kernel_forward"),
        "cached contact-init forward symbol, not friction role header",
    )
    ids = binding["observed_object_ids"]
    need(
        type(ids) is dict
        and set(ids) == {"kernel", "module", "device", "executable", "hooks"}
        and all(type(item) is int and item > 0 for item in ids.values())
        and ids["device"] == device["object_id"],
        "actual retained contact kernel/module/executable identities",
    )
    options = value["module_options"]
    plain(value["output_arch"], 1)
    need(
        type(options) is dict
        and options.get("enable_backward") is False
        and options.get("strip_hash") is False
        and type(options.get("block_dim")) is int
        and options["block_dim"] == 256
        and value["output_arch"] == device["arch"],
        "actual contact forward-only compile options",
    )
    paths = {}
    for kind in ("generated_source", "binary", "metadata"):
        info = value[kind]
        need(
            type(info) is dict and set(info) == {"path", "bytes", "sha256"},
            "retained contact compilation file anchor",
        )
        path = Path(info["path"])
        need(
            path.parent == Path(directory) / "compiled-contact",
            "contact artifacts use an explicit compiled-contact directory",
        )
        name = str(path.relative_to(directory))
        need(
            name in raw
            and inventory[name] == {key: info[key] for key in ("bytes", "sha256")},
            "contact compile files belong to whole authenticated run inventory",
        )
        paths[kind] = name
    need(
        raw[paths["binary"]][:4] == b"\x7fELF"
        and paths["binary"] == "compiled-contact/contact.cubin"
        and paths["generated_source"].endswith(".cu")
        and b"efc_contact_init" in raw[paths["generated_source"]]
        and paths["metadata"].endswith(".meta"),
        "fresh contact generated source and ELF CUBIN role",
    )
    meta = decode(raw[paths["metadata"]])
    need(
        type(meta) is dict
        and all(
            type(key) is str
            and key.endswith(
                ("_cuda_kernel_forward_smem_bytes", "_cuda_kernel_backward_smem_bytes")
            )
            and type(item) is int
            and 0 <= item <= 1024**2
            for key, item in meta.items()
        )
        and meta.get(binding["symbol"] + "_smem_bytes")
        == binding["forward_smem_bytes"],
        "contact whole metadata agrees with its explicit forward hook",
    )
    for kind in ("binary", "metadata"):
        need(
            binding[kind + "_path"] == value[kind]["path"]
            and binding[kind + "_bytes"] == value[kind]["bytes"]
            and binding[kind + "_sha256"] == value[kind]["sha256"],
            "contact loaded input file identity",
        )
    load = value["explicit_load"]
    need(
        load
        == {
            "module_object_id": ids["module"],
            "returned_executable_id": ids["executable"],
            "device_object_id": ids["device"],
            "block_dim": 256,
            "binary_path": value["binary"]["path"],
            "meta_path": value["metadata"]["path"],
            "output_arch": device["arch"],
            "fresh_cache_before": True,
        }
        and load["fresh_cache_before"] is True,
        "actual fresh explicit contact module load",
    )
    runtime = value["runtime_entries"]
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
        "held frozen contact compile/load runtime entry points",
    )
    for record in runtime.values():
        need(
            type(record) is dict
            and set(record) == {"object_id", "code_id", "qualname"}
            and type(record["qualname"]) is str,
            "held contact runtime code identity",
        )
        plain(record["object_id"], 1)
        plain(record["code_id"], 1)
    return {
        "kernel_object_id": ids["kernel"],
        "module_object_id": ids["module"],
        "runtime_entries": runtime,
    }


def verify_historical_negative(declaration, historical_root=None):
    """Authenticate the retained real-model negative before citing it."""
    need(
        type(declaration) is dict
        and set(declaration)
        == {"source", "decision", "anchors", "raw_files", "raw_bytes", "flags"}
        and declaration["source"] == HISTORICAL_NEGATIVE["source"]
        and declaration["decision"] == "real-model-one-tick-candidate-repeat-negative"
        and declaration["anchors"] == HISTORICAL_NEGATIVE_FILES
        and declaration["raw_files"] == 414
        and declaration["raw_bytes"] == 526737887
        and declaration["flags"] == FLAGS,
        "exact separately retained real-model negative anchors",
    )
    historical_root = Path(pins.ROOT if historical_root is None else historical_root)
    need(
        historical_root.is_absolute()
        and str(historical_root)
        in (
            "/home/yanbo/work/microduck_rl-com-entry-20261006",
            "/Users/yanbo/Projects/microduckPlayground/microduck_rl",
        )
        and historical_root.resolve(strict=True) == historical_root,
        "canonical native or Mac root for retained historical copies",
    )
    closeout = (
        historical_root / "artifacts/tools/friction-runtime-tick-closeout-321a72a6b7d3"
    )
    closeout_raw = {}
    for name, anchor in HISTORICAL_NEGATIVE_FILES.items():
        path = closeout / name
        need(
            path.resolve(strict=True) == path
            and path.is_file()
            and not path.is_symlink(),
            "plain retained historical closeout file " + name,
        )
        raw = path.read_bytes()
        need(
            len(raw) == anchor["bytes"] and sha256(raw).hexdigest() == anchor["sha256"],
            "whole-byte historical closeout anchor " + name,
        )
        closeout_raw[name] = raw
    inventory = json_packet(closeout_raw["inventory.json"])
    run = (
        historical_root / "artifacts/evaluations/friction-runtime-tick-run-321a72a6b7d3"
    )
    raw_run = historical_runtime.authenticate(run, inventory)
    old_declaration = json_packet(raw_run["declaration.json"])
    need(
        old_declaration["source_binding"]["source"] == HISTORICAL_NEGATIVE["source"],
        "historical run exact execution source",
    )
    old_report = historical_runtime.verify_run(
        run,
        inventory,
        expected_source=old_declaration["source_binding"],
        expected_tests_sha=old_declaration["tests_inventory_sha256"],
    )
    receiver_raw = closeout_raw["receiver.json"]
    replay = json_packet(closeout_raw["mac-replay.json"])
    delta = json_packet(closeout_raw["mac-first-delta.json"])
    need(
        historical_runtime.canonical(old_report) == receiver_raw
        and old_report["decision"] == "real-model-one-tick-candidate-repeat-negative"
        and old_report["source_binding"]["source"] == HISTORICAL_NEGATIVE["source"],
        "recomputed historical negative receiver and exact source",
    )
    need(
        replay["source"] == HISTORICAL_NEGATIVE["source"]
        and replay["raw_inventory"] == HISTORICAL_NEGATIVE_FILES["inventory.json"]
        and replay["receiver"] == HISTORICAL_NEGATIVE_FILES["receiver.json"]
        and replay["decision"] == old_report["decision"]
        and replay["all_whole_leaves_authenticated"] is True,
        "independent replay bound to authenticated negative",
    )
    need(
        delta["execution_source"] == HISTORICAL_NEGATIVE["source"]
        and delta["diagnosis_source"] == "ae6814d77c5e4b8672b2fa4a57ad2849e33df7c4"
        and delta["raw_inventory_sha256"]
        == HISTORICAL_NEGATIVE_FILES["inventory.json"]["sha256"]
        and delta["independent_receiver_sha256"]
        == HISTORICAL_NEGATIVE_FILES["receiver.json"]["sha256"]
        and delta["protocol"] == "microduck-friction-retained-first-delta-oct8-v1",
        "bit-addressed first-delta report bound to negative source and inventory",
    )
    return {
        "source": HISTORICAL_NEGATIVE["source"],
        "decision": old_report["decision"],
        "anchors": HISTORICAL_NEGATIVE_FILES,
        "raw_files": len(raw_run),
        "raw_bytes": sum(len(value) for value in raw_run.values()),
        "flags": dict(FLAGS),
        "receiver_recomputed": True,
        "mac_replay_authenticated": True,
        "first_delta_authenticated": True,
    }


def _first_raw_word(left, right):
    need(len(left) == len(right) and len(left) % 4 == 0, "equal complete raw fields")
    for index in range(0, len(left), 4):
        if left[index : index + 4] != right[index : index + 4]:
            return {
                "word_index": index // 4,
                "left_word_le_hex": left[index : index + 4].hex(),
                "right_word_le_hex": right[index : index + 4].hex(),
            }
    return None


def _int_words(carrier):
    need(
        carrier["dtype"] == "<i4" and len(carrier["raw"]) % 4 == 0,
        "whole little-endian int32 count carrier",
    )
    return tuple(
        int.from_bytes(carrier["raw"][offset : offset + 4], "little", signed=True)
        for offset in range(0, len(carrier["raw"]), 4)
    )


def _active_segments(stage, name, carrier, nacon, nefc):
    """Return bounded active-prefix spans without changing retained raw bytes."""
    shape, raw = carrier["shape"], carrier["raw"]
    need(
        type(shape) is list
        and len(raw) == 4 * prod(shape)
        and type(nacon) is int
        and 0 <= nacon <= CONTACT_CAPACITY
        and len(nefc) == 64
        and all(type(count) is int and 0 <= count <= CONTACT_NJMAX for count in nefc),
        "bounded active-span carrier and plain extents",
    )
    if not raw:
        return []

    def bounded(spans):
        need(
            all(
                type(offset) is int
                and type(size) is int
                and 0 <= offset <= len(raw)
                and 0 <= size <= len(raw) - offset
                for offset, size in spans
            ),
            "active spans stay inside the authenticated raw carrier",
        )
        return spans

    if stage.startswith("contact_"):
        prefix_names = {
            "contact.dist",
            "contact.dim",
            "contact.includemargin",
            "contact.worldid",
            "contact.geom",
            "contact.flex",
            "contact.vert",
            "contact.type",
            "context.geomcollisionid",
            "context.pos",
            "context.frame",
            "contact.efc_address",
        }
        if name in prefix_names:
            width = 4 * prod(shape[1:])
            return bounded([(0, nacon * width)] if nacon else [])
        if name == "efc.id":
            return bounded(
                [
                    (world * shape[1] * 4, count * 4)
                    for world, count in enumerate(nefc)
                    if count
                ]
            )
        return bounded([(0, len(raw))])
    if (
        name.startswith("efc.")
        and name != "efc.Ma"
        and len(shape) >= 2
        and shape[0] == 64
    ):
        row_width = 4 * prod(shape[2:])
        return bounded(
            [
                (world * shape[1] * row_width, count * row_width)
                for world, count in enumerate(nefc)
                if count
            ]
        )
    return bounded([(0, len(raw))])


def _active_first_difference(left_stage, right_stage, stage):
    order = {
        "contact_before": CONTACT_PACKET_ORDER,
        "contact_after": CONTACT_PACKET_ORDER,
        "complete": COMPLETE_ORDER,
    }[stage]
    if stage == "complete":
        left_nefc = _int_words(left_stage["data.nefc"])
        right_nefc = _int_words(right_stage["data.nefc"])
        nacon_left = nacon_right = CONTACT_CAPACITY
    else:
        left_nacon = _int_words(left_stage["contact.nacon"])
        right_nacon = _int_words(right_stage["contact.nacon"])
        need(len(left_nacon) == len(right_nacon) == 1, "one shared contact count")
        nacon_left, nacon_right = left_nacon[0], right_nacon[0]
        left_nefc = _int_words(left_stage["data.nefc"])
        right_nefc = _int_words(right_stage["data.nefc"])
    need(
        0 <= nacon_left <= CONTACT_CAPACITY
        and 0 <= nacon_right <= CONTACT_CAPACITY
        and len(left_nefc) == len(right_nefc) == 64
        and all(0 <= count <= CONTACT_NJMAX for count in left_nefc + right_nefc),
        "authenticated active contact and per-world EFC extents",
    )
    if nacon_left != nacon_right:
        difference = _first_raw_word(
            left_stage["contact.nacon"]["raw"], right_stage["contact.nacon"]["raw"]
        )
        return {"field": "contact.nacon", **difference, "extent_counts_equal": False}
    if left_nefc != right_nefc:
        difference = _first_raw_word(
            left_stage["data.nefc"]["raw"], right_stage["data.nefc"]["raw"]
        )
        return {"field": "data.nefc", **difference, "extent_counts_equal": False}
    for name in order:
        a, b = left_stage[name], right_stage[name]
        seg_a = _active_segments(stage, name, a, nacon_left, left_nefc)
        seg_b = _active_segments(stage, name, b, nacon_left, right_nefc)
        need(seg_a == seg_b, "same active extents for raw candidate comparison")
        for (offset_a, size_a), (offset_b, size_b) in zip(seg_a, seg_b):
            need(offset_a == offset_b and size_a == size_b, "same active raw span")
            word = _first_raw_word(
                a["raw"][offset_a : offset_a + size_a],
                b["raw"][offset_b : offset_b + size_b],
            )
            if word is not None:
                return {
                    "field": name,
                    "byte_offset": offset_a + word["word_index"] * 4,
                    "word_index": (offset_a // 4) + word["word_index"],
                    "left_word_le_hex": word["left_word_le_hex"],
                    "right_word_le_hex": word["right_word_le_hex"],
                    "extent_counts_equal": True,
                }
    return None


def _checked_boundary_view(bank, stage):
    need(
        type(stage) is str and stage in ("contact_before", "contact_after", "complete"),
        "literal boundary stage for raw side view",
    )
    specs = COMPLETE_CARRIERS if stage == "complete" else CONTACT_CARRIERS
    need(
        type(bank) is dict and set(bank) == set(specs),
        "exact whole boundary carrier bank for raw side view",
    )
    for name, (shape, dtype) in specs.items():
        carrier = bank[name]
        need(
            type(carrier) is dict
            and set(carrier) == {"shape", "dtype", "raw"}
            and type(carrier["shape"]) is list
            and all(type(value) is int for value in carrier["shape"])
            and tuple(carrier["shape"]) == shape
            and type(carrier["dtype"]) is str
            and carrier["dtype"] == dtype
            and type(carrier["raw"]) is bytes
            and len(carrier["raw"]) == 4 * prod(shape),
            "literal boundary carrier for raw side view: " + name,
        )


def describe_first_active_difference(left, right, stage):
    """Add storage coordinates/stale labels, never physical identity or cause.

    This side view needs caller-authenticated packets. It does not change the
    captured report, supply provenance or grant admission.
    """
    _checked_boundary_view(left, stage)
    _checked_boundary_view(right, stage)
    difference = _active_first_difference(left, right, stage)
    if difference is None:
        return None
    name = difference["field"]
    shape = left[name]["shape"]
    index = difference["word_index"]
    coordinate = []
    for axis, size in enumerate(shape):
        width = prod(shape[axis + 1 :])
        coordinate.append(index // width)
        index %= width
        need(coordinate[-1] < size, "raw coordinate inside literal carrier")
    left_nefc, right_nefc = (
        _int_words(left["data.nefc"]),
        _int_words(right["data.nefc"]),
    )
    extent = {"kind": "whole-retained-carrier"}
    contact_prefix = stage != "complete" and (
        name.startswith("contact.")
        and name != "contact.nacon"
        or name.startswith("context.")
    )
    if name == "contact.nacon":
        extent = {
            "kind": "global-active-contact-count",
            "left_count": _int_words(left["contact.nacon"])[0],
            "right_count": _int_words(right["contact.nacon"])[0],
        }
    elif contact_prefix:
        contact_index = coordinate[0]
        extent = {
            "kind": "active-contact-prefix",
            "contact_index": contact_index,
            "left_count": _int_words(left["contact.nacon"])[0],
            "right_count": _int_words(right["contact.nacon"])[0],
            "left_world": _int_words(left["contact.worldid"])[contact_index],
            "right_world": _int_words(right["contact.worldid"])[contact_index],
        }
        need(
            0 <= extent["left_world"] < 64 and 0 <= extent["right_world"] < 64,
            "bounded recorded contact world for raw coordinate",
        )
    elif name == "data.nefc" or (
        name.startswith("efc.")
        and name != "efc.Ma"
        and len(shape) >= 2
        and shape[0] == 64
    ):
        world = coordinate[0]
        extent = {
            "kind": "per-world-constraint-count"
            if name == "data.nefc"
            else "per-world-active-EFC-rows",
            "world": world,
            "left_count": left_nefc[world],
            "right_count": right_nefc[world],
        }
    return {
        "protocol": "microduck-boundary-raw-coordinate-view-oct8-v2",
        "stage": stage,
        "delta": dict(difference),
        "shape": list(shape),
        "row_major_coordinate": coordinate,
        "byte_offset": difference["word_index"] * 4,
        "active_extent": extent,
        "stale_prior_solver_storage": stage == "complete"
        and name in SIDE_VIEW_PRIOR_SOLVER_FIELDS,
        "historical_capture_stale_label": stage == "complete"
        and name in STALE_PRIOR_FIELDS,
        "recomputed_during_dense_construction": stage == "complete"
        and name == "efc.Jqvel",
        "phase_source_sha256": DENSE_CONSTRUCTION_SOURCE_SHA256,
        "raw_packets_unchanged": True,
        "flags": dict(FLAGS),
        "interpretation": (
            "raw storage coordinate and source-corrected phase label only; "
            "not physical-contact identity, newly solved force, cause or qualification"
        ),
    }


def compare_active_contact_payloads(left, right):
    """Describe recorded input/context payload order, never physical identity.

    Call only on whole-authenticated, independently validated contact packets.
    Counting a byte-tuple side view neither sorts nor rewrites raw records.
    Equal payloads do not establish constraint/solver equivalence or a cause.
    Duplicate payloads remain explicitly ambiguous and are never paired.
    """
    groups = []
    counts = []
    signatures = []
    payload_fields = (
        tuple(
            name
            for name in CONTACT_INPUT_ORDER
            if name.startswith("contact.") and name != "contact.nacon"
        )
        + CONTACT_CONTEXT_ORDER
    )
    for bank in (left, right):
        _checked_boundary_view(bank, "contact_before")
        count = _int_words(bank["contact.nacon"])[0]
        need(0 <= count <= CONTACT_CAPACITY, "bounded active payload count")
        counts.append(count)
        rows, by_payload = [], {}
        for index in range(count):
            signature = tuple(
                bank[name]["raw"][
                    index * 4 * prod(bank[name]["shape"][1:]) : (index + 1)
                    * 4
                    * prod(bank[name]["shape"][1:])
                ]
                for name in payload_fields
            )
            rows.append(signature)
            by_payload.setdefault(signature, []).append(index)
        signatures.append(rows)
        groups.append(by_payload)
    left_groups, right_groups = groups
    unique_links, ambiguous, unmatched = [], [], []
    # Insertion order is the retained left order, followed by right-only keys.
    # Pairing uses exact byte tuples, not hashes or decoded numeric equality.
    keys = dict.fromkeys((*left_groups, *right_groups))
    for payload in keys:
        left_indices = left_groups.get(payload, [])
        right_indices = right_groups.get(payload, [])
        if len(left_indices) == len(right_indices) == 1:
            unique_links.append(
                {"left_index": left_indices[0], "right_index": right_indices[0]}
            )
        else:
            info = {
                "payload_sha256": sha256(b"".join(payload)).hexdigest(),
                "left_indices": list(left_indices),
                "right_indices": list(right_indices),
            }
            if len(left_indices) != len(right_indices):
                unmatched.append(info)
            if len(left_indices) > 1 or len(right_indices) > 1:
                ambiguous.append(info)
    return {
        "protocol": "microduck-active-contact-payload-view-oct8-v1",
        "payload_fields": list(payload_fields),
        "contact_counts": {"left": counts[0], "right": counts[1]},
        "model_inputs_exact": all(
            left[name]["raw"] == right[name]["raw"]
            for name in CONTACT_INPUT_ORDER
            if name.startswith("model.")
        ),
        "record_order_exact": signatures[0] == signatures[1],
        "payload_multiset_equal": not unmatched,
        "unique_payload_links": unique_links,
        "ambiguous_payload_groups": ambiguous,
        "unmatched_payload_groups": unmatched,
        "output_fields_compared": False,
        "raw_packets_unchanged": True,
        "flags": dict(FLAGS),
        "interpretation": (
            "active raw input/context payloads only; unique byte payload links "
            "are not physical-contact identity, constraint/solver equivalence, "
            "causal proof or qualification; duplicate payloads remain unpaired"
        ),
    }


def describe_contact_constraint_rows(contact_after, complete):
    """Describe captured addresses and row backlinks, not allocation or cause.

    Caller must authenticate same-forward, same-arm packets and the pinned dense
    pyramidal source. Start at completed contact-typed rows, then check addresses
    and backlinks. No host float predicate replaces GPU eligibility. Addresses
    of contacts without observed complete rows are not read or classified;
    unused slots can retain arbitrary prior bytes.
    """
    _checked_boundary_view(contact_after, "contact_after")
    _checked_boundary_view(complete, "complete")
    count = _int_words(contact_after["contact.nacon"])[0]
    need(0 <= count <= CONTACT_CAPACITY, "bounded row-view contact count")
    nefc = _int_words(complete["data.nefc"])
    need(
        nefc == _int_words(contact_after["data.nefc"])
        and all(0 <= value <= CONTACT_NJMAX for value in nefc),
        "same bounded captured contact-after and complete row extents",
    )
    worlds = _int_words(contact_after["contact.worldid"])
    dims = _int_words(contact_after["contact.dim"])
    kinds = _int_words(contact_after["contact.type"])
    addresses = _int_words(contact_after["contact.efc_address"])
    ids, types = _int_words(complete["efc.id"]), _int_words(complete["efc.type"])
    observed = {}
    for world, extent in enumerate(nefc):
        for row in range(extent):
            index = world * CONTACT_NJMAX + row
            if types[index] not in (5, 6):
                continue
            contact_index = ids[index]
            need(
                0 <= contact_index < count,
                "complete contact row id in active contact prefix",
            )
            need(
                worlds[contact_index] == world and kinds[contact_index] & 1,
                "complete contact row belongs to captured constraint contact world",
            )
            observed.setdefault(contact_index, []).append((world, row))
    rows, occupied = [], set()
    for contact_index in range(count):
        world = worlds[contact_index]
        need(0 <= world < 64, "bounded contact world for row side view")
        if contact_index not in observed:
            rows.append(
                {
                    "contact_index": contact_index,
                    "world": world,
                    "status": "no-complete-contact-row-not-read",
                }
            )
            continue
        condim = dims[contact_index]
        need(condim in (1, 3), "scoped condim fits four captured address slots")
        width = 1 if condim == 1 else 4
        block = list(addresses[contact_index * 4 : contact_index * 4 + width])
        need(
            all(0 <= row < nefc[world] for row in block),
            "observed contact rows inside captured active EFC extent",
        )
        need(
            block == list(range(block[0], block[0] + width)),
            "observed contact address block contiguous in recorded order",
        )
        need(
            observed[contact_index] == [(world, row) for row in block],
            "exact completed contact-typed row block, no missing or extra backlink",
        )
        expected_type = 5 if condim == 1 else 6
        for row in block:
            need(
                (world, row) not in occupied,
                "unique observed contact-to-row storage association",
            )
            occupied.add((world, row))
            index = world * CONTACT_NJMAX + row
            need(
                ids[index] == contact_index and types[index] == expected_type,
                "complete captured EFC id/type backlink to recorded contact slot",
            )
        rows.append(
            {
                "contact_index": contact_index,
                "world": world,
                "condim": condim,
                "status": "observed-address-backlink",
                "row_indices": block,
                "efc_type": expected_type,
            }
        )
    return {
        "protocol": "microduck-observed-contact-row-view-oct8-v1",
        "phase_source_sha256": DENSE_CONSTRUCTION_SOURCE_SHA256,
        "contact_count": count,
        "per_world_nefc": list(nefc),
        "contacts": rows,
        "joined_row_count": len(occupied),
        "gpu_eligibility_recomputed": False,
        "uncaptured_contact_parameters": [
            "friction",
            "solref",
            "solreffriction",
            "solimp",
        ],
        "raw_packets_unchanged": True,
        "flags": dict(FLAGS),
        "interpretation": (
            "observed storage addresses and captured id/type backlinks only; "
            "not proof of fresh allocation, physical-contact identity, complete "
            "contact-parameter equality, solver equivalence, cause or qualification"
        ),
    }


def compare_linked_contact_construction(
    left_after, left_complete, right_after, right_complete
):
    """Compare a captured row-field subset at original, payload-linked offsets.

    This caller-authenticated, source-pinned side view never rewrites IDs/rows,
    asserts full contact-parameter equality or grants numerical admission.
    Duplicate payloads and contacts without observed rows are not compared.
    """
    payload = compare_active_contact_payloads(left_after, right_after)
    need(
        payload["model_inputs_exact"],
        "same captured model metadata for linked row comparison",
    )
    views = [
        describe_contact_constraint_rows(after, complete)
        for after, complete in (
            (left_after, left_complete),
            (right_after, right_complete),
        )
    ]
    maps = [{row["contact_index"]: row for row in view["contacts"]} for view in views]
    totals = {
        name: {
            "compared_words": 0,
            "differing_words": 0,
            "exact": None,
            "first_difference": None,
        }
        for name in CONSTRUCTION_ROW_FIELDS
    }
    links, unobserved = [], []
    for link in payload["unique_payload_links"]:
        a, b = maps[0][link["left_index"]], maps[1][link["right_index"]]
        if (
            a["status"] != "observed-address-backlink"
            or b["status"] != "observed-address-backlink"
        ):
            unobserved.append(
                {**link, "left_status": a["status"], "right_status": b["status"]}
            )
            continue
        need(
            a["world"] == b["world"] and a["condim"] == b["condim"],
            "linked captured world and local contact row dimension",
        )
        exact_fields = {}
        for name in CONSTRUCTION_ROW_FIELDS:
            words = prod(left_complete[name]["shape"][2:])
            byte_width = 4 * words
            total = totals[name]
            field_exact = True
            for ordinal, (left_row, right_row) in enumerate(
                zip(a["row_indices"], b["row_indices"])
            ):
                left_offset = (a["world"] * CONTACT_NJMAX + left_row) * byte_width
                right_offset = (b["world"] * CONTACT_NJMAX + right_row) * byte_width
                x = left_complete[name]["raw"][left_offset : left_offset + byte_width]
                y = right_complete[name]["raw"][
                    right_offset : right_offset + byte_width
                ]
                need(
                    len(x) == len(y) == byte_width,
                    "whole captured linked construction row",
                )
                different = sum(
                    x[i : i + 4] != y[i : i + 4] for i in range(0, byte_width, 4)
                )
                total["compared_words"] += words
                total["differing_words"] += different
                field_exact = field_exact and different == 0
                if different and total["first_difference"] is None:
                    word = _first_raw_word(x, y)
                    total["first_difference"] = {
                        **link,
                        "world": a["world"],
                        "local_row_ordinal": ordinal,
                        "component": word["word_index"],
                        "left_row": left_row,
                        "right_row": right_row,
                        "left_byte_offset": left_offset + word["word_index"] * 4,
                        "right_byte_offset": right_offset + word["word_index"] * 4,
                        "left_word_le_hex": word["left_word_le_hex"],
                        "right_word_le_hex": word["right_word_le_hex"],
                    }
            exact_fields[name] = field_exact
        links.append(
            {
                **link,
                "world": a["world"],
                "left_rows": list(a["row_indices"]),
                "right_rows": list(b["row_indices"]),
                "row_fields_exact": exact_fields,
            }
        )
    for total in totals.values():
        if total["compared_words"]:
            total["exact"] = total["differing_words"] == 0
    return {
        "protocol": "microduck-linked-construction-row-view-oct8-v1",
        "phase_source_sha256": DENSE_CONSTRUCTION_SOURCE_SHA256,
        "fields": list(CONSTRUCTION_ROW_FIELDS),
        "field_comparisons": totals,
        "compared_payload_links": links,
        "unobserved_payload_links": unobserved,
        "ambiguous_payload_groups": payload["ambiguous_payload_groups"],
        "unmatched_payload_groups": payload["unmatched_payload_groups"],
        "model_inputs_exact": True,
        "captured_state_inputs_exact": {
            name: left_complete[name]["raw"] == right_complete[name]["raw"]
            for name in ("data.qpos", "data.qvel", "data.ctrl")
        },
        "all_construction_drivers_asserted_equal": False,
        "uncaptured_contact_parameters": views[0]["uncaptured_contact_parameters"],
        "prior_solver_fields_excluded": list(SIDE_VIEW_PRIOR_SOLVER_FIELDS),
        "raw_packets_unchanged": True,
        "flags": dict(FLAGS),
        "interpretation": (
            "captured construction-field bytes aligned by unique captured payload "
            "and local row ordinal, at separately retained original offsets; "
            "not complete contact equality, fresh allocation, physical identity, "
            "solver equivalence, cause or qualification; no compared words means null exact"
        ),
    }


def compare_all_observed_construction(
    left_after, left_complete, right_after, right_complete
):
    """Partition active storage rows, not physical constraints or solver inputs.

    Contact rows use unique captured payload links; non-contact rows can only
    match identical type/id markers at the same original world/row. Never search
    for or normalize moved markers. Full coverage means an explicit row-set
    partition for these captured fields, not full solver equivalence.
    """
    contact = compare_linked_contact_construction(
        left_after, left_complete, right_after, right_complete
    )
    counts = [
        list(_int_words(bank["data.nefc"])) for bank in (left_complete, right_complete)
    ]
    need(
        counts[0] == counts[1], "same bounded per-world extents for full row side view"
    )
    active = {
        (world, row) for world, count in enumerate(counts[0]) for row in range(count)
    }
    covered = [set(), set()]
    for link in contact["compared_payload_links"]:
        for arm, key in enumerate(("left_rows", "right_rows")):
            for row in link[key]:
                point = (link["world"], row)
                need(
                    point in active and point not in covered[arm],
                    "unique active contact row coverage",
                )
                covered[arm].add(point)
    types = [_int_words(bank["efc.type"]) for bank in (left_complete, right_complete)]
    totals = {
        name: {
            "compared_words": 0,
            "differing_words": 0,
            "exact": None,
            "first_difference": None,
        }
        for name in CONSTRUCTION_ROW_FIELDS
    }
    matches, unpaired = [], []
    noncontact = set()
    for world, count in enumerate(counts[0]):
        for row in range(count):
            index = world * CONTACT_NJMAX + row
            a, b = types[0][index], types[1][index]
            if a in (5, 6) and b in (5, 6):
                continue
            if not (a in range(5) and b in range(5)) or any(
                left_complete[name]["raw"][index * 4 : (index + 1) * 4]
                != right_complete[name]["raw"][index * 4 : (index + 1) * 4]
                for name in ("efc.id", "efc.type")
            ):
                unpaired.append(
                    {
                        "world": world,
                        "row": row,
                        "left_type": a,
                        "right_type": b,
                        "left_id_word_le_hex": left_complete["efc.id"]["raw"][
                            index * 4 : (index + 1) * 4
                        ].hex(),
                        "right_id_word_le_hex": right_complete["efc.id"]["raw"][
                            index * 4 : (index + 1) * 4
                        ].hex(),
                        "status": "same-offset-markers-unpaired-or-unsupported",
                    }
                )
                continue
            point = (world, row)
            need(
                all(point not in bank for bank in covered),
                "noncontact/contact coverage disjoint",
            )
            noncontact.add(point)
            matches.append({"world": world, "row": row, "type": a})
            for name in CONSTRUCTION_ROW_FIELDS:
                words = prod(left_complete[name]["shape"][2:])
                start, end = index * words * 4, (index + 1) * words * 4
                x, y = (
                    left_complete[name]["raw"][start:end],
                    right_complete[name]["raw"][start:end],
                )
                total = totals[name]
                total["compared_words"] += words
                total["differing_words"] += sum(
                    x[i : i + 4] != y[i : i + 4] for i in range(0, len(x), 4)
                )
                if x != y and total["first_difference"] is None:
                    word = _first_raw_word(x, y)
                    total["first_difference"] = {
                        "world": world,
                        "row": row,
                        "component": word["word_index"],
                        "byte_offset": start + word["word_index"] * 4,
                        "left_word_le_hex": word["left_word_le_hex"],
                        "right_word_le_hex": word["right_word_le_hex"],
                    }
    for total in totals.values():
        if total["compared_words"]:
            total["exact"] = total["differing_words"] == 0
    combined = [bank | noncontact for bank in covered]
    partition = all(bank == active for bank in combined)
    payload_complete = not any(
        contact[name]
        for name in (
            "unobserved_payload_links",
            "ambiguous_payload_groups",
            "unmatched_payload_groups",
        )
    )
    uncovered = {
        label: [
            [world, row]
            for world, count in enumerate(counts[0])
            for row in range(count)
            if (world, row) not in bank
        ]
        for label, bank in zip(("left", "right"), combined)
    }
    return {
        "protocol": "microduck-active-construction-row-partition-oct8-v1",
        "phase_source_sha256": DENSE_CONSTRUCTION_SOURCE_SHA256,
        "active_row_count_per_arm": len(active),
        "contact_rows_covered_per_arm": [len(bank) for bank in covered],
        "same_offset_noncontact_rows_covered": len(noncontact),
        "row_partition_complete": partition,
        "captured_payload_links_complete": payload_complete,
        "full_active_row_coverage": (partition and payload_complete)
        if active
        else None,
        "coverage_status": "no-active-rows"
        if not active
        else "complete-captured-subset"
        if partition and payload_complete
        else "partial-inconclusive",
        "payload_link_exclusion_counts": {
            name: len(contact[name])
            for name in (
                "unobserved_payload_links",
                "ambiguous_payload_groups",
                "unmatched_payload_groups",
            )
        },
        "uncovered_active_row_counts": {
            label: len(rows) for label, rows in uncovered.items()
        },
        "uncovered_active_rows": {
            label: rows[:MAX_ROW_VIEW_SAMPLES] for label, rows in uncovered.items()
        },
        "same_offset_noncontact_matches": matches[:MAX_ROW_VIEW_SAMPLES],
        "unpaired_noncontact_offset_count": len(unpaired),
        "unpaired_noncontact_offsets": unpaired[:MAX_ROW_VIEW_SAMPLES],
        "sample_limit": MAX_ROW_VIEW_SAMPLES,
        "samples_truncated": any(
            len(rows) > MAX_ROW_VIEW_SAMPLES
            for rows in (*uncovered.values(), matches, unpaired)
        ),
        "noncontact_field_comparisons": totals,
        "contact_field_comparisons": contact["field_comparisons"],
        "uncaptured_contact_parameters": contact["uncaptured_contact_parameters"],
        "raw_packets_unchanged": True,
        "flags": dict(FLAGS),
        "interpretation": (
            "explicit active storage row-set partition for the captured eight-field subset; "
            "noncontact markers matched only at identical offsets; partial coverage is "
            "inconclusive; complete equality is consistency, not physical identity, "
            "all-driver equality, solver equivalence, causal proof or qualification"
        ),
    }


def compare_linked_post_forward_load(
    left_after,
    left_complete,
    left_load,
    right_after,
    right_complete,
    right_load,
    *,
    forward,
    load_call,
):
    """Compare retained BAM-boundary loads, never stale construction forces.

    Caller authenticates the same-arm recipe, exact call receipt and source
    schedule. Only the four sampled even-forward/load pairs are permitted.
    Active id/type/count continuity must hold before reading force rows. This
    pure helper neither establishes snapshot provenance nor proves a cause.
    """
    need(
        type(forward) is int
        and type(load_call) is int
        and (forward, load_call) in POST_FORWARD_LOAD_PAIRS,
        "literal predeclared sampled post-forward/load pair",
    )
    payload = compare_active_contact_payloads(left_after, right_after)
    need(
        payload["model_inputs_exact"],
        "same captured model metadata for linked load view",
    )
    views = []
    for after, complete, load in (
        (left_after, left_complete, left_load),
        (right_after, right_complete, right_load),
    ):
        view = describe_contact_constraint_rows(after, complete)
        need(
            type(load) is dict and set(load) == set(SOLVED_LOAD_CARRIERS),
            "exact retained post-forward load carrier bank",
        )
        for name, (shape, dtype) in SOLVED_LOAD_CARRIERS.items():
            value = load[name]
            need(
                type(value) is dict
                and set(value) == {"shape", "dtype", "raw"}
                and type(value["shape"]) is list
                and all(type(n) is int for n in value["shape"])
                and tuple(value["shape"]) == shape
                and type(value["dtype"]) is str
                and value["dtype"] == dtype
                and type(value["raw"]) is bytes
                and len(value["raw"]) == 4 * prod(shape),
                "literal retained post-forward load carrier: " + name,
            )
        counts = _int_words(load["data.nefc"])
        need(
            list(counts) == view["per_world_nefc"],
            "same captured post-forward load row counts",
        )
        for world, count in enumerate(counts):
            start, end = world * CONTACT_NJMAX * 4, (world * CONTACT_NJMAX + count) * 4
            for name in ("efc.id", "efc.type"):
                need(
                    complete[name]["raw"][start:end] == load[name]["raw"][start:end],
                    "exact active captured construction-to-load id/type continuity",
                )
        views.append(view)
    maps = [{row["contact_index"]: row for row in view["contacts"]} for view in views]
    links, unobserved = [], []
    count, differences, first = 0, 0, None
    for link in payload["unique_payload_links"]:
        a, b = maps[0][link["left_index"]], maps[1][link["right_index"]]
        if (
            a["status"] != "observed-address-backlink"
            or b["status"] != "observed-address-backlink"
        ):
            unobserved.append(
                {**link, "left_status": a["status"], "right_status": b["status"]}
            )
            continue
        need(
            a["world"] == b["world"] and a["condim"] == b["condim"],
            "linked captured world and local load row dimension",
        )
        exact = True
        for ordinal, (left_row, right_row) in enumerate(
            zip(a["row_indices"], b["row_indices"])
        ):
            left_offset = (a["world"] * CONTACT_NJMAX + left_row) * 4
            right_offset = (b["world"] * CONTACT_NJMAX + right_row) * 4
            x = left_load["efc.force"]["raw"][left_offset : left_offset + 4]
            y = right_load["efc.force"]["raw"][right_offset : right_offset + 4]
            count += 1
            if x != y:
                differences += 1
                exact = False
                if first is None:
                    first = {
                        **link,
                        "world": a["world"],
                        "local_row_ordinal": ordinal,
                        "left_row": left_row,
                        "right_row": right_row,
                        "left_byte_offset": left_offset,
                        "right_byte_offset": right_offset,
                        "left_word_le_hex": x.hex(),
                        "right_word_le_hex": y.hex(),
                    }
        links.append(
            {
                **link,
                "world": a["world"],
                "left_rows": a["row_indices"],
                "right_rows": b["row_indices"],
                "load_force_exact": exact,
            }
        )
    x, y = (
        left_load["data.qfrc_constraint"]["raw"],
        right_load["data.qfrc_constraint"]["raw"],
    )
    aggregate = _first_raw_word(x, y)
    if aggregate is not None:
        aggregate = {
            **aggregate,
            "world": aggregate["word_index"] // 20,
            "dof": aggregate["word_index"] % 20,
        }
    return {
        "protocol": "microduck-linked-post-forward-load-view-oct8-v1",
        "forward": forward,
        "load_call": load_call,
        "active_id_type_count_continuity_checked": True,
        "force_phase": (
            "scheduled post-forward BAM load observation under caller-authenticated "
            "receipt; not construction efc.force"
        ),
        "snapshot_provenance_established_by_helper": False,
        "observed_linked_force_words": count,
        "differing_force_words": differences,
        "force_exact": differences == 0 if count else None,
        "first_force_difference": first,
        "compared_payload_links": links,
        "unobserved_payload_links": unobserved,
        "ambiguous_payload_groups": payload["ambiguous_payload_groups"],
        "unmatched_payload_groups": payload["unmatched_payload_groups"],
        "aggregate_qfrc_constraint": {
            "compared_words": 64 * 20,
            "differing_words": sum(
                x[i : i + 4] != y[i : i + 4] for i in range(0, len(x), 4)
            ),
            "exact": x == y,
            "first_difference": aggregate,
            "contact_isolated": False,
        },
        "uncaptured_contact_parameters": views[0]["uncaptured_contact_parameters"],
        "raw_packets_unchanged": True,
        "flags": dict(FLAGS),
        "interpretation": (
            "retained load bytes at observed payload-linked row offsets with active "
            "id/type/count continuity; caller must authenticate exact recipe timing; "
            "not snapshot provenance, physical identity, fresh allocation, complete "
            "solver equivalence, cause or qualification; aggregate load spans constraint families"
        ),
    }


def _nonoverlapping(layouts, label):
    ranges = sorted(
        (start, end, name) for name, (start, end) in layouts.items() if end > start
    )
    need(
        all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])),
        "nonoverlapping raw storage spans " + label,
    )


def contact_boundary(record, raw, arm, device, compiled_role, friction_entries, recipe):
    """Validate all seven passive contact-call and pre-solver snapshots."""
    need(
        type(record) is dict
        and set(record)
        == {"protocol", "arm", "entries", "packet_count", "captured_bytes", "flags"}
        and record["protocol"] == BOUNDARY_CONTROL_PROTOCOL
        and record["arm"] == arm,
        "complete per-arm contact boundary receipt",
    )
    component.flags(record["flags"])
    entries = record["entries"]
    need(
        type(entries) is list
        and len(entries) == BOUNDARY_FORWARDS
        and record["packet_count"] == BOUNDARY_FORWARDS * 3,
        "exact seven sampled forwards and 21 primary packets per arm",
    )
    need(
        type(record["captured_bytes"]) is int
        and 0 <= record["captured_bytes"] <= MAX_BOUNDARY_ARM_BYTES,
        "bounded per-arm passive capture bytes",
    )
    result, stable_contact, stable_complete, total = [], None, None, 0
    identities = None
    packet_fields = {}
    for forward, row in enumerate(entries):
        need(
            type(row) is dict
            and set(row)
            == {
                "forward",
                "phase",
                "contact_call",
                "contact",
                "complete_layouts",
                "contact_before",
                "contact_after",
                "complete",
                "complete_fields",
                "context_fields",
                "stale_prior_fields",
            },
            "exact sampled boundary entry schema",
        )
        need(
            type(row["forward"]) is int
            and row["forward"] == forward
            and row["phase"] == "construction-complete-BEFORE-solver"
            and row["context_fields"]
            == ["context.geomcollisionid", "context.pos", "context.frame"]
            and row["stale_prior_fields"] == list(STALE_PRIOR_FIELDS),
            "literal sampled phase, explicit context, and stale solver-storage labels",
        )
        call = row["contact_call"]
        need(
            type(call) is dict
            and set(call)
            == {
                "kernel_key",
                "kernel_object_id",
                "kernel_function_id",
                "kernel_code_id",
                "module_object_id",
                "factory_object_id",
                "factory_function_id",
                "factory_code_id",
                "cone",
                "is_sparse",
                "dim",
                "njmax",
                "njmax_nnz",
                "input_count",
                "output_count",
            },
            "contact cached-kernel, exact ABI, and actual call identity witness",
        )
        for key in (
            "factory_object_id",
            "factory_function_id",
            "factory_code_id",
            "kernel_object_id",
            "kernel_function_id",
            "kernel_code_id",
            "module_object_id",
        ):
            plain(call[key], 1)
        for key in ("cone", "dim", "njmax", "njmax_nnz", "input_count", "output_count"):
            plain(call[key], 0)
        call_ids = {
            key: call[key]
            for key in (
                "factory_object_id",
                "factory_function_id",
                "factory_code_id",
                "kernel_object_id",
                "kernel_function_id",
                "kernel_code_id",
                "module_object_id",
            )
        }
        need(
            type(call["kernel_key"]) is str
            and "efc_contact_init" in call["kernel_key"]
            and call["kernel_object_id"] == compiled_role["kernel_object_id"]
            and call["module_object_id"] == compiled_role["module_object_id"]
            and call["cone"] == 0
            and type(call["cone"]) is int
            and call["is_sparse"] is False
            and call["dim"] == CONTACT_CAPACITY
            and call["njmax"] == CONTACT_NJMAX
            and call["njmax_nnz"] == CONTACT_NJMAX_NNZ
            and call["input_count"] == 18
            and call["output_count"] == 6,
            "contact factory/code role and exact frozen18/6 ABI invocation",
        )
        scope_call = recipe["control_scope"]["calls"][forward]
        friction = friction_entries[forward]
        need(
            friction["identities"]["model_object_id"] == scope_call["model_id"]
            and friction["identities"]["data_object_id"] == scope_call["data_id"]
            and friction["stream"] == device["stream"]
            and friction["dim"] == [64, 20],
            "contact and friction witnesses refer to the same forward/model/data/stream",
        )
        need(
            identities is None or identities == call_ids,
            "stable contact factory/kernel/model/data identities per arm",
        )
        identities = call_ids

        contact_layouts = row["contact"]
        complete_layouts = row["complete_layouts"]
        need(
            type(contact_layouts) is dict
            and set(contact_layouts) == set(CONTACT_PACKET_ORDER)
            and type(complete_layouts) is dict
            and set(complete_layouts) == set(COMPLETE_ORDER),
            "complete raw contact and construction-complete layouts",
        )
        packet_records = {
            "contact_before": (row["contact_before"], CONTACT_PACKET_ORDER),
            "contact_after": (row["contact_after"], CONTACT_PACKET_ORDER),
            "complete": (row["complete"], COMPLETE_ORDER),
        }
        packet_suffixes = {
            "contact_before": "contact.before",
            "contact_after": "contact.after",
            "complete": "complete",
        }
        decoded = {}
        for key, (packet, order) in packet_records.items():
            expected_path = (
                f"boundary/{arm}/forward-{forward:02d}.{packet_suffixes[key]}.bin"
            )
            need(type(packet) is dict, "plain boundary packet record")
            if key.startswith("contact_"):
                need(
                    set(packet)
                    == {"path", "bytes", "sha256", "fields", "context_fields"}
                    and packet["path"] == expected_path
                    and packet["context_fields"] == list(CONTACT_CONTEXT_ORDER),
                    "contact packet explicitly separates raw context fields",
                )
                packet = {
                    name: packet[name] for name in ("path", "bytes", "sha256", "fields")
                }
            else:
                need(
                    set(packet) == {"path", "bytes", "sha256"}
                    and packet["path"] == expected_path
                    and type(row["complete_fields"]) is dict,
                    "complete packet exact record schema",
                )
                packet = packet | {"fields": row["complete_fields"]}
            values = unpack_packet(packet, raw, order=order)
            expected_carriers = (
                CONTACT_CARRIERS if key.startswith("contact_") else COMPLETE_CARRIERS
            )
            need(
                all(
                    (tuple(carrier["shape"]), carrier["dtype"])
                    == expected_carriers[name]
                    for name, carrier in values.items()
                ),
                "literal complete-capacity shape and dtype schema " + key,
            )
            decoded[key] = values
            total += packet["bytes"]
            layouts = (
                contact_layouts if key.startswith("contact_") else complete_layouts
            )
            for name, carrier in values.items():
                start, end = layout(layouts[name], carrier, device)
            need(end >= start, "valid packet storage range")
        contact_carriers = decoded["contact_before"]
        after_carriers = decoded["contact_after"]
        complete_carriers = decoded["complete"]
        need(
            all(
                contact_carriers[name]["raw"] == after_carriers[name]["raw"]
                for name in CONTACT_INPUT_ORDER + CONTACT_CONTEXT_ORDER
            ),
            "full ABI inputs and explicitly separate context unchanged by allocation",
        )
        _nonoverlapping(
            {
                name: layout(contact_layouts[name], contact_carriers[name], device)
                for name in CONTACT_PACKET_ORDER
            },
            "contact invocation",
        )
        _nonoverlapping(
            {
                name: layout(complete_layouts[name], complete_carriers[name], device)
                for name in COMPLETE_ORDER
            },
            "construction-complete bank",
        )
        aliases = (
            ("data.nefc", "data.nefc"),
            ("efc.id", "efc.id"),
            ("efc.J_rownnz", "efc.J_rownnz"),
            ("efc.J_rowadr", "efc.J_rowadr"),
            ("local.efc_nnz", "local.efc_nnz"),
        )
        for contact_name, complete_name in aliases:
            need(
                contact_layouts[contact_name] == complete_layouts[complete_name],
                "same actual output allocation across call and completed bank: "
                + contact_name,
            )
        need(
            contact_layouts["local.efc_nnz"]
            == friction["layouts"]["bank.efc_nnz"]
            == complete_layouts["local.efc_nnz"],
            "same actual local NNZ scratch across friction, contact, and completion",
        )
        stable_c = {k: v for k, v in contact_layouts.items() if k != "local.efc_nnz"}
        stable_f = {k: v for k, v in complete_layouts.items() if k != "local.efc_nnz"}
        need(
            stable_contact is None or stable_contact == stable_c,
            "stable actual contact storage across sampled forwards",
        )
        need(
            stable_complete is None or stable_complete == stable_f,
            "stable actual completed-bank storage across sampled forwards",
        )
        stable_contact, stable_complete = stable_c, stable_f
        need(
            sum(len(complete_carriers[name]["raw"]) for name in EFC_FIELDS) == EFC_BYTES
            and tuple(row["stale_prior_fields"]) == STALE_PRIOR_FIELDS,
            "entire stale/current EFC capacity retained and stale fields labeled",
        )
        result.append(
            {
                "forward": forward,
                "contact_call": call,
                "contact_before": contact_carriers,
                "contact_after": after_carriers,
                "complete": complete_carriers,
                "packet_hashes": {
                    key: packet["sha256"]
                    for key, (packet, _order) in packet_records.items()
                },
            }
        )
        packet_fields[forward] = decoded
    need(
        total == record["captured_bytes"] and total <= MAX_BOUNDARY_ARM_BYTES,
        "exact bounded per-arm boundary packet byte total",
    )
    return {"entries": result, "packet_fields": packet_fields, "captured_bytes": total}


def compare_contact_boundaries(arms):
    """Separate whole-capacity raw timing from active-extent byte comparison."""
    fields = {
        "contact_before": CONTACT_PACKET_ORDER,
        "contact_after": CONTACT_PACKET_ORDER,
        "complete": COMPLETE_ORDER,
    }
    first = None
    first_active = None
    pair_exact = True
    active_pair_exact = True
    summaries = []
    left_rows = arms["candidate0"]["entries"]
    right_rows = arms["candidate1"]["entries"]
    for forward, (left, right) in enumerate(zip(left_rows, right_rows)):
        for stage, order in fields.items():
            left_fields = arms["candidate0"]["packet_fields"][forward][stage]
            right_fields = arms["candidate1"]["packet_fields"][forward][stage]
            stage_equal = all(
                left_fields[name]["raw"] == right_fields[name]["raw"] for name in order
            )
            pair_exact &= stage_equal
            if stage_equal:
                active_delta = _active_first_difference(
                    left_fields, right_fields, stage
                )
                active_pair_exact &= active_delta is None
                if first_active is None and active_delta is not None:
                    first_active = {
                        "forward": forward,
                        "stage": stage,
                        **active_delta,
                    }
                continue
            active_delta = _active_first_difference(left_fields, right_fields, stage)
            active_pair_exact &= active_delta is None
            if first_active is None and active_delta is not None:
                first_active = {
                    "forward": forward,
                    "stage": stage,
                    **active_delta,
                }
            for name in order:
                a, b = left_fields[name]["raw"], right_fields[name]["raw"]
                difference = _first_raw_word(a, b)
                if difference is not None:
                    summary = {
                        "forward": forward,
                        "stage": stage,
                        "field": name,
                        **difference,
                        "left_field_sha256": sha256(a).hexdigest(),
                        "right_field_sha256": sha256(b).hexdigest(),
                    }
                    summaries.append(summary)
                    if first is None:
                        first = summary
                    break
    return {
        "candidate0_candidate1_exact_through_sampled_boundaries": pair_exact,
        "first_raw_capacity_difference": first,
        "first_active_extent_difference": first_active,
        "active_extent_repeat_exact": active_pair_exact,
        "changed_packet_fields_first_per_forward_stage": summaries,
        "active_extent_definition": {
            "contact": "nacon prefix; per-world efc.id prefix uses pre/post nefc",
            "construction_complete": "per-world EFC rows below authenticated nefc",
            "other_fields": "complete retained field carrier",
            "raw_packets_unchanged": True,
        },
        "flags": dict(FLAGS),
        "interpretation": (
            "whole-capacity raw-byte timing and bounded active-extent comparison only; "
            "neither a physical-cause proof nor qualification"
        ),
    }


def expected_capture_plan():
    return {
        "protocol": BOUNDARY_CONTROL_PROTOCOL,
        "forwards": list(range(BOUNDARY_FORWARDS)),
        "arms": list(ARMS),
        "packet_count_per_arm": 21,
        "max_bytes_per_arm": MAX_BOUNDARY_ARM_BYTES,
        "max_bytes_all_arms": MAX_BOUNDARY_BYTES,
        "contact_input_order": list(CONTACT_INPUT_ORDER),
        "contact_output_order": list(CONTACT_OUTPUT_ORDER),
        "context_order": list(CONTACT_CONTEXT_ORDER),
        "complete_order": list(COMPLETE_ORDER),
        "stale_prior_fields": list(STALE_PRIOR_FIELDS),
        "phase": "construction-complete-BEFORE-solver",
        "flags": dict(FLAGS),
    }


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
        Id=f"microduck-contact-boundary-tick-{mode}-{source[:12]}.service",
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


def cpu_thread_settings(proof):
    from mjlab_microduck import stance_contact_boundary_probe as producer

    need(
        type(proof) is dict
        and type(proof.get("cpu_test_threads")) is dict
        and proof["cpu_test_threads"] == producer.CPU_TEST_THREADS,
        "exact CPU-only child thread settings",
    )


def cpu_proof(declaration, raw, expected_source, expected_tests_sha):
    from mjlab_microduck import stance_contact_boundary_probe as producer

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
        cpu_thread_settings(proof)
        component.source(proof["source_binding"], expected_source)
        component.flags(proof["flags"])
        need(
            proof["packages"] == pins.VERSIONS
            and type(proof["files"]) is list
            and len(proof["files"]) == len(set(proof["files"])) == 94
            and sha256(canonical(proof["files"])).hexdigest()
            == producer.TEST_FILES_SHA256
            and type(proof["tests"]) is int
            and proof["tests"] == producer.EXPECTED_TESTS > 0,
            "exact94-file current-source CPU proof",
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
    from mjlab_microduck import stance_contact_boundary_probe as producer

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


def caller_rng_receipt(value):
    need(
        type(value) is dict
        and set(value) == {"seeds", "states"}
        and type(value["seeds"]) is dict
        and value["seeds"] == {"cpu": 673, "cuda": 677}
        and all(type(seed) is int for seed in value["seeds"].values())
        and type(value["states"]) is dict
        and set(value["states"]) == {"cpu", "cuda"},
        "explicit fresh caller RNG setup",
    )
    for name, state in value["states"].items():
        need(
            type(state) is dict and set(state) == {"bytes", "sha256"},
            "whole caller RNG anchor",
        )
        plain(state["bytes"], 1, 16384)
        sha(state["sha256"])
        if name == "cpu":
            need(
                state["bytes"] == 5056
                and state["sha256"] == rng_checks.CPU_SEED_HASHES["caller_cpu_before"],
                "literal seeded CPU caller anchor",
            )
    return value["states"]


def _expected_thread_phase_order():
    phases = [
        ("torch-import-start", None),
        ("torch-import-done", None),
        ("caller-rng-initialized", None),
        ("warp-init-start", None),
        ("warp-init-done", None),
    ]
    for role in ("original", "candidate", "contact"):
        for phase in (
            "compile-start",
            "compile-done",
            "load-start",
            "load-done",
            "hooks-start",
            "hooks-done",
            "bind-start",
            "bind-done",
        ):
            phases.append((phase, role))
    phases.extend(
        (("three-explicit-loads-complete", None), ("contact-factory-checked", None))
    )
    for arm in ARMS:
        phases.extend(
            (
                ("arm-start", arm),
                ("recipe-case-start", arm),
                ("recipe-case-done", arm),
            )
        )
    need(len(phases) == 40, "literal forty-phase thread diagnostic sequence")
    return phases


def _thread_settings(value, expected, label):
    need(
        type(value) is dict
        and set(value) == set(THREAD_ENV_REQUIRED)
        and value == expected
        and all(type(key) is str and type(item) is str for key, item in value.items()),
        "exact literal thread settings " + label,
    )
    return value


def _thread_snapshot(value, role, pid, settings, *, one_thread=False):
    need(
        type(value) is dict
        and set(value) == {"role", "pid", "settings", "threads"}
        and value["role"] == role,
        "exact thread observation schema and role",
    )
    plain(value["pid"], 1)
    need(value["pid"] == pid, "thread observation process identity")
    _thread_settings(value["settings"], settings, role)
    threads = plain(value["threads"], 1, 64)
    need(not one_thread or threads == 1, "single-thread pre-import observation")
    return value


def _thread_phase_records(raw_child_log, child_pid, child_settings):
    need(
        type(raw_child_log) is bytes and 0 < len(raw_child_log) <= 16 * 1024**2,
        "whole authenticated bounded child log for thread proof",
    )
    marker = (DATA_PROTOCOL + ":phase").encode()
    records = []
    for line in raw_child_log.splitlines(keepends=True):
        phase_like = marker in line or b'"phase"' in line
        if not phase_like:
            continue
        need(
            line.endswith(b"\n") and line.count(b"\n") == 1 and len(line) <= 1024,
            "each phase record is one bounded newline-terminated child-log line",
        )
        record = decode(line[:-1])
        need(canonical(record) == line, "canonical compact child phase JSON")
        need(
            type(record) is dict
            and set(record)
            == {
                "protocol",
                "phase",
                "role",
                "pid",
                "threads",
                "observed_cpu_thread_env",
            }
            and record["protocol"] == DATA_PROTOCOL + ":phase",
            "exact diagnostic phase record schema and protocol",
        )
        plain(record["pid"], 1)
        need(record["pid"] == child_pid, "phase emitted by the owned child")
        plain(record["threads"], 1, 64)
        _thread_settings(
            record["observed_cpu_thread_env"], child_settings, "child phase"
        )
        records.append(record)
    expected = _expected_thread_phase_order()
    observed = [(record["phase"], record["role"]) for record in records]
    need(
        len(records) == len(expected) == 40 and observed == expected,
        "all forty child phases present exactly once and in declared order",
    )
    return records


def thread_budget_proof(declaration, child, owner, rawChildLog):
    """Validate bounded owner/child thread settings and their authenticated timeline."""
    from mjlab_microduck import stance_contact_boundary_probe as producer

    need(
        producer.OWNER_THREAD_ENV == THREAD_ENV_REQUIRED
        and producer.CUDA_CHILD_THREAD_ENV == THREAD_ENV_REQUIRED,
        "literal four-variable owner and CUDA-child thread environment maps",
    )
    expected = {
        "owner": dict(THREAD_ENV_REQUIRED),
        "cuda_child": dict(THREAD_ENV_REQUIRED),
    }
    owner_pid = plain(declaration["owner_pid"], 1)
    child_pid = plain(child["child_pid"], 1)
    need(
        owner["owner_pid"] == child["owner_pid"] == owner_pid
        and owner["child_pid"] == child_pid
        and child_pid != owner_pid,
        "thread evidence attached to the same distinct owner and child",
    )
    for record in (declaration, child, owner):
        need(
            type(record["thread_budget"]) is dict
            and set(record["thread_budget"]) == {"owner", "cuda_child"},
            "exact owner/child thread budget schema",
        )
        _thread_settings(
            record["thread_budget"]["owner"], expected["owner"], "owner budget"
        )
        _thread_settings(
            record["thread_budget"]["cuda_child"],
            expected["cuda_child"],
            "child budget",
        )
        need(record["thread_budget"] == expected, "identical literal thread budgets")

    owner_start = _thread_snapshot(
        declaration["owner_thread_start"],
        "owner",
        owner_pid,
        expected["owner"],
        one_thread=True,
    )
    owner_observations = owner["thread_observations"]
    need(
        type(owner_observations) is dict
        and set(owner_observations) == {"pre_import", "after_child"},
        "exact owner thread observation boundaries",
    )
    owner_pre = _thread_snapshot(
        owner_observations["pre_import"],
        "owner",
        owner_pid,
        expected["owner"],
        one_thread=True,
    )
    need(owner_pre == owner_start, "declared owner startup equals retained observation")
    owner_after = _thread_snapshot(
        owner_observations["after_child"], "owner", owner_pid, expected["owner"]
    )

    child_observations = child["thread_observations"]
    need(
        type(child_observations) is dict
        and set(child_observations)
        == {"pre_import", "after_warp_init", "after_recipe"},
        "exact child thread observation boundaries",
    )
    child_pre = _thread_snapshot(
        child_observations["pre_import"],
        "cuda_child",
        child_pid,
        expected["cuda_child"],
        one_thread=True,
    )
    child_warp = _thread_snapshot(
        child_observations["after_warp_init"],
        "cuda_child",
        child_pid,
        expected["cuda_child"],
    )
    child_recipe = _thread_snapshot(
        child_observations["after_recipe"],
        "cuda_child",
        child_pid,
        expected["cuda_child"],
    )

    phases = _thread_phase_records(rawChildLog, child_pid, expected["cuda_child"])
    by_key = {(record["phase"], record["role"]): record for record in phases}
    for snapshot, key in (
        (child_pre, ("torch-import-start", None)),
        (child_warp, ("warp-init-done", None)),
        (child_recipe, ("recipe-case-done", "candidate1")),
    ):
        phase = by_key[key]
        need(
            snapshot
            == {
                "role": "cuda_child",
                "pid": phase["pid"],
                "settings": phase["observed_cpu_thread_env"],
                "threads": phase["threads"],
            },
            "snapshot derived from its exact written phase observation",
        )
    return {
        "thread_budget": expected,
        "phase_count": len(phases),
        "phase_order": [
            {"phase": record["phase"], "role": record["role"]} for record in phases
        ],
        "owner_threads": {
            "pre_import": owner_pre["threads"],
            "after_child": owner_after["threads"],
        },
        "child_threads": {
            "pre_import": child_pre["threads"],
            "after_warp_init": child_warp["threads"],
            "after_recipe": child_recipe["threads"],
        },
    }


def verify_run(
    directory,
    inventory,
    *,
    expected_source,
    expected_tests_sha,
    historical_root=None,
):
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
                "negative_dependency",
                "capture_plan",
                "case_order",
                "caller_rng_seeds",
                "started_utc_ns",
                "cutoff_utc",
                "child_timeout_seconds",
                "thread_budget",
                "owner_thread_start",
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
                "caller_rng",
                "thread_budget",
                "thread_observations",
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
                "thread_budget",
                "thread_observations",
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
    thread_proof = thread_budget_proof(declaration, child, owner, raw["child.log"])
    need(
        declaration["case_order"] == list(ARMS)
        and declaration["cutoff_utc"] == "2026-10-07T23:30:00Z"
        and type(declaration["caller_rng_seeds"]) is dict
        and declaration["caller_rng_seeds"] == {"cpu": 673, "cuda": 677}
        and all(type(seed) is int for seed in declaration["caller_rng_seeds"].values()),
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
    negative = verify_historical_negative(
        declaration["negative_dependency"], historical_root=historical_root
    )
    need(
        declaration["capture_plan"] == expected_capture_plan(),
        "exact passive contact capture plan, phase, packet order and caps",
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
        / ("contact-boundary-tick-run-" + expected_source["source"][:12])
    )
    need(
        set(child["compiled"]) == {"original", "candidate", "contact"},
        "three fresh loaded modules",
    )
    component.compiled(
        {key: child["compiled"][key] for key in ("original", "candidate")},
        raw,
        inventory,
        native_directory,
        device,
    )
    contact_role = compiled_contact(
        child["compiled"]["contact"], raw, inventory, native_directory, device
    )
    need(
        contact_role["runtime_entries"]
        == child["compiled"]["original"]["runtime_entries"]
        == child["compiled"]["candidate"]["runtime_entries"],
        "same frozen Warp runtime entries across friction and contact modules",
    )
    arms = child["arms"]
    caller_states = caller_rng_receipt(child["caller_rng"])
    need(type(arms) is dict and set(arms) == set(ARMS), "exact fresh three-arm records")
    entries, packets = {}, {}
    for arm in ARMS:
        need(
            type(arms[arm]) is dict
            and set(arms[arm]) == {"recipe", "observer", "packets", "boundary"},
            "complete arm schema",
        )
        packets[arm] = recipe(arms[arm]["recipe"], arms[arm]["packets"], raw, arm)
        need(
            all(
                arms[arm]["recipe"]["rng_metadata"][f"caller_{dev}_{phase}"]
                == caller_states[dev]
                for dev in ("cpu", "cuda")
                for phase in ("before", "after")
            ),
            "actual seeded caller streams preserved across each fresh arm",
        )
        entries[arm] = observer(
            arms[arm]["observer"],
            raw,
            arm,
            device,
            child["compiled"],
            arms[arm]["recipe"],
        )
    boundary_reports = {}
    boundary_total = 0
    for arm in ARMS:
        boundary_reports[arm] = contact_boundary(
            arms[arm]["boundary"],
            raw,
            arm,
            device,
            contact_role,
            entries[arm],
            arms[arm]["recipe"],
        )
        boundary_total += boundary_reports[arm]["captured_bytes"]
    need(
        boundary_total <= MAX_BOUNDARY_BYTES,
        "bounded aggregate passive contact capture across three arms",
    )
    boundary_comparison = compare_contact_boundaries(boundary_reports)
    report = numerical.audit_entries(numerical_entry_projection(entries))
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
        "decision": "passive-contact-boundary-capture-complete",
        "recipe_candidate_repeat_exact": passed,
        "synthetic_dependency": dependency,
        "historical_negative": negative,
        "thread_budget_proof": thread_proof,
        "passive_contact_boundary": {
            "capture_plan": declaration["capture_plan"],
            "arms": {
                arm: {
                    "entries": BOUNDARY_FORWARDS,
                    "captured_bytes": boundary_reports[arm]["captured_bytes"],
                }
                for arm in ARMS
            },
            "captured_bytes": boundary_total,
            "candidate_comparison": boundary_comparison,
            "flags": dict(FLAGS),
        },
        "entries": 63,
        "proposals": 30,
        "numerical": report,
        "recipe_packets_repeat": repeat,
        "recipe_state_repeat": recipe_repeat,
        "matched_original_entries_exact": matched_entries_exact,
        "flags": FLAGS,
    }
