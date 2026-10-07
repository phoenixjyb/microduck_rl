"""Independent pure-Python audit of the ten-case dense-friction prefix packet.

Whole-file authentication is deliberately the caller's responsibility.  This
module accepts already parsed plain JSON data and never imports Warp or the
fixture that produced the packet.
"""

from hashlib import sha256
import json
import math
import struct


PROTOCOL = "microduck-dense-friction-prefix-packet-audit-oct7-v1"
PACKET_PROTOCOL = "microduck-dense-friction-prefix-ten-case-cpu-matrix-oct7-v1"
BASE_SHA256 = "8e96c5077f248aad06cacab4c3a45f17cd6b3c35c09c7a46457281916e8046d9"
PREFIX_SHA256 = "0ff83581832f3264d9f6b447b089fee285f845f12a5a4eb3614a057b82ba0bad"
TREE_SHA256 = "188d58bfac6ab54e51adfc80a6c504d782aec43856cbb15e54b4109f0ce30a1d"
CONSTRAINT_SHA256 = "b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53"

_CASES = {
    "all-dofs": (
        1,
        20,
        (0,),
        "f8712ec70a1d25d707e8d34c50fdd8cf4b5dfdef0231e58180ba77e749ce8622",
        "55b52d16c70792ac4e857cc9879f040a1ed8276a3f342974902a5b8e41da2a24",
    ),
    "append-exact-fill": (
        3,
        5,
        (2, 2, 2),
        "baaed53bbec568b0922fe67aaa799927dbb0c2be575f8ad830b0ce90ee40215f",
        "cbd69ea0b59f277ad77b0d1dbdd290c8967c5ae82ec1bc88a4f0de7fe7d4cda2",
    ),
    "broadcast": (
        3,
        32,
        (0, 2, 5),
        "a0fe67f72739f028a8b8d1bb1f81903a76db0684a75f69e0f7a358c86fe7fcec",
        "71013a1a1e6e47fd8dcf1c43b4a587e26262489a8bb87c6e701ac86b5519c2a0",
    ),
    "direct-solref": (
        2,
        32,
        (4, 8),
        "7b0d86cb04e816f0fcf2159d1e76554d9c8e892025288df00bfd0fd9785e78a0",
        "dcf870049764349cb40bdb2d3f7bd9967c48ea9e84c84d40c51f36b823345867",
    ),
    "empty": (
        1,
        32,
        (32,),
        "c71c28239a020fe542461bbe8e1dd50fcc17c76534d752d9ac2f006eb13b1b9b",
        "e2c1d58caaa681b6eb920781ddaee47bc978902ffa2a9e14fb80e4319a96530e",
    ),
    "maximum": (
        4,
        32,
        (0, 1, 16, 32),
        "7577ee43cc3add744f3609b8f7b8c0aea1604238f547b718f871cc67f1e1530e",
        "8e5926181b48988afad932f65fbd596c8c2d32341f92c18ca7c67222bef5e686",
    ),
    "mixed-broadcast": (
        4,
        32,
        (1, 2, 3, 4),
        "8fb6a29895edd3c6d3e6f7dd75383a575a3c96e4fa6b3280989d0bafea97407d",
        "517d66564b7e716500ba97fef7b59cff7f4432d31823147f0a63058a5357b2f1",
    ),
    "overflow": (
        2,
        2,
        (1, 2),
        "c9425d43717ad6d3ff2ba7a34cd351dcd2c191aa5c42b53d4f168ad0791eff2f",
        "7903bd1350024fab84ca4ed9661311adb53b2eb42c218538234bc98a9d0a2353",
    ),
    "per-world": (
        4,
        32,
        (0, 2, 5, 8),
        "a31f29c4fe2fd01fbedfc0619977e97dc54340c1fb7f9b83482c1d0ca3ea004f",
        "4da5734981fa0a8df5022788d1205bc4866ed8cffc0482722724d18c1541c1d0",
    ),
    "signed-zero": (
        1,
        32,
        (2,),
        "6c71587de42bb814baadff3f932b14bcd98e513d2a9878464a4777017889c8c8",
        "f712b046015cf86cbf58ffd7f78db31b6833b6ad46e87df3f3f3cea25c2c08b6",
    ),
}
_BANK_DTYPES = {
    "nf": "i32",
    "nefc": "i32",
    "type": "i32",
    "id": "i32",
    "row_nnz": "i32",
    "row_adr": "i32",
    "col_ind": "i32",
    "J": "u32",
    "pos": "u32",
    "margin": "u32",
    "D": "u32",
    "vel": "u32",
    "aref": "u32",
    "frictionloss": "u32",
    "efc_nnz": "i32",
}
_ROW_FIELDS = ("type", "id", "J", "pos", "margin", "D", "vel", "aref", "frictionloss")
_FLOAT_FIELDS = ("J", "pos", "margin", "D", "vel", "aref", "frictionloss")
_INPUT_NAMES = ("frictionloss", "qvel", "invweight", "solref", "solimp", "timestep")
_TOP_FLAGS = (
    "full_window_qualified",
    "native_qualified",
    "physical_acceptance",
    "runtime_cause_proven",
    "training_authorized",
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _plain_int(value, low, high, message):
    _require(type(value) is int and low <= value <= high, message)
    return value


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _bits_array(value, shape, dtype, message):
    _require(
        type(value) is dict and set(value) == {"shape", "dtype", "bits"},
        message + " record",
    )
    _require(
        type(value["shape"]) is list and all(type(x) is int for x in value["shape"]),
        message + " plain shape integers",
    )
    _require(
        value["shape"] == list(shape) and value["dtype"] == dtype,
        message + " shape/dtype",
    )
    bits = value["bits"]
    size = math.prod(shape)
    _require(
        type(bits) is list and len(bits) == size, message + " complete flat payload"
    )
    if dtype == "<u4":
        for bit in bits:
            _plain_int(bit, 0, 0xFFFFFFFF, message + " u32")
    else:
        for bit in bits:
            _plain_int(bit, -(1 << 31), (1 << 31) - 1, message + " i32")
    return bits


def _float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _source_binding(value):
    _require(type(value) is dict, "source binding object")
    _require(
        value.get("fixture_module_sha256") == BASE_SHA256, "pinned base fixture source"
    )
    _require(
        value.get("constraint_sha256") == CONSTRAINT_SHA256, "pinned constraint source"
    )
    _require(
        type(value.get("python_tree_files")) is int
        and value.get("python_tree_sha256") == TREE_SHA256
        and value.get("python_tree_files") == 69,
        "pinned package tree",
    )
    _require(value.get("fixture_module_sha256") == BASE_SHA256, "base fixture hash")
    platform = value.get("platform")
    _require(
        type(platform) is dict and set(platform) == {"system", "machine"},
        "literal CPU platform record",
    )
    envelope = (
        value.get("python"),
        platform["system"],
        platform["machine"],
    )
    _require(
        envelope in {("3.12.12", "Darwin", "arm64"), ("3.12.13", "Linux", "x86_64")},
        "one of two frozen CPU environments",
    )
    _require(
        value.get("warp_lang") == "1.12.0" and value.get("mujoco_warp") == "3.8.1",
        "frozen Warp versions",
    )
    _require(
        set(value)
        == {
            "candidate",
            "constraint_sha256",
            "efc_row",
            "fixture_module_sha256",
            "friction_kernel",
            "launch",
            "mujoco_warp",
            "platform",
            "python",
            "python_tree_files",
            "python_tree_sha256",
            "warp_lang",
        },
        "source binding keys",
    )
    for name, expected in (
        ("candidate", "_ascending_dense_friction"),
        ("efc_row", "_efc_row"),
        ("friction_kernel", "_friction_dof"),
        ("launch", "launch"),
    ):
        row = value[name]
        _require(
            type(row) is dict
            and type(row.get("object_id")) is int
            and row["object_id"] > 0
            and type(row.get("code_id")) is int
            and row["code_id"] > 0,
            "positive live identity IDs",
        )
        _require(row.get("code_name") == expected, "pinned source symbol " + name)
        if name in {"efc_row", "friction_kernel"}:
            _require(
                type(row.get("name")) is str and type(row.get("first_line")) is int,
                "pinned source location",
            )
            expected_line = 52 if name == "efc_row" else 1353
            _require(
                row["name"] == expected and row["first_line"] == expected_line,
                "literal original source location",
            )
    return value


def _helper_binding(value):
    _require(
        type(value) is dict
        and set(value)
        == {
            "_addressed_rows",
            "_allocation_record",
            "_as_bits",
            "_bank_record",
            "_base_source_binding",
            "_capture_active",
            "_check_base_helpers",
            "_input_record",
            "_launch_candidate",
            "_launch_original",
            "_module_sha256",
            "_need",
            "_new_wp_bank",
            "_numpy_bank",
            "_own_source_binding",
            "_prefix_values",
            "_validate_prefix",
            "_wp_inputs",
            "_unchanged",
            "run_prefix_cpu_fixture",
        },
        "prefix helper binding keys",
    )
    for name, row in value.items():
        _require(
            type(row) is dict
            and set(row) == {"code_name", "object_id", "code_id"}
            and row.get("code_name") == name
            and type(row.get("object_id")) is int
            and row["object_id"] > 0
            and type(row.get("code_id")) is int
            and row["code_id"] > 0,
            "prefix helper live identity",
        )
    return value


def _input_snapshot(case, worlds):
    snap = case["input_snapshot"]
    _require(
        type(snap) is dict and set(snap) == set(_INPUT_NAMES), "input snapshot fields"
    )
    shapes = {
        "frictionloss": (None, 20),
        "qvel": (worlds, 20),
        "invweight": (None, 20),
        "solref": (None, 20, 2),
        "solimp": (None, 20, 5),
        "timestep": (None,),
    }
    result = {}
    for name in _INPUT_NAMES:
        row = snap[name]
        _require(
            type(row) is dict
            and set(row) == {"dtype", "shape", "u32"}
            and row.get("dtype") == "<f4",
            "input snapshot schema " + name,
        )
        shape = row["shape"]
        _require(
            type(shape) is list and len(shape) == len(shapes[name]),
            "input rank " + name,
        )
        _require(
            all(type(x) is int and x > 0 for x in shape), "input dimensions " + name
        )
        if shapes[name][0] is None:
            _require(shape[0] in {1, worlds}, "bounded broadcast input rows " + name)
        if shapes[name][0] is not None:
            _require(shape[0] == shapes[name][0], "input world dimension " + name)
        for actual, expected in zip(shape[1:], shapes[name][1:], strict=True):
            _require(actual == expected, "input tail shape " + name)
        bits = row["u32"]
        _require(
            type(bits) is list and len(bits) == math.prod(shape),
            "complete input bits " + name,
        )
        for bit in bits:
            _plain_int(bit, 0, 0xFFFFFFFF, "input u32 " + name)
            _require(math.isfinite(_float(bit)), "finite input " + name)
        result[name] = {"shape": shape, "bits": bits}
    return result


def _bank(run, phase, worlds, cap):
    value = run[phase]
    _require(
        type(value) is dict and set(value) == set(_BANK_DTYPES), phase + " bank names"
    )
    shapes = {
        "nf": (worlds,),
        "nefc": (worlds,),
        "type": (worlds, cap),
        "id": (worlds, cap),
        "row_nnz": (worlds, cap),
        "row_adr": (worlds, cap),
        "col_ind": (worlds, 1, cap * 20),
        "J": (worlds, cap, 20),
        "pos": (worlds, cap),
        "margin": (worlds, cap),
        "D": (worlds, cap),
        "vel": (worlds, cap),
        "aref": (worlds, cap),
        "frictionloss": (worlds, cap),
        "efc_nnz": (worlds,),
    }
    decoded = {}
    for name, dtype in _BANK_DTYPES.items():
        shape, tag = shapes[name], "<u4" if dtype == "u32" else "<i4"
        decoded[name] = _bits_array(value[name], shape, tag, phase + "." + name)
    return value, decoded


def _at(bank, name, world, row, cap, dof=None):
    width = 20 if name == "J" else 1
    index = world * cap * width + row * width
    if dof is not None:
        index += dof
    return bank[name][index]


def _signature(bank, world, row, cap):
    return tuple(
        (
            name,
            tuple(_at(bank, name, world, row, cap, i) for i in range(20))
            if name == "J"
            else _at(bank, name, world, row, cap),
        )
        for name in _ROW_FIELDS
    )


def _verify_input_records(run, inputs):
    for phase in ("inputs_before", "inputs_after"):
        record = run[phase]
        _require(
            type(record) is dict and set(record) == set(_INPUT_NAMES), phase + " fields"
        )
        for name in _INPUT_NAMES:
            item = record[name]
            _require(
                type(item) is dict
                and set(item)
                == {
                    "shape",
                    "dtype",
                    "device",
                    "warp_dtype",
                    "warp_shape",
                    "pointer",
                    "u32",
                },
                phase + " input schema",
            )
            expected = inputs[name]
            _require(
                _canonical(item["shape"]) == _canonical(expected["shape"])
                and item["dtype"] == "<f4",
                phase + " exact input layout",
            )
            _require(
                type(item["u32"]) is list and len(item["u32"]) == len(expected["bits"]),
                phase + " complete input payload",
            )
            for bit in item["u32"]:
                _plain_int(bit, 0, 0xFFFFFFFF, phase + " input u32")
            _require(
                _canonical(item["u32"]) == _canonical(expected["bits"]),
                phase + " exact input bits",
            )
            warp_shape = (
                expected["shape"][:-1]
                if name in {"solref", "solimp"}
                else expected["shape"]
            )
            _require(
                item["device"] == "cpu"
                and all(type(x) is int for x in item["warp_shape"])
                and _canonical(item["warp_shape"]) == _canonical(warp_shape)
                and type(item["pointer"]) is int
                and item["pointer"] > 0,
                phase + " CPU input layout",
            )
            dtype_name = {
                "solref": "<class 'warp._src.types.vec2f'>",
                "solimp": "<class 'mujoco_warp._src.types.vec5f'>",
            }.get(name, "<class 'warp._src.types.float32'>")
            _require(item["warp_dtype"] == dtype_name, phase + " exact Warp dtype")
            allocation = run["allocations"]["inputs"][name]
            _require(
                all(
                    _canonical(item[field]) == _canonical(allocation[field])
                    for field in ("device", "warp_dtype", "warp_shape", "pointer")
                ),
                phase + " input record bound to owned allocation",
            )
    _require(
        run["inputs_before"] == run["inputs_after"], "actual input records unchanged"
    )


def _verify_allocations(runs, inputs, worlds, cap):
    intervals, objects, pointers = [], set(), set()
    input_shapes = {name: row["shape"] for name, row in inputs.items()}
    input_shapes["solref"] = input_shapes["solref"][:-1]
    input_shapes["solimp"] = input_shapes["solimp"][:-1]
    output_shapes = {
        "nf": [worlds],
        "nefc": [worlds],
        "type": [worlds, cap],
        "id": [worlds, cap],
        "row_nnz": [worlds, cap],
        "row_adr": [worlds, cap],
        "col_ind": [worlds, 1, cap * 20],
        "J": [worlds, cap, 20],
        "pos": [worlds, cap],
        "margin": [worlds, cap],
        "D": [worlds, cap],
        "vel": [worlds, cap],
        "aref": [worlds, cap],
        "frictionloss": [worlds, cap],
        "efc_nnz": [worlds],
    }
    for run in runs.values():
        alloc = run["allocations"]
        _require(
            type(alloc) is dict and set(alloc) == {"inputs", "outputs"},
            "allocation groups",
        )
        for group, names in (("inputs", _INPUT_NAMES), ("outputs", _BANK_DTYPES)):
            items = alloc[group]
            _require(
                type(items) is dict and set(items) == set(names), "allocation names"
            )
            for name, row in items.items():
                _require(
                    type(row) is dict
                    and set(row)
                    == {
                        "object_id",
                        "pointer",
                        "device",
                        "warp_dtype",
                        "warp_shape",
                        "warp_strides",
                        "bytes",
                    },
                    "allocation record schema",
                )
                _require(
                    row["device"] == "cpu"
                    and type(row["object_id"]) is int
                    and row["object_id"] > 0
                    and type(row["pointer"]) is int
                    and row["pointer"] > 0,
                    "CPU allocation identities",
                )
                _require(
                    type(row["bytes"]) is int
                    and row["bytes"] > 0
                    and type(row["warp_strides"]) is list,
                    "allocation size/strides",
                )
                _require(
                    type(row["warp_dtype"]) is str and type(row["warp_shape"]) is list,
                    "allocation dtype/shape",
                )
                shape = input_shapes[name] if group == "inputs" else output_shapes[name]
                _require(
                    all(type(x) is int for x in row["warp_shape"])
                    and _canonical(row["warp_shape"]) == _canonical(shape),
                    "allocation shape matches serialized bank",
                )
                storage_shape = inputs[name]["shape"] if group == "inputs" else shape
                _require(
                    row["bytes"] == math.prod(storage_shape) * 4,
                    "complete float32/int32 allocation bytes",
                )
                strides = _contiguous_strides(storage_shape)
                if group == "inputs" and name in {"solref", "solimp"}:
                    strides = strides[:-1]
                _require(
                    all(type(x) is int for x in row["warp_strides"])
                    and _canonical(row["warp_strides"])
                    == _canonical([stride * 4 for stride in strides]),
                    "contiguous allocation strides",
                )
                if group == "outputs":
                    dtype_name = (
                        "<class 'warp._src.types.int32'>"
                        if _BANK_DTYPES[name] == "i32"
                        else "<class 'warp._src.types.float32'>"
                    )
                else:
                    dtype_name = {
                        "solref": "<class 'warp._src.types.vec2f'>",
                        "solimp": "<class 'mujoco_warp._src.types.vec5f'>",
                    }.get(name, "<class 'warp._src.types.float32'>")
                _require(
                    row["warp_dtype"] == dtype_name, "allocation exact scalar dtype"
                )
                _require(
                    row["object_id"] not in objects and row["pointer"] not in pointers,
                    "three disjoint live run allocations",
                )
                objects.add(row["object_id"])
                pointers.add(row["pointer"])
                _require(
                    row["pointer"] + row["bytes"] <= 1 << 64,
                    "bounded 64-bit CPU allocation range",
                )
                intervals.append((row["pointer"], row["pointer"] + row["bytes"]))
    intervals.sort()
    _require(
        all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])),
        "nonoverlapping run allocation ranges",
    )


def _contiguous_strides(shape):
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return list(reversed(strides))


def audit_packet(packet):
    """Validate a fully parsed ten-case packet and recompute its component facts."""
    _require(
        type(packet) is dict
        and set(packet)
        == {
            "caller_binding_after",
            "caller_binding_before",
            "cases",
            "expected_negative",
            "flags",
            "held_checker_code_id",
            "held_checker_object_id",
            "held_entrypoint_code_id",
            "held_entrypoint_object_id",
            "protocol",
        },
        "top-level packet schema",
    )
    _require(packet["protocol"] == PACKET_PROTOCOL, "packet protocol")
    _require(
        packet["expected_negative"] == ["overflow", "maximum"],
        "literal expected negative cases",
    )
    _require(
        type(packet["flags"]) is dict
        and set(packet["flags"]) == set(_TOP_FLAGS)
        and all(packet["flags"].get(key) is False for key in _TOP_FLAGS),
        "top-level non-admission flags",
    )
    before_helpers = _helper_binding(packet["caller_binding_before"])
    after_helpers = _helper_binding(packet["caller_binding_after"])
    for field in (
        "held_checker_code_id",
        "held_checker_object_id",
        "held_entrypoint_code_id",
        "held_entrypoint_object_id",
    ):
        _plain_int(packet[field], 1, (1 << 64) - 1, "literal held caller identity")
    _require(before_helpers == after_helpers, "caller helper identities unchanged")
    _require(
        packet["held_checker_object_id"]
        == before_helpers["_own_source_binding"]["object_id"]
        and packet["held_checker_code_id"]
        == before_helpers["_own_source_binding"]["code_id"],
        "held checker identity",
    )
    _require(
        packet["held_entrypoint_object_id"]
        == before_helpers["run_prefix_cpu_fixture"]["object_id"]
        and packet["held_entrypoint_code_id"]
        == before_helpers["run_prefix_cpu_fixture"]["code_id"],
        "held entrypoint identity",
    )
    cases = packet["cases"]
    _require(type(cases) is dict and set(cases) == set(_CASES), "exact ten named cases")
    expected_rows = {}
    common_source = common_helpers = None
    for name, (worlds, cap, prefix, input_sha, bank_sha) in _CASES.items():
        case = cases[name]
        _require(type(case) is dict, "case object " + name)
        required = {
            "all_actual_inputs_unchanged",
            "all_dense_sparse_scratch_unchanged",
            "all_inactive_suffix_poison_preserved",
            "all_prefix_rows_preserved",
            "all_run_buffers_independent",
            "candidate_replay_addressed_exact",
            "candidate_replay_full_bank_bit_identical",
            "candidate_rows_ascending",
            "component_exact_without_overflow",
            "counts_and_addresses_complete",
            "device",
            "fixture_decision",
            "flags",
            "full_window_qualified",
            "initial_nefc",
            "initial_nf",
            "input_snapshot",
            "native_qualified",
            "old_fixture_module_sha256",
            "original_candidate_addressed_exact",
            "overflow_decision",
            "overflow_negative",
            "physical_acceptance",
            "prefix_helpers_after",
            "prefix_helpers_before",
            "prefix_module_sha256_after",
            "prefix_module_sha256_before",
            "protocol",
            "qualification",
            "runs",
            "runtime_cause_proven",
            "source_binding_after",
            "source_binding_before",
            "training_authorized",
            "worlds",
        }
        _require(set(case) == required, "case fields " + name)
        _require(
            case["protocol"] == "microduck-dense-friction-prefix-cpu-fixture-oct7-v1"
            and case["device"] == "cpu",
            "literal case protocol/device",
        )
        _require(
            case["old_fixture_module_sha256"] == BASE_SHA256
            and case["prefix_module_sha256_before"] == PREFIX_SHA256
            and case["prefix_module_sha256_after"] == PREFIX_SHA256,
            "pinned fixture source hashes",
        )
        src0, src1 = (
            _source_binding(case["source_binding_before"]),
            _source_binding(case["source_binding_after"]),
        )
        helpers0, helpers1 = (
            _helper_binding(case["prefix_helpers_before"]),
            _helper_binding(case["prefix_helpers_after"]),
        )
        _require(
            src0 == src1 and helpers0 == helpers1, "case before/after identity closure"
        )
        if common_source is None:
            common_source, common_helpers = src0, helpers0
        _require(
            src0 == common_source
            and helpers0 == common_helpers
            and helpers0 == before_helpers,
            "ten-case common source/helper identities",
        )
        for flag in (
            "qualification",
            "runtime_cause_proven",
            "native_qualified",
            "full_window_qualified",
            "training_authorized",
            "physical_acceptance",
        ):
            _require(case[flag] is False, "case permanently non-admitting flag " + flag)
        _require(
            type(case["flags"]) is dict
            and set(case["flags"]) == set(_TOP_FLAGS)
            and all(case["flags"].get(key) is False for key in _TOP_FLAGS),
            "case flags false",
        )
        _require(
            _canonical(case["initial_nefc"])
            == _canonical({"shape": [worlds], "dtype": "<i4", "i32": list(prefix)}),
            "expected prefix boundary",
        )
        _require(
            _canonical(case["initial_nf"])
            == _canonical({"shape": [worlds], "dtype": "<i4", "i32": [0] * worlds}),
            "zero initial nf",
        )
        inputs = _input_snapshot(case, worlds)
        _require(
            _digest(
                {
                    "input_snapshot": case["input_snapshot"],
                    "initial_nefc": case["initial_nefc"],
                    "initial_nf": case["initial_nf"],
                }
            )
            == input_sha,
            "predeclared exact fixture inputs " + name,
        )
        runs = case["runs"]
        _require(
            type(runs) is dict
            and set(runs) == {"original", "candidate0", "candidate1"},
            "all three runs",
        )
        _verify_allocations(runs, inputs, worlds, cap)
        banks = {}
        for side, run in runs.items():
            _require(
                type(run) is dict
                and set(run)
                == {"inputs_before", "inputs_after", "allocations", "before", "after"},
                "run record " + side,
            )
            _verify_input_records(run, inputs)
            before_rec, before = _bank(run, "before", worlds, cap)
            after_rec, after = _bank(run, "after", worlds, cap)
            banks[side] = (before_rec, before, after_rec, after)
        canonical_before = banks["original"][0]
        _require(
            _digest(canonical_before) == bank_sha
            and all(banks[side][0] == canonical_before for side in banks),
            "complete expected identical initial banks",
        )
        nfric = []
        for world in range(worlds):
            flrow = world % inputs["frictionloss"]["shape"][0]
            dofs = [
                dof
                for dof in range(20)
                if _float(inputs["frictionloss"]["bits"][flrow * 20 + dof]) > 0.0
            ]
            nfric.append(dofs)
        case_worlds = case["worlds"]
        _require(
            type(case_worlds) is list and len(case_worlds) == worlds,
            "complete per-world reports",
        )
        computed_worlds = []
        prefix_ok = suffix_ok = scratch_ok = counts_ok = ascending_ok = (
            replay_rows_ok
        ) = original_rows_ok = True
        any_overflow = False
        for world, dofs in enumerate(nfric):
            start = prefix[world]
            stored = min(len(dofs), cap - start)
            end = start + stored
            over = start + len(dofs) > cap
            any_overflow |= over
            counts = {"nf": len(dofs), "nefc": start + len(dofs), "stored_rows": end}
            sides = {}
            world_prefix_ok = world_suffix_ok = world_scratch_ok = True
            _require(
                type(case_worlds[world]) is dict
                and set(case_worlds[world])
                == {
                    "world",
                    "initial_nefc",
                    "expected_active_dofs",
                    "expected_counts",
                    "overflow",
                    "original",
                    "candidate0",
                    "candidate1",
                    "candidate_rows_ascending",
                    "candidate_replay_addressed_exact",
                    "original_candidate_addressed_exact",
                },
                "world report schema",
            )
            for side, (_, before, _, after) in banks.items():
                _require(
                    before["nf"][world] == 0 and before["nefc"][world] == start,
                    "initial bank counters",
                )
                _require(
                    after["nf"][world] == counts["nf"]
                    and after["nefc"][world] == counts["nefc"],
                    "independently recomputed counters",
                )
                p_ok = s_ok = True
                for field in _ROW_FIELDS:
                    width = 20 if field == "J" else 1
                    offset = world * cap * width
                    p_ok &= (
                        before[field][offset : offset + start * width]
                        == after[field][offset : offset + start * width]
                    )
                    s_ok &= (
                        before[field][offset + end * width : offset + cap * width]
                        == after[field][offset + end * width : offset + cap * width]
                    )
                sc_ok = all(
                    before[field] == after[field]
                    for field in ("row_nnz", "row_adr", "col_ind", "efc_nnz")
                )
                prefix_ok &= p_ok
                suffix_ok &= s_ok
                scratch_ok &= sc_ok
                world_prefix_ok &= p_ok
                world_suffix_ok &= s_ok
                world_scratch_ok &= sc_ok
                _require(
                    p_ok and s_ok and sc_ok, "prefix/suffix/sparse scratch preserved"
                )
                _require(
                    type(case_worlds[world][side]) is dict
                    and set(case_worlds[world][side])
                    == {
                        "rows",
                        "counts",
                        "prefix_preserved",
                        "suffix_preserved",
                        "sparse_scratch_unchanged",
                    },
                    "side report schema",
                )
                _require(
                    _canonical(case_worlds[world][side]["counts"]) == _canonical(counts)
                    and case_worlds[world][side]["prefix_preserved"] is p_ok
                    and case_worlds[world][side]["suffix_preserved"] is s_ok
                    and case_worlds[world][side]["sparse_scratch_unchanged"] is sc_ok,
                    "per-world producer summary agrees with raw bank",
                )
                rows = []
                for row in range(start, end):
                    dof = _at(after, "id", world, row, cap)
                    _require(
                        _at(after, "type", world, row, cap) == 1
                        and 0 <= dof < 20
                        and dof in dofs,
                        "active row type/address",
                    )
                    _require(
                        all(
                            _at(after, "id", world, old, cap) != dof
                            for old in range(start, row)
                        ),
                        "unique appended DOF",
                    )
                    jac = [_at(after, "J", world, row, cap, col) for col in range(20)]
                    _require(
                        jac == [0x3F800000 if col == dof else 0 for col in range(20)],
                        "exact one-hot f32 Jacobian",
                    )
                    for field in _FLOAT_FIELDS:
                        vals = (
                            [
                                _at(after, field, world, row, cap, col)
                                for col in range(20)
                            ]
                            if field == "J"
                            else [_at(after, field, world, row, cap)]
                        )
                        _require(
                            all(math.isfinite(_float(bits)) for bits in vals),
                            "finite active row field",
                        )
                    frow = world % inputs["frictionloss"]["shape"][0]
                    qrow = world * 20 + dof
                    _require(
                        _at(after, "frictionloss", world, row, cap)
                        == inputs["frictionloss"]["bits"][frow * 20 + dof],
                        "active frictionloss matches fixture input bits",
                    )
                    _require(
                        _at(after, "vel", world, row, cap)
                        == inputs["qvel"]["bits"][qrow],
                        "active velocity matches fixture input bits",
                    )
                    _require(
                        _at(after, "pos", world, row, cap) == 0
                        and _at(after, "margin", world, row, cap) == 0,
                        "zero friction-row position and margin",
                    )
                    row_value = {"row": row, "dof": dof}
                    row_value.update(
                        {
                            field: [
                                _at(after, field, world, row, cap, col)
                                for col in range(20)
                            ]
                            if field == "J"
                            else [_at(after, field, world, row, cap)]
                            for field in _FLOAT_FIELDS
                        }
                    )
                    rows.append(row_value)
                if side != "original":
                    _require(
                        [row["dof"] for row in rows] == dofs[:stored],
                        "candidate ascending appended addresses",
                    )
                _require(
                    _canonical(case_worlds[world][side]["rows"]) == _canonical(rows),
                    "per-world row report agrees with complete raw bank",
                )
                sides[side] = rows
            candidate0_sig = sorted(
                (
                    row["dof"],
                    tuple((field, tuple(row[field])) for field in _FLOAT_FIELDS),
                )
                for row in sides["candidate0"]
            )
            candidate1_sig = sorted(
                (
                    row["dof"],
                    tuple((field, tuple(row[field])) for field in _FLOAT_FIELDS),
                )
                for row in sides["candidate1"]
            )
            original_sig = sorted(
                (
                    row["dof"],
                    tuple((field, tuple(row[field])) for field in _FLOAT_FIELDS),
                )
                for row in sides["original"]
            )
            cand_exact = candidate0_sig == candidate1_sig
            orig_exact = original_sig == candidate0_sig
            ascending = [row["dof"] for row in sides["candidate0"]] == dofs[
                :stored
            ] and [row["dof"] for row in sides["candidate1"]] == dofs[:stored]
            replay_rows_ok &= cand_exact
            original_rows_ok &= orig_exact
            ascending_ok &= ascending
            _require(
                type(case_worlds[world]["world"]) is int
                and case_worlds[world]["world"] == world
                and case_worlds[world]["candidate_rows_ascending"] is ascending
                and case_worlds[world]["candidate_replay_addressed_exact"] is cand_exact
                and case_worlds[world]["original_candidate_addressed_exact"]
                is orig_exact
                and case_worlds[world]["overflow"] is over
                and _canonical(case_worlds[world]["expected_active_dofs"])
                == _canonical(dofs)
                and _canonical(case_worlds[world]["expected_counts"])
                == _canonical(counts)
                and type(case_worlds[world]["initial_nefc"]) is int
                and case_worlds[world]["initial_nefc"] == start,
                "per-world recomputed summary",
            )
            counts_ok &= all(
                banks[side][3]["nf"][world] == len(dofs)
                and banks[side][3]["nefc"][world] == start + len(dofs)
                for side in banks
            )
            computed_worlds.append(
                {
                    "world": world,
                    "positive_dofs": dofs,
                    "counts": counts,
                    "overflow": over,
                    "candidate_rows_ascending": ascending,
                    "candidate_replay_addressed_exact": cand_exact,
                    "original_candidate_addressed_exact": orig_exact,
                    "prefix_preserved": world_prefix_ok,
                    "suffix_preserved": world_suffix_ok,
                    "sparse_scratch_unchanged": world_scratch_ok,
                }
            )
        full_candidate = banks["candidate0"][2] == banks["candidate1"][2]
        _require(full_candidate, "candidate full-bank replay bit identity")
        _require(
            case["candidate_replay_full_bank_bit_identical"] is full_candidate,
            "producer full-bank replay decision",
        )
        candidate_replay = replay_rows_ok and full_candidate
        comp_exact = (
            prefix_ok
            and suffix_ok
            and scratch_ok
            and counts_ok
            and ascending_ok
            and candidate_replay
            and original_rows_ok
            and not any_overflow
        )
        flags = {
            "all_actual_inputs_unchanged": True,
            "all_dense_sparse_scratch_unchanged": scratch_ok,
            "all_inactive_suffix_poison_preserved": suffix_ok,
            "all_prefix_rows_preserved": prefix_ok,
            "all_run_buffers_independent": True,
            "candidate_replay_addressed_exact": candidate_replay,
            "candidate_replay_full_bank_bit_identical": full_candidate,
            "candidate_rows_ascending": ascending_ok,
            "component_exact_without_overflow": comp_exact,
            "counts_and_addresses_complete": counts_ok,
            "original_candidate_addressed_exact": original_rows_ok,
            "overflow_negative": any_overflow,
        }
        for field, expected in flags.items():
            _require(
                case[field] is expected,
                "producer decision disagrees with raw evidence: " + field,
            )
        _require(
            case["overflow_decision"]
            == ("overflow-negative-no-qualification" if any_overflow else "no-overflow")
            and case["fixture_decision"]
            == (
                "dense-prefix-by-address-exact"
                if comp_exact
                else "dense-prefix-negative-or-overflow"
            ),
            "producer final decision disagrees with recomputation",
        )
        _require(
            (name in {"maximum", "overflow"}) is any_overflow,
            "predeclared overflow negatives",
        )
        expected_rows[name] = {
            "worlds": computed_worlds,
            **flags,
            "overflow_negative": any_overflow,
            "fixture_decision": "dense-prefix-by-address-exact"
            if comp_exact
            else "dense-prefix-negative-or-overflow",
        }
    return {
        "protocol": PROTOCOL,
        "packet_protocol": PACKET_PROTOCOL,
        "cases": expected_rows,
        "case_count": len(expected_rows),
        "source_pins": {
            "base_fixture_sha256": BASE_SHA256,
            "prefix_fixture_sha256": PREFIX_SHA256,
            "constraint_sha256": CONSTRAINT_SHA256,
            "python_tree_sha256": TREE_SHA256,
        },
        "qualification": False,
        "runtime_cause_proven": False,
        "native_qualified": False,
        "full_window_qualified": False,
        "training_authorized": False,
        "physical_acceptance": False,
        "flags": {key: False for key in _TOP_FLAGS},
    }
