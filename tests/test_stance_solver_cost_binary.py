"""Synthetic CUBIN and SASS checks only; never run CUDA or cuobjdump."""

from hashlib import sha256
import struct

import pytest

from mjlab_microduck import stance_solver_cost_binary as audit


def symbol(stage, suffix="1234abcd"):
    return f"{audit.STAGE_PREFIXES[stage]}_{suffix}_cuda_kernel_forward"


def cubin(functions):
    names = b"\0.shstrtab\0.strtab\0.symtab\0.text\0"
    strings = b"\0" + b"".join(name.encode("ascii") + b"\0" for name, _ in functions)
    code = b"".join(payload for _, payload in functions)
    rows = [b"\0" * 24]
    offset = 0
    for name, payload in functions:
        rows.append(struct.pack(
            "<IBBHQQ", strings.index(name.encode("ascii") + b"\0"),
            18, 16, 4, offset, len(payload),
        ))
        offset += len(payload)
    symbols = b"".join(rows)
    data = bytearray(b"\0" * 64)
    sections = [(0,) * 10]
    for name, kind, flags, payload, link, entry in (
        (b".shstrtab", 3, 0, names, 0, 0),
        (b".strtab", 3, 0, strings, 0, 0),
        (b".symtab", 2, 0, symbols, 2, 24),
        (b".text", 1, 6, code, 0, 0),
    ):
        sections.append((names.index(name + b"\0"), kind, flags, 0,
                         len(data), len(payload), link, 0, 1, entry))
        data.extend(payload)
    section_offset = len(data)
    data.extend(b"".join(struct.pack("<IIQQQQIIQQ", *row) for row in sections))
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into("<16sHHIQQQIHHHHHH", data, 0, ident, 2, 190, 1, 0, 0,
                     section_offset, 0, 64, 0, 0, 64, 5, 1)
    return bytes(data)


def sass(target, code):
    assert len(code) % 16 == 0
    lines = ["", "\tcode for sm_120", f"\t\tFunction : {target}",
             '\t.headerflags\t@"EF_CUDA_64BIT_ADDRESS EF_CUDA_SM120 '
             'EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"']
    for offset in range(0, len(code), 16):
        first = int.from_bytes(code[offset:offset + 8], "little")
        second = int.from_bytes(code[offset + 8:offset + 16], "little")
        lines.append(f"/*{offset:04x}*/ MOV R1, R2 ; /* 0x{first:016x} */")
        lines.append(f"                    /* 0x{second:016x} */")
    lines.append("\t\t..........")
    return ("\n".join(lines) + "\n").encode("ascii")


def test_shared_init_dense_cubin_selects_each_exact_function():
    init_symbol, dense_symbol = symbol("init_cost"), symbol("dense")
    init_code, dense_code = bytes(range(16)), bytes(range(16, 32))
    raw = cubin([(init_symbol, init_code), (dense_symbol, dense_code)])

    init = audit.select_cubin(raw, "init_cost", init_symbol)
    dense = audit.select_cubin(raw, "dense", dense_symbol)
    assert init["target"]["function_sha256"] == sha256(init_code).hexdigest()
    assert dense["target"]["function_sha256"] == sha256(dense_code).hexdigest()
    assert init["cubin"] == dense["cubin"] == {
        "bytes": len(raw), "sha256": sha256(raw).hexdigest()
    }
    assert all(init[name] is False for name in (
        "loaded_binary_bytes_observed", "driver_jit_machine_code_observed",
        "actual_dispatch_observed",
    ))
    assert not any(init["flags"].values())


@pytest.mark.parametrize("stage", ("efc", "gauss"))
def test_unique_factory_cubin_and_disassembly_bind_exact_span(stage):
    target = symbol(stage)
    code = bytes(range(16))
    raw = cubin([(target, code)])
    record = audit.verify_disassembly(raw, sass(target, code), stage, target)
    assert record["target"]["symbol"] == target
    assert record["disassembly"]["target_symbol"] == target
    assert record["compile_input_function_bytes_match"] is True
    assert record["loaded_binary_bytes_observed"] is False
    assert record["driver_jit_machine_code_observed"] is False
    assert record["actual_dispatch_observed"] is False
    assert not any(record["flags"].values())


def test_stage_and_supplied_runtime_symbol_must_match():
    raw = cubin([(symbol("init_cost"), bytes(range(16)))])
    with pytest.raises(ValueError, match="stage-specific"):
        audit.select_cubin(raw, "dense", symbol("init_cost"))
    with pytest.raises(ValueError, match="stage-specific"):
        audit.select_cubin(raw, "init_cost", symbol("dense"))
    with pytest.raises(ValueError, match="literal caller cost stage"):
        audit.select_cubin(raw, "unknown", symbol("init_cost"))


def test_wrong_or_duplicate_stage_function_is_refused():
    raw = cubin([(symbol("init_cost"), bytes(range(16))),
                 (symbol("init_cost", "87654321"), bytes(range(16)))])
    with pytest.raises(ValueError, match="one exact stage function"):
        audit.select_cubin(raw, "init_cost", symbol("init_cost"))
    with pytest.raises(ValueError, match="one exact stage function"):
        audit.select_cubin(cubin([(symbol("dense"), bytes(range(16)))]),
                           "init_cost", symbol("init_cost"))


@pytest.mark.parametrize("mutation", (
    "architecture", "function", "flags", "offset", "encoding", "terminator",
    "trailing", "duplicate", "mismatch",
))
def test_disassembly_closed_envelope_and_exact_encoding(mutation):
    target = symbol("dense")
    code = bytes(range(16))
    raw = cubin([(target, code)])
    text = sass(target, code)
    if mutation == "architecture":
        text = text.replace(b"code for sm_120", b"code for sm_90", 1)
    elif mutation == "function":
        text = text.replace(target.encode(), b"other_symbol", 1)
    elif mutation == "flags":
        text = text.replace(b"EF_CUDA_SM120", b"EF_CUDA_SM90", 1)
    elif mutation == "offset":
        text = text.replace(b"/*0000*/", b"/*0010*/", 1)
    elif mutation == "encoding":
        text = text.replace(b"MOV R1, R2", b"", 1)
    elif mutation == "terminator":
        text = text.replace(b"\t\t..........\n", b"", 1)
    elif mutation == "trailing":
        text += b"warning: extra output\n"
    elif mutation == "duplicate":
        text += sass(target, code).lstrip(b"\n")
    elif mutation == "mismatch":
        text = sass(target, bytes(reversed(code)))
    with pytest.raises(ValueError):
        audit.verify_disassembly(raw, text, "dense", target)


def test_function_span_with_multiple_rows_requires_contiguous_offsets():
    target = symbol("efc")
    code = bytes(range(32))
    text = sass(target, code).replace(b"/*0010*/", b"/*0020*/", 1)
    with pytest.raises(ValueError, match="contiguous"):
        audit.verify_disassembly(cubin([(target, code)]), text, "efc", target)


def test_import_is_inert():
    import subprocess
    import sys

    code = (
        "from mjlab_microduck import stance_solver_cost_binary; import sys; "
        "assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} "
        "for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
