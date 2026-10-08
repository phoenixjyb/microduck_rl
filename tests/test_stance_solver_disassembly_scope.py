"""Synthetic byte and text fixtures only; never run CUDA or cuobjdump."""

from hashlib import sha256
import struct

import pytest

from mjlab_microduck import stance_solver_disassembly_scope as audit


def elf(symbols, code=b"\x10\x20\x30\x40"):
    section_names = b"\0.shstrtab\0.strtab\0.symtab\0.text\0"
    strings = b"\0" + b"".join(symbol.encode("ascii") + b"\0" for symbol in symbols)
    symbols_raw = b"\0" * 24
    for symbol in symbols:
        symbols_raw += struct.pack(
            "<IBBHQQ",
            strings.index(symbol.encode("ascii") + b"\0"),
            18,
            16,
            4,
            0,
            len(code),
        )
    data = bytearray(b"\0" * 64)
    sections = [(0,) * 10]
    for name, kind, flags, payload, link, entry in (
        (b".shstrtab", 3, 0, section_names, 0, 0),
        (b".strtab", 3, 0, strings, 0, 0),
        (b".symtab", 2, 0, symbols_raw, 2, 24),
        (b".text", 1, 6, code, 0, 0),
    ):
        sections.append(
            (
                section_names.index(name + b"\0"),
                kind,
                flags,
                0,
                len(data),
                len(payload),
                link,
                0,
                1,
                entry,
            )
        )
        data.extend(payload)
    section_header_offset = len(data)
    data.extend(b"".join(struct.pack("<IIQQQQIIQQ", *row) for row in sections))
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        data,
        0,
        ident,
        2,
        190,
        1,
        0,
        0,
        section_header_offset,
        0,
        64,
        0,
        0,
        64,
        5,
        1,
    )
    return bytes(data)


def target(suffix="1234abcd"):
    return f"{audit.TARGET}_{suffix}_cuda_kernel_forward"


def word_bytes(first, second):
    first_bytes = int(first, 16).to_bytes(8, "little")
    second_bytes = int(second, 16).to_bytes(8, "little")
    return first_bytes + second_bytes


def sass(symbol, body=None):
    if body is None:
        body = (
            "/*0000*/ MOV R1, R2 &wr=0x0 ?trans1; /* 0x0123456789abcdef */\n"
            "                    /* 0xfedcba9876543210 */\n"
        )
    return (
        f"\n\tcode for sm_120\n"
        f"\t\tFunction : {symbol}\n"
        '\t.headerflags\t@"EF_CUDA_64BIT_ADDRESS EF_CUDA_SM120 '
        'EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"\n'
        + body
        + "\t\t..........\n"
    ).encode("ascii")


def test_cubin_binds_exact_target_function_and_section_bytes():
    code = b"compiled target bytes"
    raw = elf(["helper_1234abcd_cuda_kernel_forward", target()], code)
    record = audit.select_target_cubin(raw)
    assert record["evidence_kind"] == "retained_compile_input_cubin"
    assert record["cubin"] == {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    assert record["target"] == {
        "symbol": target(),
        "section": ".text",
        "function_bytes": len(code),
        "function_sha256": sha256(code).hexdigest(),
        "code_section_bytes": len(code),
        "code_section_sha256": sha256(code).hexdigest(),
    }
    assert record["loaded_binary_bytes_observed"] is False
    assert record["actual_dispatch_observed"] is False
    assert not any(record["flags"].values())


def test_cubin_sibling_only_and_multiple_target_variants_are_refused():
    with pytest.raises(ValueError, match="exactly one"):
        audit.select_target_cubin(
            elf(["ascending_friction_dof_1234abcd_cuda_kernel_forward"])
        )
    with pytest.raises(ValueError, match="exactly one"):
        audit.select_target_cubin(elf([target(), target("87654321")]))


def test_disassembly_binds_single_exact_target_header_and_offsets():
    raw = sass(
        target(),
        "/*0000*/ MOV R1, R2 ; /* 0x0123456789abcdef */\n"
        "                    /* 0xfedcba9876543210 */\n"
        "/*0010*/ EXIT ; /* 0x1123456789abcdef */\n"
        "                    /* 0x0edcba9876543210 */\n",
    )
    record = audit.parse_cuobjdump_sass(raw, target())
    assert record["evidence_kind"] == "offline_disassembler_stdout"
    assert record["target_symbol"] == target()
    assert record["instruction_offsets_hex"] == ["0000", "0010"]
    code = word_bytes("0123456789abcdef", "fedcba9876543210") + word_bytes(
        "1123456789abcdef", "0edcba9876543210"
    )
    assert record["reconstructed_code"] == {
        "bytes": len(code),
        "sha256": sha256(code).hexdigest(),
    }
    assert record["output_sha256"] == sha256(raw).hexdigest()
    assert record["actual_dispatch_observed"] is False
    assert record["driver_jit_machine_code_observed"] is False
    assert not any(record["flags"].values())


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"Error: no cubin found\n",
        b"\n\tcode for sm_120\n",
        sass("other_symbol"),
        sass(target()).replace(b"code for sm_120", b"code for sm_120\n\tcode for sm_120"),
        sass(target()).replace(b"Function :", b"Function : other\n\t\tFunction :"),
        sass(target()).replace(b"EF_CUDA_SM120", b"EF_CUDA_SM90", 1),
        sass(target()).replace(b"/*0000*/", b"/*oops*/", 1),
        sass(target()).replace(b"0xfedcba9876543210", b"0xxyz", 1),
        sass(target()).replace(b"\t\t..........\n", b""),
        sass(target()).replace(b"/* 0x0123456789abcdef */", b"warning: failed", 1),
        sass(target()).replace(b"\tcode for sm_120", b"\tcode for sm_90", 1),
        sass(target()).replace(b"/* 0x0123456789abcdef */", b"/* 0x0123456789abcdef */ \xff", 1),
    ],
)
def test_bad_empty_error_other_function_and_unrecognized_output_refused(raw):
    with pytest.raises(ValueError):
        audit.parse_cuobjdump_sass(raw, target())


def test_non_ascii_and_duplicate_offsets_are_refused():
    with pytest.raises(ValueError, match="ASCII"):
        audit.parse_cuobjdump_sass(sass(target()) + b"\xff", target())
    body = (
        "/*0000*/ MOV R1, R2 ; /* 0x0123456789abcdef */\n"
        "                    /* 0xfedcba9876543210 */\n"
        "/*0000*/ EXIT ; /* 0x1123456789abcdef */\n"
        "                    /* 0x0edcba9876543210 */\n"
    )
    repeated = sass(target(), body)
    with pytest.raises(ValueError, match="contiguous"):
        audit.parse_cuobjdump_sass(repeated, target())


def test_offsets_must_be_contiguous_from_zero():
    first = "0123456789abcdef"
    second = "fedcba9876543210"
    gap = sass(
        target(),
        f"/*0010*/ MOV R1, R2 ; /* 0x{first} */\n"
        f"                    /* 0x{second} */\n",
    )
    with pytest.raises(ValueError, match="contiguous"):
        audit.parse_cuobjdump_sass(gap, target())
    two_rows_gap = sass(
        target(),
        f"/*0000*/ MOV R1, R2 ; /* 0x{first} */\n"
        f"                    /* 0x{second} */\n"
        f"/*0020*/ EXIT ; /* 0x{first} */\n"
        f"                    /* 0x{second} */\n",
    )
    with pytest.raises(ValueError, match="contiguous"):
        audit.parse_cuobjdump_sass(two_rows_gap, target())


def test_verify_binds_reconstructed_sass_bytes_to_cubin_function_span():
    code = word_bytes("0123456789abcdef", "fedcba9876543210")
    raw = elf([target()], code)
    record = audit.verify_target_disassembly(raw, sass(target()))
    assert (
        record["evidence_kind"]
        == "compile_input_cubin_with_matching_offline_disassembly"
    )
    assert record["target"]["function_sha256"] == sha256(code).hexdigest()
    assert (
        record["disassembly"]["reconstructed_code"]["sha256"]
        == sha256(code).hexdigest()
    )
    assert record["compile_input_function_bytes_match"] is True
    assert record["loaded_binary_bytes_observed"] is False
    assert record["actual_dispatch_observed"] is False
    assert not any(record["flags"].values())


def test_verify_refuses_cubin_code_or_span_mismatch():
    text_code = word_bytes("0123456789abcdef", "fedcba9876543210")
    with pytest.raises(ValueError, match="exactly matches"):
        audit.verify_target_disassembly(
            elf([target()], b"X" * len(text_code)), sass(target())
        )
    short_code = text_code[:8]
    with pytest.raises(ValueError, match="exactly matches"):
        audit.verify_target_disassembly(elf([target()], short_code), sass(target()))


@pytest.mark.parametrize("bad", [True, "bytes", b"x" * (audit.MAX_SASS_BYTES + 1)])
def test_disassembler_output_type_and_cap_are_enforced(bad):
    with pytest.raises(ValueError):
        audit.parse_cuobjdump_sass(bad, target())


def test_cubin_type_and_cap_are_enforced():
    with pytest.raises(ValueError):
        audit.select_target_cubin(bytearray(elf([target()])))
    with pytest.raises(ValueError):
        audit.select_target_cubin(b"x" * (audit.MAX_CUBIN_BYTES + 1))


@pytest.mark.parametrize(
    "cubin,stdout", [(b"bad", b"bad"), (elf([target()]), "text")]
)
def test_verify_requires_plain_cubin_and_stdout_bytes(cubin, stdout):
    with pytest.raises(ValueError):
        audit.verify_target_disassembly(cubin, stdout)
