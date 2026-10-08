"""Pure-file solver CUBIN and offline SASS-output scope, never runtime proof.

The CUBIN is a retained compile input and cuobjdump text is offline output.
Neither establishes loaded/driver code, actual dispatch, numerical cause, or
qualification.  This module never starts a process or imports CUDA packages.
"""

from hashlib import sha256
import re
import struct

from mjlab_microduck import stance_solver_binary_scope as binary_scope

PROTOCOL = "microduck-solver-disassembly-scope-oct8-v1"
TARGET = binary_scope.TARGET
MAX_CUBIN_BYTES = 16 * 1024**2
MAX_SASS_BYTES = 8 * 1024**2
_TARGET_SYMBOL = re.compile(
    re.escape(TARGET) + r"_[0-9a-f]{8}_cuda_kernel_forward\Z"
)
_FUNCTION_HEADER = re.compile(r"Function\s*:\s*(\S+)\Z")
_HEADER_FLAGS = re.compile(
    r'\t\.headerflags\t@"EF_CUDA_64BIT_ADDRESS EF_CUDA_SM120 '
    r'EF_CUDA_VIRTUAL_SM\(EF_CUDA_SM120\)"\Z'
)
_INSTRUCTION = re.compile(
    r"\s*/\*([0-9a-fA-F]{4,16})\*/\s+(.+?)\s*;\s*/\*\s*"
    r"0x([0-9a-fA-F]{16})\s*\*/\s*\Z"
)
_ENCODING_CONTINUATION = re.compile(r"\s*/\*\s*0x([0-9a-fA-F]{16})\s*\*/\s*\Z")
_TERMINATOR = re.compile(r"\t\t\.{10}\Z")

FLAGS = {
    "native_qualified": False,
    "full_window_qualified": False,
    "runtime_cause_proven": False,
    "training_authorized": False,
    "physical_acceptance": False,
}


def need(value, message):
    if not value:
        raise ValueError(message)


def _code_section(raw, section_name):
    """Return the unique validated section payload matching an ELF name."""
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    section_offset, section_size, name_index = header[6], header[12], header[13]
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", raw, section_offset + i * 64)
        for i in range(section_size)
    ]
    names_section = sections[name_index]
    names = raw[names_section[4] : names_section[4] + names_section[5]]
    matches = []
    for section in sections:
        start = section[0]
        end = names.find(b"\0", start, min(len(names), start + 1025))
        need(end >= start, "bounded terminated ELF section name")
        name = names[start:end].decode("ascii")
        if name == section_name:
            matches.append(section)
    need(len(matches) == 1, "one exact target code section")
    section = matches[0]
    return section[3], section[4], section[5], raw[section[4] : section[4] + section[5]]


def select_target_cubin(raw):
    """Bind one literal target function to bounded retained compile-input bytes."""
    need(type(raw) is bytes and len(raw) <= MAX_CUBIN_BYTES, "bounded CUBIN bytes")
    parsed = binary_scope.elf_functions(raw)
    matches = [
        row for row in parsed["functions"] if _TARGET_SYMBOL.fullmatch(row["symbol"])
    ]
    need(len(matches) == 1, "exactly one literal dense solver target variant")
    function = matches[0]
    section_address, _, section_bytes, section_raw = _code_section(
        raw, function["section"]
    )
    relative = function["value"] - section_address
    need(
        relative >= 0
        and function["bytes"] > 0
        and relative + function["bytes"] <= section_bytes,
        "whole selected target function in code section",
    )
    code = section_raw[relative : relative + function["bytes"]]
    need(len(code) == function["bytes"], "whole selected target code bytes")
    return {
        "protocol": PROTOCOL,
        "evidence_kind": "retained_compile_input_cubin",
        "cubin": {"bytes": len(raw), "sha256": sha256(raw).hexdigest()},
        "target": {
            "symbol": function["symbol"],
            "section": function["section"],
            "function_bytes": function["bytes"],
            "function_sha256": sha256(code).hexdigest(),
            "code_section_bytes": section_bytes,
            "code_section_sha256": sha256(section_raw).hexdigest(),
        },
        "loaded_binary_bytes_observed": False,
        "driver_jit_machine_code_observed": False,
        "actual_dispatch_observed": False,
        "flags": dict(FLAGS),
    }


def parse_cuobjdump_sass(raw, target_symbol):
    """Validate the closed cuobjdump 12.8 standalone sm_120 SASS envelope."""
    need(type(raw) is bytes and 0 < len(raw) <= MAX_SASS_BYTES, "bounded SASS bytes")
    need(
        type(target_symbol) is str and _TARGET_SYMBOL.fullmatch(target_symbol),
        "literal target forward symbol",
    )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("ASCII-only cuobjdump output") from error
    lines = text.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    need(
        index < len(lines)
        and lines[index] == "\tcode for sm_120",
        "one literal sm_120 code header",
    )
    index += 1
    need(index < len(lines), "target Function header after architecture")
    function_line = lines[index]
    function_match = _FUNCTION_HEADER.fullmatch(function_line.strip())
    need(
        function_line.startswith("\t\t")
        and function_match is not None
        and function_match.group(1) == target_symbol,
        "one indented exact target Function header",
    )
    index += 1
    need(index < len(lines), "sm_120 headerflags after target Function")
    flags_line = lines[index]
    flags_match = _HEADER_FLAGS.fullmatch(flags_line)
    need(flags_match is not None, "one literal CUDA 12.8 sm_120 headerflags line")
    index += 1
    offsets = []
    code_chunks = []
    while index < len(lines):
        line = lines[index]
        if _TERMINATOR.fullmatch(line):
            index += 1
            break
        instruction = _INSTRUCTION.fullmatch(line)
        need(instruction is not None, "recognized offset and inline 64-bit encoding")
        need(bool(instruction.group(2).strip()), "nonempty SASS instruction")
        offset = int(instruction.group(1), 16)
        need(offset == len(offsets) * 16, "contiguous SASS offsets from zero by 16")
        offsets.append(f"{offset:04x}")
        code_chunks.append(int(instruction.group(3), 16).to_bytes(8, "little"))
        index += 1
        need(index < len(lines), "second 64-bit encoding continuation")
        continuation = _ENCODING_CONTINUATION.fullmatch(lines[index])
        need(continuation is not None, "exact 64-bit encoding continuation")
        code_chunks.append(int(continuation.group(1), 16).to_bytes(8, "little"))
        index += 1
    need(offsets, "at least one target instruction row")
    need(
        index > 0 and _TERMINATOR.fullmatch(lines[index - 1]),
        "literal two-tab ten-dot terminator",
    )
    need(all(not line.strip() for line in lines[index:]), "no text after terminator")
    code = b"".join(code_chunks)
    return {
        "protocol": PROTOCOL,
        "evidence_kind": "offline_disassembler_stdout",
        "target_symbol": target_symbol,
        "instruction_count": len(offsets),
        "instruction_offsets_hex": offsets,
        "reconstructed_code": {"bytes": len(code), "sha256": sha256(code).hexdigest()},
        "output_bytes": len(raw),
        "output_sha256": sha256(raw).hexdigest(),
        "loaded_binary_bytes_observed": False,
        "driver_jit_machine_code_observed": False,
        "actual_dispatch_observed": False,
        "flags": dict(FLAGS),
    }


def verify_target_disassembly(cubin, stdout):
    """Bind one exact offline disassembly to the selected compile-input function."""
    selected = select_target_cubin(cubin)
    disassembly = parse_cuobjdump_sass(stdout, selected["target"]["symbol"])
    need(
        disassembly["reconstructed_code"]["bytes"]
        == selected["target"]["function_bytes"]
        and disassembly["reconstructed_code"]["sha256"]
        == selected["target"]["function_sha256"],
        "offline SASS encoding exactly matches selected CUBIN function span",
    )
    return {
        "protocol": PROTOCOL,
        "evidence_kind": "compile_input_cubin_with_matching_offline_disassembly",
        "cubin": selected["cubin"],
        "target": selected["target"],
        "disassembly": disassembly,
        "compile_input_function_bytes_match": True,
        "loaded_binary_bytes_observed": False,
        "driver_jit_machine_code_observed": False,
        "actual_dispatch_observed": False,
        "flags": dict(FLAGS),
    }
