"""CPU-only CUBIN and offline SASS binding for the four caller cost kernels.

This receiver binds a retained compile-input function span to one exact
cuobjdump sm_120 text envelope. It does not observe loaded machine code,
dispatch, or qualification.
"""

from hashlib import sha256
import re

from mjlab_microduck import stance_solver_binary_scope as binary_scope
from mjlab_microduck import stance_solver_disassembly_scope as disassembly_scope

PROTOCOL = "microduck-caller-cost-binary-oct9-v1"
MAX_CUBIN_BYTES = disassembly_scope.MAX_CUBIN_BYTES
MAX_SASS_BYTES = disassembly_scope.MAX_SASS_BYTES
STAGE_PREFIXES = {
    "init_cost": "update_constraint_init_cost",
    "dense": binary_scope.TARGET,
    "efc": "update_constraint_efc__locals__kernel",
    "gauss": "update_constraint_gauss_cost__locals__kernel",
}
_FUNCTION_HEADER = re.compile(r"Function\s*:\s*(\S+)\Z")
_STAGE_SYMBOLS = {
    stage: re.compile(re.escape(prefix) + r"_[0-9a-f]{8}_cuda_kernel_forward\Z")
    for stage, prefix in STAGE_PREFIXES.items()
}
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


def _need(value, message):
    if not value:
        raise ValueError(message)


def _target(raw, stage, symbol):
    _need(type(stage) is str and stage in STAGE_PREFIXES, "literal caller cost stage")
    _need(type(symbol) is str and _STAGE_SYMBOLS[stage].fullmatch(symbol),
          "exact stage-specific runtime forward symbol")
    _need(type(raw) is bytes and 64 <= len(raw) <= MAX_CUBIN_BYTES,
          "bounded CUBIN bytes")
    parsed = binary_scope.elf_functions(raw)
    family = [row for row in parsed["functions"]
              if _STAGE_SYMBOLS[stage].fullmatch(row["symbol"])]
    _need(len(family) == 1 and family[0]["symbol"] == symbol,
          "one exact stage function variant in CUBIN")
    function = family[0]
    section_address, _, section_bytes, section_raw = disassembly_scope._code_section(
        raw, function["section"]
    )
    relative = function["value"] - section_address
    _need(relative >= 0 and function["bytes"] > 0
          and relative + function["bytes"] <= section_bytes,
          "whole selected function in executable section")
    code = section_raw[relative:relative + function["bytes"]]
    _need(len(code) == function["bytes"], "whole selected function code bytes")
    return {
        "symbol": symbol,
        "section": function["section"],
        "function_bytes": len(code),
        "function_sha256": sha256(code).hexdigest(),
        "code_section_bytes": section_bytes,
        "code_section_sha256": sha256(section_raw).hexdigest(),
    }


def select_cubin(raw, stage, symbol):
    """Select one exact cost-kernel function from retained compile-input bytes.

    ``symbol`` is the complete runtime forward name returned by Warp's kernel
    object with ``_cuda_kernel_forward`` appended.
    """
    target = _target(raw, stage, symbol)
    return {
        "protocol": PROTOCOL,
        "stage": stage,
        "evidence_kind": "retained_compile_input_cubin",
        "cubin": {"bytes": len(raw), "sha256": sha256(raw).hexdigest()},
        "target": target,
        "loaded_binary_bytes_observed": False,
        "driver_jit_machine_code_observed": False,
        "actual_dispatch_observed": False,
        "flags": dict(FLAGS),
    }


def _parse_sass(raw, symbol):
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_SASS_BYTES,
          "bounded SASS output bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("ASCII-only cuobjdump output") from error
    lines = text.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    _need(index < len(lines) and lines[index] == "\tcode for sm_120",
          "one literal sm_120 code header")
    index += 1
    _need(index < len(lines), "target Function header after architecture")
    function_line = lines[index]
    function_match = _FUNCTION_HEADER.fullmatch(function_line.strip())
    _need(function_line.startswith("\t\t") and function_match is not None
          and function_match.group(1) == symbol,
          "one indented exact target Function header")
    index += 1
    _need(index < len(lines) and _HEADER_FLAGS.fullmatch(lines[index]) is not None,
          "one literal CUDA 12.8 sm_120 headerflags line")
    index += 1

    offsets, chunks = [], []
    while index < len(lines):
        line = lines[index]
        if _TERMINATOR.fullmatch(line):
            index += 1
            break
        instruction = _INSTRUCTION.fullmatch(line)
        _need(instruction is not None, "recognized offset and inline 64-bit encoding")
        _need(bool(instruction.group(2).strip()), "nonempty SASS instruction")
        offset = int(instruction.group(1), 16)
        _need(offset == len(offsets) * 16, "contiguous SASS offsets from zero by 16")
        offsets.append(f"{offset:04x}")
        chunks.append(int(instruction.group(3), 16).to_bytes(8, "little"))
        index += 1
        _need(index < len(lines), "second 64-bit encoding continuation")
        continuation = _ENCODING_CONTINUATION.fullmatch(lines[index])
        _need(continuation is not None, "exact 64-bit encoding continuation")
        chunks.append(int(continuation.group(1), 16).to_bytes(8, "little"))
        index += 1
    _need(offsets, "at least one target instruction row")
    _need(index > 0 and _TERMINATOR.fullmatch(lines[index - 1]) is not None,
          "literal two-tab ten-dot terminator")
    _need(all(not line.strip() for line in lines[index:]),
          "no text or duplicate function after terminator")
    code = b"".join(chunks)
    return {
        "protocol": PROTOCOL,
        "evidence_kind": "offline_disassembler_stdout",
        "target_symbol": symbol,
        "instruction_count": len(offsets),
        "instruction_offsets_hex": offsets,
        "reconstructed_code": {"bytes": len(code), "sha256": sha256(code).hexdigest()},
        "output_bytes": len(raw),
        "output_sha256": sha256(raw).hexdigest(),
        "loaded_binary_bytes_observed": False,
        "driver_jit_machine_code_observed": False,
        "actual_dispatch_observed": False,
        "flags": dict(FLAGS),
    }, code


def verify_disassembly(cubin, sass, stage, symbol):
    """Bind one closed offline sm_120 SASS envelope to its exact CUBIN span."""
    selected = select_cubin(cubin, stage, symbol)
    parsed, code = _parse_sass(sass, symbol)
    _need(len(code) == selected["target"]["function_bytes"]
          and sha256(code).hexdigest() == selected["target"]["function_sha256"],
          "offline SASS encoding exactly matches selected CUBIN function span")
    return {
        "protocol": PROTOCOL,
        "stage": stage,
        "evidence_kind": "compile_input_cubin_with_matching_offline_disassembly",
        "cubin": selected["cubin"],
        "target": selected["target"],
        "disassembly": parsed,
        "compile_input_function_bytes_match": True,
        "loaded_binary_bytes_observed": False,
        "driver_jit_machine_code_observed": False,
        "actual_dispatch_observed": False,
        "flags": dict(FLAGS),
    }
