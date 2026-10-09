"""Pure two-entry gradient CUBIN/offline-SASS binding, never runtime proof."""
from hashlib import sha256
import re

from mjlab_microduck import stance_solver_binary_scope as elf
from mjlab_microduck import stance_solver_cost_binary as envelope
from mjlab_microduck import stance_solver_disassembly_scope as sections
from mjlab_microduck import stance_solver_gradient_prefix as prefix

PROTOCOL = "microduck-gradient-binary-oct9-v1"
PREFIXES = dict(zip(prefix.STAGES, ("update_gradient_zero_grad_dot", "update_gradient_grad")))
SYMBOLS = {s: re.compile(re.escape(n) + r"_[0-9a-f]{8}_cuda_kernel_forward\Z") for s, n in PREFIXES.items()}
need = prefix.need


def select_cubin(raw, stage, symbol):
    """Bind a target span in a whole CUBIN containing both unique entry families."""
    need(type(stage) is str and stage in PREFIXES and type(symbol) is str
         and SYMBOLS[stage].fullmatch(symbol), "literal gradient stage and exact forward symbol")
    need(type(raw) is bytes and 64 <= len(raw) <= sections.MAX_CUBIN_BYTES, "bounded whole gradient CUBIN")
    functions = elf.elf_functions(raw)["functions"]
    selected, spans = {}, []
    for name, pattern in SYMBOLS.items():
        family = [f for f in functions if pattern.fullmatch(f["symbol"])]
        need(len(family) == 1, "one exact variant of each gradient function family")
        f = family[0]
        address, offset, size, code = sections._code_section(raw, f["section"])
        relative = f["value"] - address
        need(relative >= 0 and f["bytes"] > 0 and relative + f["bytes"] <= size,
             "whole gradient function in unique executable code section")
        payload = code[relative:relative + f["bytes"]]
        selected[name] = dict(symbol=f["symbol"], section=f["section"], function_bytes=len(payload),
            function_sha256=sha256(payload).hexdigest(), code_section_bytes=size,
            code_section_sha256=sha256(code).hexdigest())
        spans.append((offset + relative, offset + relative + len(payload)))
    a, b = sorted(spans)
    need(a[1] <= b[0], "disjoint complete gradient function spans")
    need(selected[stage]["symbol"] == symbol, "selected symbol equals actual runtime gradient entry")
    return dict(protocol=PROTOCOL, stage=stage, evidence_kind="retained_compile_input_cubin",
        cubin=dict(bytes=len(raw), sha256=sha256(raw).hexdigest()), target=selected[stage],
        paired_targets=selected, loaded_binary_bytes_observed=False,
        driver_jit_machine_code_observed=False, actual_dispatch_observed=False, flags=dict(prefix.FLAGS))


def verify_disassembly(cubin, sass, stage, symbol):
    """Reuse the unchanged closed sm120 envelope parser; bind reconstructed bytes."""
    selected = select_cubin(cubin, stage, symbol)
    parsed, code = envelope._parse_sass(sass, symbol)
    need(len(code) == selected["target"]["function_bytes"]
         and sha256(code).hexdigest() == selected["target"]["function_sha256"],
         "offline SASS encodings match the entire selected gradient function")
    parsed = dict(parsed, protocol=PROTOCOL, flags=dict(prefix.FLAGS))
    return dict(selected, evidence_kind="compile_input_cubin_with_matching_offline_disassembly",
                disassembly=parsed, compile_input_function_bytes_match=True)
