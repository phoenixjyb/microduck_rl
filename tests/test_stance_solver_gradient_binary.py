"""Synthetic ELF and SASS encodings; never instruction-semantics/GPU proof."""
from hashlib import sha256
import struct
import subprocess
import sys

import pytest
from test_stance_solver_cost_binary import cubin, sass
from mjlab_microduck import stance_solver_gradient_binary as b


def symbol(stage, suffix="1234abcd"):
    return b.PREFIXES[stage] + "_" + suffix + "_cuda_kernel_forward"


def pair():
    return [(symbol(s), bytes(range(i * 16, (i + 1) * 16))) for i, s in enumerate(b.PREFIXES)]


@pytest.mark.parametrize("stage", b.PREFIXES)
def test_two_entries_bind_same_whole_cubin_and_distinct_spans(stage):
    rows = pair()
    raw = cubin(rows)
    selected = b.select_cubin(raw, stage, symbol(stage))
    result = b.verify_disassembly(raw, sass(symbol(stage), dict(rows)[symbol(stage)]), stage, symbol(stage))
    assert selected["cubin"] == dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
    assert len(selected["paired_targets"]) == 2
    assert len({t["function_sha256"] for t in selected["paired_targets"].values()}) == 2
    assert result["compile_input_function_bytes_match"] is True
    assert result["disassembly"]["protocol"] == b.PROTOCOL
    for record in (selected, result, result["disassembly"]):
        assert len(record["flags"]) == 6 and not any(record["flags"].values())
        assert all(record[n] is False for n in ("loaded_binary_bytes_observed",
            "driver_jit_machine_code_observed", "actual_dispatch_observed"))


@pytest.mark.parametrize("stage", b.PREFIXES)
@pytest.mark.parametrize("mutation", ("missing-other", "duplicate-self", "duplicate-other", "wrong-symbol",
    "bad-suffix", "backward", "unknown-stage", "nonliteral-stage", "missing-self", "truncated", "bytearray"))
def test_closed_binary_target_refusals(stage, mutation):
    rows = pair()
    other = next(s for s in b.PREFIXES if s != stage)
    supplied, requested = symbol(stage), stage
    if mutation == "missing-other": rows = [r for r in rows if r[0] != symbol(other)]
    elif mutation == "missing-self": rows = [r for r in rows if r[0] != symbol(stage)]
    elif mutation.startswith("duplicate-"):
        target = stage if mutation == "duplicate-self" else other
        rows.append((symbol(target, "abcdef12"), bytes(range(16))))
    elif mutation == "wrong-symbol": supplied = symbol(other)
    elif mutation == "bad-suffix": supplied = symbol(stage, "1234ABCD")
    elif mutation == "backward": supplied = supplied.replace("forward", "backward")
    elif mutation == "unknown-stage": requested = "dense"
    elif mutation == "nonliteral-stage": requested = True
    raw = cubin(rows)
    if mutation == "truncated": raw = raw[:-1]
    if mutation == "bytearray": raw = bytearray(raw)
    with pytest.raises(ValueError): b.select_cubin(raw, requested, supplied)


def test_aliased_gradient_function_spans_refuse():
    raw = bytearray(cubin(pair()))
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    sym = struct.unpack_from("<IIQQQQIIQQ", raw, header[6] + 3 * 64)
    # Second function's st_value aliases the first's complete span.
    struct.pack_into("<Q", raw, sym[4] + 2 * 24 + 8, 0)
    with pytest.raises(ValueError, match="disjoint"): b.select_cubin(bytes(raw), "gradient", symbol("gradient"))


@pytest.mark.parametrize("stage", b.PREFIXES)
@pytest.mark.parametrize("mutation", ("wrong-entry", "wrong-code", "wrong-arch", "wrong-header", "gap",
    "missing-second-word", "missing-terminator", "duplicate-function", "trailing", "non-ascii", "bytearray"))
def test_closed_offline_sass_refusals(stage, mutation):
    rows = pair()
    code = dict(rows)[symbol(stage)]
    text = sass(symbol(stage), code)
    if mutation == "wrong-entry": text = text.replace(symbol(stage).encode(), symbol(next(s for s in b.PREFIXES if s != stage)).encode())
    elif mutation == "wrong-code": text = sass(symbol(stage), code[::-1])
    elif mutation == "wrong-arch": text = text.replace(b"sm_120", b"sm_100")
    elif mutation == "wrong-header": text = text.replace(b"EF_CUDA_SM120", b"EF_CUDA_SM100")
    elif mutation == "gap": text = text.replace(b"/*0000*/", b"/*0010*/")
    elif mutation == "missing-second-word": text = b"\n".join(text.splitlines()[:-2] + text.splitlines()[-1:]) + b"\n"
    elif mutation == "missing-terminator": text = text.replace(b"\t\t..........", b"")
    elif mutation == "duplicate-function": text += text
    elif mutation == "trailing": text += b"unbound text\n"
    elif mutation == "non-ascii": text += b"\xff"
    elif mutation == "bytearray": text = bytearray(text)
    with pytest.raises(ValueError): b.verify_disassembly(cubin(rows), text, stage, symbol(stage))


def test_import_is_inert():
    subprocess.run([sys.executable, "-c", "from mjlab_microduck import stance_solver_gradient_binary; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"], check=True, timeout=10)
