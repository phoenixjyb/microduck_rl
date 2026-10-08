"""CPU-only arithmetic side view; never a CUDA or training admission gate.

Authenticate the historical initialized packet and frozen solver source before
decoding. Keep both rounding hypotheses visible; neither identifies GPU code.
"""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
import subprocess

PROTOCOL = "microduck-retained-gauss-arithmetic-oct8-v1"
BASE = "0a66611a00b57bc28d95e2cec18e86b6211ecefe"
OWN = {
    "src/mjlab_microduck/stance_solver_gauss_audit.py",
    "tests/test_stance_solver_gauss_audit.py",
    "docs/experiments/2026-10-08-retained-gauss-arithmetic.md",
}
NAMES = ("efc.Ma", "data.qfrc_smooth", "data.qacc", "data.qacc_smooth")
FLAGS = dict(numerical_qualified=False, native_qualified=False,
             full_window_qualified=False, runtime_cause_proven=False,
             training_authorized=False, physical_acceptance=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def round32(value):
    """Exact rational to IEEE binary32 RNE, including gradual underflow.

    Arithmetic exact-zero is positive here: hypotheses start from +0 and are
    numerical models, not a signed-zero device identity claim. Overflow refuses.
    """
    need(type(value) is Fraction, "exact rational rounding input")
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    if not value:
        return 0.0
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n < d << exponent) if exponent >= 0 else (n << -exponent < d):
        exponent -= 1
    shift = max(exponent, -126) - 23
    numerator, denominator = (n, d << shift) if shift >= 0 else (n << -shift, d)
    mantissa, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and mantissa % 2):
        mantissa += 1
    if exponent < -126:
        bits = mantissa  # includes rounding up to the minimum normal.
    else:
        if mantissa == 1 << 24:
            exponent += 1
            mantissa >>= 1
        need(exponent < 128, "finite binary32 intermediate")
        bits = ((exponent + 127) << 23) | (mantissa - (1 << 23))
    return struct.unpack("<f", struct.pack("<I", sign | bits))[0]


def _fraction(value):
    need(type(value) is float and math.isfinite(value), "finite decoded binary32")
    return Fraction(value)


def gauss_world(vectors):
    """Exact mathematical result, two explicitly rounded models, error bound.

    All subtractions round to binary32. Separate mode rounds product then sum;
    contracted mode rounds product-plus-sum once. Both halve the final sum.
    The gamma_80 bound against unrounded input magnitudes is conservative for
    these finite, gradual-underflow RNE models, not a bound on arbitrary CUDA.
    """
    need(type(vectors) is tuple and len(vectors) == 4
         and all(type(v) is tuple and len(v) == 20 for v in vectors),
         "literal four twenty-DOF vectors")
    exact = magnitude = Fraction(0)
    separate = contracted = 0.0
    for ma, smooth, acc, acc_smooth in zip(*vectors):
        ma, smooth, acc, acc_smooth = map(_fraction, (ma, smooth, acc, acc_smooth))
        a, b = ma - smooth, acc - acc_smooth
        exact += a * b
        magnitude += (abs(ma) + abs(smooth)) * (abs(acc) + abs(acc_smooth))
        a32, b32 = Fraction(round32(a)), Fraction(round32(b))
        product = Fraction(round32(a32 * b32))
        separate = round32(Fraction(separate) + product)
        contracted = round32(Fraction(contracted) + a32 * b32)
    separate = round32(Fraction(separate) / 2)
    contracted = round32(Fraction(contracted) / 2)
    unit_roundoff = Fraction(1, 1 << 24)
    gamma = 80 * unit_roundoff / (1 - 80 * unit_roundoff)
    # Absolute floor covers RNE subnormal error across <80 operations, including
    # final halving. It is intentionally loose rather than a near-zero epsilon.
    bound = gamma * magnitude / 2 + Fraction(80, 1 << 149)
    return exact / 2, separate, contracted, bound


def analyze(fields):
    """Pure all-active 64-world Gauss view, with complete input-byte bindings."""
    expected = set(NAMES) | {"context.gauss", "context.done"}
    need(type(fields) is dict and set(fields) == expected, "closed Gauss field set")
    for name, raw in fields.items():
        size = 64 if name == "context.done" else 64 * (20 if name in NAMES else 1) * 4
        need(type(raw) is bytes and len(raw) == size, "literal Gauss field bytes " + name)
    need(fields["context.done"] == bytes(64), "all-active retained Gauss boundary")
    rows = []
    for world in range(64):
        vectors = tuple(struct.unpack_from("<20f", fields[name], world * 80) for name in NAMES)
        exact, separate, contracted, bound = gauss_world(vectors)
        actual = struct.unpack_from("<f", fields["context.gauss"], world * 4)[0]
        delta = abs(_fraction(actual) - exact)
        raw = fields["context.gauss"][world * 4:world * 4 + 4]
        rows.append(dict(world=world, captured=actual, mathematical=float(exact),
                         absolute_error=float(delta), rne_bound=float(bound),
                         within_rne_bound=delta <= bound,
                         separate_product_sum=separate, contracted_product_sum=contracted,
                         separate_bits_equal=raw == struct.pack("<f", separate),
                         contracted_bits_equal=raw == struct.pack("<f", contracted)))
    consistent = all(row["within_rne_bound"] for row in rows)
    return dict(protocol=PROTOCOL,
                decision="retained-gauss-arithmetic-consistent-only" if consistent else
                         "retained-gauss-arithmetic-inconsistent",
                worlds=64, dofs_per_world=20, all_active=True,
                rounding="exact-rational-binary32-rne-gradual-underflow-two-hypotheses",
                bound="gamma80-times-input-magnitude-half-plus-80-times-min-subnormal",
                fields={name: dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
                        for name, raw in sorted(fields.items())},
                outside_bound=sum(not row["within_rne_bound"] for row in rows),
                separate_bit_mismatches=sum(not row["separate_bits_equal"] for row in rows),
                contracted_bit_mismatches=sum(not row["contracted_bits_equal"] for row in rows),
                rows=rows, qualification=dict(FLAGS))


def audit_retained(root, solver_path):
    # Lazy project imports are inert; never import the installed solver or Warp.
    from mjlab_microduck import stance_solver_replay_probe as replay
    from mjlab_microduck import stance_solver_replay_receiver as receiver
    from mjlab_microduck import stance_solver_scratch as scratch
    from mjlab_microduck import stance_solver_target_binding as target

    source = replay.read_plain(solver_path)
    target.verify_source(source)
    inv_raw = replay.read_plain(root / replay.INVENTORY)
    meta_raw = replay.read_plain(root / replay.HISTORY / "child.json")
    raw = replay.read_plain(root / replay.HISTORY / replay.RAW_NAME, scratch.MAX_PACKET_BYTES)
    need(sha256(inv_raw).hexdigest() == replay.INPUT["inventory_sha256"]
         and sha256(meta_raw).hexdigest() == replay.INPUT["metadata_sha256"]
         and len(raw) == replay.INPUT["raw_bytes"]
         and sha256(raw).hexdigest() == replay.INPUT["raw_sha256"],
         "whole historical anchors before JSON parsing")
    inventory = receiver._json(inv_raw, "historical inventory")
    metadata = receiver._json(meta_raw, "historical metadata")
    need(inventory["child.json"] == dict(bytes=len(meta_raw), sha256=sha256(meta_raw).hexdigest())
         and inventory[replay.RAW_NAME] == dict(bytes=len(raw), sha256=sha256(raw).hexdigest()),
         "authenticated named inventory leaves")
    record = metadata["arms"]["original"]["solver_init"]
    packet = scratch.decode_packet(raw, record["snapshot"], replay.INPUT["raw_sha256"],
                                   record["recipe"], elliptic_type=7)
    result = analyze({name: packet.fields[name] for name in (*NAMES, "context.gauss", "context.done")})
    result["anchors"] = dict(replay.INPUT, solver_bytes=len(source), solver_sha256=target.SOLVER_SHA256)
    result["boundary"] = dict(phase="initialized-before-search", forward=4,
                              active_elliptic_rows=0, contact_friction_restored=False,
                              total_cost_checked=False, init_cost_transition_checked=False,
                              efc_ma_production_checked=False, gpu_execution_checked=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    need(Path.cwd().resolve() == root, "repository working directory")

    def git(*words):
        return subprocess.check_output(("git", *words), cwd=root, timeout=15).decode().strip()

    need(git("branch", "--show-current") == "feat/athletics-obstacle-curriculum"
         and not git("status", "--porcelain"), "clean exact feature branch")
    source = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", BASE, source)
    need(set(git("diff", "--name-only", BASE, source).splitlines()) == OWN,
         "distinct three-path CPU side-view source fence")
    path = (root / ".venv/lib/python3.12/site-packages/mujoco_warp/_src/solver.py").resolve(strict=True)
    result = audit_retained(root, path)
    result["source"] = dict(commit=source, tree=git("rev-parse", "HEAD^{tree}"), base=BASE)
    need(not git("status", "--porcelain") and git("rev-parse", "HEAD") == source,
         "source unchanged after CPU audit")
    output = args.output.absolute()
    need(output.parent.resolve(strict=True) == output.parent
         and output.parent == root / "artifacts/tools/retained-gauss-arithmetic",
         "dedicated canonical existing CPU artifact directory")
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), bytes=len(encoded), sha256=sha256(encoded).hexdigest(),
                          decision=result["decision"], outside_bound=result["outside_bound"],
                          separate_bit_mismatches=result["separate_bit_mismatches"],
                          contracted_bit_mismatches=result["contracted_bit_mismatches"]), sort_keys=True))
    return 0 if result["outside_bound"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
