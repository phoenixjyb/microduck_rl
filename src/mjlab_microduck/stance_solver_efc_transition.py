"""Pure non-elliptic EFC response/control preparation, never GPU admission.

The row model assumes finite binary32 RNE, gradual underflow and bounded
positive stiffness. Its two friction-cost models and arbitrary-order atomic
envelope are numerical hypotheses, not device lowering or timing evidence.
"""
import argparse
from fractions import Fraction
from hashlib import sha1, sha256
import math
import os
from pathlib import Path
import struct
import sys

from mjlab_microduck import stance_solver_cost_probe as prior
from mjlab_microduck import stance_solver_cost_receiver as prior_receiver
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_gauss_audit as arithmetic
from mjlab_microduck import stance_solver_scratch as scratch

PROTOCOL = "microduck-efc-transition-preparation-oct9-v1"
CONTROL_PROTOCOL = "microduck-efc-positive-transition-control-oct9-v1"
BASE = "b83bc7208b378d5ed949b75505ff4c916058139b"
OWN = frozenset({"src/mjlab_microduck/stance_solver_efc_transition.py",
    "tests/test_stance_solver_efc_transition.py",
    "docs/experiments/2026-10-09-efc-transition-preparation.md"})
TESTS = prior.TESTS + ("tests/test_stance_solver_efc_transition.py",)
FLAGS = dict(stages.FLAGS)
MATH_SHA256 = "4c58d5e3864f7b841633d403d78e7702bb547359ee846d43810e3b31c63966c2"
TYPES_SHA256 = "8f0b19d7b2bc039a419fa547ba6b732837327c0735f688ad34ace5afc4843e50"
PRIOR_SOURCE = "3f6ab6919db420b72de2b232de7eb0f2c150af6f"
PRIOR_INVENTORY_SHA256 = "69735a75f2185a43a03b5f955d284863f661edc4e6e5de72bf81d513860ba54d"
PRIOR_RESULT_SHA256 = "7c94ba163d484578ea8f448567ec55480a283f16169032e6c6238afbcd69b290"
need = arithmetic.need


def _float(value):
    need(type(value) is float and math.isfinite(value), "finite literal binary32 input")
    try:
        decoded = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise ValueError("finite exact binary32 input") from error
    need(decoded == value, "exact binary32 input")
    return Fraction(value)


def _mul(a, b):
    product = _float(a) * _float(b)
    if not product:
        return math.copysign(0.0, math.copysign(1.0, a) * math.copysign(1.0, b))
    return arithmetic.round32(product)


def _add(a, b):
    total = _float(a) + _float(b)
    if not total:
        return -0.0 if a == b == 0.0 and math.copysign(1.0, a) < 0 and math.copysign(1.0, b) < 0 else 0.0
    return arithmetic.round32(total)


def _half_fma(a, b):
    total = _float(a) / 2 + _float(b)
    if not total:
        return -0.0 if a == b == 0.0 and math.copysign(1.0, a) < 0 and math.copysign(1.0, b) < 0 else 0.0
    return arithmetic.round32(total)


def response(kind, stiffness, residual, frictionloss):
    """Source-order force/state and separate versus fused friction-cost models.

    Zero stiffness is deliberately not handled: safe_div's MJ_MINVAL fallback
    and nonzero subnormal division need a separate protocol. No elliptic cone.
    """
    need(type(kind) is str and kind in ("equality", "friction", "unilateral"), "supported literal row class")
    d, j, f = map(_float, (stiffness, residual, frictionloss))
    need(Fraction(1, 1 << 20) <= d <= 1 << 20 and abs(j) <= 1 << 20
         and 0 <= f <= 1 << 20, "bounded positive stiffness and nonnegative frictionloss")
    if kind == "friction":
        rf = math.copysign(0.0, frictionloss) if not f else arithmetic.round32(f / d)
        if residual <= -rf:  # inclusive negative threshold wins, including zero
            state, force, branch, signed_j = 2, frictionloss, "friction-linear-negative", residual
        elif residual >= rf:
            state, force, branch, signed_j = 3, -frictionloss, "friction-linear-positive", -residual
        else:
            state, force, branch = 1, _mul(-stiffness, residual), "friction-quadratic"
        if state != 1:
            half = _mul(0.5, rf)
            separate = _add(half, signed_j)
            fused = _half_fma(rf, signed_j)
            costs = (_mul(-frictionloss, separate), _mul(-frictionloss, fused))
    elif kind == "unilateral" and residual >= 0.0:
        return dict(state=0, force=0.0, branch="unilateral-satisfied", cost_models=(0.0, 0.0), atomic_add=False)
    else:
        state, force, branch = 1, _mul(-stiffness, residual), kind + "-quadratic"
    if state == 1:
        value = _mul(_mul(_mul(0.5, stiffness), residual), residual)
        costs = (value, value)
    need(all(math.isfinite(c) and c >= 0 for c in costs), "finite nonnegative row cost models")
    return dict(state=state, force=force, branch=branch, cost_models=costs, atomic_add=True)


def cost_interval(terms):
    """Conditional RNE envelope for any serial permutation of nonnegative terms.

    Per-row terms may take either modeled rounding. gamma_n uses a conservative
    n rather than n-1, plus n minimum-subnormal units. Exact rationals decide;
    no reported float endpoint is used as an acceptance threshold.
    """
    need(type(terms) is tuple and len(terms) <= 512
         and all(type(t) is tuple and len(t) == 2 for t in terms), "bounded closed row-cost model pairs")
    decoded = tuple(tuple(_float(v) for v in t) for t in terms)
    need(all(v >= 0 for t in decoded for v in t), "nonnegative atomic contributions")
    low, high = sum((min(t) for t in decoded), Fraction(0)), sum((max(t) for t in decoded), Fraction(0))
    n, u = len(terms), Fraction(1, 1 << 24)
    bound = n * u / (1 - n * u) * high + Fraction(n, 1 << 149)
    need(high + bound < (1 << 128) - (1 << 103), "finite arbitrary-order accumulation without overflow")
    return max(Fraction(0), low - bound), high + bound, low, high


def _rows(fields):
    stages.pack_bank(fields)  # closed complete byte/layout contract first
    counts = stages._domain(fields)
    ne, nf = (struct.unpack("<64i", fields[n]) for n in ("data.ne", "data.nf"))
    for world, count in enumerate(counts):
        for row in range(count):
            offset = (world * 512 + row) * 4
            kind = "equality" if row < ne[world] else "friction" if row < ne[world] + nf[world] else "unilateral"
            ty = struct.unpack_from("<i", fields["efc.type"], offset)[0]
            need(0 <= ty < 7, "supported non-elliptic active row type")
            values = tuple(struct.unpack_from("<f", fields[n], offset)[0]
                           for n in ("efc.D", "context.Jaref", "efc.frictionloss"))
            yield world, row, offset, response(kind, *values)


def analyze(before, after):
    """Analyze decoded whole banks; caller must authenticate capture separately."""
    stages.pack_bank(after)
    rows = list(_rows(before))
    need(all(before[n] == after[n] for n in stages.ORDER if n not in stages.WRITES["efc"]),
         "unchanged non-output EFC bank fields")
    need(before["context.cost"] == bytes(256), "positive-zero initialized EFC input cost")
    counts = stages._domain(before)
    for world, count in enumerate(counts):
        start, end = (world * 512 + count) * 4, (world + 1) * 512 * 4
        need(all(before[n][start:end] == after[n][start:end] for n in ("efc.force", "efc.state")),
             "unchanged inactive force/state tails")
    worlds = [dict(terms=[], branches={}, force_mismatches=0, state_mismatches=0,
                   force_changes=0, state_changes=0) for _ in range(64)]
    for world, row, offset, expected in rows:
        w = worlds[world]
        name = expected["branch"]
        w["branches"][name] = w["branches"].get(name, 0) + 1
        force, state = (after[n][offset:offset + 4] for n in ("efc.force", "efc.state"))
        w["force_mismatches"] += force != struct.pack("<f", expected["force"])
        w["state_mismatches"] += state != struct.pack("<i", expected["state"])
        w["force_changes"] += force != before["efc.force"][offset:offset + 4]
        w["state_changes"] += state != before["efc.state"][offset:offset + 4]
        if expected["atomic_add"]: w["terms"].append(expected["cost_models"])
    for index, w in enumerate(worlds):
        lower, upper, model_low, model_high = cost_interval(tuple(w.pop("terms")))
        value = struct.unpack_from("<f", after["context.cost"], index * 4)[0]
        need(value != 0.0 or after["context.cost"][index * 4:index * 4 + 4] == bytes(4),
             "positive-zero aggregate under initialized nonnegative RNE-add model")
        actual = _float(value)
        w.update(world=index, captured_cost=value, cost_within_interval=lower <= actual <= upper,
                 interval_exact=[str(lower), str(upper)], model_sum_exact=[str(model_low), str(model_high)])
    failures = dict(force_mismatches=sum(w["force_mismatches"] for w in worlds),
        state_mismatches=sum(w["state_mismatches"] for w in worlds),
        cost_outside_interval=sum(not w["cost_within_interval"] for w in worlds))
    return dict(protocol=PROTOCOL, decision="efc-row-response-consistent-under-declared-models" if not any(failures.values())
                else "efc-row-response-inconsistent", worlds=64, active_rows=len(rows), **failures,
                force_changes=sum(w["force_changes"] for w in worlds), state_changes=sum(w["state_changes"] for w in worlds),
                rounding="binary32-rne-gradual-underflow-source-order-and-friction-half-fma",
                atomic_order_qualified=False, device_rounding_observed=False, rows=worlds, qualification=dict(FLAGS))


def control_fields(packet):
    """Thirteen literal overrides; active rows only for six new bank fields."""
    stages.initial_banks(packet)  # sealed complete historical packet and digest
    fields = {n: packet.fields[n] for n in stages.ORDER}
    counts = stages._domain(fields)
    ne, nf = (struct.unpack("<64i", fields[n]) for n in ("data.ne", "data.nf"))
    names = ("context.Jaref", "efc.D", "efc.frictionloss", "efc.force", "efc.state")
    new = {n: bytearray(fields[n]) for n in names}
    for world, count in enumerate(counts):
        for row in range(count):
            offset = (world * 512 + row) * 4
            if row < ne[world]: residual = (-1.0, 0.0, 1.0)[(world + row) % 3]
            elif row < ne[world] + nf[world]:
                residual = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)[(world + row) % 7]
                struct.pack_into("<f", new["efc.frictionloss"], offset, 2.0)
            else: residual = (-1.0, -0.5, 0.0, 0.5, 1.0)[(world + row) % 5]
            for name, value in (("context.Jaref", residual), ("efc.D", 2.0),
                                ("efc.force", float(-4096 - world * 512 - row))):
                struct.pack_into("<f", new[name], offset, value)
            struct.pack_into("<i", new["efc.state"], offset, 4)  # CONE canary, never a claimed elliptic row
    new["data.qfrc_constraint"] = struct.pack("<1280f", *(float(-8192 - index) for index in range(1280)))
    return stages.control_fields() | {n: bytes(raw) for n, raw in new.items()}


def control_manifest(packet):
    values = control_fields(packet)
    return dict(protocol=CONTROL_PROTOCOL, synthetic_only=True, inherited_gauss_control=stages.CONTROL_PROTOCOL,
        historical_raw_sha256=packet.sha256,
        overrides={n: dict(bytes=len(raw), sha256=sha256(raw).hexdigest()) for n, raw in sorted(values.items())},
        expected_active_force_state_changes=True, expected_dense_change_each_world=True,
        stiffness=2.0, friction_threshold=1.0, expected_gauss=10.0, qualification=dict(FLAGS))


def receive_control(packets, anchors, packet):
    """Whole eight-bank authentication before interpretation; no native proof."""
    need(type(packets) is dict and type(anchors) is dict
         and set(packets) == set(anchors) == set(stages.PHASES), "closed eight-bank transition envelope")
    for phase in stages.PHASES:
        raw, a = packets[phase], anchors[phase]
        need(type(raw) is bytes and len(raw) == stages.PACKET_BYTES and type(a) is dict
             and set(a) == {"bytes", "sha256"} and type(a["bytes"]) is int
             and a["bytes"] == len(raw) and a["sha256"] == sha256(raw).hexdigest(), "whole external transition anchor")
    overrides = control_fields(packet)  # validate type, seal and whole digest first
    initial = {n: packet.fields[n] for n in stages.ORDER} | overrides
    need(packets["init_cost.before"] == stages.pack_bank(initial), "exact historical bank plus thirteen overrides")
    stage_result = stages.analyze_arm(packets, arm="control")
    efc = analyze(stages.unpack_bank(packets["efc.before"]), stages.unpack_bank(packets["efc.after"]))
    need(not any(efc[k] for k in ("force_mismatches", "state_mismatches", "cost_outside_interval")), "positive EFC response model consistency")
    need(efc["force_changes"] == efc["state_changes"] == efc["active_rows"], "every active force and state canary replaced")
    need(all(w["model_sum_exact"][0] == w["model_sum_exact"][1]
             and Fraction(w["captured_cost"]) == Fraction(w["model_sum_exact"][0]) for w in efc["rows"]),
         "exact dyadic synthetic EFC aggregate cost")
    before, after = (stages.unpack_bank(packets[p])["data.qfrc_constraint"] for p in ("dense.before", "dense.after"))
    need(all(before[w * 80:(w + 1) * 80] != after[w * 80:(w + 1) * 80] for w in range(64)),
         "positive dense output change in every world")
    return dict(protocol=PROTOCOL, decision="synthetic-transition-packet-contract-only", control=control_manifest(packet),
                stages=stage_result, efc=efc, capture_origin_authenticated=False, native_execution_authenticated=False,
                physical_stability_checked=False, qualification=dict(FLAGS))


def historical(root):
    """Authenticate all historical leaves before metadata interpretation."""
    raw_inv, raw_meta, raw = (prior.read_plain(root / p) for p in (prior.retained.INVENTORY,
        prior.retained.HISTORY + "/child.json", prior.retained.HISTORY + "/" + prior.retained.RAW_NAME))
    inp = prior.retained.INPUT
    need(sha256(raw_inv).hexdigest() == inp["inventory_sha256"]
         and sha256(raw_meta).hexdigest() == inp["metadata_sha256"]
         and len(raw) == inp["raw_bytes"] and sha256(raw).hexdigest() == inp["raw_sha256"], "whole historical anchors before decode")
    inv, meta = (prior_receiver.old._json(v, name) for v, name in ((raw_inv, "inventory"), (raw_meta, "metadata")))
    need(inv["child.json"] == dict(bytes=len(raw_meta), sha256=sha256(raw_meta).hexdigest())
         and inv[prior.retained.RAW_NAME] == dict(bytes=len(raw), sha256=sha256(raw).hexdigest()), "same named historical leaves")
    s = meta["arms"]["original"]["solver_init"]
    return scratch.decode_packet(raw, s["snapshot"], inp["raw_sha256"], s["recipe"], elliptic_type=7)


def audit_retained(root):
    """Add a pure row oracle to independently re-received prior native evidence."""
    packet = historical(root)
    closed = root / "artifacts/tools" / ("caller-cost-closeout-" + PRIOR_SOURCE[:12])
    inv_raw, result_raw = prior.read_plain(closed / "inventory.json"), prior.read_plain(closed / "native-receiver.json")
    need(sha256(inv_raw).hexdigest() == PRIOR_INVENTORY_SHA256
         and sha256(result_raw).hexdigest() == PRIOR_RESULT_SHA256, "literal previously received native evidence anchors")
    inv, closeout = prior.read_json(closed / "inventory.json"), prior.read_json(closed / "retirement.json")
    directory = root / "artifacts/evaluations" / ("caller-cost-" + PRIOR_SOURCE[:12])
    received = prior_receiver.receive(directory, inv, closeout, packet)
    need(prior.canonical(received) == result_raw, "independently reproduced prior receiver bytes")
    # Reauthenticate every additionally read bank before decoding any of them.
    bank_raw = {arm + "-" + phase + ".bin": prior.read_plain(directory / (arm + "-" + phase + ".bin"))
                for arm in ("reference", "control") for phase in ("efc.before", "efc.after")}
    need(all(inv[n] == dict(bytes=len(raw), sha256=sha256(raw).hexdigest()) for n, raw in bank_raw.items()),
         "whole re-read EFC bank anchors before additional decode")
    arms = {arm: analyze(*(stages.unpack_bank(bank_raw[arm + "-" + phase + ".bin"])
            for phase in ("efc.before", "efc.after"))) for arm in ("reference", "control")}
    need(all(not r[k] for r in arms.values() for k in ("force_mismatches", "state_mismatches", "cost_outside_interval")),
         "retained EFC model consistency")
    return dict(protocol=PROTOCOL, decision="retained-efc-response-model-audited-not-native-transition",
        prior_source=PRIOR_SOURCE, inventory_sha256=PRIOR_INVENTORY_SHA256, prior_result_sha256=PRIOR_RESULT_SHA256,
        arms=arms, prospective_control=control_manifest(packet), positive_transition_gpu_run=False, qualification=dict(FLAGS))


def source_binding(root):
    cmd = prior.retained.command
    need(Path.cwd().resolve() == root and cmd("git", "branch", "--show-current").decode().strip() == prior.BRANCH
         and not cmd("git", "status", "--porcelain").strip(), "clean exact local CPU feature branch")
    source = cmd("git", "rev-parse", "HEAD").decode().strip()
    cmd("git", "merge-base", "--is-ancestor", BASE, source)
    need(set(cmd("git", "diff", "--name-only", BASE, source).decode().splitlines()) == OWN, "new exact three-path preparation fence")
    leaves = []
    for row in cmd("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row: continue
        header, name = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed CPU source file")
        raw = prior.read_plain(root / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid, "whole committed CPU blob")
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(commit=source, tree=cmd("git", "rev-parse", source + "^{tree}").decode().strip(), base=BASE, leaves=leaves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-only preparation")
    root = Path(__file__).resolve().parents[2]
    source = source_binding(root)
    prefix = Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src"
    prior.static.verify_source(prior.read_plain((prefix / "solver.py").resolve(strict=True)))
    frontend = {n: sha256(prior.read_plain((prefix / n).resolve(strict=True))).hexdigest() for n in ("math.py", "types.py")}
    need(frontend == {"math.py": MATH_SHA256, "types.py": TYPES_SHA256}, "whole frozen row arithmetic and enum source pins")
    result = audit_retained(root) | dict(source=source, frontend=frontend)
    need(source_binding(root) == source and not any(n in sys.modules for n in ("warp", "torch", "mujoco", "mujoco_warp")),
         "unchanged source and no runtime imports")
    output = args.output.absolute()
    need(output.parent == root / "artifacts/tools/efc-transition-preparation"
         and output.parent.resolve(strict=True) == output.parent, "dedicated canonical CPU output directory")
    prior.frozen.write(output, result)
    raw = prior.read_plain(output)
    print(dict(decision=result["decision"], bytes=len(raw), sha256=sha256(raw).hexdigest(), qualification=FLAGS))


if __name__ == "__main__": main()
