"""Pure force-to-gradient prefix preparation, never a solver/GPU admission.

The two frozen kernels precede both full and incremental gradient updates.
Their byte/rounding contracts do not cover Hessian construction, factorization,
line search, search update, convergence, integration or physical stability.
"""
import argparse
import ast
from fractions import Fraction
from hashlib import sha1, sha256
import os
from pathlib import Path
import struct
import sys

from mjlab_microduck import stance_solver_efc_probe as prior
from mjlab_microduck import stance_solver_efc_receiver as receiver
from mjlab_microduck import stance_solver_efc_transition as efc
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_gauss_audit as arithmetic
from mjlab_microduck import stance_solver_target_binding as static

PROTOCOL = "microduck-gradient-prefix-preparation-oct9-v1"
BASE = "d79ecf5a384aed92a9292b3f34bed9cced2af1af"
OWN = frozenset({"src/mjlab_microduck/stance_solver_gradient_prefix.py",
    "tests/test_stance_solver_gradient_prefix.py",
    "docs/experiments/2026-10-09-gradient-prefix-preparation.md"})
TESTS = prior.TESTS + ("tests/test_stance_solver_gradient_prefix.py",)
FLAGS = dict(stages.FLAGS)
PRIOR_SOURCE = "7717b4cbef3aad06de6c3bf33463076f067e4253"
PRIOR_INVENTORY_SHA256 = "bac5c1f3958504649db5d443640e47782b8b16110d6be7a38e63102afcfcee4f"
PRIOR_RETIREMENT_SHA256 = "5222b93af19c5826d5f133b873601597af3dffc6eed52183a74d70c4010ebeba"
PRIOR_RESULT_SHA256 = "bb493d31cdc47d47248e851629a9c4761e754346a78098c2360eddcb2fc38f7b"
STAGES = ("zero_grad_dot", "gradient")
PHASES = tuple(s + "." + p for s in STAGES for p in ("before", "after"))
EXTRA = ("context.grad", "context.grad_dot")
ORDER = stages.ORDER + EXTRA
PACKET_BYTES = stages.PACKET_BYTES + 1280 * 4 + 64 * 4
MAX_TOTAL_BYTES = 32 * 1024**2
need = static.need

KERNELS = '''
@wp.kernel
def update_gradient_zero_grad_dot(ctx_done_in: wp.array[bool], ctx_grad_dot_out: wp.array[float]):
    worldid = wp.tid()
    if ctx_done_in[worldid]:
        return
    ctx_grad_dot_out[worldid] = 0.0

@wp.kernel
def update_gradient_grad(
    qfrc_smooth_in: wp.array2d[float], qfrc_constraint_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float], ctx_done_in: wp.array[bool],
    ctx_grad_out: wp.array2d[float], ctx_grad_dot_out: wp.array[float],
):
    worldid, dofid = wp.tid()
    if ctx_done_in[worldid]:
        return
    grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]
    ctx_grad_out[worldid, dofid] = grad
    wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)
'''
PREFIX = '''
def _update_gradient(m: types.Model, d: types.Data, ctx: SolverContext):
    wp.launch(update_gradient_zero_grad_dot, dim=(d.nworld), inputs=[ctx.done], outputs=[ctx.grad_dot])
    wp.launch(update_gradient_grad, dim=(d.nworld, m.nv),
              inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, ctx.done],
              outputs=[ctx.grad, ctx.grad_dot])
'''
INITIALIZATION_TAIL = '''
_update_constraint(m, d, ctx)
if grad:
    _update_gradient(m, d, ctx)
'''
ITERATION_ROUTE = '''
incremental = m.opt.solver == types.SolverType.NEWTON and m.opt.cone != types.ConeType.ELLIPTIC
if incremental:
    ctx.changed_efc_count.zero_()
_update_constraint(m, d, ctx, track_changes=incremental)
if incremental:
    _update_gradient_incremental(m, d, ctx)
else:
    _update_gradient(m, d, ctx)
'''


def source_profile(raw):
    """Structural helper only; verify_source adds the whole installed-byte pin."""
    need(type(raw) is bytes and 0 < len(raw) <= static.MAX_SOURCE_BYTES, "bounded literal solver source")
    tree, expected = ast.parse(raw.decode()), ast.parse(KERNELS)
    defs = {}
    names = {n.name for n in expected.body} | {"_update_gradient", "_update_gradient_incremental", "init_context", "_solver_iteration"}
    for name in names:
        found = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        need(len(found) == 1 and found[0] in tree.body and type(found[0]) is ast.FunctionDef,
             "unique top-level gradient contract function " + name)
        defs[name] = found[0]
    for kernel in expected.body:
        need(static.shape(defs[kernel.name]) == static.shape(kernel), "exact gradient kernel signature/arithmetic")
    prefix = ast.parse(PREFIX).body[0]
    callers, launches = {}, {}
    for name in ("_update_gradient", "_update_gradient_incremental"):
        fn = defs[name]
        body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
        need(not fn.decorator_list and fn.returns is None and static.shape(fn.args) == static.shape(prefix.args)
             and [static.shape(n) for n in body[:2]] == [static.shape(n) for n in prefix.body],
             "exact two-launch gradient prefix " + name)
        launches[name] = body[:2]
        callers[name] = [n.lineno for n in launches[name]]
    need([static.shape(n) for n in defs["init_context"].body[-2:]] ==
         [static.shape(n) for n in ast.parse(INITIALIZATION_TAIL).body], "exact constraint-to-gradient initialization tail")
    route = [static.shape(n) for n in ast.parse(ITERATION_ROUTE).body]
    body = [static.shape(n) for n in defs["_solver_iteration"].body]
    need(sum(body[i:i+len(route)] == route for i in range(len(body))) == 1,
         "exact distinct Newton incremental iteration route")
    for kernel in expected.body:
        refs = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == kernel.name]
        selected = [launches[name][i].value.args[0]
                    for name in callers for i in range(2)
                    if launches[name][i].value.args[0].id == kernel.name]
        need(len(refs) == 2 and {id(n) for n in refs} == {id(n) for n in selected}, "only the two declared gradient references")
    return dict(kernels={k.name: dict(line=defs[k.name].lineno,
        ast_sha256=sha256(static.shape(defs[k.name]).encode()).hexdigest()) for k in expected.body},
        caller_lines=callers, initialization_track_changes=False,
        newton_pyramidal_iteration_track_changes=True, iteration_uses_incremental_gradient=True)


def verify_source(raw):
    need(type(raw) is bytes and 0 < len(raw) <= static.MAX_SOURCE_BYTES, "bounded literal solver source")
    need(sha256(raw).hexdigest() == static.SOLVER_SHA256, "whole frozen solver source hash")
    return dict(protocol=PROTOCOL, source_sha256=static.SOLVER_SHA256, source_bytes=len(raw),
        profile=source_profile(raw), actual_dispatch_observed=False, qualification=dict(FLAGS))


def gradient(ma, smooth, constraint):
    """Left-associated finite binary32 RNE subtraction with signed zeros."""
    for value in (ma, smooth, constraint): efc._float(value)
    return efc._add(efc._add(ma, -smooth), -constraint)


def prediction(fields):
    """Input-only prediction; never substitute it for captured output bytes."""
    stages.pack_bank(fields)
    stages._domain(fields)  # unchanged all-active non-elliptic domain
    vectors = {n: stages._floats(fields[n]) for n in ("efc.Ma", "data.qfrc_smooth", "data.qfrc_constraint")}
    rows, packed = [], []
    for w in range(64):
        grad = tuple(gradient(*(vectors[n][w * 20 + i] for n in vectors)) for i in range(20))
        squares = tuple(arithmetic.round32(efc._float(g) ** 2) for g in grad)
        lo, hi, total, high = efc.cost_interval(tuple((s, s) for s in squares))
        need(total == high, "one rounded-square gradient model")
        packed.extend(grad)
        rows.append(dict(world=w, gradient=list(grad), rounded_squares=list(squares),
                         norm_interval_exact=[str(lo), str(hi)], square_sum_exact=str(total)))
    raw = struct.pack("<1280f", *packed)
    return dict(gradient_bytes=raw, rows=rows)


def pack_bank(fields):
    need(type(fields) is dict and set(fields) == set(ORDER), "closed complete gradient-prefix bank")
    need(all(type(fields[n]) is bytes and len(fields[n]) == size
             for n, size in zip(EXTRA, (5120, 256))), "literal gradient output field bytes")
    return stages.pack_bank({n: fields[n] for n in stages.ORDER}) + b"".join(fields[n] for n in EXTRA)


def unpack_bank(raw):
    need(type(raw) is bytes and len(raw) == PACKET_BYTES, "literal complete gradient-prefix packet")
    return stages.unpack_bank(raw[:stages.PACKET_BYTES]) | {
        EXTRA[0]: raw[stages.PACKET_BYTES:stages.PACKET_BYTES+5120], EXTRA[1]: raw[-256:]}


def initial_banks(parent_after, packet):
    efc.control_fields(packet)  # sealed whole historical bank; never arbitrary namespace
    need(type(parent_after) is dict and set(parent_after) == {"reference", "control"}, "two parent terminal banks")
    control = {EXTRA[0]: struct.pack("<1280f", *(float(-16384-i) for i in range(1280))),
               EXTRA[1]: struct.pack("<64f", *(float((-1 if w % 2 else 1)*(8192+w)) for w in range(64)))}
    return {arm: pack_bank(stages.unpack_bank(parent_after[arm]) |
            (control if arm == "control" else {n: packet.fields[n] for n in EXTRA}))
            for arm in ("reference", "control")}


def analyze_prefix(packets, arm):
    need(type(packets) is dict and set(packets) == set(PHASES) and arm in ("reference", "control"), "closed gradient boundary set")
    fields = {phase: unpack_bank(packets[phase]) for phase in PHASES}
    stages._domain({n: fields[PHASES[0]][n] for n in stages.ORDER})
    need(packets[PHASES[1]] == packets[PHASES[2]], "exact adjacent gradient-bank continuity")
    writes = ({EXTRA[1]}, set(EXTRA))
    for stage, allowed in zip(STAGES, writes):
        before, after = (fields[stage + "." + phase] for phase in ("before", "after"))
        need(all(before[n] == after[n] for n in ORDER if n not in allowed), "unchanged non-output gradient-prefix fields")
    need(fields[PHASES[1]][EXTRA[1]] == bytes(256), "positive-zero norm reset")
    before, after = fields[PHASES[2]], fields[PHASES[3]]
    predicted = prediction({n: before[n] for n in stages.ORDER})
    actual_grad, actual_dot = after[EXTRA[0]], after[EXTRA[1]]
    stages._floats(actual_grad); stages._floats(actual_dot)
    mismatches = sum(actual_grad[i:i+4] != predicted["gradient_bytes"][i:i+4] for i in range(0, 5120, 4))
    outside = 0
    for row in predicted["rows"]:
        w = row["world"]
        bits = actual_dot[w*4:w*4+4]
        dot = struct.unpack("<f", bits)[0]
        need(dot >= 0 and (dot != 0.0 or bits == bytes(4)), "finite nonnegative positive-zero gradient norm")
        lo, hi = map(Fraction, row["norm_interval_exact"])
        outside += not lo <= Fraction(dot) <= hi
    changes = sum(before[EXTRA[0]][i:i+4] != actual_grad[i:i+4] for i in range(0, 5120, 4))
    nonzero = sum(struct.unpack_from("<f", actual_dot, w*4)[0] > 0 for w in range(64))
    need(mismatches == outside == 0, "gradient and squared-norm model consistency")
    if arm == "control": need(changes == 1280 and nonzero == 64, "nonvacuous gradient control in every world")
    return dict(arm=arm, compared_dofs=1280, gradient_bit_mismatches=mismatches,
        norm_outside_interval=outside, gradient_replacements=changes, nonzero_norm_worlds=nonzero,
        rounding="left-associated-binary32-rne-gradual-underflow",
        norm_model="rounded-square-then-arbitrary-order-nonnegative-rne-add",
        device_rounding_observed=False, atomic_order_qualified=False, qualification=dict(FLAGS))


def receive_banks(arms, anchors, parent_after, packet):
    """Pure prospective envelope. Parent authentication belongs to the owner."""
    need(type(arms) is dict and type(anchors) is dict and set(arms) == set(anchors) == {"reference", "control"}, "closed two-arm gradient envelope")
    need(8 * PACKET_BYTES < MAX_TOTAL_BYTES, "bounded gradient bank budget")
    for arm in arms:
        need(type(arms[arm]) is dict and type(anchors[arm]) is dict
             and set(arms[arm]) == set(anchors[arm]) == set(PHASES), "exact four-bank gradient arm")
        for phase, raw in arms[arm].items():
            a = anchors[arm][phase]
            need(type(raw) is bytes and len(raw) == PACKET_BYTES and type(a) is dict
                 and set(a) == {"bytes", "sha256"} and type(a["bytes"]) is int
                 and a == dict(bytes=len(raw), sha256=sha256(raw).hexdigest()), "whole external gradient bank anchor")
    initial = initial_banks(parent_after, packet)  # all eight hashes precede any decode
    need(all(arms[a][PHASES[0]] == initial[a] for a in arms), "exact parent outputs and declared gradient canaries")
    return dict(protocol=PROTOCOL, decision="prospective-gradient-prefix-packet-contract-only",
        packet_bytes=PACKET_BYTES, total_bytes=8*PACKET_BYTES,
        arms={a: analyze_prefix(arms[a], a) for a in sorted(arms)},
        capture_origin_authenticated=False, native_execution_authenticated=False,
        parent_capture_authenticated=False, full_solver_iteration_checked=False, qualification=dict(FLAGS))


def audit_retained(root, solver_path):
    source = verify_source(prior.read_plain(solver_path))
    packet = efc.historical(root)
    closed = root / "artifacts/tools/efc-transition-closeout-7717b4cbef3a"
    raw_inv, raw_ret, raw_result = (prior.read_plain(closed / name) for name in ("inventory.json", "retirement.json", "wsl-receiver.json"))
    need(tuple(sha256(raw).hexdigest() for raw in (raw_inv, raw_ret, raw_result)) ==
         (PRIOR_INVENTORY_SHA256, PRIOR_RETIREMENT_SHA256, PRIOR_RESULT_SHA256), "whole literal predecessor closeout anchors")
    inv, ret = (receiver.old._json(raw, name) for raw, name in ((raw_inv, "inventory"), (raw_ret, "retirement")))
    directory = root / "artifacts/evaluations/efc-transition-7717b4cbef3a"
    result = receiver.receive(directory, inv, ret, packet)
    need(result["source"] == PRIOR_SOURCE and prior.canonical(result) == raw_result, "independently reproduced predecessor receipt")
    banks = {a: prior.read_plain(directory / (a + "-gauss.after.bin")) for a in ("reference", "control")}
    need(all(inv[a + "-gauss.after.bin"] == dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
             for a, raw in banks.items()), "whole re-read parent banks before prediction")
    predictions, gradients = {}, {}
    for arm, raw in banks.items():
        p = prediction(stages.unpack_bank(raw))
        gradients[arm] = p["gradient_bytes"]
        predictions[arm] = dict(parent_bank_sha256=sha256(raw).hexdigest(),
            predicted_gradient_sha256=sha256(p["gradient_bytes"]).hexdigest(), rows=p["rows"],
            captured_gradient_available=False, captured_norm_available=False)
    differences = [sum(gradients["reference"][i:i+4] != gradients["control"][i:i+4]
                       for i in range(w*80, (w+1)*80, 4)) for w in range(64)]
    need(all(n > 0 for n in differences), "nonvacuous paired gradient prediction in each world")
    return dict(protocol=PROTOCOL, decision="retained-force-to-gradient-prediction-only-not-execution",
        solver=source, prior_source=PRIOR_SOURCE, prior_inventory_sha256=PRIOR_INVENTORY_SHA256,
        prior_retirement_sha256=PRIOR_RETIREMENT_SHA256, prior_receiver_sha256=PRIOR_RESULT_SHA256,
        arms=predictions, paired_predicted_gradient_differences=differences,
        gradient_prefix_gpu_run=False, full_solver_iteration_checked=False,
        qualification=dict(FLAGS))


def source_binding(root):
    cmd = prior.retained.command
    need(Path.cwd().resolve() == root and cmd("git", "branch", "--show-current").decode().strip() == prior.BRANCH
         and not cmd("git", "status", "--porcelain").strip(), "clean exact CPU feature branch")
    source = cmd("git", "rev-parse", "HEAD").decode().strip()
    cmd("git", "merge-base", "--is-ancestor", BASE, source)
    need(set(cmd("git", "diff", "--name-only", BASE, source).decode().splitlines()) == OWN, "separate exact three-path gradient fence")
    leaves = []
    for row in cmd("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row: continue
        header, name = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split(); name = name.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed CPU leaf")
        raw = prior.read_plain(root / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid, "whole committed CPU blob")
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(commit=source, tree=cmd("git", "rev-parse", source + "^{tree}").decode().strip(), base=BASE, leaves=leaves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-only gradient preparation")
    root = Path(__file__).resolve().parents[2]
    source = source_binding(root)
    path = (Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py").resolve(strict=True)
    result = audit_retained(root, path) | dict(source=source)
    need(source_binding(root) == source and not any(n.split('.')[0] in {"warp", "torch", "mujoco", "mujoco_warp"} for n in sys.modules),
         "unchanged source and no device imports")
    output = args.output.absolute()
    need(output.parent == root / "artifacts/tools/gradient-prefix-preparation"
         and output.parent.resolve(strict=True) == output.parent, "dedicated canonical CPU output directory")
    prior.frozen.write(output, result)
    raw = prior.read_plain(output)
    print(dict(decision=result["decision"], bytes=len(raw), sha256=sha256(raw).hexdigest(), qualification=FLAGS))


if __name__ == "__main__": main()
