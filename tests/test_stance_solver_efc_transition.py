"""CPU-only response hypotheses and prospective synthetic packet checks."""
import ast
import enum
from fractions import Fraction
from hashlib import sha256
import itertools
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mjlab_microduck import stance_solver_efc_transition as efc
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_gauss_audit as arithmetic


@pytest.mark.parametrize("kind,j,state,force,cost", [
    ("equality", -2.0, 1, 4.0, 4.0), ("equality", 0.5, 1, -1.0, 0.25),
    ("friction", -2.0, 2, 2.0, 3.0), ("friction", -1.0, 2, 2.0, 1.0),
    ("friction", -0.5, 1, 1.0, 0.25), ("friction", 0.0, 1, -0.0, 0.0),
    ("friction", 0.5, 1, -1.0, 0.25), ("friction", 1.0, 3, -2.0, 1.0),
    ("friction", 2.0, 3, -2.0, 3.0), ("unilateral", -1.0, 1, 2.0, 1.0),
    ("unilateral", 0.0, 0, 0.0, 0.0), ("unilateral", 1.0, 0, 0.0, 0.0),
])
def test_exact_dyadic_rows_and_thresholds(kind, j, state, force, cost):
    r = efc.response(kind, 2.0, j, 2.0)
    assert r["state"] == state and struct.pack("<f", r["force"]) == struct.pack("<f", force)
    assert r["cost_models"] == (cost, cost)
    assert r["atomic_add"] is (kind != "unilateral" or j < 0)


@pytest.mark.parametrize("kind,negative,expected", [
    ("equality", False, 0x80000000), ("equality", True, 0),
    ("unilateral", False, 0), ("unilateral", True, 0),
])
def test_signed_zero_force_model(kind, negative, expected):
    r = efc.response(kind, 2.0, -0.0 if negative else 0.0, 0.0)
    assert struct.unpack("<I", struct.pack("<f", r["force"]))[0] == expected


def test_zero_friction_negative_comparison_wins_at_zero():
    r = efc.response("friction", 2.0, 0.0, 0.0)
    assert r["state"] == 2 and r["branch"] == "friction-linear-negative"


@pytest.mark.parametrize("loss", (0.0, -0.0))
def test_linear_friction_cost_preserves_negative_zero_term(loss):
    r = efc.response("friction", 2.0, -0.0, loss)
    assert r["state"] == 2 and struct.pack("<f", r["force"]) == struct.pack("<f", loss)
    assert all(struct.unpack("<I", struct.pack("<f", c))[0] == 0x80000000 for c in r["cost_models"])


@pytest.mark.parametrize("args", [
    ("elliptic", 2.0, 1.0, 1.0), (False, 2.0, 1.0, 1.0),
    ("friction", 0.0, 1.0, 1.0), ("friction", -2.0, 1.0, 1.0),
    ("friction", 2.0**-21, 1.0, 1.0), ("friction", 2.0**21, 1.0, 1.0),
    ("friction", 2.0, 2.0**21, 1.0), ("friction", 2.0, 1.0, -1.0),
    ("friction", 2.0, 1.0, 2.0**21), ("friction", 2, 1.0, 1.0),
    ("friction", 2.0, float("nan"), 1.0), ("friction", float("inf"), 1.0, 1.0),
    ("friction", 2.0, 0.1, 1.0), ("friction", 2.0, 1e300, 1.0),
])
def test_explicit_unsupported_response_domain(args):
    with pytest.raises(ValueError): efc.response(*args)


def test_gradual_underflow_and_finite_boundary_model():
    r = efc.response("equality", 2.0**-20, 2.0**-149, 0.0)
    assert r["cost_models"] == (0.0, 0.0)
    assert struct.pack("<f", r["force"]) == struct.pack("<I", 0x80000000)
    assert efc.response("equality", 2.0**20, 2.0**20, 0.0)["cost_models"] == (2.0**59, 2.0**59)


def test_any_atomic_permutation_inside_conditional_interval():
    terms = (2.0**24, 1.0, 0.5, 2.0**-149)
    lo, hi, _, _ = efc.cost_interval(tuple((t, t) for t in terms))
    observed = set()
    for perm in itertools.permutations(terms):
        value = 0.0
        for t in perm: value = arithmetic.round32(Fraction(value) + Fraction(t))
        observed.add(value)
        assert lo <= Fraction(value) <= hi
    assert len(observed) > 1  # order is deliberately not bitwise qualified


@pytest.mark.parametrize("terms", [([],), ((-1.0, 0.0),), ((1.0,),), ((1, 1.0),),
    ((float("nan"), 0.0),), ((float("inf"), 0.0),), ((2.0**127, 2.0**127),) * 2,
    ((1.0, 1.0),) * 513])
def test_invalid_atomic_envelopes_refuse(terms):
    with pytest.raises(ValueError): efc.cost_interval(terms)


def test_empty_and_subnormal_atomic_interval():
    assert efc.cost_interval(()) == (Fraction(0),) * 4
    lo, hi, low, high = efc.cost_interval(((2.0**-149, 2.0**-149),))
    assert low == high == Fraction(1, 1 << 149) and lo <= low <= hi


@pytest.fixture(scope="module")
def packet():
    return efc.historical(Path(efc.__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def prospective(packet):
    fields = {n: packet.fields[n] for n in stages.ORDER} | efc.control_fields(packet)
    banks = {}
    counts, ne, nf = (struct.unpack("<64i", fields[n]) for n in ("data.nefc", "data.ne", "data.nf"))
    for stage in stages.STAGES:
        banks[stage + ".before"] = stages.pack_bank(fields)
        if stage == "init_cost":
            fields["context.prev_cost"] = fields["context.cost"]
            fields["context.gauss"] = fields["context.cost"] = bytes(256)
        elif stage == "efc":
            force, state, costs = bytearray(fields["efc.force"]), bytearray(fields["efc.state"]), []
            # Independent dyadic equations, not calls to the response oracle.
            for w, count in enumerate(counts):
                cost = 0.0
                for row in range(count):
                    off = (w * 512 + row) * 4
                    j = struct.unpack_from("<f", fields["context.Jaref"], off)[0]
                    if row < ne[w]: f, s, c = -2*j, 1, j*j
                    elif row < ne[w] + nf[w]:
                        if j <= -1: f, s, c = 2.0, 2, 2*abs(j)-1
                        elif j >= 1: f, s, c = -2.0, 3, 2*abs(j)-1
                        else: f, s, c = -2*j, 1, j*j
                    elif j >= 0: f, s, c = 0.0, 0, 0.0
                    else: f, s, c = -2*j, 1, j*j
                    struct.pack_into("<f", force, off, f)
                    struct.pack_into("<i", state, off, s)
                    cost += c
                costs.append(cost)
            fields["efc.force"], fields["efc.state"] = bytes(force), bytes(state)
            fields["context.cost"] = struct.pack("<64f", *costs)
        elif stage == "dense":
            jac = np.frombuffer(fields["efc.J"], dtype="<f4").reshape(64,512,20)
            force = np.frombuffer(fields["efc.force"], dtype="<f4").reshape(64,512)
            result = np.zeros((64,20), dtype="<f4")
            for w, count in enumerate(counts):
                for row in range(count): result[w] = np.float32(result[w] + np.float32(jac[w,row] * force[w,row]))
            fields["data.qfrc_constraint"] = result.tobytes()
        else:
            fields["context.gauss"] = struct.pack("<64f", *([10.0] * 64))
            fields["context.cost"] = struct.pack("<64f", *(c + 10.0 for c in struct.unpack("<64f", fields["context.cost"])))
        banks[stage + ".after"] = stages.pack_bank(fields)
    anchors = {p: dict(bytes=len(raw), sha256=sha256(raw).hexdigest()) for p, raw in banks.items()}
    return NS(banks=banks, anchors=anchors)


def test_control_exact_thirteen_overrides_preserve_complete_inactive_tails(packet):
    values = efc.control_fields(packet)
    assert len(values) == 13 and set(stages.control_fields()) < set(values)
    counts = struct.unpack("<64i", packet.fields["data.nefc"])
    for name in ("context.Jaref", "efc.D", "efc.frictionloss", "efc.force", "efc.state"):
        for w, count in enumerate(counts):
            start, end = (w*512+count)*4, (w+1)*512*4
            assert values[name][start:end] == packet.fields[name][start:end]
    manifest = efc.control_manifest(packet)
    assert manifest["synthetic_only"] and not any(manifest["qualification"].values())


def test_whole_positive_transition_contract(packet, prospective):
    r = efc.receive_control(prospective.banks, prospective.anchors, packet)
    assert r["efc"]["force_changes"] == r["efc"]["state_changes"] == 2944
    assert r["stages"]["dense"]["mismatches"] == 0
    branches = {name for w in r["efc"]["rows"] for name in w["branches"]}
    assert branches == {"friction-linear-negative", "friction-linear-positive", "friction-quadratic",
                        "unilateral-satisfied", "unilateral-quadratic"}
    assert not any(r["qualification"].values()) and not r["native_execution_authenticated"]


@pytest.mark.parametrize("fault", ("force", "state", "cost", "tail", "non-output", "done", "elliptic", "zero-d"))
def test_efc_response_rejects_or_reports_wrong_complete_bank(prospective, fault):
    before, after = (stages.unpack_bank(prospective.banks[p]) for p in ("efc.before", "efc.after"))
    if fault in ("force", "state", "cost"):
        name = {"force":"efc.force", "state":"efc.state", "cost":"context.cost"}[fault]
        after[name] = bytes(len(after[name]))
        r = efc.analyze(before, after)
        assert any(r[k] for k in ("force_mismatches", "state_mismatches", "cost_outside_interval"))
        return
    if fault == "tail": after["efc.force"] = after["efc.force"][:-1] + b"x"
    elif fault == "non-output": after["efc.D"] = bytes(len(after["efc.D"]))
    elif fault == "done": before["context.done"] = b"\1" + bytes(63)
    elif fault == "elliptic": before["efc.type"] = struct.pack("<i", 7) + before["efc.type"][4:]
    else: before["efc.D"] = bytes(4) + before["efc.D"][4:]
    with pytest.raises(ValueError): efc.analyze(before, after)


@pytest.mark.parametrize("phase", stages.PHASES)
def test_every_boundary_anchor_before_any_decode(packet, prospective, monkeypatch, phase):
    banks = dict(prospective.banks)
    banks[phase] = b"x" + banks[phase][1:]
    monkeypatch.setattr(stages, "unpack_bank", lambda _: pytest.fail("premature decode"))
    with pytest.raises(ValueError, match="whole external"): efc.receive_control(banks, prospective.anchors, packet)


def test_reanchoring_wrong_control_initial_bank_does_not_admit(packet, prospective):
    banks = dict(prospective.banks)
    p = "init_cost.before"
    banks[p] = b"x" + banks[p][1:]
    anchors = dict(prospective.anchors, **{p:dict(bytes=len(banks[p]), sha256=sha256(banks[p]).hexdigest())})
    with pytest.raises(ValueError, match="exact historical"): efc.receive_control(banks, anchors, packet)


def test_unsealed_external_packet_does_not_admit(prospective):
    with pytest.raises(ValueError, match="sealed authenticated"):
        efc.receive_control(prospective.banks, prospective.anchors, NS(_sealed=True))


@pytest.mark.parametrize("fault", ("missing", "extra", "mutable", "short", "nonzero-cost", "nan-cost", "negative-zero-cost"))
def test_malformed_or_nonfinite_bank_contract(prospective, fault):
    before, after = (stages.unpack_bank(prospective.banks[p]) for p in ("efc.before", "efc.after"))
    if fault == "missing": del before["efc.D"]
    elif fault == "extra": after["other"] = b""
    elif fault == "mutable": before["efc.D"] = bytearray(before["efc.D"])
    elif fault == "short": after["efc.D"] = after["efc.D"][:-1]
    elif fault == "nonzero-cost": before["context.cost"] = struct.pack("<64f", *([1.0]*64))
    elif fault == "negative-zero-cost": after["context.cost"] = struct.pack("<64I", *([0x80000000]*64))
    else: after["context.cost"] = struct.pack("<64f", *([float("nan")]*64))
    with pytest.raises(ValueError): efc.analyze(before, after)


def test_historical_hashes_precede_all_metadata_decode(monkeypatch):
    root = Path(efc.__file__).resolve().parents[2]
    original = efc.prior.read_plain
    def read(path, *args):
        raw = original(path, *args)
        return b"x" + raw[1:] if path == root / efc.prior.retained.INVENTORY else raw
    monkeypatch.setattr(efc.prior, "read_plain", read)
    monkeypatch.setattr(efc.prior_receiver.old, "_json", lambda *a: pytest.fail("premature historical decode"))
    with pytest.raises(ValueError, match="before decode"): efc.historical(root)


@pytest.mark.parametrize("bank", ("reference-efc.before.bin", "reference-efc.after.bin", "control-efc.before.bin", "control-efc.after.bin"))
def test_reread_bank_hashes_precede_additional_arithmetic(monkeypatch, bank):
    original = efc.prior.read_plain
    def read(path, *args):
        raw = original(path, *args)
        return b"x" + raw[1:] if path.name == bank else raw
    monkeypatch.setattr(efc.prior, "read_plain", read)
    monkeypatch.setattr(efc, "analyze", lambda *a: pytest.fail("premature additional arithmetic"))
    with pytest.raises(ValueError, match="re-read EFC bank"): efc.audit_retained(Path(efc.__file__).resolve().parents[2])


def test_real_cpu_enum_values_match_frozen_state_and_type_literals():
    code = "import mujoco as m; assert [int(getattr(m.mjtConstraintState,'mjCNSTRSTATE_'+n)) for n in ('SATISFIED','QUADRATIC','LINEARNEG','LINEARPOS','CONE')]==list(range(5)); assert int(m.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)==7"
    subprocess.run([sys.executable,"-c",code],check=True,timeout=10)


def test_frozen_kernel_python_body_matches_dyadic_prospective_response(prospective):
    path = Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py"
    raw = path.read_bytes()
    efc.prior.static.verify_source(raw)
    factory = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "update_constraint_efc")
    factory.decorator_list = []  # execute authenticated body with a CPU-only fake Warp namespace
    class ArrayType:
        @classmethod
        def __class_getitem__(cls, _): return object
    class State(enum.IntEnum): SATISFIED=0; QUADRATIC=1; LINEARNEG=2; LINEARPOS=3; CONE=4
    class Kind(enum.IntEnum): CONTACT_ELLIPTIC=7
    cursor = [0,0]
    def atomic(a,w,v): a[w] = np.float32(a[w] + np.float32(v))
    wp = NS(array=ArrayType, array2d=ArrayType, kernel=lambda **kw: lambda f:f,
            tid=lambda: tuple(cursor), static=lambda v:v, atomic_add=atomic)
    ns = dict(wp=wp, types=NS(vec5=object, ConstraintState=State, ConstraintType=Kind),
              math=NS(safe_div=lambda x,y:x/y))
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(path), "exec"), ns)
    kernel = ns["update_constraint_efc"](False)
    fields = stages.unpack_bank(prospective.banks["efc.before"])
    arrays = {n:np.frombuffer(raw, dtype="<i4" if n in ("data.ne","data.nf","data.nefc","efc.type","efc.id","efc.state")
                             else "bool" if n=="context.done" else "<f4").copy()
              for n,raw in fields.items()}
    for n in ("efc.type","efc.id","efc.D","efc.frictionloss","context.Jaref","efc.force","efc.state"):
        arrays[n] = arrays[n].reshape(64,512)
    for w in range(64):
        for row in range(int(arrays["data.nefc"][w])):
            cursor[:] = [w,row]
            kernel(None, arrays["data.ne"], arrays["data.nf"], arrays["data.nefc"], None,None,None,
                arrays["efc.type"], arrays["efc.id"], arrays["efc.D"], arrays["efc.frictionloss"], None,
                arrays["context.Jaref"], arrays["context.done"], arrays["efc.force"], arrays["efc.state"],
                arrays["context.cost"], None,None)
    expected = stages.unpack_bank(prospective.banks["efc.after"])
    assert all(arrays[n].tobytes()==expected[n] for n in ("efc.force","efc.state","context.cost"))


def test_retained_native_response_audit_is_not_a_new_native_transition():
    r = efc.audit_retained(Path(efc.__file__).resolve().parents[2])
    assert not r["positive_transition_gpu_run"] and not any(r["qualification"].values())
    for arm in r["arms"].values():
        assert arm["force_changes"] == arm["state_changes"] == 0
        assert arm["force_mismatches"] == arm["state_mismatches"] == arm["cost_outside_interval"] == 0


def test_preparation_fence_and_inert_import():
    assert len(efc.OWN)==3 and len(efc.TESTS)==18 and efc.prior.BASE=="98aa8a1d1a413f76cadc397492eddb8cb953135e"
    code = "import sys; from mjlab_microduck import stance_solver_efc_transition; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp','numpy'} for n in sys.modules)"
    subprocess.run([sys.executable,"-c",code],check=True,timeout=10)
