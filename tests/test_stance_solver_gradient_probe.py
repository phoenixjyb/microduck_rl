"""CPU staging/bootstrap/admission checks; synthetic copies are not CUDA proof."""
import ast
from hashlib import sha256
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_stance_solver_gradient_prefix import parents, packet
from mjlab_microduck import stance_solver_gradient_probe as probe
from mjlab_microduck import stance_solver_gradient_prefix as prefix
from mjlab_microduck import stance_solver_gradient_dispatch as dispatch


@pytest.fixture(scope="module")
def initial(parents, packet):
    return prefix.initial_banks(parents, packet)["control"]


@pytest.mark.parametrize("index", (0, 12, 24, 25))
@pytest.mark.parametrize("fault", (None, "pin", "dtype", "bytes", "shape", "gpu", "host-alias",
                                  "host-dst-alias", "dst-alias", "entry", "stream"))
def test_complete_typed_preflight_precedes_first_copy(initial, index, fault):
    gpu, cpu = NS(is_cuda=True, is_cpu=False), NS(is_cuda=False, is_cpu=True)
    dtypes = {n: object() for n in ("float32", "int32", "bool")}
    stream, events, counter = NS(device=gpu, cuda_stream=55), [], [0]
    class Array:
        is_contiguous, requires_grad = True, False
        def __init__(self, host, *, dtype, device, pinned, requires_grad):
            i = counter[0]; counter[0] += 1
            self.host, self.dtype, self.device, self.pinned = host, dtype, cpu, pinned
            self.shape, self.ptr = host.shape, 200_000_000 + i * 4_000_000
            if i == index:
                if fault == "pin": self.pinned = False
                elif fault == "dtype": self.dtype = object()
                elif fault == "bytes": self.host = host.copy(); self.host.flat[0] += 1
                elif fault == "shape": self.shape = (host.size + 1,)
                elif fault == "gpu": self.device = gpu
                elif fault == "host-alias": self.ptr = 200_000_000 if index else 204_000_000
                elif fault == "host-dst-alias": self.ptr = 10_000_000
                elif fault == "entry": wp.copy = lambda *a, **k: None
                elif fault == "stream": stream.cuda_stream += 1
        def numpy(self):
            assert self.device is cpu, "GPU numpy access forbidden"
            return self.host
    arrays = {}
    for i, (n, logical, _, dtype) in enumerate(dispatch.SPECS):
        a = object.__new__(Array)
        a.shape, a.dtype, a.device, a.ptr = logical, dtypes[dtype], gpu, 10_000_000 + i * 4_000_000
        arrays[n] = a
    if fault == "dst-alias": arrays[prefix.ORDER[index]].ptr = arrays[prefix.ORDER[(index + 1) % 26]].ptr
    def copy(dst, src, *, stream):
        assert src.device is cpu and dst.device is gpu and stream.device is gpu
        events.append(("copy", dst.dtype, src.dtype))
    def sync(s):
        assert s is stream
        events.append(("sync",))
    wp = NS(array=Array, copy=copy, synchronize_stream=sync, **dtypes)
    anchor = dict(bytes=len(initial), sha256=sha256(initial).hexdigest())
    if fault is None:
        result = probe.restore_bank(wp, arrays, stream, lambda: None, initial, anchor)
        assert result == probe.restoration_record(initial)
        assert len(events) == 52 and sum(e[0] == "copy" for e in events) == 26
        assert all(e[1] is e[2] for e in events if e[0] == "copy")
    else:
        with pytest.raises(ValueError): probe.restore_bank(wp, arrays, stream, lambda: None, initial, anchor)
        assert events == []


@pytest.mark.parametrize("fault", ("hash", "short", "mutable", "extra", "bool-length"))
def test_restoration_authentication_before_decode(initial, fault, monkeypatch):
    anchor = dict(bytes=len(initial), sha256=sha256(initial).hexdigest())
    raw = initial
    if fault == "hash": anchor["sha256"] = "0" * 64
    elif fault == "short": raw = raw[:-1]
    elif fault == "mutable": raw = bytearray(raw)
    elif fault == "extra": anchor["extra"] = True
    else: anchor["bytes"] = True
    monkeypatch.setattr(prefix, "unpack_bank", lambda _: pytest.fail("premature decode"))
    with pytest.raises(ValueError): probe.restore_bank(None, None, None, None, raw, anchor)


def test_owner_refuses_cuda_or_missing_evidence_before_read(monkeypatch):
    monkeypatch.setattr(probe, "source_binding", lambda *_: pytest.fail("premature source"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"): probe.owner(NS(child=False))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="explicit paired"): probe.owner(NS(child=False, cpu_evidence=None))


def test_child_refuses_wrong_parent_before_read(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe, "read_plain", lambda *_: pytest.fail("premature read"))
    with pytest.raises(ValueError, match="fresh inherited"):
        probe.child(NS(source="a" * 40, child=True, owner_pid=-1))


@pytest.mark.parametrize("fault", (None, "source", "files", "flags", "hash", "case", "skip", "count", "canonical"))
def test_complete_paired_cpu_xml_and_case_multiset(monkeypatch, fault):
    source = "a" * 40
    path = probe.ROOT / "artifacts/tools/mock-cpu.json"
    cases = ''.join(f'<testcase classname="fixture" name="test_{i}"/>' for i in range(1200))
    xml = ('<testsuites><testsuite tests="1200" errors="0" failures="0" skipped="0">' + cases + '</testsuite></testsuites>').encode()
    native = xml
    if fault == "case": native = native.replace(b'test_0"', b'test_changed"')
    elif fault == "skip": native = native.replace(b'skipped="0"', b'skipped="1"')
    elif fault == "count": native = native.replace(b'tests="1200"', b'tests="1201"')
    anchors = {p: dict(path=str(probe.ROOT / ("artifacts/tools/mock-" + p + ".xml")),
                     bytes=len(v), sha256=sha256(v).hexdigest(), tests=1200)
               for p, v in (("mac", xml), ("native", native))}
    value = dict(source=source, test_files=list(probe.TESTS), flags=dict(probe.FLAGS), **anchors)
    if fault == "source": value["source"] = "b" * 40
    elif fault == "files": value["test_files"].pop()
    elif fault == "flags": value["flags"]["training_authorized"] = True
    elif fault == "hash": value["native"]["sha256"] = "0" * 64
    raw = probe.canonical(value) + (b' ' if fault == "canonical" else b'')
    monkeypatch.setattr(probe, "read_plain", lambda p: raw if p == path else native if str(p).endswith("native.xml") else xml)
    monkeypatch.setattr(probe, "read_json", lambda _: value)
    if fault is None: assert probe.tests_record(path, source)["evidence"] == value
    else:
        with pytest.raises(ValueError): probe.tests_record(path, source)


@pytest.mark.parametrize("fault", (None, "file-root", "symlink"))
def test_private_cache_inventory_closed_plain_files(tmp_path, fault):
    p = tmp_path / "warp-cache"
    if fault == "file-root": p.write_bytes(b"bad")
    else:
        p.mkdir()
        (p / "leaf").write_bytes(b"whole")
        if fault == "symlink": (p / "escape").symlink_to(p / "leaf")
    if fault is None:
        assert probe.cache_record(tmp_path)["leaves"] == {"warp-cache/leaf": dict(bytes=5, sha256=sha256(b"whole").hexdigest())}
    else:
        with pytest.raises(ValueError): probe.cache_record(tmp_path)


@pytest.mark.parametrize("fault", (None, "preinstalled", "method", "init", "mutated", "retry", "sealed"))
def test_bootstrap_in_fresh_cpu_only_process(fault, tmp_path):
    # CUDA is hidden; init loads the CPU runtime only, never a compile/launch.
    code = '''
import sys
import warp as wp
from warp._src import context as wc, types as wt, build as wb
from mjlab_microduck import stance_solver_gradient_probe as p
fault = sys.argv[1]
guard = p.api.WarpApiGuard(wp=wp, context=wc, types=wt, build=wb)
if fault == 'preinstalled': wc.runtime = object()
elif fault == 'method': wc.Runtime.load_dll = lambda *a: None
elif fault == 'init': wc.init = wp.init = lambda: None
try:
    binding = p.BootstrapBinding(wp, wc, guard)
    if fault == 'mutated': wc.Runtime.load_dll = lambda *a: None
    elif fault == 'sealed': binding.used = True
    binding.initialize()
    assert binding.record()['held_constructor_completed']
    assert not any(binding.record()['flags'].values())
    if fault == 'retry': binding.initialize()
except (ValueError, AttributeError):
    assert fault != 'None'
else:
    assert fault == 'None'
assert not any(n.split('.')[0] in {'torch','mujoco','mujoco_warp'} for n in sys.modules)
'''
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", WARP_CACHE_PATH=str(tmp_path / "cache"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-c", code, str(fault)], env=env, check=True, timeout=30,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.mark.parametrize("name", ("GradientModuleExecutable", "GradientDispatchObserver"))
def test_actual_immutable_constructor_signature(name):
    constructor = getattr(probe.executable, name) if name == "GradientModuleExecutable" else getattr(dispatch, name)
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(probe.child))) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == name]
    assert len(calls) == 1 and not calls[0].args
    inspect.signature(constructor).bind(**{k.arg: object() for k in calls[0].keywords})


def test_new_fence_limits_and_inert_import():
    assert len(probe.OWN) == 5 and len(probe.TESTS) == 26 and probe.TESTS[:-2] == probe.api.TESTS
    assert probe.BASE == "df973002b20fa99771aa353b27d8111e5a97166a"
    assert probe.BOUNDS["child_seconds"] == 240 and probe.UNIT_CAPS["TasksMax"] == "64"
    assert not any(probe.FLAGS.values())
    code = "from mjlab_microduck import stance_solver_gradient_probe,stance_solver_gradient_receiver; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp','numpy'} for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_literal_parent_receiver_and_reread_anchors(parents, packet):
    root = Path(probe.__file__).resolve().parents[2]
    record, banks, initial, historical = probe.parent_inputs(root, probe.solver_source_path())
    assert banks == parents and historical.raw == packet.raw and historical.sha256 == packet.sha256
    assert initial == prefix.initial_banks(parents, packet)
    assert record["inventory_sha256"] == prefix.PRIOR_INVENTORY_SHA256
    assert record["initial_banks"] == {a: dict(bytes=len(v), sha256=sha256(v).hexdigest()) for a, v in initial.items()}


def test_changed_parent_bank_after_audit_is_rejected_before_decode(monkeypatch):
    root = Path(probe.__file__).resolve().parents[2]
    actual_read = probe.read_plain
    monkeypatch.setattr(prefix, "audit_retained", lambda *_: None)
    def read(path):
        raw = actual_read(path)
        return b"x" + raw[1:] if path.name == "control-gauss.after.bin" else raw
    monkeypatch.setattr(probe, "read_plain", read)
    monkeypatch.setattr(prefix, "initial_banks", lambda *_: pytest.fail("premature parent decode"))
    with pytest.raises(ValueError, match="whole re-read"):
        probe.parent_inputs(root, probe.solver_source_path())


@pytest.mark.parametrize("fault", (None, "supervisor", "xml", "services", "capacity", "source", "runtime", "cgroup"))
def test_capped_owner_orchestration_and_owned_fd_cleanup(tmp_path, monkeypatch, fault):
    source, calls = "a" * 40, []
    out = tmp_path / "evaluations" / "owned"
    out.parent.mkdir()
    fixtures = {}
    for name in ("mac", "native"):
        path = tmp_path / (name + ".xml")
        path.write_bytes(b"xml")
        fixtures[name] = dict(path=str(path), bytes=3, sha256=sha256(b"xml").hexdigest(), tests=1200)
    if fault == "xml": (tmp_path / "native.xml").write_bytes(b"bad")
    tests = dict(path="evidence", bytes=1, sha256="a" * 64, evidence=fixtures)
    monkeypatch.setattr(probe, "os", NS(environ={"CUDA_VISIBLE_DEVICES": ""}, getpid=lambda: 77,
        open=lambda *a: 99, O_RDONLY=os.O_RDONLY, O_NOFOLLOW=os.O_NOFOLLOW, O_NONBLOCK=os.O_NONBLOCK,
        close=lambda fd: calls.append(("close", fd)), fsync=lambda fd: calls.append(("fsync",))))
    monkeypatch.setattr(probe, "fcntl", NS(LOCK_EX=2, LOCK_NB=4, flock=lambda fd, flags: calls.append(("lock", fd, flags))))
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "output", lambda _: out)
    monkeypatch.setattr(probe, "unit_record", lambda *_: dict(unit="owned"))
    monkeypatch.setattr(probe, "tests_record", lambda *_: tests)
    monkeypatch.setattr(probe, "solver_source_path", lambda: tmp_path / "solver.py")
    monkeypatch.setattr(probe, "parent_inputs", lambda *_: ({}, None, None, None))
    source_count, runtime_count, services_count = [0], [0], [0]
    def source_binding(_):
        source_count[0] += 1
        return dict(source=source if fault != "source" or source_count[0] == 1 else "b" * 40)
    def runtime_binding():
        runtime_count[0] += 1
        return dict(version=1 if fault != "runtime" or runtime_count[0] == 1 else 2)
    def services():
        services_count[0] += 1
        return dict(protected="inactive" if fault != "services" or services_count[0] == 1 else "active")
    def capacity(sample, baseline=None):
        if fault == "capacity": raise ValueError("capacity changed")
        return sample
    def cgroup(*args):
        calls.append(("cgroup",))
        if fault == "cgroup": raise ValueError("non-owner retained")
    monkeypatch.setattr(probe, "source_binding", source_binding)
    monkeypatch.setattr(probe.retained, "runtime_binding", runtime_binding)
    monkeypatch.setattr(probe.retained, "check_deadline", lambda *a: calls.append(("deadline", *a)))
    monkeypatch.setattr(probe.retained, "assert_cgroup_owner_only", cgroup)
    monkeypatch.setattr(probe.shared, "lease_identity", lambda fd: dict(inode=4))
    monkeypatch.setattr(probe.shared, "service_snapshot", services)
    monkeypatch.setattr(probe.shared, "telemetry", lambda: dict(used=1))
    monkeypatch.setattr(probe.shared, "capacity", capacity)
    def supervise(argv, **kwargs):
        assert kwargs["pass_fds"] == (99,) and kwargs["timeout"] == 240
        assert kwargs["probe_timeout"] == 25 and kwargs["grace"] == 5
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert all(kwargs["env"][k] == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
        assert "--lease-fd" in argv and "--declaration-sha256" in argv
        kwargs["probe"]()
        if fault == "supervisor": raise ValueError("owned child failed")
        return dict(decision="synthetic")
    monkeypatch.setattr(probe.supervisor, "supervise", supervise)
    args = NS(child=False, cpu_evidence="evidence", deadline=1.0, source=source)
    if fault is None:
        probe.owner(args)
        dcl = probe.read_json(out / "declaration.json")
        assert set(dcl) == probe.DCL_KEYS and dcl["exclusive_gpu_claimed"] is False
        assert (out / "mac-cpu.xml").read_bytes() == (out / "native-cpu.xml").read_bytes() == b"xml"
        assert probe.read_json(out / "supervision.json")["kernel_cgroup_only_owner"] is True
        assert not (out / "failure.json").exists()
    else:
        with pytest.raises(ValueError): probe.owner(args)
        assert not (out / "supervision.json").exists()
        if fault not in ("xml", "capacity"):
            assert (out / "failure.json").exists()
    assert calls[-1] == ("close", 99)
