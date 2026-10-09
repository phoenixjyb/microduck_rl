"""Admission/receiver CPU fixtures; no services, CUDA or native disassembler."""
from copy import deepcopy
from hashlib import sha256
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from test_stance_solver_cost_stages import example
from test_stance_solver_cost_binary import cubin, sass, symbol
from test_stance_solver_replay_receiver import _runtime_record
from mjlab_microduck import stance_solver_cost_probe as probe
from mjlab_microduck import stance_solver_cost_receiver as receiver
from mjlab_microduck import stance_solver_cost_dispatch as dispatch
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_cost_executable as executable
from mjlab_microduck import stance_solver_target_binding as static


@pytest.mark.parametrize("name,constructor", (("CostDispatchObserver", dispatch.CostDispatchObserver),
                                              ("CostModuleExecutable", executable.CostModuleExecutable)))
def test_native_constructor_calls_match_actual_keyword_only_contract(name, constructor):
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(probe.child)))
             if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == name)
                or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]
    assert len(calls) == 1
    for c in calls:
        assert not c.args and all(k.arg is not None for k in c.keywords)
        inspect.signature(constructor).bind(**{k.arg: object() for k in c.keywords})


def test_new_owner_refuses_cuda_before_source_or_lease(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe, "source_binding", lambda _: pytest.fail("unexpected source check"))
    with pytest.raises(ValueError, match="CPU-only"): probe.owner(NS(child=False))


def test_new_child_refuses_wrong_parent_before_read_or_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe, "read_plain", lambda _: pytest.fail("unexpected decode"))
    with pytest.raises(ValueError, match="fresh inherited"): probe.child(NS(source="a" * 40, child=True, owner_pid=-1))


def test_owner_requires_cpu_evidence_before_any_source_or_lease(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe, "source_binding", lambda _: pytest.fail("unexpected source check"))
    with pytest.raises(ValueError, match="explicit paired"): probe.owner(NS(child=False, cpu_evidence=None))


@pytest.mark.parametrize("source", (True, "A" * 40, "a" * 39, "../else", None))
def test_unique_native_unit_requires_full_source(source):
    with pytest.raises(ValueError): probe.unit(source)


def test_exact_new_fence_and_old_contracts_unchanged():
    assert len(probe.OWN) == 8 and len(probe.TESTS) == 17
    assert probe.BASE == "98aa8a1d1a413f76cadc397492eddb8cb953135e"
    assert probe.retained.BASE == "e6cf84d9615ee626390e2b691fbf08889efb0137"
    assert probe.BOUNDS["child_seconds"] == 240 and probe.UNIT_CAPS["KillMode"] == "control-group"


def _refresh(root):
    return {str(p.relative_to(root)): {"bytes": len(p.read_bytes()), "sha256": sha256(p.read_bytes()).hexdigest()}
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def bundle(tmp_path, monkeypatch, example):
    source, owner, child, inv = "a" * 40, 101, 202, "c" * 32
    native = probe.output(source)
    device = dict(alias="cuda:0", arch=120, context=31, object_id=13, gpu_uuid=probe.shared.GPU)
    props = dict(probe.UNIT_CAPS, Id=probe.unit(source), MainPID=str(owner), ActiveState="active",
                 ControlGroup=probe.retained.CGROUP_PARENT + probe.unit(source), InvocationID=inv)
    view = dict(uuid=probe.shared.GPU, driver=probe.shared.DRIVER, total_mib=24467, used_mib=8000,
                free_mib=16467, temperature_c=33, utilization_percent=2)
    sample = dict(wsl=view, windows=dict(view))
    dcl = dict(protocol=probe.PROTOCOL, source=source, owner_pid=owner, native_root=str(native),
        source_binding=dict(source=source, branch=probe.BRANCH, tree="0" * 40, leaves=[dict(path="example.py")]),
        runtime=_runtime_record(monkeypatch), service=dict(name=probe.unit(source), **props), bounds=probe.BOUNDS,
        replay_input=probe.retained.INPUT, control=stages.control_manifest(), exclusive_gpu_claimed=False,
        baseline=sample, flags=dict(stages.FLAGS))
    files, modules, bindings = {}, {}, {}
    for index, (role, names) in enumerate(executable.GROUPS.items()):
        dirname = "compiled-" + role
        code = bytes(range(16))
        raw = cubin([(symbol(n), code) for n in names])
        meta = json.dumps({symbol(n) + "_smem_bytes": 0 for n in names}).encode()
        generated = {}
        for kind, name, payload in (("binary", "module.cubin", raw), ("metadata", "module.meta", meta),
                                    ("source", "module.cu", b"// mock generated source\n")):
            files[dirname + "/" + name] = payload
            generated[kind] = dict(path=str(native / dirname / name), bytes=len(payload), sha256=sha256(payload).hexdigest(),
                                   identity=[1, 2, len(payload), 3, 4])
        load = dict(binary_path=generated["binary"]["path"], metadata_path=generated["metadata"]["path"],
                    output_arch=120, block_dim=256, returned_executable_is_cache_entry=True,
                    module_object_id=50+index, executable_object_id=60+index, device_object_id=13)
        bs, offline = {}, {}
        for n in names:
            b = dict(artifact_format="cubin", binary_path=load["binary_path"], metadata_path=load["metadata_path"],
                binary_sha256=sha256(raw).hexdigest(), metadata_sha256=sha256(meta).hexdigest(), module_hash="b" * 64,
                context=31, device="cuda:0", device_arch=120, block_dim=256, module_handle=70+index,
                forward_handle=80+list(stages.STAGES).index(n), forward_smem_bytes=0, symbol=symbol(n),
                observed_object_ids=dict(module=50+index, executable=60+index, device=13))
            bs[n] = bindings[n] = b
            files[n+".sass"], files[n+".stderr"] = sass(symbol(n), code), b""
            offline[n] = receiver.binary.verify_disassembly(raw, files[n+".sass"], n, symbol(n))
        modules[role] = dict(protocol=executable.PROTOCOL, role=role, generated=generated, module_hash="b" * 64,
            offline_disassembly=offline, explicit_load=load, bindings=bs, flags=dict(stages.FLAGS),
            loaded_binary_bytes_observed=False, driver_jit_machine_code_observed=False, actual_dispatch_observed=False)
    arms = {}
    for arm in ("reference", "control"):
        packets = {}
        for phase in stages.PHASES:
            payload = example.arms[arm][phase]
            files[arm+"-"+phase+".bin"] = payload
            packets[phase] = dict(bytes=len(payload), sha256=sha256(payload).hexdigest())
        capture = dict(protocol=stages.PROTOCOL, stages=list(stages.STAGES), timing_changed_by_readback=True,
                       packets=packets, qualification=dict(stages.FLAGS))
        observer = dict(protocol=dispatch.PROTOCOL, base=dispatch.BASE, caller="_update_constraint", track_changes=False,
            observed_stages=list(stages.STAGES), call_site_lines=[2156,2180,2197,2214], call_sites_observed=True,
            timing_changed_by_readback=True, bindings=bindings, capture=capture, qualification=dict(stages.FLAGS),
            explicit_load_provenance_authenticated=False, native_gpu_execution_authenticated=False,
            capture_origin_authenticated=False, external_unit_retirement_authenticated=False)
        arms[arm] = dict(observer=observer, scratch=dict(protocol=receiver.scratch.PROTOCOL,
            decision="dense-solver-scratch-restored-no-dispatch-or-qualification",
            source_sha256=example.packet.sha256, source_bytes=len(example.packet.raw),
            restored_fields=list(receiver.scratch.RESTORE_ORDER), retained_source_bytes=True,
            solver_called=False, forward_called=False, flags=dict(receiver.scratch.FLAGS)))
    receipt = dict(protocol=probe.PROTOCOL, source=source, owner_pid=owner, child_pid=child, native_root=str(native),
        device=device, modules=modules, arms=arms, flags=dict(stages.FLAGS), declaration_sha256=sha256(probe.canonical(dcl)).hexdigest())
    result = dict(decision="reviewed-owned-root-exited-no-native-qualification", returncode=0,
                  root=[child, 555], elapsed=1.0, observed_session_members=[], process_tree_retirement_proven=False,
                  native_qualified=False, training_authorized=False)
    records = dict(receipt=receipt, declaration=dcl,
        supervision=dict(result=result, kernel_cgroup_only_owner=True, unit_retirement_independently_required=True, flags=dict(static.FLAGS)),
        telemetry=dict(samples=[sample], flags=dict(stages.FLAGS)))
    for n, value in records.items(): files[n+".json"] = probe.canonical(value)
    files["child.log"] = b"mock child, not a native execution\n"
    root = tmp_path / "bundle"
    root.mkdir()
    for name, raw in files.items():
        p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    inventory = _refresh(root)
    close = dict(source=source, declaration_sha256=sha256(probe.canonical(dcl)).hexdigest(),
        inventory_sha256=sha256(probe.canonical(inventory)).hexdigest(),
        service=dict(props, MainPID="0", ActiveState="active", SubState="exited", ControlGroup="", ExecMainStatus="0", Result="success"),
        cgroup=dict(path=props["ControlGroup"], absent=True, observed_pids=[]))
    return NS(root=root, inventory=inventory, closeout=close, packet=example.packet, receipt=receipt, records=records)


def test_complete_independent_two_arm_receiver(bundle):
    r = receiver.receive(bundle.root, bundle.inventory, bundle.closeout, bundle.packet)
    assert r["external_retirement_evidence_checked"] and r["timing_changed_by_readback"]
    assert not any(r["qualification"].values()) and r["driver_loaded_bytes_observed"] is False


@pytest.mark.parametrize("fault", ("pid", "invocation", "cgroup", "exit", "inventory", "declaration"))
def test_external_retirement_binding_refuses_substitution(bundle, fault):
    c = deepcopy(bundle.closeout)
    if fault == "pid": c["service"]["MainPID"] = "10"
    elif fault == "invocation": c["service"]["InvocationID"] = "d" * 32
    elif fault == "cgroup": c["cgroup"]["absent"] = False
    elif fault == "exit": c["service"]["ExecMainStatus"] = "1"
    elif fault == "inventory": c["inventory_sha256"] = "0" * 64
    else: c["declaration_sha256"] = "0" * 64
    with pytest.raises(ValueError): receiver.receive(bundle.root, bundle.inventory, c, bundle.packet)


@pytest.mark.parametrize("fault", ("gpu", "stage", "binding", "sass", "packet", "flags", "scratch", "scratch-missing"))
def test_reanchored_receipt_cannot_bypass_contracts(bundle, fault):
    receipt = deepcopy(bundle.receipt)
    if fault == "gpu": receipt["device"]["gpu_uuid"] = "other"
    elif fault == "stage": receipt["arms"]["control"]["observer"]["call_site_lines"][3] += 1
    elif fault == "binding": receipt["modules"]["efc"]["bindings"]["efc"]["context"] += 1
    elif fault == "sass": (bundle.root/"efc.sass").write_bytes(b"forged")
    elif fault == "packet": (bundle.root/"control-gauss.after.bin").write_bytes(bytes(stages.PACKET_BYTES))
    elif fault == "scratch": receipt["arms"]["control"]["scratch"]["source_sha256"] = "0" * 64
    elif fault == "scratch-missing": del receipt["arms"]["control"]["scratch"]
    else: receipt["flags"]["native_qualified"] = True
    (bundle.root/"receipt.json").write_bytes(probe.canonical(receipt))
    inventory = _refresh(bundle.root)
    close = dict(bundle.closeout, inventory_sha256=sha256(probe.canonical(inventory)).hexdigest())
    with pytest.raises(ValueError): receiver.receive(bundle.root, inventory, close, bundle.packet)


def test_inventory_is_authenticated_before_any_record_decode(bundle, monkeypatch):
    (bundle.root/"control-dense.after.bin").write_bytes(b"forged")
    monkeypatch.setattr(receiver, "_json", lambda *a: pytest.fail("premature decode"))
    with pytest.raises(ValueError): receiver.receive(bundle.root, bundle.inventory, bundle.closeout, bundle.packet)


def test_import_is_inert():
    code = "from mjlab_microduck import stance_solver_cost_probe, stance_solver_cost_receiver; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


@pytest.mark.parametrize("fault", (None, "unpinned", "gpu-source", "bytes", "shape", "alias"))
def test_control_copy_validates_only_cpu_staging_and_seven_held_copies(fault):
    import numpy as np
    gpu, cpu, dtype = NS(is_cuda=True, is_cpu=False), NS(is_cuda=False, is_cpu=True), object()
    stream, events = NS(device=gpu), []
    shapes = {n: h for n, _, h, _ in stages.SPECS}
    class Array:
        is_contiguous, requires_grad = True, False
        def __init__(self, host, *, dtype, device, pinned, requires_grad):
            self.host, self.dtype, self.device, self.pinned = host, dtype, cpu, pinned
            self.shape, self.ptr = host.shape, 20_000_000
            if fault == "unpinned": self.pinned = False
            elif fault == "gpu-source": self.device = gpu
            elif fault == "bytes": self.host = np.zeros_like(host)
            elif fault == "shape": self.shape = (host.size,)
            elif fault == "alias": self.ptr = 10_000_000
        def numpy(self):
            assert self.device is cpu, "GPU numpy readback forbidden"
            return self.host
    arrays = {}
    for name, raw in stages.control_fields().items():
        a = object.__new__(Array)
        a.shape, a.dtype, a.device, a.ptr, a.pinned = shapes[name], dtype, gpu, 10_000_000, False
        arrays[name] = a
    def copy(dst, src, *, stream):
        assert dst in arrays.values() and src.device is cpu and stream.device is gpu
        events.append("copy")
    def sync(s):
        assert s is stream
        events.append("sync")
    wp = NS(array=Array, copy=copy, synchronize_stream=sync, float32=dtype)
    if fault is None:
        probe.copy_control(wp, arrays, stream, lambda: None)
        assert events == ["copy", "sync"] * 7
    else:
        with pytest.raises(ValueError): probe.copy_control(wp, arrays, stream, lambda: None)
        assert not events
