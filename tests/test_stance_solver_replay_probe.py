"""CPU-only runner admission tests; no CUDA or service mutation."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_solver_replay_probe as probe
from mjlab_microduck.stance_solver_init_control import SOLVER_RECIPE


def test_import_is_inert_in_fresh_process():
    result = subprocess.run((sys.executable, "-c", "import sys; import mjlab_microduck.stance_solver_replay_probe; "
        "assert not any(n in sys.modules for n in ('warp','torch','mujoco','mujoco_warp')); print('inert')"),
        check=True, capture_output=True, text=True, timeout=10,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(Path(probe.__file__).parents[1])))
    assert result.stdout.strip() == "inert"


@pytest.mark.parametrize("deadline,reserve", [(100,0), (1000,901), (2201,1),
    (float('nan'),1), (float('inf'),1), (True,1), (200,-1), (200,True)])
def test_deadline_rejects_expired_oversized_or_untyped_windows(deadline, reserve):
    with pytest.raises(ValueError, match="deadline"):
        probe.check_deadline(deadline, reserve, now=100)


def test_full_deadline_reserve_is_new_not_historical_cutoff():
    probe.check_deadline(1000, 660, now=100)
    assert probe.BOUNDS["service_seconds"] == 480
    assert probe.BOUNDS["child_seconds"] == 240
    assert probe.UNIT_CAPS["Type"] == "exec"
    assert probe.UNIT_CAPS["KillMode"] == "control-group"


def test_read_plain_accepts_authenticated_empty_file(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert probe.read_plain(path) == b""


def test_read_plain_rejects_symlink_and_oversize(tmp_path):
    path = tmp_path / "original"
    path.write_bytes(b"abcd")
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="canonical"):
        probe.read_plain(link)
    with pytest.raises(ValueError, match="bounded"):
        probe.read_plain(path, 3)


def test_historical_anchors_are_checked_before_any_json_decode(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    def raw(path, *args):
        return b"untrusted { json"
    monkeypatch.setattr(probe, "read_plain", raw)
    from mjlab_microduck import stance_solver_replay_receiver as receiver
    monkeypatch.setattr(receiver, "_json", lambda *a: pytest.fail("decoded before authentication"))
    with pytest.raises(ValueError, match="whole historical"):
        probe.historical_packet()


def test_exact_historical_bank_authenticates_and_decodes_without_gpu():
    # Retained fixtures are present in both campaign worktrees. This is source
    # authenticity/CPU schema evidence, not proof of the historical GPU run.
    root = Path(probe.__file__).resolve().parents[2]
    previous = probe.ROOT
    try:
        probe.ROOT = root
        packet = probe.historical_packet()
    finally:
        probe.ROOT = previous
    assert sha256(packet.raw).hexdigest() == probe.INPUT["raw_sha256"]
    assert len(packet.raw) == probe.INPUT["raw_bytes"]


def test_child_environment_uses_absent_private_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    env = probe.child_env(tmp_path)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["WARP_CACHE_PATH"] == str(tmp_path / "warp-cache")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    (tmp_path / "warp-cache").mkdir()
    with pytest.raises(ValueError, match="absent"):
        probe.child_env(tmp_path)


@pytest.mark.parametrize("bad", ["", "HEAD", "g"*40, "0"*39, "0"*41, 1])
def test_unit_requires_exact_commit(bad):
    with pytest.raises(ValueError, match="source"):
        probe.unit(bad)


def mocked_unit(source, pid):
    return dict(probe.UNIT_CAPS, Id=probe.unit(source), MainPID=str(pid), ActiveState="active",
                ControlGroup="/user.slice/user-1000.slice/user@1000.service/app.slice/" + probe.unit(source),
                InvocationID="1" * 32)


def test_unit_requires_effective_exec_caps_and_invocation(monkeypatch):
    source, pid = "a"*40, 42
    props = mocked_unit(source, pid)
    monkeypatch.setenv("INVOCATION_ID", "1"*32)
    monkeypatch.setattr(probe, "command", lambda *a: "\n".join(f"{k}={v}" for k,v in props.items()).encode())
    assert probe.unit_record(source, pid)["Type"] == "exec"
    props["Type"] = "oneshot"
    with pytest.raises(ValueError, match="capped"):
        probe.unit_record(source, pid)


@pytest.mark.parametrize("key,value", [("RuntimeMaxUSec", "infinity"), ("MemoryMax", "infinity"),
    ("KillMode", "process"), ("NRestarts", "1"), ("MainPID", "43"), ("InvocationID", "2"*32),
    ("Id", "other.service"), ("ControlGroup", "/user.slice/other.service")])
def test_unit_rejects_drifted_properties(monkeypatch, key, value):
    source = "a"*40
    props = mocked_unit(source, 42)
    props[key] = value
    monkeypatch.setenv("INVOCATION_ID", "1"*32)
    monkeypatch.setattr(probe, "command", lambda *a: "\n".join(f"{k}={v}" for k,v in props.items()).encode())
    with pytest.raises(ValueError, match="capped"):
        probe.unit_record(source, 42)


def recipe_objects():
    opt = SimpleNamespace(**{k:v for k,v in SOLVER_RECIPE.items()
                             if k not in ("nworld","njmax","njmax_nnz","nv","nv_pad","is_sparse","block_dim")})
    model = SimpleNamespace(opt=opt, nv=20, nv_pad=20, is_sparse=False,
                            block_dim=SimpleNamespace(**SOLVER_RECIPE["block_dim"]))
    data = SimpleNamespace(nworld=64, njmax=512, njmax_nnz=10240)
    return model, data


def test_recipe_validation_performs_no_gpu_array_readback():
    model, data = recipe_objects()
    assert probe.recipe_signature(model, data) == SOLVER_RECIPE
    model.opt.solver = True
    with pytest.raises(ValueError, match="recipe"):
        probe.recipe_signature(model, data)


def test_scratch_mapping_is_exact_and_resolves_nested_fields():
    roots = {key: SimpleNamespace() for key in ("model", "data", "contact", "efc", "context")}
    roots["data"].contact, roots["data"].efc = roots["contact"], roots["efc"]
    roots["data"].nacon = object()
    roots["model"].opt = SimpleNamespace()
    for name in probe.scratch.RESTORE_ORDER:
        parts = name.split(".")
        obj = roots[parts[0]]
        for part in parts[1:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], object())
    arrays = probe.scratch_arrays(roots["model"], roots["data"], roots["context"])
    assert tuple(arrays) == probe.scratch.RESTORE_ORDER
    assert arrays["model.opt.impratio_invsqrt"] is roots["model"].opt.impratio_invsqrt


def test_actual_cpu_allocation_recipe_and_restore_layout_without_solver_dispatch(tmp_path):
    code = '''
import json
import platform
from pathlib import Path
import numpy as np
import warp as wp
wp.init()
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected kernel launch in allocation/restore')
wp.launch = forbidden
import mujoco, mujoco_warp as mw
from mujoco_warp._src import solver
from mjlab_microduck.stance_warp_runtime import build_entity
from mjlab_microduck.stance_plant_evidence import describe
from mjlab_microduck import stance_solver_replay_probe as probe
from mjlab_microduck import stance_solver_scratch as scratch
probe.ROOT = Path.cwd().resolve()
with wp.ScopedDevice('cpu'):
    native = build_entity().compile()
    assert (native.nq,native.nv,native.nu) == (21,20,14)
    fingerprint = probe.sha256(probe.canonical(describe(native))).hexdigest()
    expected = {('Darwin','arm64'): '2308c6a2e24f74646cefb264df0e3ce2e94957e2396f2af3fc7ee954e0118b85',
                ('Linux','x86_64'): '91838baeac031299a008709e44cef879bdbc9bc769e84fcb2b318453ac115501'}
    assert fingerprint == expected[(platform.system(),platform.machine())]
    model = mw.put_model(native)
    data = mw.put_data(native, mujoco.MjData(native), nworld=64, nconmax=128, njmax=512)
    context = solver.create_solver_context(model,data)
    probe.recipe_signature(model,data)
    packet = probe.historical_packet()
    arrays = probe.scratch_arrays(model,data,context)
    def stage(name,raw,logical,shape,dtype,wdtype):
        return wp.array(np.frombuffer(raw,dtype=dtype).reshape(shape).copy(),
                        dtype=arrays[name].dtype,device='cpu',pinned=False)
    restored = scratch.DenseSolverScratchRestorer(packet,arrays,wp.get_device('cpu'),None,
        stage=stage,copy=wp.copy,synchronize=lambda stream: None,guard=lambda: None)
    restored.restore()
    assert restored.receipt()['restored_fields'] == list(scratch.RESTORE_ORDER)
    assert solver.update_constraint_init_qfrc_constraint_dense.module.execs == {}
    for name in scratch.RESTORE_ORDER:
        assert arrays[name].numpy().tobytes(order='C') == packet.fields[name]
print('CPU_ALLOCATION_RESTORE_ONLY')
'''
    result = subprocess.run((sys.executable, "-c", code), cwd=Path(probe.__file__).resolve().parents[2],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", WARP_CACHE_PATH=str(tmp_path / "cache"),
                 PYTHONPATH=str(Path(probe.__file__).resolve().parents[1])),
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU_ALLOCATION_RESTORE_ONLY" in result.stdout


def test_cpu_evidence_requires_whole_source_bound_matching_test_xml(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    directory = tmp_path / "artifacts/tools"
    directory.mkdir(parents=True)
    path = directory / "paired.json"
    evidence = dict(source="a"*40, test_files=list(probe.TESTS), flags=probe.FLAGS)
    for platform in ("mac", "native"):
        xml = directory / (platform + ".xml")
        raw = b'<testsuites><testsuite tests="360" errors="0" failures="0" skipped="0"/></testsuites>'
        xml.write_bytes(raw)
        evidence[platform] = dict(path=str(xml), bytes=len(raw), sha256=sha256(raw).hexdigest(), tests=360)
    path.write_bytes(probe.canonical(evidence))
    assert probe.tests_record(path, "a"*40)["evidence"] == evidence
    (directory / "native.xml").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="XML bytes"):
        probe.tests_record(path, "a"*40)


def test_owner_refuses_nonempty_cuda_visibility_before_source_or_lease(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe, "source_binding", lambda *a: pytest.fail("source reached"))
    with pytest.raises(ValueError, match="CPU-only"):
        probe.owner(SimpleNamespace(child=False))


def test_child_refuses_wrong_parent_before_declaration_or_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="owned CUDA child"):
        probe.child(SimpleNamespace(child=True, source="a"*40, owner_pid=-1))
