"""CPU/synthetic contract checks, never CUDA collection or training evidence."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mjlab_microduck import ada_duck_contact as p


def test_inert_import_in_fresh_child():
    subprocess.run([sys.executable, "-c", "import sys; import mjlab_microduck.ada_duck_contact; assert not {'torch','warp','mujoco','mujoco_warp','numpy'} & sys.modules.keys()"], check=True, timeout=15)


def test_fixed_profile_authority_and_bounds():
    assert p.base.selected_profile(p) is p
    assert p.BOUNDS == p.base.BOUNDS
    assert all(x is False for x in p.specification()["flags"].values())
    assert p.specification()["motor_preparations"] == 1
    assert p.specification()["fixture_forwards_before_motor"] == 1
    assert p.specification()["integration_steps"] == p.specification()["optimizer_steps"] == 0
    with pytest.raises(ValueError): p.base.selected_profile(SimpleNamespace(MODULE="another", BOUNDS=p.BOUNDS, FLAGS=p.FLAGS))
    with pytest.raises(ValueError): p.base.selected_profile(SimpleNamespace(MODULE=p.MODULE, BOUNDS={}, FLAGS=p.FLAGS))


@pytest.fixture(scope="module")
def cpu_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("ada-contact-cpu")
    code = ("import os; import warp as wp; wp.config.use_precompiled_headers=False; "
            "wp.config.kernel_cache_dir=os.environ['CPU_PROBE_CACHE']; "
            "from pathlib import Path; from mjlab_microduck import ada_duck_contact as p; "
            "root=Path(os.environ['CPU_PROBE_ROOT']); r=p.physics(root,device='cpu'); "
            "v=dict(specification=p.specification(),gpu_uuid=p.base.host.GPU,name=p.base.host.NAME,capability=[8,9],"
            "torch_cuda='12.8',warp_arch=89,warp_precompiled_headers=False,**r); "
            "p.validate_child(v); p.receive_payloads(root,v); p.base.host.write_json(root/'synthetic-cpu-receipt.json',v); "
            "assert not __import__('torch').cuda.is_initialized()")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", CPU_PROBE_ROOT=str(root), CPU_PROBE_CACHE=str(root / "cache"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return root, json.loads((root / "synthetic-cpu-receipt.json").read_bytes())


def load(root, name):
    with np.load(root / name, allow_pickle=False) as bank: return {k: bank[k].copy() for k in bank.files}


def test_real_cpu_fixture_has_fresh_contact_load_and_per_world_motor_fields(cpu_fixture):
    root, value = cpu_fixture
    source, motor = load(root, "bam-source.npz"), load(root, "motor.npz")
    assert not source["qfrc_constraint"][0].any() and source["qfrc_constraint"][1].any()
    assert not np.array_equal(motor["dof_frictionloss"][0], motor["dof_frictionloss"][1])
    assert value["counters"]["gpu"]["nf"] == [14, 14]
    assert p.receive_payloads(root, value)["active_structure_recomputed"] is True
    assert value["active_analysis"]["contact_status"] == "unresolved-correspondence"
    # Do not force this source/architecture's contact counts onto CUDA.
    assert value["active_analysis"]["issues"] and not value["active_analysis"]["solver_qualified"]


@pytest.mark.parametrize("damage", ["nan", "id", "J", "address", "slot", "sidecar", "missing", "foreign"])
def test_every_active_row_and_contact_is_checked(cpu_fixture, damage):
    root, value = cpu_fixture
    cpu, gpu = load(root, "cpu-active.npz"), load(root, "gpu-active.npz")
    if damage == "nan": gpu["rows/1/force"][-1] = np.nan
    if damage == "id": gpu["rows/0/id"][0] = 999
    if damage == "J": gpu["rows/0/J"][0, 0] = .5
    if damage == "address": gpu["contacts/efc_address"][0] = 999
    if damage == "slot": gpu["contacts/slot"][0] = 999
    if damage == "sidecar": gpu["sidecar/type"] = gpu["sidecar/type"][:-1]
    if damage == "missing": gpu.pop("rows/1/D")
    if damage == "foreign": gpu["contacts/geom"][0, 1] = 999
    plant = value["plant"]
    with pytest.raises((ValueError, IndexError)): p.analyze_active(cpu, gpu, plant["dofs"], plant["floor"], plant["feet"])


@pytest.mark.parametrize("damage", ["residual", "coverage", "snapshot", "offset", "motor", "stale", "counter", "load", "gain"])
def test_independent_reception_rejects_forged_metadata_or_rebound_payload(cpu_fixture, tmp_path, damage):
    root, original = cpu_fixture
    value = deepcopy(original)
    # Symlink unchanged fixture files is refused by receiver; copy only the
    # altered file and hardlink the immutable others in this owned temp dir.
    target = {"snapshot": "prepared-inputs.npz", "motor": "motor.npz", "stale": "bam-source.npz", "load": "bam-source.npz", "gain": "bam-source.npz"}.get(damage)
    for item in value["payloads"]:
        if item["file"] != target: os.link(root / item["file"], tmp_path / item["file"])
    if damage == "residual": value["residuals"]["qacc"]["max_abs"] += 1.
    if damage == "coverage": value["active_analysis"]["contact_status"] = "unique-key-candidates"
    if damage == "offset": value["initial_qpos"][1][2] += .01
    if damage == "counter": value["counters"]["gpu"]["nefc"][1] += 1
    if target:
        data = load(root, target)
        if damage == "snapshot": data["/data/qpos"][0] ^= 1
        if damage == "motor": data["dof_frictionloss"][1, 7] += .01
        if damage == "stale": data["qfrc_constraint"][1] = 0
        if damage == "load": data["qfrc_constraint"][1] += .01
        if damage == "gain": data["kp"][1] += 1.
        # Rebinding raw file hash is not enough to forge source semantics.
        rebound = p.retain(tmp_path, target, data)
        value["payloads"] = [rebound if item["file"] == target else item for item in value["payloads"]]
    with pytest.raises(ValueError): p.receive_payloads(tmp_path, value)


def test_retention_quota_and_no_overwrite(tmp_path):
    with pytest.raises(ValueError): p.retain(tmp_path, "motor.npz", {"x": np.zeros(p.PAYLOAD_MAX, dtype=np.uint8)})
    data = {"x": np.zeros(4, dtype=np.float32)}
    item = p.retain(tmp_path, "motor.npz", data)
    assert sha256((tmp_path / item["file"]).read_bytes()).hexdigest() == item["sha256"]
    with pytest.raises(FileExistsError): p.retain(tmp_path, "motor.npz", data)


@pytest.mark.parametrize("damage", ["dtype", "shape", "strides"])
def test_input_layout_metadata_cannot_be_relabelled(cpu_fixture, damage):
    _, original = cpu_fixture
    value = deepcopy(original)
    row = value["input_manifest"]["/data/qpos"]
    if damage == "dtype": row["dtype"] = "guessed-dtype"
    if damage == "shape": row["shape"][1] += 1
    if damage == "strides": row["strides"][1] += 1
    with pytest.raises(ValueError): p.validate_child(value)


def test_no_integration_or_optimizer_call_source():
    source = Path(p.__file__).read_text()
    assert "env._forward()" in source
    for forbidden in ("env.step(", "mj_step(", ".integrate(", "optimizer.step(", "enable_forward_graph("):
        assert forbidden not in source


def test_contact_profile_uses_own_command_quota_inventory_and_pending_decision(cpu_fixture, monkeypatch, tmp_path):
    root, value = cpu_fixture
    output = tmp_path / "profile-output"
    lock = tmp_path / "existing.lock"; lock.touch()
    base, host = p.base, p.base.host
    monkeypatch.setattr(p, "identity", lambda source: {"source": source})
    monkeypatch.setattr(p, "owned_service", lambda source: {"unit": "synthetic-contact-unit"})
    monkeypatch.setattr(p, "directory", lambda source: output)
    monkeypatch.setattr(base, "sys", SimpleNamespace(modules={}, executable=sys.executable))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(host, "LOCK", lock)
    monkeypatch.setattr(host, "READ_DEADLINE", None)
    telemetry = {"gpu": host.parse_gpu(f"{host.GPU},{host.DRIVER},8.9,16376,961,15232,0,44")}
    monkeypatch.setattr(host, "telemetry", lambda: telemetry)
    monkeypatch.setattr(host, "service_snapshot", lambda: {"mission": "inactive"})
    monkeypatch.setattr(host, "foreign_processes", lambda owned_pid=None: [])
    class Child:
        pid = 12345; returncode = 0
        def __init__(self, command, **kwargs):
            assert command[1:3] == ["-m", p.MODULE] and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            for item in value["payloads"]: os.link(root / item["file"], output / item["file"])
            host.write_json(output / "child.json", value)
        def poll(self): return 0
    monkeypatch.setattr(base.subprocess, "Popen", Child)
    base.supervise("a" * 40, probe=p)
    report = json.loads((output / "report.json").read_bytes())
    assert report["decision"] == p.COLLECTION_DECISION
    assert set(report["files"]) == set(p.FILES)
    assert report["specification"] == p.specification() and report["child_exit"] == 0
    assert (output / "child.json").stat().st_size > base.CHILD_JSON_LIMIT  # Correct contact, not baseline JSON cap.
