"""CPU contract and actual-Duck forward checks, not Ada or training evidence."""
from copy import deepcopy
from hashlib import sha256
import os
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mjlab_microduck import ada_duck_forward as probe


def arrays(dtype):
    return {key: np.zeros((probe.WORLDS, probe.WIDTHS[key]), dtype=dtype) for key in probe.FIELDS}


def receipt():
    cpu, gpu = arrays(np.float64), arrays(np.float32)
    return dict(specification=probe.specification(), gpu_uuid=probe.host.GPU, name=probe.host.NAME,
                capability=[8, 9], torch_cuda="12.8", warp_arch=89, warp_precompiled_headers=False,
                plant=dict(protocol="football-b1n-compiled-plant-v1", assets={"test.xml": "a" * 64},
                           topology=[21, 20, 14, 1, 0, 0], joints=[], qids=[], dofs=[], floor=0, feet=[], ranges=[],
                           initial_qpos=[], selected_fields=[], selected_fields_sha256="b" * 64, options={}),
                placement_gaps=[.0001, .0001],
                counters=dict(cpu=dict(nf=0, nefc=0, ncon=0), gpu=dict(nf=[0, 0], nefc=[0, 0], nacon=[0])),
                residuals=probe.comparison(cpu, gpu),
                payloads=[dict(file=name, bytes=100, sha256="a" * 64) for name in ("cpu-fields.npz", "gpu-fields.npz")])


def test_import_is_inert_in_fresh_process():
    code = "import sys; import mjlab_microduck.ada_duck_forward; assert not {'torch','warp','mujoco','mujoco_warp','numpy'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_specification_is_zero_integration_unconstrained_without_admission():
    spec = probe.specification()
    assert spec["worlds"] == 2
    assert spec["motor_preparations"] == spec["optimizer_steps"] == spec["integration_steps"] == 0
    assert spec["numerical_acceptance_tolerance"] is None
    assert all(value is False for value in spec["flags"].values())
    assert set(probe.WIDTHS) == set(probe.FIELDS)


@pytest.mark.parametrize("field", probe.FIELDS)
def test_every_element_and_both_worlds_are_compared(field):
    cpu, gpu = arrays(np.float64), arrays(np.float32)
    gpu[field][-1, -1] = 2.
    row = probe.comparison(cpu, gpu)[field]
    assert row["float32_bit_mismatches"] == 1 and row["max_abs"] == 2.
    assert row["values"] == gpu[field].size
    assert row["gpu_sha256"] == sha256(gpu[field].tobytes()).hexdigest()


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "nan", "inf"])
def test_incomplete_or_nonfinite_buffers_are_rejected(change):
    cpu, gpu = arrays(np.float64), arrays(np.float32)
    if change == "missing": gpu.pop("crb")
    if change == "extra": gpu["unknown"] = np.zeros(2)
    if change == "shape": gpu["crb"] = gpu["crb"][:, :-1]
    if change == "dtype": gpu["crb"] = gpu["crb"].astype(np.float64)
    if change == "nan": gpu["crb"][-1, -1] = np.nan
    if change == "inf": cpu["crb"][-1, -1] = np.inf
    with pytest.raises(ValueError): probe.comparison(cpu, gpu)


def test_signed_zero_bit_difference_is_not_hidden_by_zero_residual():
    cpu, gpu = arrays(np.float64), arrays(np.float32)
    gpu["crb"][-1, -1] = -0.
    row = probe.comparison(cpu, gpu)["crb"]
    assert row["max_abs"] == 0. and row["float32_bit_mismatches"] == 1


def test_large_finite_dynamics_difference_is_reported_not_accepted_as_training():
    value = receipt()
    value["residuals"]["qacc"]["max_abs"] = 999.
    value["residuals"]["qacc"]["rms"] = 99.
    value["residuals"]["qacc"]["float32_bit_mismatches"] = 40
    probe.validate_child(value)
    assert value["specification"]["flags"]["training_authorized"] is False


@pytest.mark.parametrize("change", ["flags", "integration", "uuid", "sm", "pch", "counter", "counter-type", "missing", "nan", "hash", "state", "payload", "gap", "plant"])
def test_strict_receipt_rejects_scope_identity_and_buffer_drift(change):
    value = deepcopy(receipt())
    if change == "flags": value["specification"]["flags"]["training_authorized"] = True
    if change == "integration": value["specification"]["integration_steps"] = 1
    if change == "uuid": value["gpu_uuid"] = "GPU-other"
    if change == "sm": value["warp_arch"] = 120
    if change == "pch": value["warp_precompiled_headers"] = True
    if change == "counter": value["counters"]["gpu"]["nf"] = [0, 14]
    if change == "counter-type": value["counters"]["gpu"]["nf"] = [False, False]
    if change == "missing": value["residuals"].pop("crb")
    if change == "nan": value["residuals"]["crb"]["max_abs"] = float("nan")
    if change == "hash": value["residuals"]["crb"]["gpu_sha256"] = "not-a-hash"
    if change == "state": value["residuals"]["qpos"]["float32_bit_mismatches"] = 1
    if change == "payload": value["payloads"][0]["file"] = "../other.npz"
    if change == "gap": value["placement_gaps"][1] = float("nan")
    if change == "plant": value["plant"] = {}
    with pytest.raises(ValueError): probe.validate_child(value)


def test_retained_arrays_are_complete_recomputable_and_never_overwritten(tmp_path):
    value = arrays(np.float64)
    descriptor = probe.retain_arrays(tmp_path, "cpu-fields.npz", value)
    raw = (tmp_path / descriptor["file"]).read_bytes()
    assert descriptor["bytes"] == len(raw) and descriptor["sha256"] == sha256(raw).hexdigest()
    with np.load(tmp_path / descriptor["file"], allow_pickle=False) as loaded:
        assert set(loaded.files) == set(probe.FIELDS)
        for key in probe.FIELDS: np.testing.assert_array_equal(loaded[key], value[key])
    with pytest.raises(FileExistsError): probe.retain_arrays(tmp_path, "cpu-fields.npz", value)


@pytest.mark.parametrize("forged", [False, True])
def test_receiver_recomputes_all_rows_not_just_child_claimed_hashes(tmp_path, forged):
    value = receipt()
    cpu, gpu = arrays(np.float64), arrays(np.float32)
    gpu["crb"][-1, -1] = .125
    value["payloads"] = [probe.retain_arrays(tmp_path, "cpu-fields.npz", cpu), probe.retain_arrays(tmp_path, "gpu-fields.npz", gpu)]
    value["residuals"] = probe.comparison(cpu, gpu)
    if forged:
        value["residuals"]["crb"]["max_abs"] = 0.
        with pytest.raises(ValueError, match="recomputed"): probe.receive_payloads(tmp_path, value)
    else:
        assert probe.receive_payloads(tmp_path, value)["residuals_recomputed"] is True


def test_actual_duck_cpu_forward_without_integration_in_fresh_process(tmp_path):
    code = ("import os; import warp as wp; "
            "wp.config.kernel_cache_dir=os.environ['PROBE_CPU_CACHE']; wp.config.use_precompiled_headers=False; "
            "from pathlib import Path; from mjlab_microduck.ada_duck_forward import physics; "
            "r=physics(Path(os.environ['PROBE_CPU_ROOT']),device='cpu'); "
            "assert r['counters']['cpu']==dict(nf=0,nefc=0,ncon=0); "
            "assert all(r['residuals'][k]['float32_bit_mismatches']==0 for k in "
            "('qpos','qvel','time','qacc_warmstart','ctrl','qfrc_applied','xfrc_applied')); "
            "assert not __import__('torch').cuda.is_initialized()")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PROBE_CPU_ROOT=str(tmp_path),
               PROBE_CPU_CACHE=str(tmp_path / "warp-cache"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_only_new_diagnostic_source_added_no_policy_integration_or_runtime_construction():
    source = Path(probe.__file__).read_text()
    for forbidden in ("WarpStanceRuntime(", "mj_step(", "EulerCandidateCommit(", ".initialize(", "optimizer.step("):
        assert forbidden not in source
    assert "mjwarp.forward(model, data)" in source


@pytest.mark.parametrize("mode", ["success", "capacity", "foreign", "services", "child", "watchdog"])
def test_new_owner_cleanup_and_foreign_preservation(monkeypatch, tmp_path, mode):
    root = tmp_path / "new-evidence"
    lock = tmp_path / "existing.lock"
    lock.touch()
    host = probe.host
    monkeypatch.setattr(probe, "identity", lambda source: {"commit": source})
    monkeypatch.setattr(probe, "owned_service", lambda source: {"unit": "owned-test"})
    monkeypatch.setattr(probe, "directory", lambda source: root)
    monkeypatch.setattr(probe, "sys", SimpleNamespace(modules={}, executable=sys.executable))
    monkeypatch.setattr(host, "LOCK", lock)
    monkeypatch.setattr(host, "READ_DEADLINE", None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    sample = {"gpu": host.parse_gpu(f"{host.GPU},{host.DRIVER},8.9,16376,961,15232,0,43")}
    if mode == "capacity": sample["gpu"]["free_mib"] = 10239
    monkeypatch.setattr(host, "telemetry", lambda: deepcopy(sample))
    foreign = [{"pid": 1592, "name": "untouched"}]
    foreign_rows = iter([foreign, [] if mode == "foreign" else foreign, foreign])
    monkeypatch.setattr(host, "foreign_processes", lambda owned_pid=None: next(foreign_rows))
    service_rows = iter([{"mission": "inactive"}, {"mission": "active" if mode == "services" else "inactive"}, {"mission": "inactive"}])
    monkeypatch.setattr(host, "service_snapshot", lambda: next(service_rows))
    actions = []

    class Child:
        pid = 12345
        returncode = None if mode in ("foreign", "services", "watchdog") else (1 if mode == "child" else 0)
        def __init__(self, command, **kwargs):
            actions.append("spawn-owned")
            assert probe.MODULE in command and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            assert len(kwargs["pass_fds"]) == 1
            if mode == "success":
                value = receipt()
                value["payloads"] = [probe.retain_arrays(root, "cpu-fields.npz", arrays(np.float64)),
                                     probe.retain_arrays(root, "gpu-fields.npz", arrays(np.float32))]
                host.write_json(root / "child.json", value)
        def poll(self): return self.returncode
        def terminate(self): actions.append("terminate-owned"); self.returncode = -15
        def kill(self): actions.append("kill-owned"); self.returncode = -9
        def wait(self, timeout): return self.returncode

    monkeypatch.setattr(probe.subprocess, "Popen", Child)
    class Watchdog:
        def __init__(self, delay, callback):
            assert 0 < delay <= probe.BOUNDS["child_seconds"]
            self.callback = callback
        def start(self):
            if mode == "watchdog": self.callback()
        def cancel(self): pass
        def join(self, timeout): pass
    monkeypatch.setattr(probe.threading, "Timer", Watchdog)
    if mode == "success":
        probe.supervise("a" * 40)
    else:
        with pytest.raises(ValueError): probe.supervise("a" * 40)
    report = json.loads((root / "report.json").read_bytes())
    assert all(value is False for value in report["specification"]["flags"].values())
    if mode == "success":
        assert report["decision"].endswith("pending-reception-not-training")
        assert actions == ["spawn-owned"] and len(report["files"]) == 5
    elif mode == "capacity":
        assert actions == [] and report["child_exit"] is None
    elif mode in ("foreign", "services"):
        assert actions == ["spawn-owned", "terminate-owned"] and report["child_exit"] == -15
    elif mode == "watchdog":
        assert actions == ["spawn-owned", "kill-owned"] and report["child_exit"] == -9
    else:
        assert actions == ["spawn-owned"] and report["child_exit"] == 1
    assert lock.exists() and lock.read_bytes() == b""
