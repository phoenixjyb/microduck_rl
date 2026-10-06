"""Real two-world CPU fixtures; not native CUDA or original replay acceptance."""

from hashlib import sha256
from threading import Thread

import numpy as np
import pytest
import torch
import warp as wp
from mujoco_warp._src import smooth

from mjlab_microduck import stance_com_entry_capture as capture
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


@pytest.fixture(autouse=True)
def hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="module")
def observed():
    scope = capture.ComEntryCapture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with scope:
            env = WarpStanceRuntime(nworld=2, device="cpu", solved_field_check="packed")
            env._forward()
    return env, scope


def test_original_constructor_and_later_entry(observed):
    env, scope = observed
    receipt = scope.receipt
    assert receipt["status"] == "complete"
    assert len(scope.entries) == 2
    assert scope.entries[0] == scope.entries[1]
    assert all(len(raw) == 384 for raw in scope.entries)
    assert all(row["original_launches"] == 11 for row in receipt["calls"])
    assert all(
        row["boundary"] == "after-init-before-first-accumulation"
        for row in receipt["calls"]
    )
    assert all(
        row["readback_and_device_sync_perturb_timing"] for row in receipt["calls"]
    )
    assert all(value is False for value in receipt["flags"].values())
    assert (env.steps == 0).all()
    assert env.forward_graph is None
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH
    receipt["calls"].clear()
    assert len(scope.receipt["calls"]) == 2


@pytest.mark.parametrize("mode,count", [("concurrent", 224), ("serial", 288)])
def test_whole_repeat_bank_resets_and_includes_root(observed, mode, count):
    env, scope = observed
    raw = scope.entries[0]
    before = env.data.subtree_com.numpy().copy()
    with wp.ScopedDevice(env.wp_device):
        bank, receipt = capture.repeat_entry(
            env.model, env.data, raw, sha256(raw).hexdigest(), mode
        )
    assert len(bank) == 32 * 384
    assert receipt["repeats"] == receipt["resets"] == 32
    assert receipt["accumulation_launches"] == count
    assert receipt["output_boundary"] == "after-accumulation-before-division"
    assert all(value is False for value in receipt["flags"].values())
    np.testing.assert_array_equal(
        env.data.subtree_com.numpy().view(np.uint32), before.view(np.uint32)
    )
    expected = np.frombuffer(raw, dtype="<f4").reshape(2, 16, 3).copy()
    for level in capture.source_checks.EXPECTED_REVERSED_LEVELS:
        for body in level:
            if body:
                parent = capture.source_checks.EXPECTED_PARENTS[body]
                np.add(expected[:, parent], expected[:, body], out=expected[:, parent])
    values = np.frombuffer(bank, dtype="<f4").reshape(32, 2, 16, 3)
    np.testing.assert_array_equal(
        values.view("<u4"), np.broadcast_to(expected.view("<u4"), values.shape)
    )
    assert (
        values[:, :, 0, :]
        != np.frombuffer(raw, dtype="<f4").reshape(2, 16, 3)[None, :, 0, :]
    ).any()


@pytest.mark.parametrize("mutation", ["bad-hash", "short", "mutable", "nan", "mode"])
def test_repeat_rejects_invalid_inputs_before_kernel(observed, monkeypatch, mutation):
    env, scope = observed
    raw, mode = scope.entries[0], "serial"
    digest = sha256(raw).hexdigest()
    if mutation == "bad-hash":
        digest = "0" * 64
    elif mutation == "short":
        raw = raw[:-4]
    elif mutation == "mutable":
        raw = bytearray(raw)
    elif mutation == "nan":
        values = np.frombuffer(raw, dtype="<f4").copy()
        values[0] = np.nan
        raw = values.tobytes()
        digest = sha256(raw).hexdigest()
    else:
        mode = "other"
    launches = []
    with monkeypatch.context() as context:

        def forbidden(*args, **kwargs):
            launches.append(args)
            raise AssertionError("invalid input must not launch")

        context.setattr(capture, "_LAUNCH", forbidden)
        context.setattr(wp, "launch", forbidden)
        with wp.ScopedDevice(env.wp_device), pytest.raises(ValueError):
            capture.repeat_entry(env.model, env.data, raw, digest, mode)
    assert not launches


@pytest.mark.parametrize(
    "field", ["body_mass", "body_subtreemass", "subtree_com", "xipos"]
)
def test_layout_refuses_storage_and_shape_changes(observed, field):
    env, _ = observed
    owner = env.model if field.startswith("body_") else env.data
    original = getattr(owner, field)
    try:
        setattr(
            owner, field, wp.zeros((2, 15), dtype=original.dtype, device=env.wp_device)
        )
        with wp.ScopedDevice(env.wp_device), pytest.raises(ValueError):
            capture._layout(env.model, env.data)
    finally:
        setattr(owner, field, original)


def test_incomplete_scope_restores_original():
    scope = capture.ComEntryCapture()
    with pytest.raises(ValueError, match="complete exact two-call"):
        with scope:
            pass
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH
    with pytest.raises(ValueError, match="one-shot"):
        scope.__enter__()


def test_exception_restores_original():
    scope = capture.ComEntryCapture()
    with pytest.raises(RuntimeError, match="synthetic"):
        with scope:
            raise RuntimeError("synthetic")
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


def test_foreign_entry_is_preserved(monkeypatch):
    scope = capture.ComEntryCapture()

    def foreign(*args, **kwargs):
        return None

    assert smooth.com_pos is capture._ENTRY
    with monkeypatch.context() as context:
        context.setattr(smooth, "com_pos", capture._ENTRY)
        with pytest.raises(ValueError, match="complete exact two-call"):
            with scope:
                context.setattr(smooth, "com_pos", foreign)
        assert smooth.com_pos is foreign
    assert smooth.com_pos is capture._ENTRY


def test_foreign_launch_refused_and_preserved(monkeypatch):
    def foreign(*args, **kwargs):
        return None

    with monkeypatch.context() as context:
        context.setattr(wp, "launch", foreign)
        scope = capture.ComEntryCapture()
        with pytest.raises(ValueError, match="unmodified CoM"):
            scope.__enter__()
        assert wp.launch is foreign
    assert wp.launch is capture._LAUNCH


def test_nonmain_thread_refused():
    errors = []

    def worker():
        try:
            capture.ComEntryCapture().__enter__()
        except ValueError as error:
            errors.append(str(error))

    thread = Thread(target=worker)
    thread.start()
    thread.join()
    assert errors == ["main-thread CoM scope"]


def test_third_call_and_same_shape_array_replacement_refused(observed):
    env, _ = observed
    scope = capture.ComEntryCapture()
    with pytest.raises(ValueError, match="bounded unmodified"):
        with scope, wp.ScopedDevice(env.wp_device):
            smooth.com_pos(env.model, env.data)
            smooth.com_pos(env.model, env.data)
            smooth.com_pos(env.model, env.data)
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH
    scope = capture.ComEntryCapture()
    original = env.data.subtree_com
    try:
        with pytest.raises(ValueError, match="unchanged CoM objects"):
            with scope, wp.ScopedDevice(env.wp_device):
                smooth.com_pos(env.model, env.data)
                env.data.subtree_com = wp.zeros(
                    (2, 16), dtype=wp.vec3, device=env.wp_device
                )
                smooth.com_pos(env.model, env.data)
    finally:
        env.data.subtree_com = original
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH
