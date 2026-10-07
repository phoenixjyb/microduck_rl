"""Two-world real CPU RNE checks, never native CUDA or learner admission."""

from hashlib import sha256
from threading import Thread
from types import MethodType

import numpy as np
import pytest
import torch
import warp as wp
from mujoco_warp._src import smooth

from mjlab_microduck import stance_rne_entry_capture as capture
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


@pytest.fixture(autouse=True)
def hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="module")
def observed():
    scope = capture.RneEntryCapture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with scope:
            env = WarpStanceRuntime(nworld=2, device="cpu", solved_field_check="packed")
            env._forward()
    return env, scope


def test_original_initializer_caller_and_both_aliased_outputs(observed):
    env, scope = observed
    receipt = scope.receipt
    assert receipt["status"] == "complete"
    assert receipt["fault"] is None
    assert receipt["unchanged_postconstraint_passthrough_calls"] == 2
    assert len(scope.entries) == len(scope.outputs) == len(receipt["calls"]) == 2
    assert scope.entries[0] == scope.entries[1]
    assert scope.outputs[0] == scope.outputs[1]
    assert all(len(raw) == 768 for raw in (*scope.entries, *scope.outputs))
    assert all(row["original_launches"] == 7 for row in receipt["calls"])
    assert all(
        row["initializer_caller_code_bound"]
        and row["input_output_alias"]
        and row["readback_and_device_sync_perturb_timing"]
        for row in receipt["calls"]
    )
    assert all(
        row["input_boundary"] == "after-original-cfrc-before-first-backward"
        and row["output_boundary"]
        == "after-original-backward-before-original-qfrc-bias"
        and row["body_rule"] == "body!=0-includes-body1-to-root0"
        for row in receipt["calls"]
    )
    assert all(value is False for value in receipt["flags"].values())
    assert (env.steps == 0).all() and env.forward_graph is None
    assert smooth._rne_cfrc_backward is capture._ENTRY and wp.launch is capture._LAUNCH
    receipt["calls"].clear()
    assert len(scope.receipt["calls"]) == 2


@pytest.mark.parametrize("mode,count", [("concurrent", 224), ("serial", 288)])
def test_detached_repeat_bank_all_scalars_including_root(observed, mode, count):
    env, scope = observed
    raw = scope.entries[0]
    before = env.data.cfrc_int.numpy().copy()
    with wp.ScopedDevice(env.wp_device):
        bank, receipt = capture.repeat_entry(
            env.model, env.data, raw, sha256(raw).hexdigest(), mode
        )
    assert len(bank) == 32 * 768
    assert receipt["repeats"] == receipt["resets"] == 32
    assert receipt["accumulation_launches"] == count
    assert receipt["live_before_sha256"] == receipt["live_after_sha256"]
    assert receipt["live_array_unchanged"]
    assert all(value is False for value in receipt["flags"].values())
    assert env.data.cfrc_int.numpy().tobytes() == before.tobytes()
    expected = np.frombuffer(raw, dtype="<f4").reshape(2, 16, 6).copy()
    for level in capture.source_checks.EXPECTED_REVERSED_LEVELS:
        for body in level:
            if body != 0:
                np.add(
                    expected[:, capture.source_checks.EXPECTED_PARENTS[body]],
                    expected[:, body],
                    out=expected[:, capture.source_checks.EXPECTED_PARENTS[body]],
                )
    values = np.frombuffer(bank, dtype="<f4").reshape(32, 2, 16, 6)
    np.testing.assert_array_equal(
        values.view("<u4"), np.broadcast_to(expected.view("<u4"), values.shape)
    )
    assert values[0].tobytes() == scope.outputs[0]
    assert (
        values[:, :, 0, :]
        != np.frombuffer(raw, dtype="<f4").reshape(2, 16, 6)[None, :, 0, :]
    ).any()


@pytest.mark.parametrize("mutation", ["bad-hash", "short", "mutable", "nan", "mode"])
def test_bad_repeat_bytes_refused_before_launch(observed, monkeypatch, mutation):
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
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("invalid repeat must not launch")

    with monkeypatch.context() as patch:
        patch.setattr(capture, "_LAUNCH", forbidden)
        patch.setattr(wp, "launch", forbidden)
        with wp.ScopedDevice(env.wp_device), pytest.raises(ValueError):
            capture.repeat_entry(env.model, env.data, raw, digest, mode)
    assert not calls


def test_direct_backward_is_not_an_initialized_callsite(observed):
    env, _ = observed
    scope = capture.RneEntryCapture()
    with pytest.raises(ValueError, match="actual unchanged RNE initializer caller"):
        with wp.ScopedDevice(env.wp_device), scope:
            smooth._rne_cfrc_backward(env.model, env.data)
    assert scope.receipt["status"] == "faulted" and not scope.entries
    assert smooth._rne_cfrc_backward is capture._ENTRY and wp.launch is capture._LAUNCH


@pytest.mark.parametrize("field", ["cfrc_int", "cacc", "cvel"])
def test_layout_rejects_wrong_spatial_storage(observed, field):
    env, _ = observed
    original = getattr(env.data, field)
    try:
        setattr(
            env.data,
            field,
            wp.zeros((2, 15), dtype=wp.spatial_vector, device=env.wp_device),
        )
        with (
            wp.ScopedDevice(env.wp_device),
            pytest.raises(ValueError, match="spatial arrays"),
        ):
            capture._layout(env.model, env.data)
    finally:
        setattr(env.data, field, original)


def test_layout_rejects_acceleration_force_alias(observed):
    env, _ = observed
    original = env.data.cacc
    try:
        env.data.cacc = env.data.cfrc_int
        with (
            wp.ScopedDevice(env.wp_device),
            pytest.raises(ValueError, match="distinct force"),
        ):
            capture._layout(env.model, env.data)
    finally:
        env.data.cacc = original


def test_incomplete_scope_is_faulted_and_restores_owned_references():
    scope = capture.RneEntryCapture()
    with pytest.raises(ValueError, match="complete exact two-call"):
        with scope:
            pass
    assert scope.receipt["status"] == "faulted"
    assert smooth._rne_cfrc_backward is capture._ENTRY
    with pytest.raises(ValueError, match="one-shot"):
        scope.__enter__()


def test_foreign_entry_is_preserved_on_fault(monkeypatch):
    def foreign(*_):
        return None

    with monkeypatch.context() as patch:
        patch.setattr(smooth, "_rne_cfrc_backward", capture._ENTRY)
        with pytest.raises(ValueError, match="complete exact two-call"):
            with capture.RneEntryCapture():
                patch.setattr(smooth, "_rne_cfrc_backward", foreign)
        assert smooth._rne_cfrc_backward is foreign
    assert smooth._rne_cfrc_backward is capture._ENTRY


def test_foreign_launch_is_preserved_on_fault(monkeypatch):
    def foreign(*_):
        return None

    with monkeypatch.context() as patch:
        patch.setattr(wp, "launch", capture._LAUNCH)
        with pytest.raises(ValueError, match="complete exact two-call"):
            with capture.RneEntryCapture():
                patch.setattr(wp, "launch", foreign)
        assert wp.launch is foreign
    assert wp.launch is capture._LAUNCH


def test_foreign_initializer_reference_is_refused_before_scope(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(smooth, "_rne_cfrc", lambda *_: None)
        with pytest.raises(ValueError, match="unmodified RNE initialization"):
            capture.RneEntryCapture().__enter__()
    # Failure must release the process-local ownership lock.
    with pytest.raises(ValueError, match="complete exact two-call"):
        with capture.RneEntryCapture():
            pass


@pytest.mark.parametrize("kind", ["subclass", "instance", "class"])
def test_changed_observer_hooks_are_not_admitted(monkeypatch, kind):
    if kind == "subclass":

        class Custom(capture.RneEntryCapture):
            pass

        scope = Custom()
    else:
        scope = capture.RneEntryCapture()
    with monkeypatch.context() as patch:
        if kind == "instance":
            patch.setattr(scope, "_call", MethodType(lambda *_: None, scope))
        elif kind == "class":
            patch.setattr(capture.RneEntryCapture, "_call", lambda *_: None)
        with pytest.raises(ValueError, match="unmodified RNE observer"):
            scope.__enter__()
    assert smooth._rne_cfrc_backward is capture._ENTRY


def test_worker_thread_cannot_open_scope():
    errors = []

    def worker():
        try:
            capture.RneEntryCapture().__enter__()
        except ValueError as error:
            errors.append(str(error))

    thread = Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and "main-thread" in errors[0]


def test_nested_scope_cannot_replace_owned_entry():
    outer = capture.RneEntryCapture()
    with pytest.raises(ValueError, match="complete exact two-call"):
        with outer:
            with pytest.raises(ValueError, match="exclusive process-local"):
                capture.RneEntryCapture().__enter__()
            assert smooth._rne_cfrc_backward is outer._wrapper
