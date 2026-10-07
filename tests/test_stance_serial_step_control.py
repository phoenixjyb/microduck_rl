"""CPU-only integration and fault-boundary tests for one-step control."""

from threading import Thread

import pytest
import numpy as np
import torch
import warp as wp
from mujoco_warp._src import smooth

from mjlab_microduck import stance_serial_step_control as control
from mjlab_microduck.stance_com_entry_receiver import FLAGS
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


@pytest.fixture(autouse=True)
def _cuda_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _build(scope):
    env = WarpStanceRuntime(nworld=2, device="cpu", solved_field_check="packed")
    scope.bind_runtime(env)
    return env


def _cpu_accumulate(raw, worlds, components):
    values = np.frombuffer(raw, dtype="<f4").reshape(worlds, 16, components).copy()
    for group in control._ACCUMULATION_GROUPS:
        for body in group:
            if body == 0:
                continue
            parent = control._PARENTS[body]
            np.add(values[:, parent], values[:, body], out=values[:, parent])
    return values


def test_actual_cpu_constructor_and_one_nominal_step_are_exactly_bounded():
    scope = control.SerialStepControl()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(781)
        with scope:
            env = _build(scope)
            result = scope.step(
                env,
                torch.zeros((2, 10), dtype=torch.float32, device="cpu"),
                capture_control=True,
            )

    receipt = scope.receipt
    assert receipt["protocol"] == control.PROTOCOL
    assert receipt["status"] == "complete"
    assert receipt["fault"] is None
    assert receipt["forward_calls"] == control.FORWARDS == 21
    assert receipt["constructor_forward_calls"] == 1
    assert receipt["step_calls"] == 1
    assert env.forward_graph is None and (env.steps == 10).all()
    assert result["optimizer_launched"] is False
    assert len(result["boundaries"]) == 11
    assert receipt["entry_counts"] == {
        "com": 21,
        "crb": 21,
        "rne_bias": 21,
        "rne_post": 21,
    }
    assert len(scope.com_initialized) == len(scope.com_weighted) == 21
    assert len(scope.rne_inputs) == len(scope.rne_outputs) == 21
    assert all(len(raw) == 2 * 16 * 3 * 4 for raw in scope.com_initialized)
    assert all(len(raw) == 2 * 16 * 3 * 4 for raw in scope.com_weighted)
    assert all(len(raw) == 2 * 16 * 6 * 4 for raw in scope.rne_inputs)
    assert all(len(raw) == 2 * 16 * 6 * 4 for raw in scope.rne_outputs)
    for initialized, weighted in zip(
        scope.com_initialized, scope.com_weighted, strict=True
    ):
        expected = _cpu_accumulate(initialized, 2, 3)
        actual = np.frombuffer(weighted, dtype="<f4").reshape(2, 16, 3)
        assert np.array_equal(actual.view("<u4"), expected.view("<u4"))
    for initialized, accumulated in zip(
        scope.rne_inputs, scope.rne_outputs, strict=True
    ):
        expected = _cpu_accumulate(initialized, 2, 6)
        actual = np.frombuffer(accumulated, dtype="<f4").reshape(2, 16, 6)
        assert np.array_equal(actual.view("<u4"), expected.view("<u4"))
    assert [row["phase"] for row in receipt["calls"]] == [
        "constructor",
        *("step-pre", "step-post") * 10,
    ]
    assert all(row["com"]["logical_launches"] == 11 for row in receipt["calls"])
    assert all(row["com"]["underlying_launches"] == 13 for row in receipt["calls"])
    assert all(
        row["crb"]["logical_accumulation_launches"] == 7 for row in receipt["calls"]
    )
    assert all(
        row["crb"]["underlying_accumulation_launches"] == 9 for row in receipt["calls"]
    )
    assert all(row["crb"]["qM_launches"] == 1 for row in receipt["calls"])
    assert all(row["rne_bias"]["logical_launches"] == 7 for row in receipt["calls"])
    assert all(row["rne_bias"]["underlying_launches"] == 9 for row in receipt["calls"])
    assert all(
        row["rne_sensory"] == "untouched-original-passthrough"
        for row in receipt["calls"]
    )
    assert receipt["dispatch"]["split_body_order"] == [2, 7, 11]
    assert receipt["dispatch"]["sensory_rne_untouched_passthrough_calls"] == 21
    assert receipt["dispatch"]["all_observation_readbacks_perturb_timing"] is True
    assert all(value is False for value in receipt["flags"].values())
    assert receipt["flags"] == dict(FLAGS, native_qualified=False)
    assert smooth.crb is control._CRB
    assert smooth.com_pos is control._COM
    assert smooth._rne_cfrc_backward is control._RNE_ENTRY
    assert wp.launch is control._LAUNCH


def test_receipts_bind_full_serial_group_order_for_every_forward():
    scope = control.SerialStepControl()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(781)
        with scope:
            env = _build(scope)
            scope.step(
                env,
                torch.zeros((2, 10), dtype=torch.float32, device="cpu"),
                capture_control=True,
            )
    expected = [
        [6, 15],
        [5, 10, 14],
        [4, 9, 13],
        [3, 8, 12],
        [2],
        [7],
        [11],
        [1],
        [0],
    ]
    for row in scope.receipt["calls"]:
        assert row["com"]["underlying_body_groups"] == expected
        assert row["com"]["observed_body_groups"] == expected
        assert row["crb"]["underlying_body_groups"] == expected
        assert row["crb"]["observed_body_groups"] == expected
        assert row["rne_bias"]["underlying_body_groups"] == expected
        assert row["rne_bias"]["observed_body_groups"] == expected
        assert len(row["array_layouts"]) == 31
        assert row["stream"] is None
    trace = scope.receipt["dispatch_trace"]
    assert len(trace) == 21 * 27
    assert [row["entry"] for row in trace[:27]] == [
        *("com",) * 9,
        *("crb",) * 9,
        *("rne_bias",) * 9,
    ]
    assert [row["body_ids"] for row in trace[:9]] == expected


def test_scope_without_constructor_and_step_is_rejected():
    scope = control.SerialStepControl()
    with pytest.raises(ValueError, match="complete exact one-constructor/one-step"):
        with scope:
            pass
    assert scope.receipt["status"] == "faulted"
    with pytest.raises(ValueError, match="one-shot"):
        scope.__enter__()


def test_constructor_only_scope_is_incomplete():
    scope = control.SerialStepControl()
    with pytest.raises(ValueError, match="complete exact one-constructor/one-step"):
        with scope:
            _build(scope)
    assert scope.receipt["forward_calls"] == 1
    assert scope.receipt["status"] == "faulted"


def test_invalid_action_is_rejected_before_step_dispatch():
    scope = control.SerialStepControl()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(781)
        with pytest.raises(ValueError, match="exact single nominal zero-action"):
            with scope:
                env = _build(scope)
                scope.step(env, torch.ones((2, 10), dtype=torch.float32))
    assert scope.receipt["forward_calls"] == 1
    assert scope.receipt["step_calls"] == 0


def test_twenty_second_forward_is_refused_before_another_kernel(monkeypatch):
    scope = control.SerialStepControl()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(781)
        with pytest.raises(ValueError, match="hard 21-forward bound"):
            with scope:
                env = _build(scope)
                scope.step(
                    env,
                    torch.zeros((2, 10), dtype=torch.float32),
                    capture_control=True,
                )
                env._forward()
    assert scope.receipt["forward_calls"] == 21
    assert scope.receipt["status"] == "faulted"


def test_foreign_entry_is_not_clobbered_on_scope_exit():
    original = control._COM

    def foreign(*_args, **_kwargs):
        return None

    scope = control.SerialStepControl()
    try:
        with pytest.raises(ValueError, match="complete exact one-constructor/one-step"):
            with scope:
                smooth.com_pos = foreign
        assert smooth.com_pos is foreign
    finally:
        smooth.com_pos = original


def test_direct_com_entry_is_not_a_pinned_forward_caller_and_fault_latches():
    scope = control.SerialStepControl()
    with pytest.raises(ValueError, match="complete exact one-constructor/one-step"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(781)
            with scope:
                env = _build(scope)
                with pytest.raises(
                    ValueError, match="actual pinned position-phase CoM"
                ):
                    smooth.com_pos(env.model, env.data)
                with pytest.raises(ValueError, match="active isolated main-thread"):
                    scope.step(
                        env,
                        torch.zeros((2, 10), dtype=torch.float32),
                        capture_control=True,
                    )
    assert scope.receipt["status"] == "faulted"
    assert scope.receipt["forward_calls"] == 1


@pytest.mark.parametrize("name", ["_forward", "reset"])
def test_runtime_method_substitution_is_rejected_before_constructor(monkeypatch, name):
    monkeypatch.setattr(WarpStanceRuntime, name, lambda *_: None)
    scope = control.SerialStepControl()
    with pytest.raises(ValueError, match="pinned forward"):
        scope.__enter__()
    assert scope.receipt["status"] == "faulted"
    assert scope.receipt["forward_calls"] == 0


def test_live_com_kernel_substitution_is_rejected_before_constructor(monkeypatch):
    monkeypatch.setattr(smooth, "_subtree_com_init", object())
    scope = control.SerialStepControl()
    with pytest.raises(ValueError, match="pinned forward, reduction kernels"):
        with scope:
            pass
    assert scope.receipt["status"] == "faulted"
    assert scope.receipt["forward_calls"] == 0


def test_live_com_accumulation_alias_replacement_is_refused_before_dispatch():
    scope = control.SerialStepControl()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(781)
        with pytest.raises(ValueError, match="unchanged live model/data arrays"):
            with scope:
                env = _build(scope)
                original = env.data.subtree_com
                env.data.subtree_com = wp.zeros(
                    (2, 16), dtype=wp.vec3, device=env.wp_device
                )
                try:
                    scope.step(
                        env,
                        torch.zeros((2, 10), dtype=torch.float32),
                        capture_control=True,
                    )
                finally:
                    env.data.subtree_com = original
    assert scope.receipt["forward_calls"] == 1
    assert scope.receipt["status"] == "faulted"


def test_split_body_declaration_cannot_be_mutated(monkeypatch):
    monkeypatch.setattr(control, "SPLIT_BODY_IDS", (2, 7, 10))
    scope = control.SerialStepControl()
    with pytest.raises(ValueError, match="immutable pinned callables"):
        with scope:
            pass
    assert scope.receipt["status"] == "faulted"


def test_scope_rejects_a_non_main_thread():
    scope = control.SerialStepControl()
    errors = []

    def enter_from_worker():
        try:
            scope.__enter__()
        except ValueError as error:
            errors.append(str(error))

    thread = Thread(target=enter_from_worker)
    thread.start()
    thread.join()
    assert errors == ["one-shot isolated main-thread serial-step scope"]
    assert scope.receipt["status"] == "faulted"
    assert scope.receipt["fault"] == "ThreadBoundaryViolation"


def test_scope_rejects_mutated_controller_method(monkeypatch):
    monkeypatch.setattr(control.SerialStepControl, "_launch_com", lambda *_args: None)
    scope = control.SerialStepControl()
    with pytest.raises(ValueError, match="exact owned serial-step control methods"):
        with scope:
            pass
    assert scope.receipt["status"] == "faulted"
