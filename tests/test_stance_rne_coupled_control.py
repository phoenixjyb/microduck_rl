"""CPU-only checks for the bounded actual-coupled serial RNE control."""

from threading import Thread
from types import MethodType

import numpy as np
import pytest
import torch
import warp as wp
from mujoco_warp._src import smooth

from mjlab_microduck import stance_rne_coupled_control as control
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


@pytest.fixture(autouse=True)
def _cuda_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="module")
def coupled_serial():
    scope = control.SerialRneCoupledControl()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with scope:
            env = WarpStanceRuntime(nworld=2, device="cpu", solved_field_check="packed")
            env._forward()
    return env, scope


def test_actual_cpu_path_splits_only_bias_and_keeps_sensory_passthrough(coupled_serial):
    env, scope = coupled_serial
    receipt = scope.receipt

    assert receipt["protocol"] == control.PROTOCOL
    assert receipt["status"] == "complete"
    assert receipt["fault"] is None
    assert receipt["unchanged_postconstraint_passthrough_calls"] == 2
    assert receipt["dispatch"] == {
        "logical_launches_per_bias_call": 7,
        "underlying_launches_per_bias_call": 9,
        "split_level": 4,
        "split_body_order": [2, 7, 11],
        "split_uses_live_initialized_alias": True,
        "sensory_passthrough_calls": 2,
        "closed": True,
    }
    assert len(scope.entries) == len(scope.outputs) == len(receipt["calls"]) == 2
    assert all(len(raw) == 768 for raw in (*scope.entries, *scope.outputs))
    for row in receipt["calls"]:
        assert row["logical_launches"] == 7
        assert row["underlying_launches"] == 9
        assert row["split_level"] == 4
        assert row["split_body_order"] == [2, 7, 11]
        assert row["underlying_body_groups"] == [
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
        assert row["split_uses_live_initialized_alias"] is True
        assert row["input_output_alias"] is True
        assert row["layout"]["object_id"] == id(env.data.cfrc_int)
        assert row["layout"]["ptr"] == int(env.data.cfrc_int.ptr)
    assert all(value is False for value in receipt["flags"].values())
    assert env.forward_graph is None and (env.steps == 0).all()
    assert smooth._rne_cfrc_backward is control._ENTRY
    assert wp.launch is control._LAUNCH
    receipt["calls"].clear()
    assert len(scope.receipt["calls"]) == 2


def test_recorder_observes_seven_logical_and_nine_underlying_launches(monkeypatch):
    observed = []
    original_launch = control._LAUNCH

    def record(*args, **kwargs):
        if not args or args[0] is not control._KERNEL:
            return original_launch(*args, **kwargs)
        row = kwargs["inputs"][2]
        body_ids = tuple(row.numpy().reshape(-1).tolist())
        observed.append(
            (
                args[0],
                kwargs["dim"],
                body_ids,
                kwargs["inputs"][1],
                kwargs["outputs"][0],
            )
        )
        return original_launch(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wp, "launch", record)
        patch.setattr(control, "_LAUNCH", record)
        scope = control.SerialRneCoupledControl()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(977)
            with scope:
                env = WarpStanceRuntime(
                    nworld=2, device="cpu", solved_field_check="packed"
                )
                env._forward()

    assert len(observed) == 32
    assert all(kernel is control._KERNEL for kernel, *_ in observed)
    serial_groups = [
        (6, 15),
        (5, 10, 14),
        (4, 9, 13),
        (3, 8, 12),
        (2,),
        (7,),
        (11,),
        (1,),
        (0,),
    ]
    original_groups = [
        (6, 15),
        (5, 10, 14),
        (4, 9, 13),
        (3, 8, 12),
        (2, 7, 11),
        (1,),
        (0,),
    ]
    assert [ids for _, _, ids, _, _ in observed[:9]] == serial_groups
    assert [ids for _, _, ids, _, _ in observed[9:16]] == original_groups
    assert [ids for _, _, ids, _, _ in observed[16:25]] == serial_groups
    assert [ids for _, _, ids, _, _ in observed[25:]] == original_groups
    assert [dim for _, dim, *_ in observed[:9]] == [
        [2, 2],
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
    ]
    assert [dim for _, dim, *_ in observed[9:16]] == [
        [2, 2],
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 1],
        [2, 1],
    ]
    assert all(inputs is outputs for _, _, _, inputs, outputs in observed)
    assert len(scope.entries) == 2
    assert env.forward_graph is None


def test_direct_backward_is_not_an_initialized_caller(coupled_serial):
    env, _ = coupled_serial
    scope = control.SerialRneCoupledControl()
    with pytest.raises(ValueError, match="actual pinned RNE initializer caller"):
        with wp.ScopedDevice(env.wp_device), scope:
            smooth._rne_cfrc_backward(env.model, env.data)
    assert scope.receipt["status"] == "faulted"
    assert not scope.entries
    assert smooth._rne_cfrc_backward is control._ENTRY


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("SPLIT_LEVEL_INDEX", 3), ("SPLIT_BODY_IDS", (2, 7, 10))],
)
def test_wrong_sibling_level_or_body_ids_are_refused(monkeypatch, attribute, value):
    with monkeypatch.context() as patch:
        patch.setattr(control, attribute, value)
        scope = control.SerialRneCoupledControl()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(977)
            with pytest.raises(ValueError):
                with scope:
                    env = WarpStanceRuntime(
                        nworld=2, device="cpu", solved_field_check="packed"
                    )
                    env._forward()
    assert scope.receipt["status"] == "faulted"
    assert smooth._rne_cfrc_backward is control._ENTRY
    assert wp.launch is control._LAUNCH


@pytest.mark.parametrize("mutation", ["shape", "kernel", "alias", "body-order"])
def test_malformed_original_request_is_refused(monkeypatch, mutation):
    original_entry = control._ENTRY
    original_launch = control._LAUNCH

    def malformed_entry(model, data):
        levels = tuple(reversed(model.body_tree))
        for index, row in enumerate(levels):
            kernel = control._KERNEL
            dim = [data.nworld, row.size]
            current = row
            output = data.cfrc_int
            if index == 4:
                if mutation == "shape":
                    dim = [data.nworld, row.size + 1]
                elif mutation == "kernel":
                    kernel = object()
                elif mutation == "alias":
                    output = data.cacc
                elif mutation == "body-order":
                    current = wp.array([2, 11, 7], dtype=wp.int32, device=row.device)
            wp.launch(
                kernel,
                dim=dim,
                inputs=[model.body_parentid, data.cfrc_int, current],
                outputs=[output],
            )
        return None

    with monkeypatch.context() as patch:
        patch.setattr(control, "_ENTRY", malformed_entry)
        patch.setattr(smooth, "_rne_cfrc_backward", malformed_entry)
        scope = control.SerialRneCoupledControl()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(977)
            with pytest.raises(
                ValueError, match="original RNE kernel and actual aliased"
            ):
                with scope:
                    env = WarpStanceRuntime(
                        nworld=2, device="cpu", solved_field_check="packed"
                    )
                    env._forward()
    assert scope.receipt["status"] == "faulted"
    assert not scope.entries
    assert smooth._rne_cfrc_backward is original_entry
    assert wp.launch is original_launch


def test_incomplete_scope_restores_owned_entry():
    scope = control.SerialRneCoupledControl()
    with pytest.raises(ValueError, match="complete exact serial RNE scope"):
        with scope:
            pass
    assert scope.receipt["status"] == "faulted"
    assert smooth._rne_cfrc_backward is control._ENTRY
    with pytest.raises(ValueError, match="one-shot"):
        scope.__enter__()


def test_foreign_entry_is_preserved_on_scope_fault(monkeypatch):
    def foreign(*_args, **_kwargs):
        return None

    with monkeypatch.context() as patch:
        patch.setattr(smooth, "_rne_cfrc_backward", control._ENTRY)
        with pytest.raises(ValueError, match="complete exact serial RNE scope"):
            with control.SerialRneCoupledControl():
                patch.setattr(smooth, "_rne_cfrc_backward", foreign)
        assert smooth._rne_cfrc_backward is foreign
    assert smooth._rne_cfrc_backward is control._ENTRY


def test_worker_thread_cannot_open_scope():
    errors = []

    def worker():
        try:
            control.SerialRneCoupledControl().__enter__()
        except ValueError as error:
            errors.append(str(error))

    thread = Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and "main-thread" in errors[0]


def test_actual_cpu_serial_outputs_match_independent_full_recurrence(coupled_serial):
    _env, scope = coupled_serial
    assert scope.entries[0] == scope.entries[1]
    for entry, output in zip(scope.entries, scope.outputs, strict=True):
        expected = np.frombuffer(entry, dtype="<f4").reshape(2, 16, 6).copy()
        for level in control.observer.source_checks.EXPECTED_REVERSED_LEVELS:
            for body in level:
                if body != 0:
                    parent = control.observer.source_checks.EXPECTED_PARENTS[body]
                    np.add(
                        expected[:, parent], expected[:, body], out=expected[:, parent]
                    )
        assert expected.tobytes() == output


@pytest.mark.parametrize("kind", ["subclass", "instance", "class"])
def test_changed_control_methods_are_refused_before_patch(monkeypatch, kind):
    if kind == "subclass":

        class Custom(control.SerialRneCoupledControl):
            pass

        scope = Custom()
    else:
        scope = control.SerialRneCoupledControl()
    with monkeypatch.context() as patch:
        if kind == "instance":
            patch.setattr(scope, "_call", MethodType(lambda *_: None, scope))
        elif kind == "class":
            patch.setattr(control.SerialRneCoupledControl, "_call", lambda *_: None)
        with pytest.raises(ValueError, match="unmodified serial RNE control methods"):
            scope.__enter__()
    assert smooth._rne_cfrc_backward is control._ENTRY


def test_foreign_launch_is_preserved_and_owned_entry_restored(monkeypatch):
    def foreign(*_args, **_kwargs):
        return None

    with monkeypatch.context() as patch:
        patch.setattr(wp, "launch", control._LAUNCH)
        with pytest.raises(ValueError, match="complete exact serial RNE scope"):
            with control.SerialRneCoupledControl():
                patch.setattr(wp, "launch", foreign)
        assert wp.launch is foreign
        assert smooth._rne_cfrc_backward is control._ENTRY
    assert wp.launch is control._LAUNCH


def test_nested_scope_cannot_replace_exclusive_owned_entry():
    outer = control.SerialRneCoupledControl()
    with pytest.raises(ValueError, match="complete exact serial RNE scope"):
        with outer:
            with pytest.raises(ValueError, match="exclusive serial RNE scope"):
                control.SerialRneCoupledControl().__enter__()
            assert smooth._rne_cfrc_backward is outer._wrapper
