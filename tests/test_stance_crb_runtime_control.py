"""CUDA-hidden tests for narrowly scoped CRB runtime launch control."""

import gc
from threading import Thread

import pytest
import torch
import warp as wp

import mujoco_warp as public_mjwarp
from mujoco_warp._src import forward as pinned_forward
from mujoco_warp._src import smooth as pinned_smooth
from mjlab_microduck import stance_crb_runtime_control as control_module
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

SOURCE = "a" * 40
CELL_IDS = ["zero-wrench", "zero-wrench"]


@pytest.fixture(autouse=True)
def _cuda_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(autouse=True)
def _collect_after_test():
    yield
    gc.collect()


def _declaration():
    return schedule.declaration(SOURCE, "dose", "training", CELL_IDS)


def _construct(mode="serial", *, max_forward_calls=4, call_count=3):
    scope = control_module.StanceCrbRuntimeControl(
        mode=mode, max_forward_calls=max_forward_calls
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with scope:
            env = ScheduledRecoveryRuntime(
                _declaration(), device="cpu", solved_field_check="packed"
            )
            scope.bind_runtime(env)
            for _ in range(call_count):
                env._forward()
        receipt = scope.receipt
    return env, receipt


def test_never_entered_scope_receipt_is_not_started():
    scope = control_module.StanceCrbRuntimeControl(mode="serial")
    assert scope.receipt["status"] == "not-started"
    assert scope.receipt["forward_calls"] == 0


def test_serial_scope_covers_constructor_and_splits_only_conflict_level():
    original_crb = pinned_smooth.crb
    original_launch = wp.launch
    env, receipt = _construct("serial")

    assert receipt["status"] == "complete"
    assert receipt["mode"] == "serial"
    assert receipt["smooth_source_sha256"] == control_module.PINNED_SMOOTH_SHA256
    assert receipt["constructor_forward_calls"] == 1
    assert receipt["constructor_forward_covered"] is True
    assert receipt["runtime_bound"] is True
    assert receipt["forward_calls"] == 4
    assert receipt["topology_id_snapshots"] == 4
    assert receipt["original_level_launch_requests"] == 28
    assert receipt["actual_accumulation_launches"] == 36
    assert receipt["split_child_launches"] == 12
    assert receipt["dense_qM_launches"] == 4
    assert receipt["topology_id_snapshots"] == 4
    assert receipt["parent_zero_noop_level_requests"] == 8
    assert receipt["singleton_child_arrays_allocated"] == 3
    assert receipt["initialization_timing_changed"] is True
    assert all(value is False for value in receipt["flags"].values())
    assert pinned_smooth.crb is original_crb
    assert wp.launch is original_launch
    assert env._forward.__func__ is WarpStanceRuntime._forward
    assert (
        env._scheduled_forward.__func__ is ScheduledRecoveryRuntime._scheduled_forward
    )
    assert "_forward" not in env.__dict__
    assert "_scheduled_forward" not in env.__dict__


def test_concurrent_mode_forwards_all_seven_original_level_launches():
    _env, receipt = _construct("concurrent")

    assert receipt["mode"] == "concurrent"
    assert receipt["constructor_forward_covered"] is True
    assert receipt["forward_calls"] == 4
    assert receipt["original_level_launch_requests"] == 28
    assert receipt["actual_accumulation_launches"] == 28
    assert receipt["split_child_launches"] == 0
    assert receipt["dense_qM_launches"] == 4
    assert receipt["topology_id_snapshots"] == 4


@pytest.mark.parametrize(
    ("mode", "max_forward_calls"),
    [("unknown", 4), ("serial", True), ("serial", 0), ("serial", 5)],
)
def test_scope_rejects_unbounded_or_unknown_configuration(mode, max_forward_calls):
    with pytest.raises(ValueError):
        control_module.StanceCrbRuntimeControl(
            mode=mode, max_forward_calls=max_forward_calls
        )


def test_scope_requires_exact_pinned_smooth_and_warp_references(monkeypatch):
    original_crb = pinned_smooth.crb

    def replacement(*_args, **_kwargs):
        return None

    monkeypatch.setattr(pinned_smooth, "crb", replacement)
    scope = control_module.StanceCrbRuntimeControl(mode="serial")

    with pytest.raises(ValueError, match="pinned smooth"):
        scope.__enter__()

    assert pinned_smooth.crb is replacement
    monkeypatch.setattr(pinned_smooth, "crb", original_crb)


@pytest.mark.parametrize(
    ("target", "attribute"),
    [
        ("forward-module", "forward"),
        ("public-forward", "forward"),
        ("public-crb", "crb"),
    ],
)
def test_scope_refuses_retargeted_forward_caller_aliases(
    monkeypatch, target, attribute
):
    owner = {
        "forward-module": pinned_forward,
        "public-forward": public_mjwarp,
        "public-crb": public_mjwarp,
    }[target]
    previous = getattr(owner, attribute)
    monkeypatch.setattr(owner, attribute, lambda *_args, **_kwargs: None)
    scope = control_module.StanceCrbRuntimeControl(mode="serial")

    with pytest.raises(ValueError, match="pinned forward module"):
        scope.__enter__()

    assert pinned_smooth.crb is control_module._PINNED_CRB
    monkeypatch.setattr(owner, attribute, previous)


def test_scope_restores_hooks_and_marks_constructor_exception_faulted():
    original_crb = pinned_smooth.crb
    original_launch = wp.launch
    scope = control_module.StanceCrbRuntimeControl(mode="serial")

    with pytest.raises(RuntimeError, match="synthetic constructor fault"):
        with scope:
            assert pinned_smooth.crb is not original_crb
            raise RuntimeError("synthetic constructor fault")

    assert scope.receipt["status"] == "faulted"
    assert pinned_smooth.crb is original_crb
    assert wp.launch is original_launch


def test_scope_preserves_unrelated_smooth_replacement_on_restore():
    original_crb = pinned_smooth.crb

    def replacement(*_args, **_kwargs):
        return None

    scope = control_module.StanceCrbRuntimeControl(mode="serial")

    try:
        with pytest.raises(ValueError, match="complete bound CRB control scope"):
            with scope:
                pinned_smooth.crb = replacement
        assert pinned_smooth.crb is replacement
    finally:
        pinned_smooth.crb = original_crb


def test_scope_rejects_non_main_thread_and_is_not_reusable():
    scope = control_module.StanceCrbRuntimeControl(mode="serial")
    failures = []
    with pytest.raises(ValueError, match="complete bound CRB control scope"):
        with scope:
            worker = Thread(target=lambda: _capture_failure(scope, failures))
            worker.start()
            worker.join()
            assert len(failures) == 1
            assert isinstance(failures[0], ValueError)
    assert scope.receipt["status"] == "faulted"

    with pytest.raises(ValueError, match="one-shot"):
        scope.__enter__()


def _capture_failure(scope, failures):
    try:
        scope.bind_runtime(None)
    except BaseException as error:
        failures.append(error)


def test_scope_enforces_forward_cap_and_preserves_changed_wp_launch(monkeypatch):
    original_launch = wp.launch
    scope = control_module.StanceCrbRuntimeControl(mode="serial", max_forward_calls=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with pytest.raises(ValueError, match="maximum CRB forward-call"):
            with scope:
                env = ScheduledRecoveryRuntime(
                    _declaration(), device="cpu", solved_field_check="packed"
                )
                scope.bind_runtime(env)
                env._forward()

    assert scope.receipt["status"] == "faulted"
    assert scope.receipt["forward_calls"] == 1
    assert wp.launch is original_launch


def test_scope_refuses_and_preserves_foreign_warp_launch_replacement():
    original_launch = wp.launch
    original_crb = pinned_smooth.crb

    def replacement(*_args, **_kwargs):
        return None

    scope = control_module.StanceCrbRuntimeControl(mode="serial")
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(977)
            with pytest.raises(ValueError, match="unmodified Warp launch"):
                with scope:
                    env = ScheduledRecoveryRuntime(
                        _declaration(), device="cpu", solved_field_check="packed"
                    )
                    scope.bind_runtime(env)
                    assert scope.receipt["constructor_forward_calls"] == 1
                    assert scope.receipt["forward_calls"] == 1
                    wp.launch = replacement
                    pinned_smooth.crb(env.model, env.data)
        assert wp.launch is replacement
        assert scope.receipt["status"] == "faulted"
        assert scope.receipt["constructor_forward_covered"] is True
        assert scope.receipt["constructor_forward_calls"] == 1
        assert scope.receipt["forward_calls"] == 1
        assert scope.receipt["fault_type"] == "ValueError"
        assert pinned_smooth.crb is original_crb
    finally:
        wp.launch = original_launch


def test_scope_rejects_same_shape_qm_input_storage_replacement():
    scope = control_module.StanceCrbRuntimeControl(mode="serial")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with pytest.raises(
            ValueError, match="bound model/data array identities unchanged"
        ):
            with scope:
                env = ScheduledRecoveryRuntime(
                    _declaration(), device="cpu", solved_field_check="packed"
                )
                scope.bind_runtime(env)
                original = env.model.dof_bodyid
                env.model.dof_bodyid = wp.array(
                    original.numpy().copy(), dtype=wp.int32, device="cpu"
                )
                env._forward()
    assert scope.receipt["status"] == "faulted"


@pytest.mark.parametrize("target", ["parent", "tree"])
def test_scope_revalidates_topology_ids_before_each_later_crb(target):
    scope = control_module.StanceCrbRuntimeControl(mode="serial")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with pytest.raises(ValueError, match="IDs remain exact"):
            with scope:
                env = ScheduledRecoveryRuntime(
                    _declaration(), device="cpu", solved_field_check="packed"
                )
                scope.bind_runtime(env)
                if target == "parent":
                    env.model.body_parentid.numpy()[2] = 0
                else:
                    env.model.body_tree[0].numpy()[0] = 5
                env._forward()
    receipt = scope.receipt
    assert receipt["forward_calls"] == 1
    assert receipt["topology_id_snapshots"] == 1
    assert receipt["dense_qM_launches"] == 1
    assert receipt["status"] == "faulted"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("n", 64), ("device", "cuda:0"), ("wp_device", "not-the-model-device")],
)
def test_runtime_binding_rejects_world_or_device_metadata_drift(field, replacement):
    scope = control_module.StanceCrbRuntimeControl(mode="serial")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with pytest.raises(
            ValueError, match="exact unmodified eager scheduled runtime"
        ):
            with scope:
                env = ScheduledRecoveryRuntime(
                    _declaration(), device="cpu", solved_field_check="packed"
                )
                setattr(env, field, replacement)
                scope.bind_runtime(env)
    assert scope.receipt["status"] == "faulted"


def test_successful_scope_requires_declared_forward_cap():
    scope = control_module.StanceCrbRuntimeControl(mode="serial", max_forward_calls=4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with pytest.raises(ValueError, match="complete bound CRB control scope"):
            with scope:
                env = ScheduledRecoveryRuntime(
                    _declaration(), device="cpu", solved_field_check="packed"
                )
                scope.bind_runtime(env)
    assert scope.receipt["forward_calls"] == 1
    assert scope.receipt["status"] == "faulted"


def test_one_forward_cpu_fixture_can_complete_with_cap_one():
    _env, receipt = _construct("serial", max_forward_calls=1, call_count=0)
    assert receipt["status"] == "complete"
    assert receipt["forward_calls"] == 1
    assert receipt["topology_id_snapshots"] == 1


def test_scope_freezes_mode_and_cap_after_entry():
    scope = control_module.StanceCrbRuntimeControl(mode="serial")
    with pytest.raises(ValueError, match="complete bound CRB control scope"):
        with scope:
            scope.mode = "concurrent"
    assert scope.receipt["status"] == "faulted"


def test_receipt_is_detached_and_never_admits_native_or_physical_result():
    _env, receipt = _construct("serial", call_count=3)
    receipt["flags"]["native_qualified"] = True
    receipt["forward_calls"] = -1
    assert receipt["status"] == "complete"
    assert receipt["constructor_forward_covered"] is True


def test_runtime_binding_refuses_wrong_runtime_type():
    scope = control_module.StanceCrbRuntimeControl(mode="serial")
    with pytest.raises(ValueError, match="exact unmodified"):
        with scope:
            scope.bind_runtime(object())
    assert scope.receipt["status"] == "faulted"
