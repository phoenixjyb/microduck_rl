"""Focused CPU tests for the bounded BAM load observer."""

import pytest
import torch
import warp as wp
import bam.mjlab as bam_bridge

from mjlab_microduck import stance_bam_load_observer as observer_module
from mjlab_microduck.stance_serial_step_control import SerialStepControl
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


@pytest.fixture(autouse=True)
def _cuda_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _run_observed(*, hook=None):
    control = SerialStepControl()
    observer = observer_module.BamLoadObserver()
    with wp.ScopedDevice("cpu"), control, observer:
        env = WarpStanceRuntime(nworld=2, device="cpu", solved_field_check="packed")
        control.bind_runtime(env)
        observer.bind_runtime(control, env)
        if hook is not None:
            hook(env, observer)
        result = control.step(
            env,
            torch.zeros((2, 10), dtype=torch.float32, device="cpu"),
            capture_control=True,
        )
    return env, control, observer, result


def test_actual_cpu_step_captures_all_ten_real_load_proposals_and_closes():
    env, control, observer, result = _run_observed()

    receipt = observer.receipt
    assert receipt["protocol"] == observer_module.PROTOCOL
    assert receipt["status"] == "complete"
    assert receipt["calls"] == receipt["expected_calls"] == 10
    assert [row["proposal_index"] for row in receipt["call_records"]] == list(range(10))
    assert all(
        row["steps_before"] == [row["proposal_index"]] * 2
        for row in receipt["call_records"]
    )
    assert all(
        row["compute_calls"]
        == row["friction_scan_calls"]
        == row["budget_calls"]
        == index + 1
        for index, row in enumerate(receipt["call_records"])
    )
    assert receipt["worlds"] == 2 and receipt["device"] == "cpu"
    assert receipt["m6_json_sha256"] == observer_module._M6_SHA256
    assert receipt["original_compute_called_once_per_proposal"] is True
    assert receipt["original_friction_scan_called_once_per_proposal"] is True
    assert receipt["original_budget_called_once_per_proposal"] is True
    assert receipt["runtime_cause_proven"] is False
    assert receipt["training_authorized"] is False
    assert receipt["physical_acceptance"] is False
    assert receipt["runtime_parameters_checked_each_budget"] is True
    assert len(receipt["runtime_parameters"]) == 10
    assert receipt["model_flags"] == {
        "name": "m6",
        "actuator": "xl330",
        "stribeck": True,
        "load_dependent": True,
        "directional": True,
        "quadratic": True,
    }
    assert result["optimizer_launched"] is False
    assert env.forward_graph is None and (env.steps == 10).all()
    assert len(receipt["staged_actuator_ids"]) == 10
    assert all(
        type(value) is int and value > 0 for value in receipt["staged_actuator_ids"]
    )

    packets = observer.packets
    assert set(packets) == set(observer_module._ALL_FIELDS)
    assert {name: len(raw) for name, raw in packets.items()} == {
        "qfrc_bias": 10 * 2 * 20 * 4,
        "qfrc_constraint": 10 * 2 * 20 * 4,
        "qfrc_actuator": 10 * 2 * 20 * 4,
        "efc_force": 10 * 2 * 512 * 4,
        "qfrc_friction": 10 * 2 * 20 * 4,
        "budget_motor": 10 * 2 * 14 * 4,
        "budget_external": 10 * 2 * 14 * 4,
        "budget_stribeck": 10 * 2 * 14 * 4,
        "budget_output": 10 * 2 * 14 * 4,
        "friction_scale": 10 * 2 * 1 * 4,
        "efc_type": 10 * 2 * 512 * 4,
        "efc_id": 10 * 2 * 512 * 4,
        "nefc": 10 * 2 * 4,
    }
    assert sum(map(len, packets.values())) <= observer_module.MAX_TOTAL_BYTES
    assert all(len(raw) <= observer_module.MAX_FIELD_BYTES for raw in packets.values())
    assert all(len(rows) == 10 for rows in receipt["calls_sha256"].values())
    assert bam_bridge.BamActuator._dof_friction_force is observer_module._DOF_FRICTION
    assert (
        observer_module.FrictionDRBamActuator._compute_friction_budget
        is observer_module._BUDGET
    )
    assert "compute" not in env.motor.__dict__
    assert control.receipt["status"] == "complete"


def test_packets_and_receipts_are_detached_immutable_values():
    _env, _control, observer, _result = _run_observed()
    first = observer.packets
    original = first["budget_output"]
    first["budget_output"] = b"changed"
    assert observer.packets["budget_output"] == original

    receipt = observer.receipt
    receipt["m6_parameters"]["model"] = "changed"
    receipt["calls_sha256"]["budget_output"].clear()
    assert observer.receipt["m6_parameters"]["model"] == "m6"
    assert len(observer.receipt["calls_sha256"]["budget_output"]) == 10


@pytest.mark.parametrize("field_change", ["missing", "extra"])
def test_closed_packet_refuses_missing_or_extra_fields(field_change):
    _env, _control, observer, _result = _run_observed()
    if field_change == "missing":
        del observer._entries[4]["efc_id"]
    else:
        observer._entries[4]["unexpected"] = b""
    with pytest.raises(ValueError, match="exact complete packet field set"):
        _ = observer.packets


def test_float_capture_rejects_nonfinite_or_wrong_shape_before_encoding():
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="finite complete tensor"):
        observer_module._tensor_bytes(
            torch.tensor([[float("inf")]], dtype=torch.float32),
            (1, 1),
            torch.float32,
            device,
            "synthetic nonfinite",
        )
    with pytest.raises(ValueError, match="exact shape"):
        observer_module._tensor_bytes(
            torch.zeros((1, 2), dtype=torch.float32),
            (1, 1),
            torch.float32,
            device,
            "synthetic wrong shape",
        )


def test_entry_refuses_changed_installed_compute_source(monkeypatch):
    original = bam_bridge.BamActuator.compute
    monkeypatch.setattr(
        bam_bridge.BamActuator, "compute", lambda self, cmd: original(self, cmd)
    )
    with pytest.raises(ValueError, match="pinned compute sources"):
        with observer_module.BamLoadObserver():
            pass


def test_compute_interposition_rejects_nonruntime_caller_without_capture():
    def wrong_caller(env, observer):
        with pytest.raises(ValueError, match="exact in-order live motor proposal"):
            observer._commit_compute_wrapper(
                env.motor,
                None,
                torch.ones((2,), dtype=torch.bool, device="cpu"),
            )

    _env, _control, observer, _result = _run_observed(hook=wrong_caller)
    assert observer.receipt["calls"] == 10


def test_scope_refuses_packets_before_a_closed_complete_capture():
    observer = observer_module.BamLoadObserver()
    with pytest.raises(ValueError, match="closed complete observer packets"):
        _ = observer.packets


def test_foreign_hook_replacement_is_rejected_and_not_clobbered(monkeypatch):
    original = observer_module._DOF_FRICTION

    def foreign(self, nv):
        return original(self, nv)

    def replace(_env, _observer):
        monkeypatch.setattr(bam_bridge.BamActuator, "_dof_friction_force", foreign)

    control = SerialStepControl()
    observer = observer_module.BamLoadObserver()
    with pytest.raises(ValueError, match="complete ten-call"):
        with wp.ScopedDevice("cpu"), control, observer:
            env = WarpStanceRuntime(nworld=2, device="cpu", solved_field_check="packed")
            control.bind_runtime(env)
            observer.bind_runtime(control, env)
            replace(env, observer)
            with pytest.raises(ValueError, match="pinned compute sources"):
                control.step(
                    env,
                    torch.zeros((2, 10), dtype=torch.float32, device="cpu"),
                    capture_control=True,
                )
    assert bam_bridge.BamActuator._dof_friction_force is foreign
    monkeypatch.undo()
    bam_bridge.BamActuator._dof_friction_force = original
    assert bam_bridge.BamActuator._dof_friction_force is original
    assert (
        observer_module.FrictionDRBamActuator._compute_friction_budget
        is observer_module._BUDGET
    )


def test_one_shot_scope_cannot_be_reused():
    observer = observer_module.BamLoadObserver()
    with pytest.raises(ValueError, match="complete ten-call"):
        with observer:
            pass
    with pytest.raises(ValueError, match="one-shot"):
        observer.__enter__()
