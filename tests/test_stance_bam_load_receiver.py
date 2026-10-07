"""Independent full-field BAM load arithmetic and fail-closed authentication."""

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_microduck import stance_bam_load_receiver as receiver


def _fields():
    return {
        name: np.zeros((10, 64, *shape), dtype=dtype)
        for name, (dtype, shape) in receiver.LOAD_FIELDS.items()
    }


def _raw(fields, case="serial0"):
    return {
        case + ".load." + name + ".bin": value.tobytes()
        for name, value in fields.items()
    }


def _valid_fields():
    result = _fields()
    result["friction_scale"][:] = 1
    result["budget_stribeck"][:] = 1
    for step in range(10):
        result["budget_output"][step] = receiver.budget_reference(
            result["budget_motor"][step],
            result["budget_external"][step],
            result["budget_stribeck"][step],
            result["friction_scale"][step],
        )
    return result


def test_literal_complete_wire_shapes_and_old_caps_unchanged():
    assert len(receiver.LOAD_FIELDS) == 13 and len(receiver.CAPS) == 48
    assert receiver.CASE_CAPS["load.efc_force.bin"] == 1310720
    assert receiver.CASE_CAPS["load.nefc.bin"] == 2560
    assert all(cap <= 2 * 1024**2 for cap in receiver.CAPS.values())
    assert receiver.old.SERVICE_CAPS["LimitFSIZE"] == "1048576"
    assert receiver.SERVICE_CAPS["LimitFSIZE"] == "2097152"
    assert receiver.FLAGS == receiver.old.FLAGS
    assert all(value is False for value in receiver.FLAGS.values())


def test_complete_packet_decode_preserves_last_world_last_substep():
    fields = _valid_fields()
    fields["qfrc_constraint"][9, 63, 19] = np.float32(0.125)
    decoded = receiver.decode_load_packets(_raw(fields), "serial0")
    assert decoded["qfrc_constraint"].shape == (10, 64, 20)
    assert decoded["qfrc_constraint"][9, 63, 19] == np.float32(0.125)
    assert decoded["nefc"].dtype == np.dtype("<i4")


@pytest.mark.parametrize(
    "mutation",
    [
        "length",
        "nan",
        "negative_count",
        "count_overflow",
        "scale",
        "stribeck",
        "budget",
    ],
)
def test_packet_decode_fails_closed_on_structural_or_nonfinite_fields(mutation):
    fields = _valid_fields()
    if mutation == "nan":
        fields["efc_force"][9, 63, 511] = np.nan
    elif mutation == "negative_count":
        fields["nefc"][9, 63] = -1
    elif mutation == "count_overflow":
        fields["nefc"][9, 63] = 513
    elif mutation == "scale":
        fields["friction_scale"][9, 63, 0] = 2
    elif mutation == "stribeck":
        fields["budget_stribeck"][9, 63, 13] = 2
    elif mutation == "budget":
        fields["budget_output"][9, 63, 13] = -1
    raw = _raw(fields)
    if mutation == "length":
        raw["serial0.load.efc_id.bin"] = raw["serial0.load.efc_id.bin"][:-4]
    with pytest.raises(ValueError):
        receiver.decode_load_packets(raw, "serial0")


def test_friction_scan_uses_only_active_friction_rows_and_complete_tail():
    types = np.zeros((64, 512), dtype="<i4")
    ids = np.full((64, 512), -123, dtype="<i4")
    forces = np.ones((64, 512), dtype="<f4")
    counts = np.zeros(64, dtype="<i4")
    counts[63] = 512
    types[63, 0] = types[63, 511] = 1
    ids[63, 0] = ids[63, 511] = 19
    forces[63, 0] = np.float32(0.5)
    forces[63, 511] = np.float32(-0.25)
    types[0, 511], ids[0, 511] = 1, 999  # inactive stale row must not participate
    value, rows = receiver.friction_scan_reference(types, ids, forces, counts)
    assert value[63, 19] == np.float32(0.25)
    assert rows[63, 19] == 2
    assert np.count_nonzero(value) == np.count_nonzero(rows) == 1


def test_active_invalid_friction_dof_is_rejected():
    types = np.ones((64, 512), dtype="<i4")
    ids = np.zeros((64, 512), dtype="<i4")
    ids[0, 0] = 20
    with pytest.raises(ValueError, match="real DOF address"):
        receiver.friction_scan_reference(
            types, ids, np.zeros((64, 512), dtype="<f4"), np.ones(64, dtype="<i4")
        )


def test_independent_m6_budget_matches_stock_cpu_pointwise_arithmetic():
    # Arithmetic fixture only: no Entity, simulation, initialized motor or CUDA.
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

    actuator = FrictionDRBamActuator.__new__(FrictionDRBamActuator)
    parameters = {
        name: SimpleNamespace(value=value)
        for name, value in receiver.M6_PARAMETERS.items()
        if type(value) is float
    }
    actuator._bam_model = SimpleNamespace(
        **parameters,
        stribeck=True,
        load_dependent=True,
        directional=True,
        quadratic=True,
    )
    actuator.friction_scale = torch.ones((64, 1), dtype=torch.float32)
    motor = np.linspace(-0.3, 0.3, 64 * 14, dtype="<f4").reshape(64, 14)
    external = motor[::-1].copy()
    stribeck = np.linspace(0, 1, 64 * 14, dtype="<f4").reshape(64, 14)
    expected = receiver.budget_reference(
        motor, external, stribeck, np.ones((64, 1), dtype="<f4")
    )
    actual = actuator._compute_friction_budget(
        torch.from_numpy(motor), torch.from_numpy(external), torch.from_numpy(stribeck)
    ).numpy()
    assert expected.tobytes() == actual.astype("<f4").tobytes()


def _receipt(raw):
    value = dict(
        protocol="microduck-bam-load-observer-oct7-v1",
        status="complete",
        fault=None,
        calls=10,
        expected_calls=10,
        worlds=64,
        device="cuda:0",
        m6_json_sha256=receiver.M6_SHA256,
        m6_parameters=deepcopy(receiver.M6_PARAMETERS),
        installed_bam_sha256="af3de252939ca868712423979c2ab52e198d382d7aca613b5b33c77d74baa440",
        adapter_sha256="e9b349a4be9910ea2c452cacfa78ccfc7fadb9f60e2f5ab8b54eed835e09b2b4",
        state_commit_sha256="7974953ded3c96e4a05ba05d15bfc82e12ea3b8255b150c8636ac07f580d26b9",
        original_compute_called_once_per_proposal=True,
        original_friction_scan_called_once_per_proposal=True,
        original_budget_called_once_per_proposal=True,
        runtime_parameters_checked_each_budget=True,
        actual_bam_inputs_share_bound_raw_storage=True,
        runtime_cause_proven=False,
        training_authorized=False,
        physical_acceptance=False,
        original_run_entry_captured=False,
        native_qualified=False,
        full_window_qualified=False,
        runtime_id=11,
        data_id=12,
        motor_id=13,
        bridge_data_id=15,
        staged_actuator_ids=[14] * 10,
        field_lengths={},
        calls_sha256={},
    )
    value.update(
        source_pins={
            "bam.mjlab": value["installed_bam_sha256"],
            "stance_control_state.py": value["state_commit_sha256"],
            "friction_dr_bam.py": value["adapter_sha256"],
        },
        model_flags=dict(
            name="m6",
            actuator="xl330",
            stribeck=True,
            load_dependent=True,
            directional=True,
            quadratic=True,
        ),
        runtime_parameters=[
            {
                name: number
                for name, number in receiver.M6_PARAMETERS.items()
                if type(number) is float
            }
        ]
        * 10,
        call_order=list(range(10)),
        raw_data_call_ids=[12] * 10,
        bridge_data_call_ids=[15] * 10,
        call_records=[
            dict(
                proposal_index=i,
                steps_before=[i] * 64,
                runtime_data_id=12,
                bridge_data_id=15,
                staged_actuator_id=14,
                compute_calls=i + 1,
                friction_scan_calls=i + 1,
                budget_calls=i + 1,
            )
            for i in range(10)
        ],
        method_pins={
            name: dict(object_id=i + 20, code_name=code)
            for i, (name, code) in enumerate(
                {
                    "BamStateCommit.compute": "context_decorator.<locals>.decorate_context",
                    "BamActuator.compute": "BamActuator.compute",
                    "BamActuator._dof_friction_force": "BamActuator._dof_friction_force",
                    "FrictionDRBamActuator._compute_friction_budget": "FrictionDRBamActuator._compute_friction_budget",
                    "BamActuator._compute_friction_budget": "BamActuator._compute_friction_budget",
                }.items()
            )
        },
        raw_array_layouts=[],
    )
    for i, name in enumerate(
        (
            "qfrc_bias",
            "qfrc_constraint",
            "qfrc_actuator",
            "nefc",
            "efc.type",
            "efc.id",
            "efc.force",
        )
    ):
        shape = (
            [64]
            if name == "nefc"
            else [64, 512]
            if name.startswith("efc.")
            else [64, 20]
        )
        dtype = "int32" if name in ("nefc", "efc.type", "efc.id") else "float32"
        value["raw_array_layouts"].append(
            dict(
                name=name,
                object_id=i + 100,
                ptr=4096 + i * 4096,
                shape=shape,
                strides=[4] if name == "nefc" else [shape[-1] * 4, 4],
                dtype=f"<class 'warp._src.types.{dtype}'>",
                device="cuda:0",
                contiguous=True,
            )
        )
    for name in receiver.LOAD_FIELDS:
        size = receiver.CASE_CAPS["load." + name + ".bin"] // 10
        packet = raw["serial0.load." + name + ".bin"]
        value["field_lengths"][name] = size
        value["calls_sha256"][name] = [
            sha256(packet[i * size : (i + 1) * size]).hexdigest() for i in range(10)
        ]
    return value


def test_receipt_requires_actual_source_data_and_all_ten_full_hashes():
    raw = _raw(_valid_fields())
    record = dict(control_scope=dict(calls=[dict(data_id=12)]))
    receipt = _receipt(raw)
    receiver.checked_load_receipt(receipt, record, raw, "serial0")
    for key, value in (
        ("data_id", 99),
        ("m6_json_sha256", "0" * 64),
        ("runtime_parameters_checked_each_budget", False),
        ("worlds", True),
        ("runtime_cause_proven", True),
    ):
        changed = deepcopy(receipt)
        changed[key] = value
        with pytest.raises(ValueError):
            receiver.checked_load_receipt(changed, record, raw, "serial0")
    changed = deepcopy(receipt)
    changed["calls_sha256"]["efc_force"][9] = "0" * 64
    with pytest.raises(ValueError, match="complete per-call"):
        receiver.checked_load_receipt(changed, record, raw, "serial0")


def test_all_raw_hashes_pass_before_any_json_or_array_decode(tmp_path, monkeypatch):
    raw = {name: b"x" for name in receiver.CAPS}
    inventory = {}
    for name, value in raw.items():
        (tmp_path / name).write_bytes(value)
        inventory[name] = dict(bytes=1, sha256=sha256(value).hexdigest())
    inventory["serial1.load.efc_force.bin"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        receiver.old,
        "_json",
        lambda *_: pytest.fail("decoded JSON before whole inventory"),
    )
    monkeypatch.setattr(
        receiver.np,
        "frombuffer",
        lambda *_args, **_kwargs: pytest.fail("decoded array before whole inventory"),
    )
    with pytest.raises(ValueError, match="SHA256|SHA-256"):
        receiver.verify_directory(
            tmp_path,
            inventory,
            b"{}",
            "0" * 64,
            source="1" * 40,
            expected_tree="2" * 40,
            expected_leaves_sha256="3" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "method",
        "parameters",
        "flag",
        "alias",
        "pointer",
        "stride",
        "bridge",
        "order",
        "step_bool",
        "counter",
    ],
)
def test_receipt_binds_real_source_parameters_storage_and_call_sequence(mutation):
    raw = _raw(_valid_fields())
    receipt = _receipt(raw)
    record = dict(control_scope=dict(calls=[dict(data_id=12)]))
    if mutation == "source":
        receipt["source_pins"]["bam.mjlab"] = "0" * 64
    elif mutation == "method":
        receipt["method_pins"]["BamActuator.compute"]["code_name"] = "replacement"
    elif mutation == "parameters":
        receipt["runtime_parameters"][9]["friction_base"] += 0.001
    elif mutation == "flag":
        receipt["model_flags"]["quadratic"] = 1
    elif mutation == "alias":
        receipt["actual_bam_inputs_share_bound_raw_storage"] = False
    elif mutation == "pointer":
        receipt["raw_array_layouts"][6]["ptr"] = 1
    elif mutation == "stride":
        receipt["raw_array_layouts"][0]["strides"] = [0, 4]
    elif mutation == "bridge":
        receipt["call_records"][9]["bridge_data_id"] = 999
    elif mutation == "order":
        receipt["call_order"][9] = 8
    elif mutation == "step_bool":
        receipt["call_records"][1]["steps_before"][63] = True
    elif mutation == "counter":
        receipt["call_records"][9]["budget_calls"] = 11
    with pytest.raises(ValueError):
        receiver.checked_load_receipt(receipt, record, raw, "serial0")


@pytest.mark.parametrize(
    "negative", ["arithmetic", "paired", "load_pair", "load_commit", "load_reference"]
)
def test_repeat_gate_preserves_each_separate_negative(negative):
    analysis = dict(
        paired_observed_loads_exact=True,
        actual_input_commit_consistency_exact=True,
        reference_decision="bam-load-reference-exact",
    )
    assert (
        receiver.repeat_decision(True, True, analysis) == "fresh-bam-load-repeat-exact"
    )
    arithmetic, paired = True, True
    if negative == "arithmetic":
        arithmetic = False
    elif negative == "paired":
        paired = False
    elif negative == "load_pair":
        analysis["paired_observed_loads_exact"] = False
    elif negative == "load_commit":
        analysis["actual_input_commit_consistency_exact"] = False
    else:
        analysis["reference_decision"] = "bam-load-reference-negative"
    assert (
        receiver.repeat_decision(arithmetic, paired, analysis)
        == "fresh-bam-load-repeat-negative"
    )
