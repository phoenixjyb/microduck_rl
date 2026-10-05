"""CPU-only compiled-coordinate mapping for persistent inertia diagnostics."""

from copy import deepcopy
import gc
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import stance_recovery_early_inertia_trace as inertia
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_inertia_component_map as component
from mjlab_microduck import stance_warp_runtime
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime

SOURCE = "d" * 40
CELL_IDS = ["zero-wrench", "zero-wrench"]


@pytest.fixture(scope="module")
def actual_case():
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    caller_rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        runtime = ScheduledRecoveryRuntime(
            declaration, device="cpu", solved_field_check="packed"
        )
    observer = inertia.EarlyInertiaTrace(runtime)
    with observer:
        result = runtime.step_with_schedule(torch.zeros(2, 10), capture_control=True)
    trace = observer.capture()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return {
        "declaration": declaration,
        "runtime": runtime,
        "trace": trace,
        "record": {"runtime_result_before_reset": result},
    }


@pytest.fixture(autouse=True)
def cleanup_runtime_objects():
    yield
    gc.collect()


@pytest.fixture(scope="module")
def mapping():
    return component.compiled_map()


@pytest.fixture(scope="module")
def topology_model():
    native = stance_warp_runtime.build_entity().compile()
    return SimpleNamespace(
        nbody=native.nbody,
        nq=native.nq,
        nv=native.nv,
        nu=native.nu,
        njnt=native.njnt,
        body_parentid=native.body_parentid.tolist(),
        jnt_dofadr=native.jnt_dofadr.tolist(),
        jnt_type=native.jnt_type.tolist(),
        jnt_bodyid=native.jnt_bodyid.tolist(),
        dof_jntid=native.dof_jntid.tolist(),
        dof_bodyid=native.dof_bodyid.tolist(),
        body_names=[
            component.mujoco.mj_id2name(native, component.mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(native.nbody)
        ],
        joint_names=[
            component.mujoco.mj_id2name(native, component.mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(native.njnt)
        ],
    )


def test_compiled_map_is_actual_valid_topology_and_preserves_torch_rng(mapping):
    assert set(mapping) == {"compiled_plant", "bodies", "dofs", "topology_sha256"}
    assert len(mapping["topology_sha256"]) == 64
    binding = mapping["compiled_plant"]
    assert binding["nbody"] == 16
    assert len(binding["body_names"]) == 16
    assert len(mapping["bodies"]) == 16
    assert len(mapping["dofs"]) == 20
    assert binding["selected_plant"]["topology"][:3] == [21, 20, 14]
    for i, body in enumerate(mapping["bodies"]):
        assert body["index"] == i
        assert body["name"] == binding["body_names"][i]
        assert body["ancestry"][-1] == {"index": i, "name": body["name"]}
        assert body["ancestry"][0] == {"index": 0, "name": "world"}
    for i, dof in enumerate(mapping["dofs"]):
        assert dof["index"] == i
        assert mapping["bodies"][dof["body_index"]]["index"] == dof["body_index"]
        assert dof["joint_name"]
        assert dof["joint_index"] >= 0
        assert dof["local_offset"] >= 0


def test_compiled_map_call_preserves_caller_rng():
    before = torch.random.get_rng_state().clone()
    component.compiled_map()
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize(
    "damage",
    [
        "world-parent",
        "parent-cycle",
        "parent-range",
        "nbody",
        "nv",
        "dof-body-range",
        "dof-body-joint-mismatch",
        "dof-joint-range",
        "unknown-joint-type",
        "local-offset",
    ],
)
def test_topology_validation_fails_closed_on_bad_model_tables(
    topology_model, monkeypatch, damage
):
    model = deepcopy(topology_model)
    monkeypatch.setattr(
        component.mujoco,
        "mj_id2name",
        lambda candidate, obj, index: (
            candidate.body_names
            if obj == component.mujoco.mjtObj.mjOBJ_BODY
            else candidate.joint_names
        )[index],
    )
    if damage == "world-parent":
        model.body_parentid[0] = 1
    elif damage == "parent-cycle":
        model.body_parentid[1] = 2
        model.body_parentid[2] = 1
    elif damage == "parent-range":
        model.body_parentid[1] = model.nbody
    elif damage == "nbody":
        model.nbody -= 1
    elif damage == "nv":
        model.nv -= 1
    elif damage == "dof-body-range":
        model.dof_bodyid[0] = model.nbody
    elif damage == "dof-body-joint-mismatch":
        model.dof_bodyid[0] = 2
    elif damage == "dof-joint-range":
        model.dof_jntid[0] = model.njnt
    elif damage == "unknown-joint-type":
        model.jnt_type[0] = 4
    else:
        model.jnt_dofadr[0] = 1
    with pytest.raises(ValueError):
        component._topology(model)


def test_identical_actual_trace_maps_all_events_without_admission(actual_case, mapping):
    trace = actual_case["trace"]
    right = deepcopy(trace)
    record = actual_case["record"]
    right_record = deepcopy(record)
    inputs = (trace, right, actual_case["declaration"], record, right_record)
    hashes_before = [
        inertia.early.forward.throughput.tree_hash(value) for value in inputs
    ]
    rng = torch.random.get_rng_state().clone()
    result = component.compare(
        trace,
        right,
        actual_case["declaration"],
        (record, right_record),
        mapping["compiled_plant"],
    )
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert [
        inertia.early.forward.throughput.tree_hash(value) for value in inputs
    ] == hashes_before
    assert result["protocol"] == component.PROTOCOL + ":comparison"
    assert result["earliest_persistent_differing_event"] is None
    assert result["events"] == []
    assert result["original_trace_comparison"]["earliest_differing_event"] is None
    assert result["reporting_order_is_not_kernel_temporal_order"] is True
    assert result["persistent_exactness_includes_signed_zero"] is True
    assert all(result[name] is False for name in component.FLAGS)
    assert all(
        field["exact"]
        for event in result["events"]
        for field in event["fields"].values()
    )


def test_signed_zero_is_raw_difference_and_maps_body_component(actual_case, mapping):
    left = deepcopy(actual_case["trace"])
    right = deepcopy(actual_case["trace"])
    tensor = right["events"][0]["persistent"]["crb"]
    zero_coordinate = tuple((tensor == 0).nonzero(as_tuple=False)[0].tolist())
    tensor[zero_coordinate] = -0.0
    result = component.compare(
        left,
        right,
        actual_case["declaration"],
        (actual_case["record"], actual_case["record"]),
        mapping["compiled_plant"],
    )
    field = result["events"][0]["fields"]["crb"]
    assert result["earliest_persistent_differing_event"] == 0
    assert field["exact"] is False
    assert field["different_elements"] == 1
    assert field["differing_worlds"] == 1
    assert field["max_abs_delta"] == 0.0
    sample = field["samples"][0]
    assert sample["coordinate"] == list(zero_coordinate)
    assert sample["body"]["index"] == zero_coordinate[1]
    assert sample["component_index"] == zero_coordinate[2]
    assert sample["left_bits"] != sample["right_bits"]
    assert sample["left"] == sample["right"] == 0.0


def test_sample_cap_retains_exact_count_and_dof_matrix_coordinates(
    actual_case, mapping
):
    left = deepcopy(actual_case["trace"])
    right = deepcopy(actual_case["trace"])
    modified_qm = right["events"][0]["persistent"]["qM"]
    modified_qm[0, :4, :10] += 0.125
    right["events"][0]["qM"] = modified_qm.clone()
    result = component.compare(
        left,
        right,
        actual_case["declaration"],
        (actual_case["record"], actual_case["record"]),
        mapping["compiled_plant"],
    )
    field = result["events"][0]["fields"]["qM"]
    assert field["different_elements"] == 40
    assert field["differing_worlds"] == 1
    assert len(field["samples"]) == component.SAMPLE_LIMIT == 32
    assert field["samples_truncated"] is True
    assert field["samples"][0]["row_dof"]["index"] == 0
    assert field["samples"][0]["column_dof"]["index"] == 0
    assert field["samples"][0]["left_bits"].startswith("0x")


def test_only_earliest_event_gets_component_coordinates(actual_case, mapping):
    left = deepcopy(actual_case["trace"])
    right = deepcopy(actual_case["trace"])
    right["events"][0]["persistent"]["crb"][0, 0, 0] += 0.125
    right["events"][2]["persistent"]["crb"][0, 0, 0] += 0.25
    record = actual_case["record"]
    result = component.compare(
        left,
        right,
        actual_case["declaration"],
        (record, record),
        mapping["compiled_plant"],
    )
    assert result["earliest_persistent_differing_event"] == 0
    assert [event["index"] for event in result["events"]] == [0]
    assert len(result["original_trace_comparison"]["events"]) == 6
    assert (
        result["original_trace_comparison"]["events"][2]["fields"]["crb"]["exact"]
        is False
    )


@pytest.mark.parametrize(
    ("field", "coordinate", "expected_mapping"),
    [
        ("subtree_com", (0, 1, 2), "body"),
        ("cinert", (0, 1, 2), "body"),
        ("cdof", (0, 1, 2), "dof"),
        ("crb", (0, 1, 2), "body"),
        ("qM", (0, 1, 2), "matrix"),
        ("qLD", (0, 1, 2), "matrix"),
        ("cvel", (0, 1, 2), "body"),
        ("cdof_dot", (0, 1, 2), "dof"),
        ("qfrc_bias", (0, 1), "dof"),
        ("qfrc_smooth", (0, 1), "dof"),
    ],
)
def test_each_persistent_field_maps_changed_coordinate(
    actual_case, mapping, field, coordinate, expected_mapping
):
    left = deepcopy(actual_case["trace"])
    right = deepcopy(actual_case["trace"])
    target = right["events"][0]["persistent"][field]
    target[coordinate] += 0.125
    if field == "qM":
        right["events"][0]["qM"] = target.clone()
    elif field == "qfrc_bias":
        right["events"][0]["solved"]["dynamics"]["qfrc_bias"] = target.clone()
    record = actual_case["record"]
    hash_inputs = (
        left,
        right,
        actual_case["declaration"],
        record,
        record,
    )
    before = [
        inertia.early.forward.throughput.tree_hash(value) for value in hash_inputs
    ]
    result = component.compare(
        left,
        right,
        actual_case["declaration"],
        (record, record),
        mapping["compiled_plant"],
    )
    after = [inertia.early.forward.throughput.tree_hash(value) for value in hash_inputs]
    assert before == after
    field_result = result["events"][0]["fields"][field]
    assert field_result["different_elements"] == 1
    assert field_result["differing_worlds"] == 1
    sample = field_result["samples"][0]
    assert sample["coordinate"] == list(coordinate)
    assert sample["world_index"] == coordinate[0]
    assert sample["left_bits"] != sample["right_bits"]
    if expected_mapping == "body":
        assert sample["body"]["index"] == coordinate[1]
        assert sample["component_index"] == coordinate[2]
    elif expected_mapping == "dof":
        assert sample["dof"]["index"] == coordinate[1]
        assert sample["dof"]["body_name"]
        assert sample["dof"]["body_ancestry"]
        if len(coordinate) > 2:
            assert sample["component_index"] == coordinate[2]
    else:
        assert sample["row_dof"]["index"] == coordinate[1]
        assert sample["column_dof"]["index"] == coordinate[2]
        assert sample["row_dof"]["body_ancestry"]
        assert sample["column_dof"]["body_ancestry"]


@pytest.mark.parametrize("damage", ["shape", "nonfinite", "dtype", "flag", "plant"])
def test_comparison_rejects_bad_trace_or_unbound_compiled_plant(
    actual_case, mapping, damage
):
    left = deepcopy(actual_case["trace"])
    right = deepcopy(actual_case["trace"])
    supplied_plant = deepcopy(mapping["compiled_plant"])
    if damage == "shape":
        right["events"][0]["persistent"]["cdof"] = torch.zeros(2, 20, 5)
    elif damage == "nonfinite":
        right["events"][0]["persistent"]["crb"][0, 0, 0] = float("nan")
    elif damage == "dtype":
        right["events"][0]["persistent"]["qLD"] = right["events"][0]["persistent"][
            "qLD"
        ].double()
    elif damage == "flag":
        right["component_map_qualified"] = True
    else:
        supplied_plant["nbody"] = 15
    record = actual_case["record"]
    with pytest.raises(ValueError):
        component.compare(
            left,
            right,
            actual_case["declaration"],
            (record, record),
            supplied_plant,
        )


def test_cuda_hidden_guard_precedes_trace_or_compiled_plant_reads(monkeypatch):
    class DoNotRead(dict):
        def __getitem__(self, key):
            raise AssertionError("trace accessed before hidden guard")

        def items(self):
            raise AssertionError("trace traversed before hidden guard")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA state read before guard")),
    )
    with pytest.raises(ValueError, match="CUDA-hidden"):
        component.compare(
            DoNotRead(), DoNotRead(), DoNotRead(), DoNotRead(), DoNotRead()
        )


def test_initialized_cuda_is_rejected_even_when_device_mask_is_empty(monkeypatch):
    class DoNotRead(dict):
        def __getitem__(self, key):
            raise AssertionError("trace accessed before initialized-CUDA guard")

        def items(self):
            raise AssertionError("trace traversed before initialized-CUDA guard")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="CUDA-hidden"):
        component.compare(
            DoNotRead(), DoNotRead(), DoNotRead(), DoNotRead(), DoNotRead()
        )
