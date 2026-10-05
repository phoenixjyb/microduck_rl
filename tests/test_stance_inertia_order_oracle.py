"""CPU-only arithmetic-order plausibility checks; no native cause qualification."""

from copy import deepcopy
import gc

import pytest
import torch

from mjlab_microduck import stance_inertia_component_map as component
from mjlab_microduck import stance_inertia_order_oracle as oracle
from mjlab_microduck import stance_recovery_early_inertia_trace as inertia
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime

SOURCE = "e" * 40
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
    replay = deepcopy(trace)
    # Change only the retained trunk-base CRB output. Inputs to the arithmetic
    # hypothesis (parent cinert and direct-child CRBs) remain bit-identical.
    replay["events"][0]["persistent"]["crb"][0, 1, 3] += 0.125
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    record = {"runtime_result_before_reset": result}
    mapping = component.compiled_map()
    return {
        "declaration": declaration,
        "trace": trace,
        "replay": replay,
        "record": record,
        "compiled_map": mapping,
        "compiled_plant": mapping["compiled_plant"],
    }


@pytest.fixture(autouse=True)
def cleanup_runtime_objects():
    yield
    gc.collect()


def _tree_hash(value):
    return inertia.early.forward.throughput.tree_hash(value)


def test_actual_pair_retains_all_six_float32_candidates_and_memberships(actual_case):
    left = actual_case["trace"]
    right = actual_case["replay"]
    record = actual_case["record"]
    declaration = actual_case["declaration"]
    plant = actual_case["compiled_plant"]
    before = [_tree_hash(value) for value in (left, right, record, declaration, plant)]
    rng = torch.random.get_rng_state().clone()
    result = oracle.compare(left, right, declaration, (record, deepcopy(record)), plant)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert before == [
        _tree_hash(value) for value in (left, right, record, declaration, plant)
    ]
    assert result["protocol"] == oracle.PROTOCOL + ":comparison"
    assert result["event"] == {"index": 0, "phase": "scheduled-pre", "step": 0}
    assert result["worlds"] == 2
    assert result["compiled_topology_sha256"]
    assert [row["order"] for row in result["orders"]] == [
        list(x) for x in oracle.ORDERS
    ]
    assert len(result["orders"]) == 6
    assert all(len(row["candidate_values"]) == 2 for row in result["orders"])
    assert all(len(row["candidate_uint32_bits"]) == 2 for row in result["orders"])
    assert len(result["capture_observed_root_crb"]["values"]) == 2
    assert len(result["replay_observed_root_crb"]["values"]) == 2
    assert len(result["cell_membership"]) == 20
    for cell in result["cell_membership"]:
        for side in ("capture", "replay"):
            mask = cell[f"{side}_membership_mask"]
            matching = cell[f"{side}_matching_orders"]
            assert 0 <= mask < (1 << 6)
            assert matching == [i for i in range(6) if mask & (1 << i)]
            observed = result[f"{side}_observed_root_crb"]["uint32_bits"][
                cell["world"]
            ][cell["component_index"]]
            expected = sum(
                1 << i
                for i, order in enumerate(result["orders"])
                if order["candidate_uint32_bits"][cell["world"]][
                    cell["component_index"]
                ]
                == observed
            )
            assert mask == expected
    assert all(result[name] is False for name in oracle.FLAGS)
    assert result["actual_kernel_order_observed"] is False
    assert "not actual kernel order" in result["interpretation"]
    assert len(result["full_comparison"]["original_trace_comparison"]["events"]) == 6
    summary = result["membership_summary"]
    assert summary["different_observed_elements"] == 1
    assert summary["differing_observed_worlds"] == 1
    assert summary["capture_unmatched_elements"] <= 20
    assert summary["replay_unmatched_elements"] <= 20
    assert summary["all_observed_cells_have_a_candidate"] is (
        summary["capture_unmatched_elements"] == 0
        and summary["replay_unmatched_elements"] == 0
    )


def test_reversing_pair_swaps_observed_labels_not_candidate_order(actual_case):
    result = oracle.compare(
        actual_case["replay"],
        actual_case["trace"],
        actual_case["declaration"],
        (actual_case["record"], actual_case["record"]),
        actual_case["compiled_plant"],
    )
    assert [row["order"] for row in result["orders"]] == [
        list(x) for x in oracle.ORDERS
    ]
    assert result["capture_observed_root_crb"]["uint32_bits"] == [
        [f"0x{v:08x}" for v in row]
        for row in oracle._bits(
            actual_case["replay"]["events"][0]["persistent"]["crb"][:, 1, :]
        ).tolist()
    ]
    assert result["replay_observed_root_crb"]["uint32_bits"] == [
        [f"0x{v:08x}" for v in row]
        for row in oracle._bits(
            actual_case["trace"]["events"][0]["persistent"]["crb"][:, 1, :]
        ).tolist()
    ]


def test_candidate_enumeration_exposes_float32_nonassociativity_and_signed_zero():
    initial = torch.zeros(1, 10, dtype=torch.float32)
    initial[0, 0] = 1.0e20
    initial[0, 1] = -0.0
    children = {index: torch.zeros_like(initial) for index, _ in oracle.CHILDREN}
    children[2][0, 0] = -1.0e20
    children[7][0, 0] = 3.14
    children[2][0, 1] = -0.0
    children[7][0, 1] = -0.0
    children[11][0, 1] = -0.0
    initial_before = initial.clone()
    results = {
        order: oracle._candidate(initial, children, order) for order in oracle.ORDERS
    }
    assert results[(2, 7, 11)][0, 0].item() == pytest.approx(3.14)
    assert results[(7, 2, 11)][0, 0].item() == 0.0
    assert (
        oracle._bits(results[(2, 7, 11)])[0, 0]
        != oracle._bits(results[(7, 2, 11)])[0, 0]
    )
    assert all(
        oracle._bits(value)[0, 1].item() == 0x80000000 for value in results.values()
    )
    assert torch.equal(initial, initial_before)


@pytest.mark.parametrize(
    "damage",
    [
        "root-dtype",
        "root-layout",
        "root-nonfinite",
        "child-dtype",
        "child-layout",
        "child-nonfinite",
        "order-list",
        "order-incomplete",
        "order-duplicate",
        "order-unknown",
        "child-missing",
        "child-extra",
        "child-not-dict",
    ],
)
def test_candidate_rejects_malformed_order_inputs(damage):
    initial = torch.zeros(2, 10, dtype=torch.float32)
    children = {index: torch.zeros_like(initial) for index, _ in oracle.CHILDREN}
    order = (2, 7, 11)
    if damage == "root-dtype":
        initial = initial.double()
    elif damage == "root-layout":
        initial = initial[:, :9]
    elif damage == "root-nonfinite":
        initial[0, 0] = float("inf")
    elif damage == "child-dtype":
        children[2] = children[2].double()
    elif damage == "child-layout":
        children[2] = children[2][:, :9]
    elif damage == "child-nonfinite":
        children[2][0, 0] = float("nan")
    elif damage == "order-list":
        order = [2, 7, 11]
    elif damage == "order-incomplete":
        order = (2, 7)
    elif damage == "order-duplicate":
        order = (2, 2, 11)
    elif damage == "order-unknown":
        order = (2, 7, 12)
    elif damage == "child-missing":
        del children[11]
    elif damage == "child-extra":
        children[12] = torch.zeros_like(initial)
    elif damage == "child-not-dict":
        children = list(children.values())
    with pytest.raises(ValueError):
        oracle._candidate(initial, children, order)


def test_candidate_rejects_float32_overflow_from_finite_inputs():
    maximum = torch.finfo(torch.float32).max
    initial = torch.full((1, 10), maximum, dtype=torch.float32)
    children = {
        index: torch.full_like(initial, maximum) for index, _ in oracle.CHILDREN
    }
    with pytest.raises(ValueError, match="remains finite"):
        oracle._candidate(initial, children, oracle.ORDERS[0])


def test_raw_membership_rejects_signed_zero_value_equivalence():
    observed = torch.zeros(1, 1, dtype=torch.float32)
    negative_zero = torch.full((1, 1), -0.0, dtype=torch.float32)
    rows = oracle._membership(observed, [negative_zero] * 6)
    assert rows == [
        {
            "world": 0,
            "component_index": 0,
            "capture_membership_mask": 0,
            "capture_matching_orders": [],
        }
    ]


@pytest.mark.parametrize(
    "damage", ["parent", "child-parent", "root-name", "body-table", "plant-topology"]
)
def test_node_binding_rejects_wrong_compiled_parent_or_topology(actual_case, damage):
    mapping = deepcopy(actual_case["compiled_map"])
    if damage == "parent":
        mapping["bodies"][1]["parent_index"] = 1
        mapping["bodies"][1]["parent_name"] = "trunk_base"
    elif damage == "child-parent":
        mapping["bodies"][2]["parent_index"] = 0
        mapping["bodies"][2]["parent_name"] = "world"
    elif damage == "root-name":
        mapping["bodies"][1]["name"] = "different_root"
    elif damage == "body-table":
        mapping["compiled_plant"]["body_names"][1] = "different_root"
    else:
        mapping["compiled_plant"]["selected_plant"]["topology"][1] = 19
    with pytest.raises(ValueError):
        oracle._node_binding(mapping)


@pytest.mark.parametrize(
    "damage",
    [
        "root-input",
        "child-input",
        "nonfinite",
        "phase",
        "plant",
        "qualification",
        "all-qualification",
        "new-qualification",
        "unknown-schema",
    ],
)
def test_oracle_rejects_invalid_premise_trace_or_plant(
    actual_case, monkeypatch, damage
):
    left = deepcopy(actual_case["trace"])
    right = deepcopy(actual_case["replay"])
    plant = deepcopy(actual_case["compiled_plant"])
    if damage == "root-input":
        right["events"][0]["persistent"]["cinert"][0, 1, 0] += 0.125
    elif damage == "child-input":
        right["events"][0]["persistent"]["crb"][0, 2, 0] += 0.125
    elif damage == "nonfinite":
        right["events"][0]["persistent"]["cinert"][0, 1, 0] = float("nan")
    elif damage == "phase":
        right["events"][0]["phase"] = "unforced-post"
    elif damage == "plant":
        plant["nbody"] = 15
    elif damage == "qualification":
        right["native_trace_qualified"] = True
    elif damage == "all-qualification":
        right.update({name: True for name in oracle.FLAGS})
    elif damage == "new-qualification":
        right["addition_order_qualified"] = True
    else:
        right["unknown_oracle_status"] = True
    if damage in (
        "qualification",
        "all-qualification",
        "new-qualification",
        "unknown-schema",
    ):
        monkeypatch.setattr(
            oracle,
            "_candidate",
            lambda *args: (_ for _ in ()).throw(
                AssertionError("schema and qualification checks must precede math")
            ),
        )
    record = actual_case["record"]
    with pytest.raises(ValueError):
        oracle.compare(
            left,
            right,
            actual_case["declaration"],
            (record, record),
            plant,
        )


def test_incomplete_or_changed_pair_premises_are_refused(actual_case):
    left = deepcopy(actual_case["trace"])
    right = deepcopy(actual_case["replay"])
    right["events"][0]["persistent"]["cinert"][0, 1, 0] = -0.0
    record = actual_case["record"]
    with pytest.raises(ValueError, match="root cinert premise"):
        oracle.compare(
            left,
            right,
            actual_case["declaration"],
            (record, record),
            actual_case["compiled_plant"],
        )


def test_cuda_hidden_guard_rejects_initialized_cuda_before_trace_access(monkeypatch):
    class DoNotRead(dict):
        def __getitem__(self, key):
            raise AssertionError("caller data read before CUDA-hidden guard")

        def items(self):
            raise AssertionError("caller tree traversed before CUDA-hidden guard")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="CUDA-hidden"):
        oracle.compare(DoNotRead(), DoNotRead(), DoNotRead(), DoNotRead(), DoNotRead())


def test_nonempty_device_mask_rejects_before_cuda_state_or_trace_access(monkeypatch):
    class DoNotRead(dict):
        def __getitem__(self, key):
            raise AssertionError("caller data read before device-mask guard")

        def items(self):
            raise AssertionError("caller tree traversed before device-mask guard")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA state read after bad mask")),
    )
    with pytest.raises(ValueError, match="CUDA-hidden"):
        oracle.compare(DoNotRead(), DoNotRead(), DoNotRead(), DoNotRead(), DoNotRead())
