"""Actual CPU persistent-inertia observation; no solver or native qualification."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
import gc

import pytest
import torch

from mjlab_microduck import stance_recovery_early_forward_trace as forward
from mjlab_microduck import stance_recovery_early_inertia_trace as inertia
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_warp_runtime as warp_runtime
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

SOURCE = "c" * 40
CELL_IDS = ["zero-wrench", "zero-wrench"]
PERSISTENT_SHAPES = {
    "subtree_com": (2, 16, 3),
    "cinert": (2, 16, 10),
    "crb": (2, 16, 10),
    "cdof": (2, 20, 6),
    "cdof_dot": (2, 20, 6),
    "qM": (2, 20, 20),
    "qLD": (2, 20, 20),
    "cvel": (2, 16, 6),
    "qfrc_bias": (2, 20),
    "qfrc_smooth": (2, 20),
}
EXCLUDED = ["cacc", "cfrc_int", "cfrc_ext"]


def _tree_equal(left, right):
    """Compare runtime snapshots recursively without tensor truthiness."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left, right)
        )
    if is_dataclass(left) or is_dataclass(right):
        return (
            is_dataclass(left)
            and is_dataclass(right)
            and type(left) is type(right)
            and all(
                _tree_equal(getattr(left, item.name), getattr(right, item.name))
                for item in fields(left)
            )
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


@pytest.fixture(scope="module")
def actual_case():
    """Collect a real 2-world CPU prefix and an identical unobserved run."""
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    caller_rng = torch.random.get_rng_state().clone()

    def fresh_runtime():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(977)
            return ScheduledRecoveryRuntime(
                declaration, device="cpu", solved_field_check="packed"
            )

    observed = fresh_runtime()
    reference = fresh_runtime()
    observer = inertia.EarlyInertiaTrace(observed)
    with observer:
        observed_result = observed.step_with_schedule(
            torch.zeros(2, 10), capture_control=True
        )
    retained = observer.capture()
    reference_result = reference.step_with_schedule(
        torch.zeros(2, 10), capture_control=True
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return {
        "declaration": declaration,
        "observer": observer,
        "observed": observed,
        "reference": reference,
        "retained": retained,
        "observed_result": observed_result,
        "reference_result": reference_result,
        "first_record": {"runtime_result_before_reset": observed_result},
    }


@pytest.fixture(autouse=True)
def cleanup_runtime_objects():
    yield
    gc.collect()


def test_observer_keeps_original_calls_rng_and_runtime_trajectory(actual_case):
    observed = actual_case["observed"]
    reference = actual_case["reference"]
    observer = actual_case["observer"]
    assert _tree_equal(actual_case["observed_result"], actual_case["reference_result"])
    assert _tree_equal(observed._control_snapshot(), reference._control_snapshot())
    assert _tree_equal(observed.snapshot(), reference.snapshot())
    assert len(observer.events) == 2 * forward.STEPS == 6
    assert "_forward" not in observed.__dict__
    assert "_scheduled_forward" not in observed.__dict__
    assert observed._forward.__func__ is WarpStanceRuntime._forward
    assert (
        observed._scheduled_forward.__func__
        is ScheduledRecoveryRuntime._scheduled_forward
    )


def test_each_observed_and_reference_tick_calls_original_forward_twenty_times():
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)

    def fresh_runtime():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(977)
            return ScheduledRecoveryRuntime(
                declaration, device="cpu", solved_field_check="packed"
            )

    observed = fresh_runtime()
    reference = fresh_runtime()
    observer = inertia.EarlyInertiaTrace(observed)
    calls = {id(observed.data): 0, id(reference.data): 0}
    original = warp_runtime.mjwarp.forward

    def counted(model, data):
        if data is observed.data:
            calls[id(data)] += 1
        elif data is reference.data:
            calls[id(data)] += 1
        return original(model, data)

    warp_runtime.mjwarp.forward = counted
    try:
        with observer:
            observed_result = observed.step_with_schedule(
                torch.zeros(2, 10), capture_control=True
            )
        reference_result = reference.step_with_schedule(
            torch.zeros(2, 10), capture_control=True
        )
    finally:
        warp_runtime.mjwarp.forward = original
    assert calls == {id(observed.data): 20, id(reference.data): 20}
    assert _tree_equal(observed_result, reference_result)


def test_persistent_inertia_layouts_metadata_and_owned_captures(actual_case):
    trace = actual_case["retained"]
    assert trace["protocol"] == inertia.PROTOCOL
    assert trace["observation_boundary"] == (
        "after-complete-forward-including-acceleration-sensors"
    )
    assert trace["excluded_intermediates"] == EXCLUDED
    assert trace["persistent_inertia_qualified"] is False
    assert len(trace["events"]) == 6
    for event in trace["events"]:
        assert set(event) == {"phase", "step", "inputs", "solved", "qM", "persistent"}
        assert set(event["inputs"]) == set(forward.INPUTS)
        assert set(event["persistent"]) == set(PERSISTENT_SHAPES)
        for name, shape in PERSISTENT_SHAPES.items():
            value = event["persistent"][name]
            assert value.shape == shape
            assert value.device.type == "cpu"
            assert value.dtype == torch.float32
            assert torch.isfinite(value).all()
        assert not (set(EXCLUDED) & set(event["persistent"]))

    saved = {
        name: value.clone() for name, value in trace["events"][0]["persistent"].items()
    }
    live = {
        name: actual_case["observed"]._view(name).clone() for name in PERSISTENT_SHAPES
    }
    try:
        for name in PERSISTENT_SHAPES:
            actual_case["observed"]._view(name).fill_(123.0)
        assert all(
            torch.equal(trace["events"][0]["persistent"][name], value)
            for name, value in saved.items()
        )
    finally:
        for name, value in live.items():
            actual_case["observed"]._view(name).copy_(value)


def test_checker_binds_persistent_double_qm_and_bias_to_existing_trace(actual_case):
    trace = actual_case["retained"]
    result = inertia.check(
        trace, actual_case["declaration"], actual_case["first_record"]
    )
    assert result["protocol"] == inertia.PROTOCOL + ":score"
    assert result["source"] == SOURCE
    assert result["worlds"] == 2
    assert result["events"] == 6
    assert all(result[name] is False for name in inertia.FLAGS)


@pytest.mark.parametrize("replacement", ["view", "sync", "graph"])
def test_observer_constructor_refuses_foreign_view_sync_or_graph(replacement):
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        env = ScheduledRecoveryRuntime(
            declaration, device="cpu", solved_field_check="packed"
        )
    if replacement == "graph":
        env.forward_graph = object()
        with pytest.raises(ValueError):
            inertia.EarlyInertiaTrace(env)
        env.forward_graph = None
    else:
        name = "_view" if replacement == "view" else "_sync"
        setattr(env, name, lambda *args: None)
        try:
            with pytest.raises(ValueError):
                inertia.EarlyInertiaTrace(env)
        finally:
            delattr(env, name)


@pytest.mark.parametrize(
    "damage",
    [
        "shape",
        "nonfinite",
        "dtype",
        "metadata",
        "unknown-top",
        "unknown-event",
        "unknown-persistent",
        "qualified-flag",
        "double-qM",
        "bias",
        "qM-signed-zero",
        "bias-signed-zero",
        "xfrc-worlds",
    ],
)
def test_checker_rejects_bad_persistent_layout_or_binding(actual_case, damage):
    trace = deepcopy(actual_case["retained"])
    if damage == "shape":
        trace["events"][0]["persistent"]["subtree_com"] = torch.zeros(2, 16, 2)
    elif damage == "nonfinite":
        trace["events"][0]["persistent"]["cinert"][0, 0, 0] = float("nan")
    elif damage == "dtype":
        trace["events"][0]["persistent"]["qLD"] = trace["events"][0]["persistent"][
            "qLD"
        ].double()
    elif damage == "metadata":
        trace["observation_boundary"] = "before-forward"
    elif damage == "unknown-top":
        trace["unexpected"] = True
    elif damage == "unknown-event":
        trace["events"][0]["unexpected"] = True
    elif damage == "unknown-persistent":
        trace["events"][0]["persistent"]["unexpected"] = torch.zeros(1)
    elif damage == "qualified-flag":
        trace["inertia_kernel_cause_proven"] = True
    elif damage == "double-qM":
        trace["events"][0]["persistent"]["qM"][0, 0, 0] += 0.25
    elif damage in ("qM-signed-zero", "bias-signed-zero"):
        name = "qM" if damage == "qM-signed-zero" else "qfrc_bias"
        tensor = trace["events"][0]["persistent"][name]
        zero_index = (tensor == 0).nonzero()[0]
        tensor[tuple(zero_index.tolist())] = -0.0
    elif damage == "xfrc-worlds":
        trace["events"][0]["inputs"]["xfrc_applied"] = torch.zeros(2, 15, 6)
    else:
        trace["events"][0]["persistent"]["qfrc_bias"][0, 0] += 0.25
    with pytest.raises(ValueError):
        inertia.check(trace, actual_case["declaration"], actual_case["first_record"])


def test_actual_trace_comparison_is_exact_and_nonadmitting(actual_case):
    trace = actual_case["retained"]
    record = actual_case["first_record"]
    result = inertia.compare(
        trace, deepcopy(trace), actual_case["declaration"], (record, deepcopy(record))
    )
    assert result["protocol"] == inertia.PROTOCOL + ":comparison"
    assert result["earliest_differing_event"] is None
    assert len(result["events"]) == 6
    assert all(result[name] is False for name in inertia.FLAGS)


def test_finite_crb_delta_is_diagnostic_only(actual_case):
    left = deepcopy(actual_case["retained"])
    right = deepcopy(actual_case["retained"])
    right["events"][0]["persistent"]["crb"][0, 0, 0] += 0.125
    record = actual_case["first_record"]
    result = inertia.compare(left, right, actual_case["declaration"], (record, record))
    field = result["events"][0]["fields"]["crb"]
    assert result["earliest_differing_event"] == 0
    assert field["exact"] is False
    assert field["different_elements"] == 1
    assert field["max_abs_delta"] == pytest.approx(0.125)
    assert all(result[name] is False for name in inertia.FLAGS)


def test_crb_signed_zero_is_a_raw_difference_with_zero_numeric_delta(actual_case):
    left = deepcopy(actual_case["retained"])
    right = deepcopy(actual_case["retained"])
    tensor = right["events"][0]["persistent"]["crb"]
    zero_index = (tensor == 0).nonzero()[0]
    tensor[tuple(zero_index.tolist())] = -0.0
    record = actual_case["first_record"]
    result = inertia.compare(left, right, actual_case["declaration"], (record, record))
    field = result["events"][0]["fields"]["crb"]
    assert result["exact"] is False
    assert result["events"][0]["exact"] is False
    assert field["exact"] is False
    assert field["different_elements"] == 1
    assert field["max_abs_delta"] == 0.0
    assert all(result[name] is False for name in inertia.FLAGS)


def test_cuda_hidden_checker_guard_precedes_input_tree_reads(monkeypatch):
    class DoNotRead(dict):
        def __getitem__(self, key):
            raise AssertionError("trace accessed before CUDA-hidden guard")

        def items(self):
            raise AssertionError("trace accessed before CUDA-hidden guard")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA state read before guard")),
    )
    with pytest.raises(ValueError, match="CUDA-hidden"):
        inertia.check(DoNotRead(), DoNotRead(), DoNotRead())
