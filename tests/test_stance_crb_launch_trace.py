"""Source-only tests for scoped observation of real CPU CRB launches."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
import gc
from hashlib import sha256
import os
import threading

import pytest
import torch
import warp as wp

from mujoco_warp._src import smooth
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_crb_launch_trace as crb_trace
from mjlab_microduck import stance_warp_runtime as warp_runtime
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.first_attempt_smoke import canonical

SOURCE = "e" * 40
CELL_IDS = ["zero-wrench", "zero-wrench"]


def _tree_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.device == right.device
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.contiguous().view(torch.uint8).numpy().tobytes()
            == right.contiguous().view(torch.uint8).numpy().tobytes()
        )
    if is_dataclass(left) or is_dataclass(right):
        return (
            is_dataclass(left)
            and is_dataclass(right)
            and type(left) is type(right)
            and all(
                _tree_equal(getattr(left, field.name), getattr(right, field.name))
                for field in fields(left)
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


def _cpu_runtime(declaration):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        return ScheduledRecoveryRuntime(
            declaration, device="cpu", solved_field_check="packed"
        )


@pytest.fixture(scope="module")
def _module_cuda_hidden():
    prior = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior


@pytest.fixture(scope="module")
def actual_case(_module_cuda_hidden):
    """Observe six real early phases beside the identical unobserved CPU tick."""
    assert not torch.cuda.is_initialized()
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    caller_rng = torch.random.get_rng_state().clone()
    observed = _cpu_runtime(declaration)
    reference = _cpu_runtime(declaration)
    forward_calls = {id(observed.data): 0, id(reference.data): 0}
    launch_calls = []
    other_launches = []
    original_forward = warp_runtime.mjwarp.forward
    original_launch = wp.launch

    def counted_forward(model, data):
        if id(data) in forward_calls:
            forward_calls[id(data)] += 1
        return original_forward(model, data)

    def counted_launch(kernel, *args, **kwargs):
        result = original_launch(kernel, *args, **kwargs)
        if kernel is smooth._crb_accumulate:
            launch_calls.append((kernel, args, kwargs, result))
        else:
            other_launches.append(kernel)
        return result

    observer = None
    try:
        warp_runtime.mjwarp.forward = counted_forward
        wp.launch = counted_launch
        observer = crb_trace.CrbLaunchTrace(
            observed, smooth_module=smooth, launch_module=wp
        )
        with observer:
            observed_result = observed.step_with_schedule(
                torch.zeros(2, 10), capture_control=True
            )
        retained = observer.capture()
        # The reference physics run is genuinely uninstrumented: no launch
        # wrapper, synchronization, snapshots, or copied CRB/Cinert arrays.
        wp.launch = original_launch
        reference_result = reference.step_with_schedule(
            torch.zeros(2, 10), capture_control=True
        )
    finally:
        warp_runtime.mjwarp.forward = original_forward
        wp.launch = original_launch
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return {
        "declaration": declaration,
        "observed": observed,
        "reference": reference,
        "observer": observer,
        "trace": retained,
        "observed_result": observed_result,
        "reference_result": reference_result,
        "record": {"runtime_result_before_reset": observed_result},
        "forward_calls": forward_calls,
        "launch_calls": launch_calls,
        "other_launches": other_launches,
        "caller_rng": caller_rng,
    }


@pytest.fixture(scope="module")
def pulse_case(_module_cuda_hidden):
    """Same-seed, real 280-step CPU pair with one declared pulse row."""
    assert not torch.cuda.is_initialized()
    pulse_cell = "+x-2n-20steps-t250"
    declaration = schedule.declaration(
        SOURCE, "dose", "training", ["zero-wrench", pulse_cell]
    )
    caller_rng = torch.random.get_rng_state().clone()
    observed = _cpu_runtime(declaration)
    reference = _cpu_runtime(declaration)
    forward_calls = {id(observed.data): 0, id(reference.data): 0}
    launch_calls = []
    original_forward = warp_runtime.mjwarp.forward
    original_launch = wp.launch
    observer_box = []

    def counted_forward(model, data):
        if id(data) in forward_calls:
            forward_calls[id(data)] += 1
        return original_forward(model, data)

    def pass_through_launch(kernel, *args, **kwargs):
        result = original_launch(kernel, *args, **kwargs)
        if kernel is smooth._crb_accumulate:
            observer = observer_box[0]
            launch_calls.append(
                (
                    (forward_calls[id(observed.data)] - 1) // 2,
                    observer._current_levels is not None,
                )
            )
        return result

    observer = None
    observed_results, observed_trajectory = [], []
    reference_results, reference_trajectory = [], []
    try:
        warp_runtime.mjwarp.forward = counted_forward
        wp.launch = pass_through_launch
        observer = crb_trace.CrbLaunchTrace(
            observed, smooth_module=smooth, launch_module=wp
        )
        observer_box.append(observer)
        with observer:
            for _ in range(28):
                observed_results.append(
                    observed.step_with_schedule(
                        torch.zeros(2, 10), capture_control=True
                    )
                )
                observed_trajectory.append(observed.snapshot())
        trace = observer.capture()
        # The reference has the original launch function and no trace observer.
        wp.launch = original_launch
        for _ in range(28):
            reference_results.append(
                reference.step_with_schedule(torch.zeros(2, 10), capture_control=True)
            )
            reference_trajectory.append(reference.snapshot())
    finally:
        warp_runtime.mjwarp.forward = original_forward
        wp.launch = original_launch
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return {
        "declaration": declaration,
        "pulse_cell": pulse_cell,
        "observed": observed,
        "reference": reference,
        "observer": observer,
        "trace": trace,
        "observed_results": observed_results,
        "reference_results": reference_results,
        "observed_trajectory": observed_trajectory,
        "reference_trajectory": reference_trajectory,
        "forward_calls": forward_calls,
        "launch_calls": launch_calls,
        "caller_rng": caller_rng,
    }


@pytest.fixture(autouse=True)
def _cuda_hidden_and_cleanup(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield
    gc.collect()


def _body_tree_ids(runtime):
    return [
        wp.to_torch(level).detach().cpu().to(torch.int64).tolist()
        for level in reversed(runtime.model.body_tree)
    ]


def _as_tensor(value):
    assert torch.is_tensor(value)
    assert value.device.type == "cpu"
    assert value.dtype == torch.float32
    assert value.shape == (2, 16, 10)
    assert torch.isfinite(value).all()
    return value


def test_actual_observer_is_trajectory_neutral_and_cpu_only(actual_case):
    case = actual_case
    observed, reference = case["observed"], case["reference"]
    assert torch.cuda.is_initialized() is False
    assert _tree_equal(case["observed_result"], case["reference_result"])
    assert _tree_equal(observed.snapshot(), reference.snapshot())
    assert _tree_equal(observed._control_snapshot(), reference._control_snapshot())
    assert case["forward_calls"] == {id(observed.data): 20, id(reference.data): 20}
    assert case["trace"]["status"] == "complete"
    assert case["trace"]["partial_crb_accumulation"] is None
    assert case["trace"]["launch_trace_boundary"] == crb_trace.BOUNDARY
    assert all(case["trace"][key] is False for key in crb_trace.FLAGS)
    assert len(case["trace"]["events"]) == 6
    assert [(event["phase"], event["step"]) for event in case["trace"]["events"]] == [
        (phase, step)
        for step in range(3)
        for phase in ("scheduled-pre", "unforced-post")
    ]
    assert torch.equal(torch.random.get_rng_state(), case["caller_rng"])


def test_280_step_pulse_pair_is_raw_identical_and_only_first_six_phases_observed(
    pulse_case,
):
    case = pulse_case
    observed, reference = case["observed"], case["reference"]
    levels = _body_tree_ids(observed)
    assert case["forward_calls"] == {id(observed.data): 560, id(reference.data): 560}
    assert _tree_equal(case["observed_results"], case["reference_results"])
    assert _tree_equal(case["observed_trajectory"], case["reference_trajectory"])
    assert _tree_equal(observed._control_snapshot(), reference._control_snapshot())
    assert _tree_equal(observed.snapshot(), reference.snapshot())
    assert not hasattr(observed, "optimizer")
    assert all(
        result["optimizer_launched"] is False for result in case["observed_results"]
    )
    assert torch.equal(torch.random.get_rng_state(), case["caller_rng"])
    assert len(case["trace"]["events"]) == 6
    assert [event["step"] for event in case["trace"]["events"]] == [0, 0, 1, 1, 2, 2]
    assert all(
        len(event["crb_accumulation_stages"]) == len(levels)
        for event in case["trace"]["events"]
    )

    scoped = [entry for entry in case["launch_calls"] if entry[1]]
    assert len(scoped) == 6 * len(levels)
    assert len(case["launch_calls"]) == 560 * len(levels)
    pulse_launches = [entry for entry in case["launch_calls"] if 250 <= entry[0] < 270]
    assert pulse_launches
    assert all(not entry[1] for entry in pulse_launches)

    pulse_records = [
        entry
        for result in case["observed_results"]
        for entry in result["scheduled_pulse_evidence"]
        if 250 <= int(entry["before_steps"][1]) < 270
    ]
    assert len(pulse_records) == 20
    trunk = observed.plant_binding["body_id"]
    for entry in pulse_records:
        assert entry["accepted"].tolist() == [True, True]
        assert torch.count_nonzero(entry["pre_xfrc"][0]) == 0
        assert entry["pre_xfrc"][1, trunk, 0].item() == 2.0
        assert torch.count_nonzero(entry["pre_xfrc"][1]) == 1
        assert torch.count_nonzero(entry["post_xfrc"]) == 0
        assert set(entry["phases"]) == {"forced_pre", "integrated", "unforced_post"}
    assert torch.count_nonzero(observed._view("xfrc_applied")) == 0
    assert torch.count_nonzero(observed._view("qfrc_applied")) == 0


def test_launch_stages_bind_real_reversed_topology_and_whole_vec10_arrays(actual_case):
    case = actual_case
    runtime = case["observed"]
    trace = case["trace"]
    expected_levels = _body_tree_ids(runtime)
    topology = trace["compiled_topology"]
    assert (
        trace["compiled_topology_sha256"]
        == sha256(canonical(topology).encode()).hexdigest()
    )
    assert topology["reversed_body_tree_ids"] == expected_levels
    assert topology["body_parentid"] == runtime.native.body_parentid.tolist()
    expected_payload_bytes = (
        2
        * 3
        * len(expected_levels)
        * 2
        * 16
        * 10
        * torch.tensor([], dtype=torch.float32).element_size()
        * 4
    )
    assert trace["stage_payload_tensor_bytes"] == expected_payload_bytes
    assert trace["stage_payload_limit_bytes"] == crb_trace.MAX_STAGE_PAYLOAD_BYTES
    assert trace["stage_payload_tensor_bytes"] <= trace["stage_payload_limit_bytes"]
    assert len(trace["events"]) == 6
    assert len(trace["events"][0]["crb_accumulation_stages"]) == len(expected_levels)

    expected_target_calls = 20 * len(expected_levels)
    observed_calls = case["launch_calls"]
    assert len(observed_calls) == expected_target_calls
    assert len(case["other_launches"]) > 0
    assert len(case["launch_calls"]) - 6 * len(expected_levels) == 14 * len(
        expected_levels
    )
    observed_early_calls = observed_calls[: 6 * len(expected_levels)]
    for event in trace["events"]:
        stages = event["crb_accumulation_stages"]
        assert len(stages) == len(expected_levels)
        for stage_index, (stage, body_ids) in enumerate(
            zip(stages, expected_levels, strict=True)
        ):
            assert set(stage) == {
                "stage_index",
                "body_tree_ids",
                "dim",
                "before_crb",
                "before_cinert",
                "after_crb",
                "after_cinert",
            }
            assert stage["stage_index"] == stage_index
            assert stage["body_tree_ids"] == body_ids
            assert stage["dim"] == [2, len(body_ids)]
            for key in ("before_crb", "before_cinert", "after_crb", "after_cinert"):
                _as_tensor(stage[key])
            assert _tree_equal(stage["before_cinert"], stage["after_cinert"])
            if stage_index == 0:
                assert _tree_equal(stage["before_crb"], stage["before_cinert"])
            else:
                assert _tree_equal(
                    stage["before_crb"], stages[stage_index - 1]["after_crb"]
                )
        assert _tree_equal(stages[-1]["after_crb"], event["persistent"]["crb"])

    for stage, (kernel, args, kwargs, _launch_result) in zip(
        [
            stage
            for event in trace["events"]
            for stage in event["crb_accumulation_stages"]
        ],
        observed_early_calls,
        strict=True,
    ):
        assert kernel is smooth._crb_accumulate
        assert args == ()
        assert set(kwargs) == {"dim", "inputs", "outputs"}
        assert kwargs["dim"] == tuple(stage["dim"])
        assert kwargs["inputs"][0] is runtime.model.body_parentid
        assert kwargs["inputs"][1] is runtime.data.crb
        assert (
            kwargs["inputs"][2]
            is runtime.model.body_tree[len(expected_levels) - 1 - stage["stage_index"]]
        )
        assert len(kwargs["outputs"]) == 1 and kwargs["outputs"][0] is runtime.data.crb


def test_checker_accepts_actual_trace_and_rejects_stage_damage(actual_case):
    case = actual_case
    score = crb_trace.check(case["trace"], case["declaration"], case["record"])
    assert score["protocol"] == crb_trace.PROTOCOL + ":score"
    assert score["source"] == SOURCE and score["worlds"] == 2
    assert score["events"] == 6
    assert all(score[key] is False for key in crb_trace.FLAGS)

    for damage in (
        "order",
        "dim",
        "shape",
        "nonfinite",
        "dtype",
        "topology",
        "source",
        "flag",
        "unknown",
        "stage-cap",
        "bool-index",
        "bool-dim",
        "cinert-chain",
    ):
        changed = deepcopy(case["trace"])
        stage = changed["events"][0]["crb_accumulation_stages"][0]
        if damage == "order":
            stage["body_tree_ids"] = list(reversed(stage["body_tree_ids"]))
        elif damage == "dim":
            stage["dim"][1] += 1
        elif damage == "shape":
            stage["after_crb"] = torch.zeros(2, 16, 9)
        elif damage == "nonfinite":
            stage["before_cinert"][0, 0, 0] = float("nan")
        elif damage == "dtype":
            stage["after_cinert"] = stage["after_cinert"].double()
        elif damage == "topology":
            changed["compiled_topology_sha256"] = "f" * 64
        elif damage == "source":
            changed["source"] = "f" * 40
        elif damage == "flag":
            changed[next(iter(crb_trace.FLAGS))] = True
        elif damage == "unknown":
            stage["unexpected"] = None
        elif damage == "bool-index":
            stage["stage_index"] = False
        elif damage == "bool-dim":
            stage["dim"][0] = True
        elif damage == "cinert-chain":
            changed["events"][0]["crb_accumulation_stages"][1]["before_cinert"][
                0, 0, 0
            ] += 0.125
        else:
            changed["events"][0]["crb_accumulation_stages"].append(deepcopy(stage))
        with pytest.raises((TypeError, ValueError)):
            crb_trace.check(changed, case["declaration"], case["record"])


def test_tree_equality_is_raw_byte_exact_for_signed_zero():
    plus_zero = torch.tensor([0.0], dtype=torch.float32)
    minus_zero = torch.tensor([-0.0], dtype=torch.float32)
    assert torch.equal(plus_zero, minus_zero)
    assert not _tree_equal(plus_zero, minus_zero)


def test_captured_launch_arrays_are_detached_from_runtime_storage(actual_case):
    case = actual_case
    trace = case["trace"]
    first = trace["events"][0]["crb_accumulation_stages"][0]
    retained = {
        key: first[key].clone()
        for key in ("before_crb", "before_cinert", "after_crb", "after_cinert")
    }
    runtime = case["observed"]
    old_crb = runtime._view("crb").clone()
    old_cinert = runtime._view("cinert").clone()
    try:
        runtime._view("crb").fill_(123.0)
        runtime._view("cinert").fill_(456.0)
        assert all(torch.equal(first[key], value) for key, value in retained.items())
    finally:
        runtime._view("crb").copy_(old_crb)
        runtime._view("cinert").copy_(old_cinert)


def test_owned_dispatch_calls_real_launch_once_with_same_arguments_and_result():
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    runtime = _cpu_runtime(declaration)
    observer = crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    observer._expected_level_arrays = list(reversed(runtime.model.body_tree))
    observer._current_levels = []
    observer._owner_thread = threading.get_ident()
    actual_launch = wp.launch
    marker_result = object()
    calls = []

    def delegate(kernel, *args, **kwargs):
        actual_launch(kernel, *args, **kwargs)
        calls.append((kernel, args, kwargs))
        return marker_result

    observer._original_launch = delegate
    level = observer._expected_level_arrays[0]
    dim = (runtime.n, level.size)
    inputs = [runtime.model.body_parentid, runtime.data.crb, level]
    outputs = [runtime.data.crb]
    captured_stages = []
    try:
        result = observer._dispatch_launch(
            observer.kernel,
            dim=dim,
            inputs=inputs,
            outputs=outputs,
        )
        captured_stages = list(observer._current_levels)
    finally:
        observer._current_levels = None
        observer._owner_thread = None
    assert result is marker_result
    assert len(calls) == 1
    kernel, args, kwargs = calls[0]
    assert kernel is observer.kernel
    assert args == ()
    assert kwargs["dim"] is dim
    assert kwargs["inputs"] is inputs
    assert kwargs["outputs"] is outputs
    assert len(captured_stages) == 1
    assert captured_stages[0]["body_tree_ids"] == _body_tree_ids(runtime)[0]

    reject = crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    reject._expected_level_arrays = list(reversed(runtime.model.body_tree))
    reject._current_levels = []
    reject._owner_thread = threading.get_ident()
    reject_calls = []
    reject._original_launch = lambda *args, **kwargs: reject_calls.append(
        (args, kwargs)
    )
    try:
        with pytest.raises(ValueError):
            reject._dispatch_launch(
                reject.kernel,
                dim=dim,
                inputs=inputs,
                outputs=outputs,
                stream=object(),
            )
    finally:
        reject._current_levels = None
        reject._owner_thread = None
    assert reject_calls == []


def test_exception_restores_owned_launch_hook_without_clobbering_foreign_replacement(
    actual_case,
):
    runtime = _cpu_runtime(actual_case["declaration"])
    original = wp.launch
    observer = crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    with pytest.raises(RuntimeError, match="synthetic body exception"):
        with observer:
            raise RuntimeError("synthetic body exception")
    assert wp.launch is original
    faulted = observer.capture()
    assert faulted["status"] == "faulted"
    with pytest.raises(ValueError):
        crb_trace.check(faulted, actual_case["declaration"], {})

    second = crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)

    def foreign(*args, **kwargs):
        return args, kwargs

    try:
        with pytest.raises((RuntimeError, ValueError)):
            with second:
                wp.launch = foreign
        assert wp.launch is foreign
        assert second.capture()["status"] == "faulted"
    finally:
        wp.launch = original


def test_owned_launch_exception_retains_partial_and_restores_launch_hook(monkeypatch):
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    runtime = _cpu_runtime(declaration)
    original = wp.launch
    calls = []

    def fail_owned(kernel, *args, **kwargs):
        if kernel is smooth._crb_accumulate:
            calls.append((kernel, args, kwargs))
            raise RuntimeError("synthetic owned CRB launch failure")
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", fail_owned)
    observer = crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    with pytest.raises(RuntimeError, match="synthetic owned CRB launch failure"):
        with observer:
            runtime.step_with_schedule(torch.zeros(2, 10), capture_control=True)
    assert wp.launch is fail_owned
    assert len(calls) == 1
    partial = observer.capture()
    assert partial["status"] == "faulted"
    assert partial["partial_crb_accumulation"] is not None
    assert partial["partial_crb_accumulation"]["stages"]
    with pytest.raises(ValueError):
        crb_trace.check(partial, declaration, {})


def test_active_hook_reset_to_original_faults_after_complete_level_sequence(
    monkeypatch,
):
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    runtime = _cpu_runtime(declaration)
    original = wp.launch
    expected_levels = len(_body_tree_ids(runtime))
    calls = []

    def reset_after_levels(kernel, *args, **kwargs):
        result = original(kernel, *args, **kwargs)
        if kernel is smooth._crb_accumulate:
            calls.append(kernel)
            if len(calls) == expected_levels:
                # Simulate foreign removal of the scope hook even though all
                # levels for this forward have already returned successfully.
                wp.launch = original
        return result

    monkeypatch.setattr(wp, "launch", reset_after_levels)
    observer = crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    with pytest.raises(ValueError, match="launch hook changed"):
        with observer:
            runtime.step_with_schedule(torch.zeros(2, 10), capture_control=True)
    assert wp.launch is original
    assert len(calls) == expected_levels
    assert runtime.faulted
    partial = observer.capture()
    assert partial["status"] == "faulted"
    assert len(partial["partial_crb_accumulation"]["stages"]) == expected_levels


def test_active_hook_foreign_replacement_is_preserved_without_overwrite(monkeypatch):
    declaration = schedule.declaration(SOURCE, "dose", "training", CELL_IDS)
    runtime = _cpu_runtime(declaration)
    original = wp.launch
    foreign_calls = []

    def foreign(kernel, *args, **kwargs):
        foreign_calls.append(kernel)
        return original(kernel, *args, **kwargs)

    def replace_during_launch(kernel, *args, **kwargs):
        result = original(kernel, *args, **kwargs)
        if kernel is smooth._crb_accumulate:
            wp.launch = foreign
        return result

    monkeypatch.setattr(wp, "launch", replace_during_launch)
    observer = crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    try:
        with pytest.raises(ValueError, match="launch hook changed"):
            with observer:
                runtime.step_with_schedule(torch.zeros(2, 10), capture_control=True)
        assert wp.launch is foreign
        assert foreign_calls
        assert runtime.faulted
        assert observer.capture()["status"] == "faulted"
    finally:
        wp.launch = original


def test_observer_lifetime_is_one_shot_and_capture_rejects_faulted_state(actual_case):
    runtime = _cpu_runtime(actual_case["declaration"])
    observer = crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)
    with observer:
        pass
    with pytest.raises(ValueError):
        with observer:
            pass
    assert observer.capture()["status"] == "faulted"


@pytest.mark.parametrize("source_kind", ["symlink", "oversized"])
def test_constructor_rejects_unsafe_smooth_source_before_observer_setup(
    tmp_path, monkeypatch, source_kind
):
    runtime = _cpu_runtime(schedule.declaration(SOURCE, "dose", "training", CELL_IDS))
    source = tmp_path / "smooth-source.py"
    if source_kind == "symlink":
        source.symlink_to(smooth.__file__)
    else:
        source.write_bytes(b"x" * (1024 * 1024 + 1))
    monkeypatch.setattr(smooth, "__file__", str(source))
    with pytest.raises(ValueError, match="exact pinned smooth module"):
        crb_trace.CrbLaunchTrace(runtime, smooth_module=smooth, launch_module=wp)


@pytest.mark.parametrize(
    "damage", ["events-custom", "stages-custom", "stages-oversized", "topology-custom"]
)
def test_reader_preflights_hostile_or_oversized_lists_before_traversal_or_compile(
    actual_case, monkeypatch, damage
):
    class HostileList(list):
        def __iter__(self):
            raise AssertionError("hostile metadata list was traversed")

    class HostileDict(dict):
        def __iter__(self):
            raise AssertionError("oversized stage element was traversed")

    changed = deepcopy(actual_case["trace"])
    if damage == "events-custom":
        changed["events"] = HostileList(changed["events"])
    elif damage == "stages-custom":
        changed["events"][0]["crb_accumulation_stages"] = HostileList(
            changed["events"][0]["crb_accumulation_stages"]
        )
    elif damage == "stages-oversized":
        changed["events"][0]["crb_accumulation_stages"] = [
            HostileDict() for _ in range(crb_trace.MAX_BODIES + 1)
        ]
    else:
        changed["compiled_topology"]["reversed_body_tree_ids"] = HostileList(
            changed["compiled_topology"]["reversed_body_tree_ids"]
        )
    monkeypatch.setattr(
        crb_trace,
        "_fresh_topology",
        lambda *_: pytest.fail("fresh CPU compilation before metadata preflight"),
    )
    with pytest.raises(ValueError):
        crb_trace.check(changed, actual_case["declaration"], actual_case["record"])


@pytest.mark.parametrize("location", ["root", "topology", "event", "stage"])
def test_schema_length_is_checked_before_key_set_materialization(
    actual_case, monkeypatch, location
):
    class HashBomb:
        armed = False

        def __hash__(self):
            if self.armed:
                raise AssertionError("unknown dictionary key was hashed")
            return 97

    changed = deepcopy(actual_case["trace"])
    mappings = {
        "root": changed,
        "topology": changed["compiled_topology"],
        "event": changed["events"][0],
        "stage": changed["events"][0]["crb_accumulation_stages"][0],
    }
    key = HashBomb()
    mappings[location][key] = None
    key.armed = True
    monkeypatch.setattr(
        crb_trace,
        "_fresh_topology",
        lambda *_: pytest.fail("compilation before exact-schema length check"),
    )
    with pytest.raises(ValueError):
        crb_trace.check(changed, actual_case["declaration"], actual_case["record"])
