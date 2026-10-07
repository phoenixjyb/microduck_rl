"""Real two-world CPU fixtures; not native CUDA or original replay acceptance."""

from hashlib import sha256
from threading import Thread

import numpy as np
import pytest
import torch
import warp as wp
from mujoco_warp._src import smooth

from mjlab_microduck import stance_com_entry_capture as capture
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


@pytest.fixture(autouse=True)
def hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="module")
def observed():
    scope = capture.ComEntryCapture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        with scope:
            env = WarpStanceRuntime(nworld=2, device="cpu", solved_field_check="packed")
            env._forward()
    return env, scope


def test_original_constructor_and_later_entry(observed):
    env, scope = observed
    receipt = scope.receipt
    assert receipt["status"] == "complete"
    assert len(scope.entries) == 2
    assert scope.entries[0] == scope.entries[1]
    assert all(len(raw) == 384 for raw in scope.entries)
    assert all(row["original_launches"] == 11 for row in receipt["calls"])
    assert all(
        row["boundary"] == "after-init-before-first-accumulation"
        for row in receipt["calls"]
    )
    assert all(
        row["readback_and_device_sync_perturb_timing"] for row in receipt["calls"]
    )
    assert all(value is False for value in receipt["flags"].values())
    assert (env.steps == 0).all()
    assert env.forward_graph is None
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH
    receipt["calls"].clear()
    assert len(scope.receipt["calls"]) == 2


@pytest.mark.parametrize("mode,count", [("concurrent", 224), ("serial", 288)])
def test_whole_repeat_bank_resets_and_includes_root(observed, mode, count):
    env, scope = observed
    raw = scope.entries[0]
    before = env.data.subtree_com.numpy().copy()
    with wp.ScopedDevice(env.wp_device):
        bank, receipt = capture.repeat_entry(
            env.model, env.data, raw, sha256(raw).hexdigest(), mode
        )
    assert len(bank) == 32 * 384
    assert receipt["repeats"] == receipt["resets"] == 32
    assert receipt["accumulation_launches"] == count
    assert receipt["output_boundary"] == "after-accumulation-before-division"
    assert all(value is False for value in receipt["flags"].values())
    np.testing.assert_array_equal(
        env.data.subtree_com.numpy().view(np.uint32), before.view(np.uint32)
    )
    expected = np.frombuffer(raw, dtype="<f4").reshape(2, 16, 3).copy()
    for level in capture.source_checks.EXPECTED_REVERSED_LEVELS:
        for body in level:
            if body:
                parent = capture.source_checks.EXPECTED_PARENTS[body]
                np.add(expected[:, parent], expected[:, body], out=expected[:, parent])
    values = np.frombuffer(bank, dtype="<f4").reshape(32, 2, 16, 3)
    np.testing.assert_array_equal(
        values.view("<u4"), np.broadcast_to(expected.view("<u4"), values.shape)
    )
    assert (
        values[:, :, 0, :]
        != np.frombuffer(raw, dtype="<f4").reshape(2, 16, 3)[None, :, 0, :]
    ).any()


@pytest.mark.parametrize("mutation", ["bad-hash", "short", "mutable", "nan", "mode"])
def test_repeat_rejects_invalid_inputs_before_kernel(observed, monkeypatch, mutation):
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
    launches = []
    with monkeypatch.context() as context:

        def forbidden(*args, **kwargs):
            launches.append(args)
            raise AssertionError("invalid input must not launch")

        context.setattr(capture, "_LAUNCH", forbidden)
        context.setattr(wp, "launch", forbidden)
        with wp.ScopedDevice(env.wp_device), pytest.raises(ValueError):
            capture.repeat_entry(env.model, env.data, raw, digest, mode)
    assert not launches


@pytest.mark.parametrize(
    "field", ["body_mass", "body_subtreemass", "subtree_com", "xipos"]
)
def test_layout_refuses_storage_and_shape_changes(observed, field):
    env, _ = observed
    owner = env.model if field.startswith("body_") else env.data
    original = getattr(owner, field)
    try:
        setattr(
            owner, field, wp.zeros((2, 15), dtype=original.dtype, device=env.wp_device)
        )
        with wp.ScopedDevice(env.wp_device), pytest.raises(ValueError):
            capture._layout(env.model, env.data)
    finally:
        setattr(owner, field, original)


def test_incomplete_scope_restores_original():
    scope = capture.ComEntryCapture()
    with pytest.raises(ValueError, match="complete exact two-call"):
        with scope:
            pass
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH
    with pytest.raises(ValueError, match="one-shot"):
        scope.__enter__()


def test_exception_restores_original():
    scope = capture.ComEntryCapture()
    with pytest.raises(RuntimeError, match="synthetic"):
        with scope:
            raise RuntimeError("synthetic")
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


def test_foreign_entry_is_preserved(monkeypatch):
    scope = capture.ComEntryCapture()

    def foreign(*args, **kwargs):
        return None

    assert smooth.com_pos is capture._ENTRY
    with monkeypatch.context() as context:
        context.setattr(smooth, "com_pos", capture._ENTRY)
        with pytest.raises(ValueError, match="complete exact two-call"):
            with scope:
                context.setattr(smooth, "com_pos", foreign)
        assert smooth.com_pos is foreign
    assert smooth.com_pos is capture._ENTRY


def test_foreign_launch_refused_and_preserved(monkeypatch):
    def foreign(*args, **kwargs):
        return None

    with monkeypatch.context() as context:
        context.setattr(wp, "launch", foreign)
        scope = capture.ComEntryCapture()
        with pytest.raises(ValueError, match="unmodified CoM"):
            scope.__enter__()
        assert wp.launch is foreign
    assert wp.launch is capture._LAUNCH


def test_nonmain_thread_refused():
    errors = []

    def worker():
        try:
            capture.ComEntryCapture().__enter__()
        except ValueError as error:
            errors.append(str(error))

    thread = Thread(target=worker)
    thread.start()
    thread.join()
    assert errors == ["main-thread CoM scope"]


def test_third_call_and_same_shape_array_replacement_refused(observed):
    env, _ = observed
    scope = capture.ComEntryCapture()
    with pytest.raises(ValueError, match="bounded unmodified"):
        with scope, wp.ScopedDevice(env.wp_device):
            smooth.com_pos(env.model, env.data)
            smooth.com_pos(env.model, env.data)
            smooth.com_pos(env.model, env.data)
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH
    scope = capture.ComEntryCapture()
    original = env.data.subtree_com
    try:
        with pytest.raises(ValueError, match="unchanged CoM objects"):
            with scope, wp.ScopedDevice(env.wp_device):
                smooth.com_pos(env.model, env.data)
                env.data.subtree_com = wp.zeros(
                    (2, 16), dtype=wp.vec3, device=env.wp_device
                )
                smooth.com_pos(env.model, env.data)
    finally:
        env.data.subtree_com = original
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


def cpu_frame(env):
    from mjlab_microduck.stance_crb_runtime_probe import FRAME_FIELDS

    assert len(FRAME_FIELDS) == 17
    result = {}
    for name, suffix in FRAME_FIELDS.items():
        value = getattr(env.data, name).numpy()
        assert value.dtype == np.dtype(np.float32)
        assert value.shape == (2, *suffix)
        assert np.isfinite(value).all()
        result[name] = value.astype("<f4", copy=False).tobytes()
    return result


@pytest.fixture(scope="module")
def coupled_cpu_pair():
    from mjlab_microduck import stance_warp_integrator as integrator

    assert not torch.cuda.is_initialized()
    original = integrator.EulerCandidateCommit.integrate
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("CPU coupled prototype must not integrate")

    pairs = []
    rng = torch.random.get_rng_state().clone()
    integrator.EulerCandidateCommit.integrate = forbidden
    try:
        with torch.random.fork_rng(devices=[]), wp.ScopedDevice("cpu"):
            for control in (capture.ComEntryCapture, capture.CpuSerialComControl):
                torch.manual_seed(991)
                scope = control()
                with scope:
                    env = WarpStanceRuntime(
                        nworld=2, device="cpu", solved_field_check="packed"
                    )
                    first, cpu_rng = (
                        cpu_frame(env),
                        torch.random.get_rng_state().clone(),
                    )
                    env._forward()
                    second = cpu_frame(env)
                    assert torch.equal(cpu_rng, torch.random.get_rng_state())
                assert (env.steps == 0).all() and env.forward_graph is None
                pairs.append((env, scope, first, second, cpu_rng))
    finally:
        if integrator.EulerCandidateCommit.integrate is forbidden:
            integrator.EulerCandidateCommit.integrate = original
    assert calls == [] and torch.equal(rng, torch.random.get_rng_state())
    assert not torch.cuda.is_initialized()
    return pairs


def test_cpu_coupled_constructor_and_later_complete_frame_equality(coupled_cpu_pair):
    (_, original, a0, a1, arng), (_, controlled, b0, b1, brng) = coupled_cpu_pair
    assert a0 == b0 and a1 == b1
    assert torch.equal(arng, brng)
    for name in ("qpos", "qvel", "time", "ctrl", "qfrc_applied", "xfrc_applied"):
        assert a0[name] == a1[name] == b0[name] == b1[name]
    assert original.entries == controlled.entries
    receipt = controlled.receipt
    assert receipt["protocol"] == capture.PROTOCOL + ":cpu-serial-control"
    assert receipt["cpu_only_prototype"] is True
    assert receipt["complete_kernel_count_matches"] is True
    assert receipt["dispatched_original_kernel_counts"] == [13, 13]
    assert all(row["original_launches"] == 11 for row in receipt["calls"])
    assert all(value is False for value in receipt["flags"].values())
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


def test_cpu_control_exact_original_kernel_dispatches(observed, monkeypatch):
    env, _ = observed
    launches = []
    real = capture._LAUNCH

    def record(kernel, *args, **kwargs):
        if kernel in capture._KERNELS:
            group = (
                tuple(int(v) for v in kwargs["inputs"][2].numpy())
                if kernel is capture._KERNELS[1]
                else None
            )
            launches.append((kernel, kwargs["dim"], group))
        return real(kernel, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(capture, "_LAUNCH", record)
        context.setattr(wp, "launch", record)
        scope = capture.CpuSerialComControl()
        with scope, wp.ScopedDevice(env.wp_device):
            smooth.com_pos(env.model, env.data)
            smooth.com_pos(env.model, env.data)
    assert len(launches) == 26
    expected_groups = [
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
    for block in (launches[:13], launches[13:]):
        assert [row[2] for row in block[1:10]] == expected_groups
        assert [row[0] for row in block] == [capture._KERNELS[0]] + [
            capture._KERNELS[1]
        ] * 9 + list(capture._KERNELS[2:])
        assert all(row[1] == (2, len(row[2])) for row in block[1:10])
    assert scope.receipt["dispatched_original_kernel_counts"] == [13, 13]
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


@pytest.mark.parametrize("domain", ["cuda", "worlds", "boolean"])
def test_cpu_control_refuses_other_domains_before_kernel(observed, monkeypatch, domain):
    env, _ = observed
    scope = capture.CpuSerialComControl()
    original_nworld = env.data.nworld
    device = "cuda:0" if domain == "cuda" else env.wp_device
    with monkeypatch.context() as context:
        context.setattr(capture, "_layout", lambda *_: device)
        context.setattr(
            scope, "_dispatch", lambda *_: pytest.fail("refused domain launched")
        )
        try:
            if domain != "cuda":
                env.data.nworld = 64 if domain == "worlds" else True
            with pytest.raises(ValueError, match="two-world CPU-only"):
                with scope, wp.ScopedDevice(env.wp_device):
                    smooth.com_pos(env.model, env.data)
        finally:
            env.data.nworld = original_nworld
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


@pytest.mark.parametrize("mutation", ["storage", "ids"])
def test_cpu_control_split_rows_cannot_change_between_calls(observed, mutation):
    env, _ = observed
    scope = capture.CpuSerialComControl()
    with pytest.raises(ValueError, match="CPU sibling-control"):
        with scope, wp.ScopedDevice(env.wp_device):
            smooth.com_pos(env.model, env.data)
            if mutation == "storage":
                scope._split_rows = (
                    wp.array([2], dtype=wp.int32, device="cpu"),
                    *scope._split_rows[1:],
                )
            else:
                scope._split_rows[0].assign(np.array([3], dtype=np.int32))
            smooth.com_pos(env.model, env.data)
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


@pytest.mark.parametrize("failure", ["incomplete", "exception", "foreign", "third"])
def test_cpu_control_faults_restore_only_owned_hooks(observed, monkeypatch, failure):
    env, _ = observed
    scope = capture.CpuSerialComControl()

    def foreign(*args, **kwargs):
        return None

    with monkeypatch.context() as context:
        context.setattr(smooth, "com_pos", capture._ENTRY)
        with pytest.raises((ValueError, RuntimeError)):
            with scope, wp.ScopedDevice(env.wp_device):
                if failure == "exception":
                    raise RuntimeError("synthetic CPU control exception")
                if failure == "foreign":
                    context.setattr(smooth, "com_pos", foreign)
                if failure == "third":
                    for _ in range(3):
                        smooth.com_pos(env.model, env.data)
        assert smooth.com_pos is (foreign if failure == "foreign" else capture._ENTRY)
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


def test_cpu_control_subclass_cannot_change_dispatch_domain(observed, monkeypatch):
    env, _ = observed

    class ForeignControl(capture.CpuSerialComControl):
        pass

    scope = ForeignControl()
    monkeypatch.setattr(scope, "_dispatch", lambda *_: pytest.fail("subclass launched"))
    with pytest.raises(ValueError, match="two-world CPU-only"):
        with scope, wp.ScopedDevice(env.wp_device):
            smooth.com_pos(env.model, env.data)
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


def test_cpu_control_wrong_kernel_count_faults_at_return(observed, monkeypatch):
    env, _ = observed
    scope = capture.CpuSerialComControl()
    original = scope._dispatch

    def corrupted(index, args, kwargs):
        result = original(index, args, kwargs)
        if index == 10:
            scope._kernel_counts[-1] -= 1
        return result

    monkeypatch.setattr(scope, "_dispatch", corrupted)
    with pytest.raises(ValueError, match="thirteen-call"):
        with scope, wp.ScopedDevice(env.wp_device):
            smooth.com_pos(env.model, env.data)
    assert scope.receipt["status"] == "faulted"
    assert scope.receipt["complete_kernel_count_matches"] is False
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


def test_original_observer_subclass_cannot_report_altered_dispatch():
    class ForeignObserver(capture.ComEntryCapture):
        def _dispatch(self, *_):
            pytest.fail("foreign observer dispatch reached")

    scope = ForeignObserver()
    with pytest.raises(ValueError, match="exact observation"):
        with scope:
            pytest.fail("foreign observer entered")
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


@pytest.mark.parametrize("name", ["_before_call", "_dispatch", "_after_call"])
def test_original_observer_instance_hooks_cannot_change_dispatch(monkeypatch, name):
    scope = capture.ComEntryCapture()
    monkeypatch.setattr(scope, name, lambda *_: pytest.fail("foreign hook reached"))
    with pytest.raises(ValueError, match="bound original observer"):
        with scope:
            pytest.fail("altered observer entered")
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


@pytest.mark.parametrize("timing", ["before-enter", "during-scope"])
def test_original_observer_class_hook_is_pinned_before_dispatch(
    observed, monkeypatch, timing
):
    env, _ = observed
    scope = capture.ComEntryCapture()
    with monkeypatch.context() as context:
        if timing == "before-enter":
            context.setattr(
                capture.ComEntryCapture,
                "_dispatch",
                lambda *_: pytest.fail("altered class dispatched"),
            )
        with pytest.raises(ValueError, match="bound original observer"):
            with scope, wp.ScopedDevice(env.wp_device):
                if timing == "during-scope":
                    context.setattr(
                        capture.ComEntryCapture,
                        "_dispatch",
                        lambda *_: pytest.fail("altered class dispatched"),
                    )
                smooth.com_pos(env.model, env.data)
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


@pytest.mark.parametrize("mode", [None, True, "cpu", "concurrent", "SERIAL"])
def test_native_coupled_mode_is_literal_before_any_device_query(mode, monkeypatch):
    monkeypatch.setattr(
        wp, "get_device", lambda *_: pytest.fail("invalid mode queried device")
    )
    with pytest.raises(ValueError, match="literal coupled"):
        capture.NativeCoupledComControl(mode=mode)


@pytest.mark.parametrize("mode", ["original", "serial"])
def test_native_coupled_scope_refuses_cpu_before_dispatch(mode):
    scope = capture.NativeCoupledComControl(mode=mode)
    with wp.ScopedDevice("cpu"), pytest.raises(ValueError, match="requires CUDA0"):
        with scope:
            pytest.fail("native scope entered CPU")
    assert not torch.cuda.is_initialized()
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH


@pytest.mark.parametrize("domain", ["cpu", "worlds", "boolean", "changed-mode"])
def test_native_coupled_domains_refuse_before_row_allocation(domain, monkeypatch):
    from types import SimpleNamespace

    scope = capture.NativeCoupledComControl(mode="serial")
    data = SimpleNamespace(
        nworld=True if domain == "boolean" else (2 if domain == "worlds" else 64)
    )
    if domain == "changed-mode":
        scope.mode = "original"
    monkeypatch.setattr(
        wp, "array", lambda *_a, **_k: pytest.fail("refused domain allocated")
    )
    with pytest.raises(ValueError, match="fixed-mode CUDA64"):
        scope._before_call(None, data, "cpu" if domain == "cpu" else "cuda:0")


@pytest.mark.parametrize("mode,expected", [("original", 11), ("serial", 13)])
def test_native_dispatch_plan_synthetic_preserves_non_target_arguments(
    mode, expected, monkeypatch
):
    # Synthetic launch recorder only. No Warp CUDA object or kernel is created.
    class Output:
        device = "cuda:0"

        def numpy(self):
            return np.zeros((64, 16, 3), dtype=np.float32)

    scope = capture.NativeCoupledComControl(mode=mode)
    scope._kernel_counts = [0]
    rows = tuple(object() for _ in (2, 7, 11))
    scope._split_rows = rows
    validations = []
    monkeypatch.setattr(
        scope,
        "_check_rows",
        lambda device: validations.append((str(device), scope._kernel_counts[-1])),
    )
    monkeypatch.setattr(wp, "synchronize_device", lambda *_: None)
    launches = []
    monkeypatch.setattr(capture, "_LAUNCH", lambda *a, **k: launches.append((a, k)))
    output = Output()
    parent = object()
    original_row = object()
    requests = []
    for index in range(11):
        kernel = (
            capture._KERNELS[0]
            if index == 0
            else capture._KERNELS[1]
            if index < 8
            else capture._KERNELS[index - 6]
        )
        inputs = [parent, output, original_row]
        kwargs = dict(dim=(64, 3), inputs=inputs, outputs=[output])
        requests.append((kernel, kwargs))
        scope._dispatch(index, (kernel,), kwargs)
    assert len(launches) == expected and scope._kernel_counts == [expected]
    assert validations == [("cuda:0", 5)]  # Same readback boundary in both modes.
    for index, (kernel, kwargs) in enumerate(requests):
        position = index + (2 if mode == "serial" and index > 5 else 0)
        if mode == "serial" and index == 5:
            split = launches[position : position + 3]
            assert [k["inputs"][2] for _, k in split] == list(rows)
            assert all(
                a == (kernel,)
                and k["dim"] == (64, 1)
                and k["inputs"][0] is parent
                and k["inputs"][1] is output
                and k["outputs"] is kwargs["outputs"]
                for a, k in split
            )
        else:
            a, k = launches[position]
            assert a == (kernel,) and k["inputs"] is kwargs["inputs"]
            assert k["outputs"] is kwargs["outputs"] and k["dim"] is kwargs["dim"]
    assert len(scope.weighted_entries) == 1 and len(scope.weighted_entries[0]) == 12288
    assert scope.receipt["protocol"].endswith(":native-coupled-control")
    assert scope.receipt["complete_kernel_count_matches"] is False
    assert all(v is False for v in scope.receipt["flags"].values())


@pytest.mark.parametrize("bad", ["nan", "short", "repeat-boundary"])
def test_native_weighted_boundary_refuses_invalid_values_before_division(
    bad, monkeypatch
):
    class Output:
        device = "cuda:0"

        def numpy(self):
            values = np.zeros((64, 16, 3), dtype=np.float32)
            if bad == "nan":
                values[0, 0, 0] = np.nan
            return values[:2] if bad == "short" else values

    scope = capture.NativeCoupledComControl(mode="original")
    scope._kernel_counts = [8]
    if bad == "repeat-boundary":
        scope._weighted_entries = [b"x"]
    monkeypatch.setattr(wp, "synchronize_device", lambda *_: None)
    monkeypatch.setattr(
        capture, "_LAUNCH", lambda *_a, **_k: pytest.fail("invalid division dispatched")
    )
    with pytest.raises(ValueError):
        scope._dispatch(8, (capture._KERNELS[2],), dict(outputs=[Output()]))


@pytest.mark.parametrize("level", ["instance", "class"])
def test_native_dispatch_method_is_pinned_before_scope(level, monkeypatch):
    scope = capture.NativeCoupledComControl(mode="serial")
    owner = scope if level == "instance" else capture.NativeCoupledComControl
    monkeypatch.setattr(
        owner, "_dispatch", lambda *_: pytest.fail("foreign native dispatch")
    )
    monkeypatch.setattr(wp, "get_device", lambda *_: "cuda:0")
    with pytest.raises(ValueError, match="bound native coupled"):
        with scope:
            pytest.fail("foreign native scope entered")
    assert scope.receipt["status"] == "faulted"
    assert smooth.com_pos is capture._ENTRY and wp.launch is capture._LAUNCH
