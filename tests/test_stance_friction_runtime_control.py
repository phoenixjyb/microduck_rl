"""CPU fake-backend tests for the scoped runtime friction observer."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from mjlab_microduck import stance_friction_runtime_control as control


class FakeArray:
    def __init__(self, host, device, pointer, *, strides=None):
        self._host = host
        self.device = device
        self.ptr = pointer if host.nbytes else None
        self.shape = tuple(host.shape)
        self.strides = tuple(host.strides) if strides is None else tuple(strides)
        self.dtype = str(host.dtype)
        self.is_contiguous = bool(host.flags.c_contiguous)

    def numpy(self):
        return self._host


class FakeKernel:
    def __init__(self, key):
        self.key = key
        self.func = self._func

    @staticmethod
    def _func():
        return None


@pytest.mark.parametrize("shape", [(64, 0), (64, 0, 0)])
def test_identity_preserves_actual_warp_empty_pointer_null(shape):
    import warp as wp

    array = wp.zeros(shape, dtype=wp.int32, device="cpu")
    layout = control.identity(array, array.device)
    assert array.ptr is None and layout["pointer"] is None
    assert layout["span"] == layout["bytes"] == 0
    assert layout["shape"] == list(shape)


def test_identity_refuses_null_pointer_for_nonempty_array():
    device = SimpleNamespace(context=77)
    array = FakeArray(np.zeros((2,), dtype=np.float32), device, 4096)
    array.ptr = None
    with pytest.raises(ValueError, match="live nonempty allocation"):
        control.identity(array, device)


def _environment(arm="original"):
    device = SimpleNamespace(context=object(), is_cuda=True, is_capturing=False)
    stream = SimpleNamespace(device=device, cuda_stream=17)
    pointer = 0x10000000
    arrays = {}

    def array(name, shape, dtype=np.float32, *, broadcast=False):
        nonlocal pointer
        if broadcast:
            base = np.arange(np.prod(shape[1:]), dtype=dtype).reshape((1, *shape[1:]))
            host = np.broadcast_to(base, shape)
            strides = (0, *host.strides[1:])
        else:
            host = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            strides = None
        value = FakeArray(host, device, pointer, strides=strides)
        pointer += 0x10000
        arrays[name] = value
        return value

    # Singleton-leading zero-stride model inputs exercise the accepted broadcast layout.
    frictionloss = array("frictionloss", (64, 20), broadcast=True)
    qvel = array("qvel", (64, 20))
    invweight = array("invweight", (1, 20), broadcast=True)
    solref = array("solref", (64, 20), broadcast=True)
    solimp = array("solimp", (64, 20), broadcast=True)
    timestep = array("timestep", (64,))

    nf = array("nf", (64,), np.int32)
    nefc = array("nefc", (64,), np.int32)
    efc_type = array("type", (64, 2), np.int32)
    efc_id = array("id", (64, 2), np.int32)
    row_nnz = array("row_nnz", (64, 2), np.int32)
    row_adr = array("row_adr", (64, 2), np.int32)
    col_ind = array("col_ind", (0,), np.int32)
    jacobian = array("J", (64, 2), np.float32)
    pos = array("pos", (64, 2), np.float32)
    margin = array("margin", (64, 2), np.float32)
    diagonal = array("D", (64, 2), np.float32)
    velocity = array("vel", (64, 2), np.float32)
    aref = array("aref", (64, 2), np.float32)
    friction_bank = array("friction_bank", (64, 2), np.float32)
    efc_nnz = array("efc_nnz", (64,), np.int32)
    force = array("force", (64, 2), np.float32)

    original = FakeKernel("frozen-friction")
    candidate = FakeKernel("ascending-friction")
    other = FakeKernel("unrelated")

    model = SimpleNamespace(
        nv=20,
        nq=21,
        nbody=16,
        nu=14,
        ntendon=0,
        is_sparse=False,
        opt=SimpleNamespace(disableflags=0, timestep=timestep),
        dof_solref=solref,
        dof_solimp=solimp,
        dof_frictionloss=frictionloss,
        dof_invweight0=invweight,
    )
    data = SimpleNamespace(
        nworld=64,
        njmax=512,
        njmax_nnz=10240,
        qvel=qvel,
        nf=nf,
        nefc=nefc,
        efc=SimpleNamespace(
            type=efc_type,
            id=efc_id,
            J_rownnz=row_nnz,
            J_rowadr=row_adr,
            J_colind=col_ind,
            J=jacobian,
            pos=pos,
            margin=margin,
            D=diagonal,
            vel=velocity,
            aref=aref,
            frictionloss=friction_bank,
            efc_nnz=efc_nnz,
            force=force,
        ),
    )
    inputs = [
        20,
        timestep,
        0,
        solref,
        solimp,
        frictionloss,
        invweight,
        False,
        qvel,
        512,
        10240,
    ]
    outputs = [
        nf,
        nefc,
        efc_type,
        efc_id,
        row_nnz,
        row_adr,
        col_ind,
        jacobian,
        pos,
        margin,
        diagonal,
        velocity,
        aref,
        friction_bank,
        efc_nnz,
    ]
    call_state = {"dim": (64, 20), "inputs": inputs, "outputs": outputs}
    launch_calls = []
    sink = OrderedDict()
    guard_calls = []
    backend = SimpleNamespace()
    backend.after_target_launch = None

    def launch(kernel, *args, **kwargs):
        launch_calls.append((kernel, args, kwargs))
        if kernel in (original, candidate) and backend.after_target_launch is not None:
            backend.after_target_launch()
        return kernel.key, kwargs

    backend.launch = launch
    backend.synchronize_stream = lambda actual_stream: actual_stream is stream

    class FakeConstraint:
        _friction_dof = original

        def __init__(self):
            self.fail = False
            self.after_target = None
            # Warp's constraint stores its dispatch callable on the instance;
            # retain the same bound-method object so identity checks are real.
            self.make_constraint = type(self).make_constraint.__get__(self, type(self))

        def make_constraint(self, actual_model, actual_data):
            if self.fail:
                raise RuntimeError("fake constraint failure")
            assert actual_model is model and actual_data is data
            result = backend.launch(
                original,
                dim=call_state["dim"],
                inputs=list(call_state["inputs"]),
                outputs=list(call_state["outputs"]),
            )
            if self.after_target is not None:
                self.after_target()
            backend.launch(other, dim=(64,), inputs=["opaque"], outputs=["opaque"])
            return result

    constraint = FakeConstraint()
    observer = control.RuntimeFrictionObserver(
        arm,
        wp=backend,
        constraint=constraint,
        candidate=candidate,
        device=device,
        stream=stream,
        sink=lambda name, raw: sink.__setitem__(name, raw),
        guard=lambda: guard_calls.append("guard"),
    )
    return SimpleNamespace(
        observer=observer,
        backend=backend,
        constraint=constraint,
        model=model,
        data=data,
        device=device,
        stream=stream,
        original=original,
        candidate=candidate,
        other=other,
        arrays=arrays,
        inputs=inputs,
        outputs=outputs,
        call_state=call_state,
        launch_calls=launch_calls,
        sink=sink,
        guard_calls=guard_calls,
    )


def _dispatch_entries(env, count=control.FORWARDS):
    results = []
    for _ in range(count):
        results.append(env.observer.constraint.make_constraint(env.model, env.data))
    return results


@pytest.mark.parametrize(
    ("arm", "selected"),
    [("original", "frozen-friction"), ("candidate0", "ascending-friction")],
)
def test_full_dispatch_window_selects_only_target_and_packs_complete_inputs_and_banks(
    arm, selected
):
    env = _environment(arm)
    with env.observer:
        _dispatch_entries(env)
    receipt = env.observer.receipt()

    target_calls = [
        call for call in env.launch_calls if call[0] in (env.original, env.candidate)
    ]
    forwarded = [call for call in env.launch_calls if call[0] is env.other]
    assert len(target_calls) == control.FORWARDS == 21
    assert [call[0].key for call in target_calls] == [selected] * 21
    assert len(forwarded) == 21
    assert all(
        call[2] == {"dim": (64,), "inputs": ["opaque"], "outputs": ["opaque"]}
        for call in forwarded
    )
    assert len(receipt["entries"]) == 21 and receipt["owned_hooks_restored"] is True
    assert [entry["index"] for entry in receipt["entries"]] == list(range(21))
    assert all(entry["dim"] == [64, 20] for entry in receipt["entries"])
    assert all(
        entry["scalars"]
        == {
            "nv": 20,
            "disableflags": 0,
            "is_sparse": False,
            "njmax": 512,
            "njmax_nnz": 10240,
        }
        for entry in receipt["entries"]
    )
    assert len(env.sink) == 21 * 4
    assert len(env.guard_calls) >= 21
    for entry in receipt["entries"]:
        packets = entry["packets"]
        for kind in ("inputs", "bank"):
            before = env.sink[packets[f"{kind}.before"]["path"]]
            after = env.sink[packets[f"{kind}.after"]["path"]]
            assert before == after

    first = receipt["entries"][0]
    input_fields = first["packets"]["inputs.before"]["fields"]
    bank_fields = first["packets"]["bank.before"]["fields"]
    assert tuple(input_fields) == control.INPUT_ORDER
    assert tuple(bank_fields) == control.BANK_ORDER
    for packet, order in (
        (input_fields, control.INPUT_ORDER),
        (bank_fields, control.BANK_ORDER),
    ):
        offset = 0
        for name in order:
            assert packet[name]["offset"] == offset
            offset += packet[name]["bytes"]
        raw = env.sink[
            first["packets"]["inputs.before"]["path"]
            if order is control.INPUT_ORDER
            else first["packets"]["bank.before"]["path"]
        ]
        assert len(raw) == offset
    assert first["layouts"]["input.solref"]["strides"][0] == 0
    assert first["layouts"]["bank.col_ind"]["span"] == 0
    assert first["layouts"]["bank.col_ind"]["shape"] == [0]
    assert env.constraint.make_constraint.__self__ is not env.observer
    assert env.backend.launch is env.observer.launch


def test_exclusive_lock_reentry_and_main_thread_guard_with_exact_21_entries():
    env = _environment()
    with env.observer:
        other = _environment()
        with pytest.raises(ValueError, match="exclusive new runtime observer"):
            other.observer.__enter__()

        failures = []

        def inspect_from_worker():
            try:
                env.observer._guard()
            except BaseException as error:
                failures.append(error)

        import threading

        worker = threading.Thread(target=inspect_from_worker)
        worker.start()
        worker.join()
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        assert "single main Python thread" in str(failures[0])
        _dispatch_entries(env)
    assert len(env.observer.receipt()["entries"]) == 21


def test_constraint_dispatch_is_nonreentrant_and_frozen_hook_guard_rejects_replacement():
    env = _environment()
    original = env.constraint._friction_dof
    foreign = FakeKernel("foreign-friction")
    with env.observer:
        env.constraint._friction_dof = foreign
        with pytest.raises(ValueError, match="held frozen dispatch and kernel code"):
            env.observer._guard()
        env.constraint._friction_dof = original

        reentry_failures = []

        def attempt_reentry():
            try:
                env.constraint.make_constraint(env.model, env.data)
            except ValueError as error:
                reentry_failures.append(error)

        env.constraint.after_target = attempt_reentry
        env.constraint.make_constraint(env.model, env.data)
        assert len(reentry_failures) == 1
        assert "bounded nonreentrant constraint entry" in str(reentry_failures[0])
        env.constraint.after_target = None
        _dispatch_entries(env, 20)
    assert len(env.observer.receipt()["entries"]) == 21


@pytest.mark.parametrize("count", [20, 22])
def test_close_refuses_any_target_count_other_than_21(count):
    env = _environment()
    expected = (
        "exact21 completed constraint targets"
        if count < 21
        else "bounded nonreentrant constraint entry"
    )
    with pytest.raises(ValueError, match=expected):
        with env.observer:
            _dispatch_entries(env, count)
    assert (
        env.constraint.make_constraint.__func__ is type(env.constraint).make_constraint
    )
    assert env.backend.launch is env.observer.launch


def test_sparse_models_and_scalar_or_mistyped_launch_dimensions_are_refused():
    env = _environment()
    env.model.is_sparse = True
    with pytest.raises(ValueError, match="literal nominal dense stance plant"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)

    env = _environment()
    env.call_state["dim"] = 128
    with pytest.raises(ValueError, match="literal dense friction launch"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)


@pytest.mark.parametrize("bad_dim", [(64.0, 20), (True, 20), 128])
def test_launch_dimension_requires_literal_integer_tuple(bad_dim):
    env = _environment()
    env.call_state["dim"] = bad_dim
    with pytest.raises(ValueError, match="literal dense friction launch"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch


def test_missing_outputs_and_invalid_scalar_or_dtype_are_refused():
    env = _environment()
    env.call_state["outputs"] = env.outputs[:-1]
    with pytest.raises(ValueError, match="26 argument lists"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)

    env = _environment()
    env.call_state["inputs"] = [*env.inputs[:1], 0.002, *env.inputs[2:]]
    with pytest.raises(ValueError, match="actual model/data input objects"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)

    env = _environment()
    bad_friction = FakeArray(
        np.ones((64, 20), dtype=np.float64), env.device, 0x70000000
    )
    env.model.dof_frictionloss = bad_friction
    env.call_state["inputs"] = [
        20,
        env.inputs[1],
        0,
        env.inputs[3],
        env.inputs[4],
        bad_friction,
        *env.inputs[6:],
    ]
    with pytest.raises(ValueError, match="raw four-byte host dtype"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)


def test_pointer_alias_between_live_inputs_is_refused():
    env = _environment()
    env.data.efc.J.ptr = env.data.qvel.ptr
    with pytest.raises(
        ValueError, match="nonoverlapping actual kernel banks and inputs"
    ):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)
    assert (
        env.constraint.make_constraint.__func__ is type(env.constraint).make_constraint
    )


def test_mutated_layout_is_detected_and_owned_hooks_are_restored():
    env = _environment()

    def mutate_layout():
        env.data.qvel.ptr += 0x100000

    env.backend.after_target_launch = mutate_layout
    with pytest.raises(ValueError, match="entry layouts closed unchanged"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch
    assert (
        env.constraint.make_constraint.__func__ is type(env.constraint).make_constraint
    )


def test_exception_and_foreign_hook_replacements_are_preserved_safely():
    env = _environment()
    env.constraint.fail = True
    with pytest.raises(RuntimeError, match="fake constraint failure"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch
    assert (
        env.constraint.make_constraint.__func__ is type(env.constraint).make_constraint
    )

    env = _environment()

    def foreign(*_args, **_kwargs):
        return "foreign launch"

    env.constraint.after_target = lambda: setattr(env.backend, "launch", foreign)
    with pytest.raises(ValueError, match="foreign hook not overwritten during close"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is foreign
    assert (
        env.constraint.make_constraint.__func__ is type(env.constraint).make_constraint
    )

    env = _environment()

    def foreign_constraint_hook(*_args):
        return "foreign constraint"

    env.constraint.after_target = lambda: setattr(
        env.constraint, "make_constraint", foreign_constraint_hook
    )
    with pytest.raises(ValueError, match="foreign hook not overwritten during close"):
        with env.observer:
            env.constraint.make_constraint(env.model, env.data)
    assert env.constraint.make_constraint is foreign_constraint_hook
    assert env.backend.launch is env.observer.launch
