"""CPU fake-backend tests for passive contact-boundary observation."""

from collections import OrderedDict
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mjlab_microduck import stance_contact_boundary_control as boundary
from mjlab_microduck import stance_friction_runtime_control as friction


class FakeArray:
    def __init__(self, host, device, pointer, *, shape=None):
        self.host = host
        self.device = device
        self.ptr = pointer if host.nbytes else None
        self.shape = tuple(host.shape if shape is None else shape)
        self.strides = tuple(host.strides[: len(self.shape)])
        self.dtype = str(host.dtype)
        self.is_contiguous = bool(host.flags.c_contiguous)

    def numpy(self):
        return self.host


class FakeKernel:
    def __init__(self, key):
        self.key = key
        self.module = SimpleNamespace(kernel_key=key)

        def kernel_func():
            return None

        self.func = kernel_func


def _environment(monkeypatch, *, arm="original", contact_first=False):
    monkeypatch.setattr(boundary, "_TOTAL_CAPTURE_BYTES", 0)
    device = SimpleNamespace(context=101, is_cuda=True, is_capturing=False)
    stream = SimpleNamespace(device=device, cuda_stream=17)
    pointer = 0x10000000
    arrays = {}

    def array(name, shape, dtype=np.float32, *, logical_shape=None):
        nonlocal pointer
        host = np.zeros(shape, dtype=dtype)
        value = FakeArray(host, device, pointer, shape=logical_shape)
        pointer += max(host.nbytes, 4) + 0x1000
        arrays[name] = value
        return value

    # Frozen dense-friction inputs.
    frictionloss = array("frictionloss", (64, 20))
    qvel = array("qvel", (64, 20))
    invweight = array("invweight", (1, 20))
    solref = array("solref", (64, 20))
    solimp = array("solimp", (64, 20))
    timestep = array("timestep", (64,))

    # Complete EFC bank uses the retained CPU-proven host layouts.
    efc = SimpleNamespace()
    for name in ("type", "id", "state"):
        setattr(efc, name, array("efc." + name, (64, 512), np.int32))
    for name in ("J_rownnz", "J_rowadr"):
        setattr(efc, name, array("efc." + name, (64, 0), np.int32))
    efc.J_colind = array("efc.J_colind", (64, 0, 0), np.int32)
    efc.J = array("efc.J", (64, 512, 20))
    for name in ("pos", "margin", "D", "vel", "aref", "frictionloss", "force"):
        setattr(efc, name, array("efc." + name, (64, 512)))
    efc.Ma = array("efc.Ma", (64, 20))
    efc.Jqvel = array("efc.Jqvel", (64, 512))
    local_nnz = array("local.efc_nnz", (64,), np.int32)
    efc.efc_nnz = local_nnz

    data_nefc = array("data.nefc", (64,), np.int32)
    data = SimpleNamespace(
        nworld=64,
        njmax=512,
        njmax_nnz=10240,
        naconmax=8192,
        naccdmax=8192,
        ne=array("data.ne", (64,), np.int32),
        nf=array("data.nf", (64,), np.int32),
        nl=array("data.nl", (64,), np.int32),
        nefc=data_nefc,
        qvel=qvel,
        qpos=array("data.qpos", (64, 21)),
        qacc=array("data.qacc", (64, 20)),
        qacc_warmstart=array("data.qacc_warmstart", (64, 20)),
        ctrl=array("data.ctrl", (64, 14)),
        efc=efc,
        nacon=array("contact.nacon", (1,), np.int32),
    )
    contact = SimpleNamespace(
        dist=array("contact.dist", (8192,)),
        dim=array("contact.dim", (8192,), np.int32),
        includemargin=array("contact.includemargin", (8192,)),
        worldid=array("contact.worldid", (8192,), np.int32),
        geom=array("contact.geom", (8192, 2), np.int32, logical_shape=(8192,)),
        flex=array("contact.flex", (8192, 2), np.int32, logical_shape=(8192,)),
        vert=array("contact.vert", (8192, 2), np.int32, logical_shape=(8192,)),
        type=array("contact.type", (8192,), np.int32),
        geomcollisionid=array("context.geomcollisionid", (8192,), np.int32),
        efc_address=array("contact.efc_address", (8192, 4), np.int32),
        pos=array("context.pos", (8192, 3), logical_shape=(8192,)),
        frame=array("context.frame", (8192, 3, 3), logical_shape=(8192, 3)),
    )
    data.contact = contact

    model = SimpleNamespace(
        nv=20,
        nq=21,
        nbody=16,
        nu=14,
        ntendon=0,
        is_sparse=False,
        opt=SimpleNamespace(disableflags=0, timestep=timestep, cone=0),
        dof_solref=solref,
        dof_solimp=solimp,
        dof_frictionloss=frictionloss,
        dof_invweight0=invweight,
        body_weldid=array("model.body_weldid", (16,), np.int32),
        body_dofnum=array("model.body_dofnum", (16,), np.int32),
        body_dofadr=array("model.body_dofadr", (16,), np.int32),
        dof_parentid=array("model.dof_parentid", (20,), np.int32),
        geom_bodyid=array("model.geom_bodyid", (82,), np.int32),
        flex_vertadr=array("model.flex_vertadr", (0,), np.int32),
        flex_vertbodyid=array("model.flex_vertbodyid", (0,), np.int32),
    )

    original = FakeKernel("friction-original")
    candidate = FakeKernel("friction-candidate")
    other = FakeKernel("other")
    contact_kernel = FakeKernel("contact-init")
    launch_calls = []
    sink = OrderedDict()
    backend = SimpleNamespace()
    backend.after_contact = None
    backend.contact_factory_result = contact_kernel
    backend.contact_kwargs = lambda value: value

    def launch(kernel, *args, **kwargs):
        launch_calls.append((kernel, args, kwargs))
        if kernel is contact_kernel and backend.after_contact is not None:
            backend.after_contact()
        return kernel.key, kwargs

    backend.launch = launch
    backend.synchronize_stream = lambda actual: actual is stream

    class Constraint:
        _friction_dof = original

        def __init__(self):
            self.make_constraint = type(self).make_constraint.__get__(self, type(self))
            self.calls = 0

        def make_constraint(self, actual_model, actual_data):
            assert actual_model is model and actual_data is data
            self.calls += 1

            def launch_contact():
                kwargs = {
                    "dim": 8192,
                    "inputs": [
                        model.body_weldid,
                        model.body_dofnum,
                        model.body_dofadr,
                        model.dof_parentid,
                        model.geom_bodyid,
                        model.flex_vertadr,
                        model.flex_vertbodyid,
                        512,
                        10240,
                        data.nacon,
                        contact.dist,
                        contact.dim,
                        contact.includemargin,
                        contact.worldid,
                        contact.geom,
                        contact.flex,
                        contact.vert,
                        contact.type,
                    ],
                    "outputs": [
                        data.nefc,
                        contact.efc_address,
                        efc.id,
                        efc.J_rownnz,
                        efc.J_rowadr,
                        local_nnz,
                    ],
                }
                backend.launch(contact_kernel, **backend.contact_kwargs(kwargs))

            def launch_friction():
                friction_inputs = [
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
                friction_outputs = [
                    data.nf,
                    data.nefc,
                    efc.type,
                    efc.id,
                    efc.J_rownnz,
                    efc.J_rowadr,
                    efc.J_colind,
                    efc.J,
                    efc.pos,
                    efc.margin,
                    efc.D,
                    efc.vel,
                    efc.aref,
                    efc.frictionloss,
                    local_nnz,
                ]
                backend.launch(
                    original,
                    dim=(64, 20),
                    inputs=friction_inputs,
                    outputs=friction_outputs,
                )

            if contact_first:
                launch_contact()
                launch_friction()
            else:
                launch_friction()
                launch_contact()
            backend.launch(other, dim=(64,), inputs=[], outputs=[])

    constraint = Constraint()

    def contact_factory(cone, sparse):
        if cone == 0 and sparse is False:
            return backend.contact_factory_result
        return other

    constraint._efc_contact_init = contact_factory
    observer = boundary.RuntimeContactBoundaryObserver(
        arm,
        wp=backend,
        constraint=constraint,
        candidate=candidate,
        device=device,
        stream=stream,
        contact_kernel=contact_kernel,
        sink=lambda name, raw: sink.__setitem__(name, raw),
        guard=lambda: None,
    )
    return SimpleNamespace(
        observer=observer,
        backend=backend,
        constraint=constraint,
        model=model,
        data=data,
        contact=contact,
        efc=efc,
        local_nnz=local_nnz,
        original=original,
        contact_kernel=contact_kernel,
        sink=sink,
        arrays=arrays,
        launch_calls=launch_calls,
    )


def _run(env, forwards=friction.FORWARDS):
    with env.observer:
        for _ in range(forwards):
            env.observer.constraint.make_constraint(env.model, env.data)


@pytest.mark.parametrize(
    ("arm", "selected"),
    [
        ("original", "friction-original"),
        ("candidate0", "friction-candidate"),
        ("candidate1", "friction-candidate"),
    ],
)
def test_contact_boundary_receipt_is_separate_bounded_and_pre_solver(
    monkeypatch, arm, selected
):
    env = _environment(monkeypatch, arm=arm)
    _run(env)
    assert len(env.observer.receipt()["entries"]) == friction.FORWARDS
    receipt = env.observer.boundary_receipt()
    assert receipt["protocol"] == "microduck-contact-boundary-control-oct8-v1"
    assert receipt["arm"] == arm
    assert receipt["packet_count"] == 21
    assert len(receipt["entries"]) == 7
    assert set(receipt["flags"].values()) == {False}
    assert all(
        entry["phase"] == "construction-complete-BEFORE-solver"
        for entry in receipt["entries"]
    )
    assert all(
        tuple(entry["stale_prior_fields"]) == boundary.STALE_PRIOR_FIELDS
        for entry in receipt["entries"]
    )
    assert all(
        tuple(entry["context_fields"]) == boundary.CONTEXT_FIELDS
        for entry in receipt["entries"]
    )
    assert receipt["captured_bytes"] < boundary.MAX_INSTANCE_BYTES
    assert [entry["kernel"] for entry in env.observer.receipt()["entries"]] == [
        selected
    ] * friction.FORWARDS
    contact_other_launches = [
        launch
        for launch in env.observer.other_launches
        if launch["kernel"] == env.contact_kernel.key
    ]
    assert len(contact_other_launches) == friction.FORWARDS
    assert (
        sum(
            launch["kernel"] == env.contact_kernel.key
            for launch in env.observer.other_launches
            if launch["forward"] >= boundary.SAMPLED_FORWARDS
        )
        == friction.FORWARDS - boundary.SAMPLED_FORWARDS
    )
    assert (
        len([call for call in env.launch_calls if call[0] is env.contact_kernel]) == 21
    )
    for entry in receipt["entries"]:
        call = entry["contact_call"]
        assert set(call) == {
            "kernel_key",
            "kernel_object_id",
            "kernel_function_id",
            "kernel_code_id",
            "module_object_id",
            "factory_object_id",
            "factory_function_id",
            "factory_code_id",
            "cone",
            "is_sparse",
            "dim",
            "njmax",
            "njmax_nnz",
            "input_count",
            "output_count",
        }
        assert call == {
            "kernel_key": env.contact_kernel.key,
            "kernel_object_id": id(env.contact_kernel),
            "kernel_function_id": id(env.observer.contact_function),
            "kernel_code_id": id(env.observer.contact_code),
            "module_object_id": id(env.contact_kernel.module),
            "factory_object_id": id(env.observer.contact_factory),
            "factory_function_id": id(env.observer.contact_factory_func),
            "factory_code_id": id(env.observer.contact_factory_code),
            "cone": 0,
            "is_sparse": False,
            "dim": 8192,
            "njmax": 512,
            "njmax_nnz": 10240,
            "input_count": 18,
            "output_count": 6,
        }
        assert call["module_object_id"] != id(env.observer.contact_module)
        assert len(entry["contact"]) == len(boundary.CONTACT_PACKET_ORDER)
        assert entry["complete"]["bytes"] > boundary.EFC_BYTES
        for key in ("contact_before", "contact_after", "complete"):
            packet = entry[key]
            assert len(env.sink[packet["path"]]) == packet["bytes"]
        assert (
            env.sink[entry["contact_before"]["path"]]
            == env.sink[entry["contact_after"]["path"]]
        )
    assert (
        env.observer.receipt()["entries"]
        and len(env.observer.receipt()["entries"]) == 21
    )
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("cone", "plain dense pyramidal cone-zero"),
        ("sparse", "plain dense pyramidal cone-zero"),
        ("capacity", "literal full contact"),
        ("wrong-dim", "friction-first contact dimensions"),
        ("extra-kw", "exact contact launch keyword ABI"),
        ("wrong-abi", "exact18 contact inputs"),
        ("wrong-input", "literal array shape and dtype contact-before"),
        ("wrong-output", "literal array shape and dtype contact-before"),
        ("wrong-shape", "array rank and strides"),
        ("alias", "nonoverlapping actual allocations contact allocation call"),
        (
            "duplicate",
            "friction-first contact dimensions|exactly one sampled contact allocation target",
        ),
        (
            "missing",
            "one target per constraint entry|exactly one sampled contact allocation target",
        ),
    ],
)
def test_malformed_contact_paths_refuse_and_restore_hooks(monkeypatch, mutation, match):
    env = _environment(monkeypatch)
    if mutation == "cone":
        env.model.opt.cone = True
    elif mutation == "sparse":
        env.model.is_sparse = True
    elif mutation == "capacity":
        env.data.naconmax = True
    elif mutation == "wrong-dim":
        env.backend.contact_kwargs = lambda kw: kw | {"dim": True}
    elif mutation == "extra-kw":
        env.backend.contact_kwargs = lambda kw: kw | {"stream": env.observer.stream}
    elif mutation == "wrong-abi":
        old = env.constraint.make_constraint

        def wrong_abi(model, data):
            kwargs = {"dim": 8192, "inputs": [], "outputs": []}
            env.backend.launch(env.contact_kernel, **kwargs)
            return old(model, data)

        env.constraint.make_constraint = wrong_abi
        env.observer.make = wrong_abi
        env.observer.make_code = wrong_abi.__code__
    elif mutation == "wrong-input":
        env.contact.dist = env.arrays["contact.worldid"]
    elif mutation == "wrong-output":
        env.contact.efc_address = env.arrays["contact.dim"]
    elif mutation == "wrong-shape":
        env.contact.pos.shape = (8192, 3)
    elif mutation == "alias":
        env.contact.geomcollisionid.ptr = env.contact.dist.ptr
    elif mutation == "duplicate":
        old = env.constraint.make_constraint

        def duplicate(model, data):
            result = old(model, data)
            call = next(c for c in env.launch_calls if c[0] is env.contact_kernel)
            env.backend.launch(call[0], **call[2])
            return result

        env.constraint.make_constraint = duplicate
        env.observer.make = duplicate
        env.observer.make_code = duplicate.__code__
    elif mutation == "missing":
        env.constraint.make_constraint = lambda model, data: None
        env.observer.make = env.constraint.make_constraint
        env.observer.make_code = env.observer.make.__code__
    with pytest.raises(ValueError, match=match):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make


@pytest.mark.parametrize("cap", ["packet", "instance", "total"])
def test_capture_caps_fail_closed_and_restore_hooks(monkeypatch, cap):
    env = _environment(monkeypatch)
    if cap == "packet":
        monkeypatch.setattr(boundary, "MAX_PACKET_BYTES", 1)
    elif cap == "instance":
        monkeypatch.setattr(boundary, "MAX_INSTANCE_BYTES", 1)
    else:
        monkeypatch.setattr(boundary, "MAX_ALL_ARMS_BYTES", 1)
    with pytest.raises(ValueError, match="bounded boundary"):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make


def test_context_inputs_are_raw_opaque_and_must_remain_unchanged(monkeypatch):
    env = _environment(monkeypatch)
    env.contact.pos.host.view(np.uint32).flat[0] = 0x7FC00001
    env.contact.frame.host.view(np.uint32).flat[0] = 0x80000000
    before = env.contact.pos.host.tobytes()

    def mutate_context():
        env.contact.pos.host.view(np.uint32).flat[0] ^= 1

    env.backend.after_contact = mutate_context
    with pytest.raises(
        ValueError, match="contact inputs unchanged|contact context fields unchanged"
    ):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.contact.pos.host.tobytes() != before


def test_contact_before_friction_refuses_and_restores_hooks(monkeypatch):
    env = _environment(monkeypatch, contact_first=True)
    with pytest.raises(ValueError, match="friction-first contact dimensions"):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make


def test_sampled_layout_drift_and_foreign_factory_replacement_refuse(monkeypatch):
    env = _environment(monkeypatch)
    with pytest.raises(ValueError, match="stable sampled contact allocation layouts"):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
            env.contact.dist.ptr += 4
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make

    env = _environment(monkeypatch)
    env.backend.contact_factory_result = env.original
    with pytest.raises(ValueError, match="exact cached cone-zero dense contact kernel"):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make

    env = _environment(monkeypatch)

    def foreign_factory(*_args):
        return env.original

    env.backend.after_contact = lambda: setattr(
        env.constraint, "_efc_contact_init", foreign_factory
    )
    with pytest.raises(ValueError, match="held cached contact factory and kernel code"):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.constraint._efc_contact_init is foreign_factory
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make


@pytest.mark.parametrize("replacement", ["kernel-code", "python-module", "warp-module"])
def test_contact_kernel_code_and_module_replacements_refuse(monkeypatch, replacement):
    env = _environment(monkeypatch)
    function = env.contact_kernel.func
    module_name = function.__module__
    original_warp_module = env.contact_kernel.module
    if replacement == "kernel-code":
        original_code = function.__code__

        def substitute():
            return None

        function.__code__ = substitute.__code__
        message = "held cached contact factory and kernel code"
    elif replacement == "python-module":
        monkeypatch.setitem(sys.modules, module_name, object())
        message = "held contact kernel module identity"
    else:
        env.contact_kernel.module = object()
        message = "held cached contact Warp kernel module"
    try:
        with pytest.raises(ValueError, match=message):
            with env.observer:
                env.observer.constraint.make_constraint(env.model, env.data)
    finally:
        if replacement == "kernel-code":
            function.__code__ = original_code
        elif replacement == "warp-module":
            env.contact_kernel.module = original_warp_module
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make


def test_sink_exception_and_foreign_launch_are_never_hidden_or_overwritten(monkeypatch):
    env = _environment(monkeypatch)
    env.observer.sink = lambda _name, _raw: (_ for _ in ()).throw(
        RuntimeError("sink failure")
    )
    with pytest.raises(RuntimeError, match="sink failure"):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is env.observer.launch
    assert env.constraint.make_constraint is env.observer.make

    env = _environment(monkeypatch)

    def foreign(*args, **kwargs):
        return None

    env.backend.after_contact = lambda: setattr(env.backend, "launch", foreign)
    with pytest.raises(ValueError, match="foreign hook not overwritten during close"):
        with env.observer:
            env.observer.constraint.make_constraint(env.model, env.data)
    assert env.backend.launch is foreign
