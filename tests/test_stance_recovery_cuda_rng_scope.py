"""Synthetic-only source tests for private CUDA0 RNG scoping.

The mocked CUDA backend below is explicitly SYNTHETIC. These tests prove only
the Python state handoff and refusal behavior; they make no native CUDA claim.
"""

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import Thread

import pytest
import torch

from mjlab_microduck import stance_recovery_cuda_rng_scope as scope_module

SYNTHETIC_BACKEND = "SYNTHETIC_MOCKED_CUDA"
SOURCE = "a" * 40
LEASE_FD = 19


def _seed_state(seed):
    return torch.tensor(
        [seed & 255, (seed + 1) & 255, 7, 11, 13, 17, 19, 23], dtype=torch.uint8
    )


@pytest.fixture
def synthetic_cuda(monkeypatch):
    """Install a SYNTHETIC CUDA API without initializing or querying real CUDA."""
    profile = scope_module.execution.select(scope_module.execution.WSL)
    monkeypatch.setattr(scope_module.execution, "PROFILE", profile)
    monkeypatch.setattr(scope_module.execution, "ROOT", Path.cwd().resolve())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    valid_cpu_state = torch.Generator().manual_seed(8).get_state()

    fake = {
        "initialized": True,
        "count": 1,
        "current": 0,
        "default": torch.tensor([91, 92, 93, 94, 95, 96, 97, 98], dtype=torch.uint8),
        "lease_calls": 0,
        "cuda_calls": [],
        "source": SOURCE,
        "lease_valid": True,
        "identity_tag": "identity-a",
        "mutate_generator": False,
        "valid_cpu_state": valid_cpu_state,
    }

    def inherited_lease(fd):
        fake["lease_calls"] += 1
        if fd != LEASE_FD:
            raise ValueError("SYNTHETIC lease refusal")
        if not fake["lease_valid"]:
            raise ValueError("SYNTHETIC lease lost")

    monkeypatch.setattr(scope_module.training_smoke, "inherited_lease", inherited_lease)
    monkeypatch.setattr(scope_module.host, "read", lambda *args: fake["source"])
    monkeypatch.setattr(
        scope_module.host,
        "identity",
        lambda source: {
            "source": source,
            "execution_profile": deepcopy(scope_module.execution.PROFILE),
            "tag": fake["identity_tag"],
        },
    )

    def initialized():
        fake["cuda_calls"].append("is_initialized")
        return fake["initialized"]

    def device_count():
        fake["cuda_calls"].append("device_count")
        return fake["count"]

    def current_device():
        fake["cuda_calls"].append("current_device")
        return fake["current"]

    def get_rng_state(device=0):
        fake["cuda_calls"].append("get_rng_state")
        if device not in (0, "cuda:0", torch.device("cuda:0")):
            raise ValueError("SYNTHETIC wrong CUDA device")
        return fake["default"].clone()

    def set_rng_state(state, device=0):
        fake["cuda_calls"].append("set_rng_state")
        if device not in (0, "cuda:0", torch.device("cuda:0")):
            raise ValueError("SYNTHETIC wrong CUDA device")
        fake["default"] = state.detach().cpu().clone()

    monkeypatch.setattr(scope_module.torch.cuda, "is_initialized", initialized)
    monkeypatch.setattr(scope_module.torch.cuda, "device_count", device_count)
    monkeypatch.setattr(scope_module.torch.cuda, "current_device", current_device)
    monkeypatch.setattr(scope_module.torch.cuda, "get_rng_state", get_rng_state)
    monkeypatch.setattr(scope_module.torch.cuda, "set_rng_state", set_rng_state)

    class SyntheticGenerator:
        def __init__(self, device):
            fake["cuda_calls"].append("private_generator:" + str(device))
            if str(device) != "cuda:0":
                raise ValueError("SYNTHETIC wrong private generator device")
            self._state = None

        def manual_seed(self, seed):
            self._state = _seed_state(seed)
            if fake["mutate_generator"]:
                torch.random.set_rng_state(fake["valid_cpu_state"])
                fake["default"] = torch.zeros_like(fake["default"])
            return self

        def get_state(self):
            return self._state.clone()

    monkeypatch.setattr(scope_module.torch, "Generator", SyntheticGenerator)

    @contextmanager
    def synthetic_fork_rng(*, devices):
        assert devices == [0]
        cpu = torch.random.get_rng_state().clone()
        cuda = fake["default"].clone()
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu)
            fake["default"] = cuda

    monkeypatch.setattr(scope_module.torch.random, "fork_rng", synthetic_fork_rng)
    return fake


def make_scope(state=None, *, seed=653, lease_fd=LEASE_FD):
    return scope_module.CudaPrivateRngScope(
        seed, _seed_state(seed) if state is None else state, lease_fd=lease_fd
    )


def test_synthetic_scope_advances_private_state_and_restores_both_callers(
    synthetic_cuda,
):
    fake = synthetic_cuda
    original_cpu = torch.random.get_rng_state().clone()
    original_cuda = fake["default"].clone()
    caller_state = _seed_state(653)
    scope = make_scope(caller_state)
    caller_state.zero_()
    exposed = scope.state
    exposed.zero_()
    assert torch.equal(scope.state, _seed_state(653))
    assert scope.seed == 653
    with scope.scope():
        assert scope.receipt["scope_active"] is True
        assert torch.equal(fake["default"], _seed_state(653))
        torch.random.set_rng_state(fake["valid_cpu_state"])
        fake["default"] = torch.tensor(
            [31, 32, 33, 34, 35, 36, 37, 38], dtype=torch.uint8
        )
    assert torch.equal(
        scope.state, fake["default"].new_tensor([31, 32, 33, 34, 35, 36, 37, 38])
    )
    assert torch.equal(torch.random.get_rng_state(), original_cpu)
    assert torch.equal(fake["default"], original_cuda)
    receipt = scope.receipt
    assert (
        receipt["private_state_sha256"]
        == sha256(scope.state.numpy().tobytes()).hexdigest()
    )
    assert receipt["caller_cpu_rng_preserved"] is True
    assert receipt["caller_cuda_rng_preserved"] is True
    assert receipt["native_cuda_rng_scope_qualified"] is False
    assert receipt["native_cuda_sampling_qualified"] is False
    assert all(
        value is False
        for key, value in receipt.items()
        if key in scope_module.FALSE_FLAGS
    )
    assert fake["lease_calls"] >= 3
    assert SYNTHETIC_BACKEND.startswith("SYNTHETIC_")


def test_synthetic_scope_captures_advanced_state_and_faults_on_body_error(
    synthetic_cuda,
):
    fake = synthetic_cuda
    original_cpu = torch.random.get_rng_state().clone()
    original_cuda = fake["default"].clone()
    scope = make_scope(seed=659)
    advanced = torch.tensor([41, 42, 43, 44, 45, 46, 47, 48], dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="synthetic action failure"):
        with scope.scope():
            fake["default"] = advanced.clone()
            raise RuntimeError("synthetic action failure")
    assert torch.equal(scope.state, advanced)
    assert torch.equal(torch.random.get_rng_state(), original_cpu)
    assert torch.equal(fake["default"], original_cuda)
    assert scope.receipt["faulted"] is True
    with pytest.raises(ValueError, match="cannot be reused"):
        with scope.scope():
            pass
    recovered = make_scope(seed=659)
    with recovered.scope():
        fake["default"] = _seed_state(659) + 1
    assert recovered.receipt["scope_count"] == 1


def test_synthetic_scope_rejects_reentry_and_faults_outer_scope(synthetic_cuda):
    scope = make_scope()
    with pytest.raises(ValueError, match="not reentrant or concurrent"):
        with scope.scope():
            with scope.scope():
                pytest.fail("reentrant scope must be rejected")
    assert scope.receipt["faulted"] is True


def test_synthetic_swallowed_same_object_reentry_faults_outer(synthetic_cuda):
    scope = make_scope()
    with pytest.raises(ValueError, match="swallowed reentry"):
        with scope.scope():
            with pytest.raises(ValueError, match="not reentrant"):
                with scope.scope():
                    pass
    assert scope.receipt["faulted"] is True
    assert scope.receipt["scope_count"] == 0


def test_synthetic_different_instance_scope_is_refused_and_process_lock_released(
    synthetic_cuda,
):
    outer, inner = make_scope(seed=653), make_scope(seed=659)
    with outer.scope():
        with pytest.raises(ValueError, match="another private CUDA0"):
            with inner.scope():
                pass
    assert inner.receipt["faulted"] is True
    assert outer.receipt["scope_count"] == 1
    following = make_scope(seed=659)
    with following.scope():
        synthetic_cuda["default"] = _seed_state(659) + 1
    assert following.receipt["scope_count"] == 1


def test_synthetic_sequential_scopes_resume_advanced_private_state(synthetic_cuda):
    scope = make_scope()
    first = _seed_state(653) + 1
    second = _seed_state(653) + 2
    with scope.scope():
        synthetic_cuda["default"] = first.clone()
    with scope.scope():
        assert torch.equal(synthetic_cuda["default"], first)
        synthetic_cuda["default"] = second.clone()
    assert torch.equal(scope.state, second)
    assert scope.receipt["scope_count"] == 2


def test_synthetic_constructor_detects_and_restores_generator_side_effects(
    synthetic_cuda,
):
    fake = synthetic_cuda
    original_cpu = torch.random.get_rng_state().clone()
    original_cuda = fake["default"].clone()
    fake["mutate_generator"] = True
    with pytest.raises(ValueError, match="preserves caller RNG streams"):
        make_scope()
    assert torch.equal(torch.random.get_rng_state(), original_cpu)
    assert torch.equal(fake["default"], original_cuda)


def test_bad_lease_is_refused_before_any_cuda_query(monkeypatch, synthetic_cuda):
    fake = synthetic_cuda
    fake["cuda_calls"].clear()

    def reject_lease(_fd):
        raise ValueError("lease not held")

    monkeypatch.setattr(scope_module.training_smoke, "inherited_lease", reject_lease)
    with pytest.raises(ValueError, match="lease not held"):
        make_scope()
    assert fake["cuda_calls"] == []


def test_synthetic_scope_refuses_lost_lease_during_scope(synthetic_cuda):
    fake = synthetic_cuda
    scope = make_scope()
    with pytest.raises(ValueError, match="lease lost"):
        with scope.scope():
            fake["lease_valid"] = False
    assert scope.receipt["faulted"] is True


@pytest.mark.parametrize("bad_seed", [True, 653.0, "653"])
def test_synthetic_scope_rejects_non_exact_seed_type(synthetic_cuda, bad_seed):
    with pytest.raises(ValueError, match="fixed private CUDA learner seed"):
        make_scope(seed=bad_seed, state=_seed_state(653))


@pytest.mark.parametrize(
    "state",
    [torch.empty((0,), dtype=torch.uint8), torch.zeros((7,), dtype=torch.uint8)],
)
def test_synthetic_scope_rejects_empty_or_wrong_length_state(synthetic_cuda, state):
    with pytest.raises(ValueError, match="CPU uint8|seed-bound"):
        make_scope(state=state)


def test_synthetic_scope_refuses_bad_fd_and_wrong_worktree_before_cuda(
    monkeypatch, synthetic_cuda
):
    fake = synthetic_cuda
    fake["cuda_calls"].clear()
    with pytest.raises(ValueError, match="inherited shared GPU lease"):
        make_scope(lease_fd=True)
    assert fake["cuda_calls"] == []
    with monkeypatch.context() as patch:
        patch.setattr(scope_module.Path, "cwd", lambda: Path("/tmp"))
        with pytest.raises(ValueError, match="exact frozen WSL worktree"):
            make_scope()
    assert fake["cuda_calls"] == []


@pytest.mark.parametrize("field", ["source", "identity_tag"])
def test_synthetic_scope_refuses_context_drift_inside_scope(synthetic_cuda, field):
    fake = synthetic_cuda
    scope = make_scope()
    with pytest.raises(ValueError, match="source and host identity unchanged"):
        with scope.scope():
            fake[field] = "b" * 40 if field == "source" else "identity-b"
    assert scope.receipt["faulted"] is True


def test_synthetic_worker_thread_refused(synthetic_cuda):
    scope = make_scope()
    result = []

    def attempt():
        try:
            with scope.scope():
                pass
        except ValueError as exc:
            result.append(str(exc))

    worker = Thread(target=attempt)
    worker.start()
    worker.join()
    assert result and "isolated main Python thread" in result[0]
    assert scope.receipt["faulted"] is True


def test_synthetic_thread_count_drift_at_constructor_and_exit_is_refused(
    monkeypatch, synthetic_cuda
):
    fake = synthetic_cuda
    fake["thread_count"] = 2
    monkeypatch.setattr(scope_module, "active_count", lambda: fake["thread_count"])
    fake["cuda_calls"].clear()
    with pytest.raises(ValueError, match="isolated main Python thread"):
        make_scope()
    assert fake["cuda_calls"] == []

    fake["thread_count"] = 1
    scope = make_scope()
    original_cpu = torch.random.get_rng_state().clone()
    original_cuda = fake["default"].clone()
    with pytest.raises(ValueError, match="isolated main Python thread"):
        with scope.scope():
            torch.random.set_rng_state(fake["valid_cpu_state"])
            fake["thread_count"] = 2
    fake["thread_count"] = 1
    assert torch.equal(torch.random.get_rng_state(), original_cpu)
    assert torch.equal(fake["default"], original_cuda)
    assert scope.receipt["scope_count"] == 0


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda fake: fake.update(initialized=False), "already be initialized"),
        (lambda fake: fake.update(count=2), "single visible CUDA0"),
        (lambda fake: fake.update(current=1), "single visible CUDA0"),
    ],
)
def test_synthetic_scope_refuses_wrong_initialized_device(
    synthetic_cuda, mutate, match
):
    fake = synthetic_cuda
    mutate(fake)
    with pytest.raises(ValueError, match=match):
        make_scope()


def test_synthetic_scope_rejects_visibility_profile_seed_and_state_mismatch(
    monkeypatch, synthetic_cuda
):
    with monkeypatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        with pytest.raises(ValueError, match="only visible device"):
            make_scope()
    with monkeypatch.context() as patch:
        profile = deepcopy(scope_module.execution.PROFILE)
        profile["driver"] = "wrong"
        patch.setattr(scope_module.execution, "PROFILE", profile)
        with pytest.raises(ValueError, match="frozen WSL10098"):
            make_scope()
    with pytest.raises(ValueError, match="fixed private CUDA learner seed"):
        make_scope(seed=651)
    with pytest.raises(ValueError, match="exact seed-bound"):
        make_scope(state=_seed_state(659))
    with pytest.raises(ValueError, match="nonempty one-dimensional CPU uint8"):
        make_scope(state=torch.zeros(8, dtype=torch.float32))
    with pytest.raises(ValueError, match="nonempty one-dimensional CPU uint8"):
        make_scope(state=torch.zeros((2, 4), dtype=torch.uint8))


def test_synthetic_scope_seed_is_read_only_and_receipts_are_detached(synthetic_cuda):
    scope = make_scope()
    state = scope.state
    receipt = scope.receipt
    state.zero_()
    receipt["private_state_sha256"] = "f" * 64
    with pytest.raises(AttributeError):
        scope.seed = 659
    assert torch.equal(scope.state, _seed_state(653))
    assert scope.receipt["private_state_sha256"] != "f" * 64
    assert scope.receipt["scope_active"] is False
    assert scope.receipt["state_bytes"] == len(_seed_state(653))


def test_synthetic_scope_never_calls_global_seed_apis(monkeypatch, synthetic_cuda):
    def forbidden(*_args, **_kwargs):
        pytest.fail("global RNG reseeding is forbidden")

    monkeypatch.setattr(scope_module.torch, "manual_seed", forbidden)
    monkeypatch.setattr(scope_module.torch.cuda, "manual_seed", forbidden)
    monkeypatch.setattr(scope_module.torch.cuda, "manual_seed_all", forbidden)
    scope = make_scope()
    with scope.scope():
        synthetic_cuda["default"] = torch.tensor(
            [51, 52, 53, 54, 55, 56, 57, 58], dtype=torch.uint8
        )
