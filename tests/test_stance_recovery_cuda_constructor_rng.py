"""Synthetic CUDA backend tests for one-shot constructor RNG accounting.

These tests prove only the Python guards and retained-state contract. They do
not initialize native CUDA, sample actions, or qualify a simulator rollout.
"""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from mjlab_microduck import stance_recovery_cuda_constructor_rng as constructor_rng
from mjlab_microduck import stance_recovery_cuda_rng_scope as sampler
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime

SYNTHETIC_BACKEND = "SYNTHETIC_MOCKED_CUDA"
SOURCE = "a" * 40
LEASE_FD = 19


def declaration(source=SOURCE):
    return schedule.declaration(source, "dose", "training", ["zero-wrench"] * 64)


def _state(seed, size=16):
    return torch.tensor([(seed + i) % 256 for i in range(size)], dtype=torch.uint8)


def _nominal_values():
    return {
        key: torch.full((64, 1), value, dtype=torch.float32)
        for key, value in constructor_rng.NOMINAL.items()
    }


@pytest.fixture
def synthetic_cuda(monkeypatch):
    """Install API-shaped CUDA behavior without querying or initializing CUDA."""
    profile = sampler.execution.select(sampler.execution.WSL)
    monkeypatch.setattr(sampler.execution, "PROFILE", profile)
    monkeypatch.setattr(sampler.execution, "ROOT", Path.cwd().resolve())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    fake = {
        "initialized": True,
        "count": 1,
        "current": 0,
        "default": _state(41),
        "cuda_calls": [],
        "lease_calls": 0,
        "generator_seeds": [],
        "runtime_calls": [],
        "identity_tag": "synthetic-host-a",
        "runtime_error": None,
        "runtime_hook": None,
    }
    cpu_generator = torch.Generator

    def inherited_lease(fd):
        fake["lease_calls"] += 1
        if fd != LEASE_FD:
            raise ValueError("SYNTHETIC inherited lease refused")

    monkeypatch.setattr(constructor_rng.training, "inherited_lease", inherited_lease)
    monkeypatch.setattr(sampler.training_smoke, "inherited_lease", inherited_lease)
    monkeypatch.setattr(sampler.host, "read", lambda *args: SOURCE)
    monkeypatch.setattr(
        sampler.host,
        "identity",
        lambda source: {
            "source": source,
            "execution_profile": deepcopy(sampler.execution.PROFILE),
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
            raise ValueError("SYNTHETIC unexpected CUDA device")
        return fake["default"].clone()

    def set_rng_state(state, device=0):
        fake["cuda_calls"].append("set_rng_state")
        if device not in (0, "cuda:0", torch.device("cuda:0")):
            raise ValueError("SYNTHETIC unexpected CUDA device")
        fake["default"] = state.detach().cpu().clone()

    for name, value in (
        ("is_initialized", initialized),
        ("device_count", device_count),
        ("current_device", current_device),
        ("get_rng_state", get_rng_state),
        ("set_rng_state", set_rng_state),
    ):
        monkeypatch.setattr(sampler.torch.cuda, name, value)

    class SyntheticGenerator:
        def __init__(self, device="cpu"):
            self.device = str(device)
            self.value = None

        def manual_seed(self, seed):
            fake["generator_seeds"].append((self.device, seed))
            if self.device == "cpu":
                self.value = cpu_generator(device="cpu").manual_seed(seed).get_state()
            elif self.device == "cuda:0":
                self.value = _state(seed)
            else:
                raise ValueError("SYNTHETIC unexpected private generator device")
            return self

        def get_state(self):
            return self.value.clone()

    monkeypatch.setattr(sampler.torch, "Generator", SyntheticGenerator)

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

    monkeypatch.setattr(torch.random, "fork_rng", synthetic_fork_rng)

    class SyntheticRuntime:
        def __init__(self, value, *, device, solved_field_check):
            fake["runtime_calls"].append((deepcopy(value), device, solved_field_check))
            if fake["runtime_hook"] is not None:
                fake["runtime_hook"]()
            torch.rand(7)
            fake["default"] = (
                (fake["default"].to(torch.int16) + 1).remainder(256).to(torch.uint8)
            )
            if fake["runtime_error"] is not None:
                raise fake["runtime_error"]
            self.motor = type("Motor", (), {"nominal_parameters": _nominal_values()})()

    monkeypatch.setattr(constructor_rng, "ScheduledRecoveryRuntime", SyntheticRuntime)
    return fake, cpu_generator


def make_scope():
    return constructor_rng.CudaConstructorRng(SOURCE, lease_fd=LEASE_FD)


def _prepared_caller(receipt):
    return {
        key: value.detach().clone() for key, value in receipt["caller_states"].items()
    }


def test_constructor_uses_distinct_fixed_private_seeds_and_advances_both_streams(
    synthetic_cuda,
):
    fake, cpu_generator = synthetic_cuda
    caller_cpu = torch.random.get_rng_state().clone()
    caller_cuda = fake["default"].clone()
    scope = make_scope()
    env = scope.construct(declaration())
    receipt = scope.receipt
    assert receipt["status"] == "success" and receipt["faulted"] is False
    assert receipt["cpu_seed"] == constructor_rng.CPU_SEED == 977
    assert receipt["cuda_seed"] == constructor_rng.CUDA_SEED == 983
    assert receipt["cpu_seed"] not in sampler.SEEDS
    assert receipt["cuda_seed"] not in sampler.SEEDS
    assert fake["generator_seeds"] == [("cpu", 977), ("cuda:0", 983)]
    assert fake["runtime_calls"] == [(declaration(), "cuda:0", "packed")]
    assert receipt["constructor_calls"] == 1 and receipt["worlds"] == 64
    assert torch.equal(torch.random.get_rng_state(), caller_cpu)
    assert torch.equal(fake["default"], caller_cuda)
    assert torch.equal(
        receipt["private_states"]["cpu_start"],
        cpu_generator(device="cpu").manual_seed(constructor_rng.CPU_SEED).get_state(),
    )
    assert not torch.equal(
        receipt["private_states"]["cpu_start"], receipt["private_states"]["cpu_end"]
    )
    assert not torch.equal(
        receipt["private_states"]["cuda_start"], receipt["private_states"]["cuda_end"]
    )
    assert receipt["caller_cpu_preserved"] is True
    assert receipt["caller_cuda_preserved"] is True
    assert all(
        torch.equal(env.motor.nominal_parameters[key], value)
        for key, value in _nominal_values().items()
    )
    assert all(receipt[key] is False for key in constructor_rng.FLAGS)
    assert fake["lease_calls"] >= 3
    assert SYNTHETIC_BACKEND.startswith("SYNTHETIC_")
    with pytest.raises(ValueError, match="one-shot"):
        scope.construct(declaration())
    assert len(fake["runtime_calls"]) == 1


def test_constructor_receipt_returns_deep_clones_and_cpu_checker_accepts_exact_record(
    monkeypatch, synthetic_cuda
):
    fake, _ = synthetic_cuda
    scope = make_scope()
    scope.construct(declaration())
    receipt = scope.receipt
    prepared = _prepared_caller(receipt)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fake["initialized"] = False
    score = constructor_rng.check(receipt, source=SOURCE, prepared_caller=prepared)
    assert score["protocol"] == constructor_rng.PROTOCOL + ":score"
    assert score["caller_streams_preserved"] is True
    assert score["nominal_fields_exact"] is True
    assert score["private_cpu_advanced"] is True
    assert score["private_cuda_advanced"] is True
    assert all(score[key] is False for key in constructor_rng.FLAGS)

    mutable = scope.receipt
    mutable["caller_states"]["cpu_before"].zero_()
    mutable["private_states"]["cuda_end"].zero_()
    mutable["nominal_parameters"]["kp_scale"].zero_()
    assert scope.receipt["caller_states"]["cpu_before"].any()
    assert scope.receipt["private_states"]["cuda_end"].any()
    assert (scope.receipt["nominal_parameters"]["kp_scale"] == 1.0).all()

    # The checker may perform its single CUDA initialization-state guard, but
    # must not inspect device topology or any RNG state / CUDA generator.
    fake["cuda_calls"].clear()
    score = constructor_rng.check(receipt, source=SOURCE, prepared_caller=prepared)
    assert score["protocol"] == constructor_rng.PROTOCOL + ":score"
    assert fake["cuda_calls"] == ["is_initialized"]


@pytest.mark.parametrize("bad", ["cpu_end", "caller", "nominal", "seed", "state_type"])
def test_cpu_checker_rejects_tampered_constructor_receipt(
    monkeypatch, synthetic_cuda, bad
):
    fake, _ = synthetic_cuda
    scope = make_scope()
    scope.construct(declaration())
    receipt = scope.receipt
    prepared = _prepared_caller(receipt)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fake["initialized"] = False
    if bad == "cpu_end":
        receipt["private_states"]["cpu_end"][0] ^= 1
    elif bad == "caller":
        prepared["cpu_after"][0] ^= 1
    elif bad == "nominal":
        receipt["nominal_parameters"]["vin_tensor"][0, 0] += 1
    elif bad == "seed":
        receipt["cuda_seed"] = True
    else:
        receipt["private_states"]["cuda_end"] = torch.zeros(16, dtype=torch.float32)
    with pytest.raises(ValueError):
        constructor_rng.check(receipt, source=SOURCE, prepared_caller=prepared)


def test_lease_refusal_precedes_any_cuda_query(monkeypatch, synthetic_cuda):
    fake, _ = synthetic_cuda
    fake["cuda_calls"].clear()

    def refuse(_fd):
        raise ValueError("SYNTHETIC constructor lease refused")

    monkeypatch.setattr(constructor_rng.training, "inherited_lease", refuse)
    with pytest.raises(ValueError, match="constructor lease refused"):
        constructor_rng.CudaConstructorRng(SOURCE, lease_fd=LEASE_FD)
    assert fake["cuda_calls"] == []


@pytest.mark.parametrize(
    "change,match",
    [
        ("hidden", "isolated exact WSL constructor context"),
        ("profile", "isolated exact WSL constructor context"),
        ("root", "isolated exact WSL constructor context"),
        ("uninitialized", "already initialized single visible CUDA0"),
        ("multiple", "already initialized single visible CUDA0"),
        ("wrong-current", "already initialized single visible CUDA0"),
    ],
)
def test_constructor_refuses_wrong_context_or_cuda_initialization_before_scope(
    monkeypatch, synthetic_cuda, change, match
):
    fake, _ = synthetic_cuda
    if change == "hidden":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    elif change == "profile":
        profile = deepcopy(sampler.execution.PROFILE)
        profile["driver"] = "SYNTHETIC_WRONG_PROFILE"
        monkeypatch.setattr(sampler.execution, "PROFILE", profile)
    elif change == "root":
        monkeypatch.setattr(sampler.execution, "ROOT", Path("/tmp/not-the-worktree"))
    elif change == "uninitialized":
        fake["initialized"] = False
    elif change == "multiple":
        fake["count"] = 2
    else:
        fake["current"] = 1
    with pytest.raises(ValueError, match=match):
        make_scope()


def test_constructor_failure_restores_callers_records_fault_and_cannot_be_reused(
    synthetic_cuda,
):
    fake, _ = synthetic_cuda
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = fake["default"].clone()
    fake["runtime_error"] = RuntimeError("SYNTHETIC constructor failure")
    scope = make_scope()
    with pytest.raises(RuntimeError, match="constructor failure"):
        scope.construct(declaration())
    receipt = scope.receipt
    assert receipt["status"] == "faulted" and receipt["faulted"] is True
    assert receipt["constructor_calls"] == 1
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["error"] == "SYNTHETIC constructor failure"
    assert torch.equal(torch.random.get_rng_state(), before_cpu)
    assert torch.equal(fake["default"], before_cuda)
    assert receipt["caller_cpu_preserved"] is True
    assert receipt["caller_cuda_preserved"] is True
    assert not torch.equal(
        receipt["private_states"]["cpu_start"], receipt["private_states"]["cpu_end"]
    )
    assert not torch.equal(
        receipt["private_states"]["cuda_start"], receipt["private_states"]["cuda_end"]
    )
    with pytest.raises(ValueError, match="one-shot"):
        scope.construct(declaration())


def test_context_drift_during_constructor_restores_callers_and_faults(synthetic_cuda):
    fake, _ = synthetic_cuda
    caller_cpu = torch.random.get_rng_state().clone()
    caller_cuda = fake["default"].clone()
    fake["runtime_hook"] = lambda: fake.update(identity_tag="synthetic-host-changed")
    scope = make_scope()
    with pytest.raises(ValueError, match="unchanged closed constructor context"):
        scope.construct(declaration())
    assert scope.receipt["status"] == "faulted"
    assert torch.equal(torch.random.get_rng_state(), caller_cpu)
    assert torch.equal(fake["default"], caller_cuda)


def test_swallowed_same_instance_reentry_faults_outer_constructor(synthetic_cuda):
    fake, _ = synthetic_cuda
    holder = {}

    def reenter():
        with pytest.raises(ValueError, match="one-shot and nonreentrant"):
            holder["scope"].construct(declaration())

    fake["runtime_hook"] = reenter
    scope = make_scope()
    holder["scope"] = scope
    with pytest.raises(ValueError, match="constructor fault cannot be swallowed"):
        scope.construct(declaration())
    assert scope.receipt["faulted"] is True
    assert scope.receipt["status"] == "faulted"


def test_constructor_refuses_while_shared_sampler_default_stream_lock_is_held(
    synthetic_cuda,
):
    scope = make_scope()
    assert sampler._PROCESS_CUDA0_SCOPE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(ValueError, match="cannot borrow an active sampler"):
            make_scope()
        with pytest.raises(ValueError, match="another CUDA default-stream scope"):
            scope.construct(declaration())
        assert scope.receipt["faulted"] is True
        assert scope.receipt["constructor_calls"] == 0
    finally:
        sampler._PROCESS_CUDA0_SCOPE_LOCK.release()


def test_constructor_requires_isolated_main_thread_before_cuda(
    monkeypatch, synthetic_cuda
):
    fake, _ = synthetic_cuda
    fake["cuda_calls"].clear()

    def refuse():
        raise ValueError("SYNTHETIC nonisolated constructor thread")

    monkeypatch.setattr(sampler, "_require_isolated_main_thread", refuse)
    with pytest.raises(ValueError, match="nonisolated constructor thread"):
        make_scope()
    assert fake["cuda_calls"] == []


def test_failed_fork_restoration_faults_without_manual_rng_repair(
    monkeypatch, synthetic_cuda
):
    fake, _ = synthetic_cuda
    scope = make_scope()
    before = fake["default"].clone()

    @contextmanager
    def broken_fork(*, devices):
        assert devices == [0]
        cpu = torch.random.get_rng_state().clone()
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu)
            # Deliberately leave mocked CUDA changed. Production must refuse,
            # not manually repair it and relabel the receipt as success.

    monkeypatch.setattr(torch.random, "fork_rng", broken_fork)
    with pytest.raises(ValueError, match="restores constructor caller streams exactly"):
        scope.construct(declaration())
    assert scope.receipt["status"] == "faulted"
    assert scope.receipt["caller_cuda_preserved"] is False
    assert not torch.equal(before, fake["default"])


def test_wrong_schedule_source_faults_before_runtime_constructor(synthetic_cuda):
    fake, _ = synthetic_cuda
    scope = make_scope()
    with pytest.raises(
        ValueError, match="exact source-bound CUDA64 constructor schedule"
    ):
        scope.construct(declaration("b" * 40))
    assert fake["runtime_calls"] == []
    assert scope.receipt["status"] == "faulted"
    assert scope.receipt["constructor_calls"] == 0


def test_actual_two_world_cpu_runtime_reports_reproducible_nominal_values():
    declaration_two = schedule.declaration(
        SOURCE, "dose", "training", ["zero-wrench", "zero-wrench"]
    )
    cpu_before = torch.random.get_rng_state().clone()
    start = torch.Generator(device="cpu").manual_seed(977).get_state()
    results = []
    for _ in range(2):
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(start.clone())
            runtime = ScheduledRecoveryRuntime(
                declaration_two, device="cpu", solved_field_check="packed"
            )
            actual = {
                key: value.detach().clone()
                for key, value in runtime.motor.nominal_parameters.items()
            }
            results.append(
                (
                    actual,
                    torch.random.get_rng_state().clone(),
                    runtime._control_snapshot(),
                    runtime.snapshot()["qpos"],
                )
            )
        assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert set(actual) == set(constructor_rng.NOMINAL)
    for key, expected in constructor_rng.NOMINAL.items():
        assert actual[key].shape == (2, 1)
        assert actual[key].dtype == torch.float32
        assert torch.isfinite(actual[key]).all()
        assert (actual[key] == expected).all()
        assert torch.equal(results[0][0][key], results[1][0][key])
    assert torch.equal(results[0][1], results[1][1])
    assert not torch.equal(start, results[0][1])
    assert torch.equal(results[0][3], results[1][3])
    for key in results[0][2]:
        assert torch.equal(results[0][2][key], results[1][2][key])
